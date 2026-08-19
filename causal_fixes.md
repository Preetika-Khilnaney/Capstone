# Causal Engine Fixes

Diagnosis: the causal engine produces random/self-pointing graphs dominated by autoregressive links (speed at t depends on speed at t-1, trivially true). Root causes span config, code bugs, missing variables, and noisy data.

---

## Fix 1: Filter all self-links (autoregression bug)

**File**: `app/pipeline/causal.py:267-268`

**Current code**:
```python
if i == tj and tau == 0:
    continue
```

**Problem**: Only skips the contemporaneous self-link (lag 0). Autoregressive links at τ>0 (e.g. `tgt_speed(t-1) → tgt_speed(t)`) pass through and dominate the output — velocity autocorrelation is ~0.88-0.91 for all objects, so this is trivially the strongest signal.

**Fix**:
```python
if i == tj:
    continue
```

**Impact**: Pure bug fix. Removes meaningless self-pointing results from all output.

---

## Fix 2: Increase `tau_max` to 15 (1.5 seconds)

**File**: `app/config.py:116`

**Current value**: `tau_max: int = 5` (0.5 seconds at 10 FPS)

**Problem**: Real human car-following reaction time is 1-2 seconds (10-20 frames at 10 FPS). The algorithm cannot detect links at the physically correct lag. The synthetic test (`test_causal.py`) masks this because it uses LAG=5 (0.5s) — unrealistically fast for real traffic.

**Fix**: `tau_max: int = 15` (1.5 seconds)

**Also update** (`config.py:117`): `tau_max_frame_frac: float = 0.2` — so clips ≥75 frames use the full tau_max=15 (currently 0.12 makes the cap redundant for 100-frame clips: `min(15, 12) = 12`).

**Tradeoff**: Larger tau_max = more hypotheses tested. PCMCI+ handles this via its PC algorithm's correction. The current value is simply too small for real traffic physics.

---

## Fix 3: Add `rel_speed` variable, replace `lead_speed`

**File**: `app/pipeline/causal.py`, `_build_variables` function (~line 141-191)

**Current variables**: `tgt_speed`, `lead_gap`, `lead_speed`, `nn_gap`, `nn_speed`

**Problem**: In car-following theory (IDM, Wiedemann), the causal variable is `relative_speed = lead_speed - target_speed`, not lead_speed alone. Both lead_speed and tgt_speed are highly autocorrelated (~0.9), so ParCorr struggles to separate their independent contributions. The relative speed is the actual signal that triggers braking.

**Fix**:
1. In `_build_variables`, after computing `lead_speed` and `tgt_speed`, add:
   ```python
   rel_speed = lead_speed - tgt_speed
   ```
2. Return `rel_speed` instead of `lead_speed` in the dict (replace, don't add — avoids multicollinearity with ParCorr since `rel_speed` is a linear combination of the other two):
   ```python
   return {"tgt_speed": tgt_speed, "lead_gap": lead_gap, "rel_speed": rel_speed,
           "nn_gap": nn_gap, "nn_speed": nn_speed}
   ```

**Tradeoff**: Replaces lead_speed (loses raw lead speed signal). Keeping both would let ParCorr decide, but adds redundancy and multicollinearity risk. Relative speed is strictly more informative for car-following causality.

---

## Fix 4: Fix Savitzky-Golay NaN handling

**File**: `app/pipeline/handoff.py:28-39`

**Current code**:
```python
def _smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window < 5 or arr.size <= window or not np.all(np.isfinite(arr)):
        return arr
    ...
```

**Problem**: If there is even ONE NaN in a position series (from depth gating), the entire Savitzky-Golay pass is skipped. Objects with depth-gated positions get unsmoothed, noisy velocity. Some objects hit the 40 m/s cap, indicating spike contamination.

**Fix**:
```python
def _smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window < 5 or arr.size <= window:
        return arr
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() <= window:
        return arr
    if window % 2 == 0:
        window += 1
    smoothed = arr.copy()
    smoothed[finite_mask] = savgol_filter(arr[finite_mask], window, polyorder=2)
    return smoothed
```

**Tradeoff**: Slightly more complex, but preserves the smoothing benefit for objects with partial depth-gating. Smoothing only the finite portion is mathematically sound — Savitzky-Golay on contiguous finite segments works correctly.

---

## Fix 5: Raise `min_track_frames` from 3 to 8

**File**: `app/config.py:78`

**Current value**: `min_track_frames: int = 3`

**Problem**: The event has 34 detected objects but only 3-4 are meaningful. Many ghost/fragment tracks (1-2 frame duplicates from BoT-SORT flickering) pass the filter and pollute the scene with noise.

**Fix**: `min_track_frames: int = 8`

**Tradeoff**: Could drop short-lived but real vehicles (e.g., a car crossing through in 6 frames). But <8-frame tracks contribute almost nothing to causal analysis (insufficient timesteps) and mostly add noise. This affects crops and RAG indexing globally, not just causal — but short tracks produce poor crops too.

---

## Fix 6: Proper NaN mask instead of MISSING=999.0 sentinel

**File**: `app/pipeline/causal.py:194-215` (`_assemble`) and `258`

**Current approach**: Fill NaN with 999.0, pass `missing_flag=MISSING` to `pp.DataFrame`.

**Problem**: 999.0 is ~40x larger than any real value (speeds 0-25, gaps 0-50). If ParCorr doesn't properly exclude these, they distort correlation computation. The sentinel approach works but is fragile.

**Fix**:
1. In `_assemble`, return the NaN mask alongside data and names:
   ```python
   d = pd.DataFrame(np.column_stack(arrays), columns=names)
   d = d.interpolate(limit=5, limit_direction="both")
   mask = d.notna().values  # True = observed, False = missing
   data = d.fillna(0.0).to_numpy(dtype=float)  # fill with 0 (arbitrary — masked out anyway)
   return data, names, mask
   ```

2. In `analyze_event`, pass the mask:
   ```python
   dataframe = pp.DataFrame(data, var_names=names, mask=~mask)
   ```

**Tradeoff**: Minimal — this is the tigramite-recommended approach. Cleaner than the sentinel value.

---

## Fix 7: Add result extraction logging

**File**: `app/pipeline/causal.py`, `analyze_event` method

**Change**: After `_assemble` and after PCMCI+ runs, log:
- Variable matrix shape (n_timesteps × n_variables)
- Which variables were kept/dropped and why
- The full PCMCI+ graph before self-link filtering
- Number of significant links found per target

**Example**:
```python
logger.info("Target %s: %d timesteps, %d variables: %s",
            target_oid, data.shape[0], len(names), names)
logger.info("Target %s: PCMCI+ found %d links (pre-filter)",
            target_oid, sum(1 for ...))
```

**Impact**: Pure observability. Makes debugging possible without running the full pipeline in a debugger.

---

## Fix 8 (optional): Stationarity pre-processing

**File**: `app/pipeline/causal.py`, new function after `_assemble`

**Problem**: Non-stationary series can produce spurious correlations in ParCorr. Velocity is usually stationary (it's already a first-difference of position), but vehicles with trends (accelerating/decelerating) may violate this.

**Fix** (if included):
```python
from statsmodels.tsa.stattools import adfuller

def _ensure_stationarity(data, names):
    for col_idx, name in enumerate(names):
        series = data[:, col_idx]
        finite = np.isfinite(series)
        if finite.sum() < 15:
            continue
        adf_p = adfuller(series[finite])[1]
        if adf_p > 0.05:
            data[1:, col_idx] = np.diff(series)
            data[0, col_idx] = data[1, col_idx]
            logger.info("Variable %s non-stationary (p=%.3f), applied first-differencing", name, adf_p)
    return data
```

**Tradeoff**: First-differencing reduces effective length by 1 frame (already short at ~100). Adds `statsmodels` dependency. For velocity (already differenced from position), this is usually unnecessary. **Recommendation: skip for now, add later if needed.**

---

## Questions to resolve before implementation

1. **Fix 3**: Replace `lead_speed` with `rel_speed`, or keep both? → Recommendation: replace.
2. **Fix 5**: Raise `min_track_frames` globally in config, or only in the causal engine? → Recommendation: globally (short tracks produce poor crops too).
3. **Fix 8 (stationarity)**: Include now or defer? → Recommendation: defer.
4. **Visualization**: Add causal graph / time-series plots from newPlan.md in this same batch, or keep focused on fixing the core engine? → Recommendation: defer to a separate step.
