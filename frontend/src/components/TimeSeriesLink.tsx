'use client';

import styles from './CausalPanel.module.css';

const W = 720;
const H = 200;
const ML = 46;
const MR = 16;
const MT = 20;
const MB = 26;
const INNER_W = W - ML - MR;
const INNER_H = H - MT - MB;

export interface TsWindow {
  label: string;
  start: number;
  end: number;
}

export interface CsvRow {
  Timestamp?: string;
  Object_ID?: string;
  Velocity_mps?: string;
}

interface Point { t: number; v: number }

function seriesFor(rows: CsvRow[], oid: string): Point[] {
  return rows
    .filter((r) => r.Object_ID === oid)
    .map((r) => ({ t: Number(r.Timestamp), v: Number(r.Velocity_mps) }))
    .filter((p) => Number.isFinite(p.t) && Number.isFinite(p.v))
    .sort((a, b) => a.t - b.t);
}

function firstDropOnset(pts: Point[]): number | null {
  if (pts.length < 2) return null;
  const maxV = Math.max(...pts.map((p) => p.v));
  const half = pts.find((p) => p.v <= maxV * 0.5);
  return half ? half.t : null;
}

function valueAt(pts: Point[], t: number): number | null {
  if (pts.length === 0) return null;
  if (t <= pts[0].t) return pts[0].v;
  if (t >= pts[pts.length - 1].t) return pts[pts.length - 1].v;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].t >= t) {
      const a = pts[i - 1];
      const b = pts[i];
      const f = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
      return a.v + f * (b.v - a.v);
    }
  }
  return null;
}

function polyline(pts: Point[], x: (t: number) => number, y: (v: number) => number): string {
  return pts.map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ');
}

function xTicks(xMin: number, xMax: number): number[] {
  const span = xMax - xMin;
  const step = Math.max(1, Math.ceil(span / 6));
  const ticks: number[] = [];
  for (let t = Math.ceil(xMin / step) * step; t <= xMax; t += step) ticks.push(t);
  return ticks;
}

/**
 * Per-link time-series overlay: cause vehicle speed (solid) versus effect
 * vehicle speed (dashed) over the clip-anchored timeline, with the t=0
 * incident anchor, episode stage bands, and a lag arrow from the cause
 * vehicle's drop onset to the effect vehicle's response.
 */
export default function TimeSeriesLink({
  causeId,
  causeName,
  effectId,
  effectClass,
  viaVar,
  lagSec,
  strength,
  rows,
  windows,
}: {
  causeId: string;
  causeName?: string;
  effectId: string;
  effectClass?: string;
  viaVar: string;
  lagSec: number;
  strength: number;
  rows: CsvRow[];
  windows?: TsWindow[];
}) {
  const cause = seriesFor(rows, causeId);
  const effect = seriesFor(rows, effectId);
  if (cause.length === 0 && effect.length === 0) return null;

  const allT = [...cause.map((p) => p.t), ...effect.map((p) => p.t)];
  const xMin = Math.min(...allT);
  const xMax = Math.max(...allT);
  const allV = [...cause.map((p) => p.v), ...effect.map((p) => p.v)];
  const vMax = Math.max(5, Math.max(...allV) * 1.12);
  const vTicks = [0, vMax / 2, vMax];

  const px = (t: number) => ML + ((t - xMin) / (xMax - xMin || 1)) * INNER_W;
  const py = (v: number) => MT + (1 - v / vMax) * INNER_H;

  const zeroX = xMin <= 0 && 0 <= xMax ? px(0) : null;

  const causeOnset = firstDropOnset(cause);
  const effOnset = firstDropOnset(effect);
  const arrowEndT = effOnset ?? (causeOnset !== null ? Math.min(causeOnset + lagSec, xMax) : null);
  const arrow = causeOnset !== null && arrowEndT !== null
    ? { x1: px(causeOnset), y1: py(valueAt(cause, causeOnset) ?? 0), x2: px(arrowEndT), y2: py(valueAt(effect, arrowEndT) ?? 0) }
    : null;

  const causeColor = 'var(--color-active)';
  const effectColor = 'var(--color-warning)';

  return (
    <div className={styles.tsWrap}>
      <div className={styles.tsHeader}>
        <span className={styles.tsTitle}>
          {causeName ? `${causeName} (${causeId})` : causeId} → {effectId}
          {effectClass ? ` (${effectClass})` : ''}
        </span>
        <span className={styles.tsSub}>
          via {viaVar} · lag {lagSec.toFixed(1)} s · r={strength}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className={styles.tsSvg}>
        {(windows || []).filter((win) => win.end >= xMin && win.start <= xMax).map((win) => (
          <g key={win.label}>
            <rect
              x={px(Math.max(win.start, xMin))}
              y={MT - 4}
              width={Math.max(2, px(Math.min(win.end, xMax)) - px(Math.max(win.start, xMin)))}
              height={INNER_H + 8}
              className={styles.tsWin}
            />
            <text x={px(Math.max(win.start, xMin)) + 4} y={MT + 2} className={styles.tsWinLabel}>
              {win.label}
            </text>
          </g>
        ))}

        {zeroX !== null && (
          <g>
            <line x1={zeroX} y1={MT} x2={zeroX} y2={MT + INNER_H} className={styles.tsZero} />
            <text x={zeroX + 3} y={MT + 8} className={styles.tsZeroLabel}>t = 0</text>
          </g>
        )}

        {vTicks.map((v) => (
          <g key={v}>
            <line x1={ML} y1={py(v)} x2={W - MR} y2={py(v)} className={styles.tsGrid} />
            <text x={ML - 6} y={py(v) + 3} textAnchor="end" className={styles.tsAxis}>{v.toFixed(1)}</text>
          </g>
        ))}
        {xTicks(xMin, xMax).map((t) => (
          <text key={t} x={px(t)} y={H - 8} textAnchor="middle" className={styles.tsAxis}>{t.toFixed(0)}s</text>
        ))}

        {cause.length > 0 && (
          <polyline points={polyline(cause, px, py)} fill="none" stroke={causeColor} strokeWidth={2.2} className={styles.tsCauseLine} />
        )}
        {effect.length > 0 && (
          <polyline points={polyline(effect, px, py)} fill="none" stroke={effectColor} strokeWidth={2.2} strokeDasharray="7 4" className={styles.tsEffectLine} />
        )}

        {arrow && (
          <g>
            <line
              x1={arrow.x1} y1={arrow.y1} x2={arrow.x2} y2={arrow.y2}
              stroke={effectColor} strokeWidth={1.8}
              markerEnd="url(#tsArrow)"
            />
            <text
              x={(arrow.x1 + arrow.x2) / 2}
              y={Math.min(arrow.y1, arrow.y2) - 8}
              textAnchor="middle"
              className={styles.tsArrowLabel}
            >
              lag {lagSec.toFixed(1)} s
            </text>
          </g>
        )}
        <defs>
          <marker id="tsArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill={effectColor} />
          </marker>
        </defs>
      </svg>
      <div className={styles.tsLegend}>
        <span><i className={styles.tsSwatchSolid} style={{ background: causeColor }} /> {causeId} (cause)</span>
        <span><i className={styles.tsSwatchDashed} style={{ borderColor: effectColor }} /> {effectId} (effect)</span>
        <span className={styles.tsMuted}>t = 0 is the incident anchor; shaded bands = episode stages</span>
      </div>
    </div>
  );
}