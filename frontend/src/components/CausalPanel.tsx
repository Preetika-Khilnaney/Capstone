'use client';

import { useEffect, useState } from 'react';
import {
  analyzeCausal, fetchCausalGraph, generateSitrep, fetchSitrep,
  FullCausalResult, SitrepResult, TemporalEvent,
} from '@/lib/api';
import CausalGraphD3 from './CausalGraph';
import styles from './CausalPanel.module.css';

// ── Markdown renderer ─────────────────────────────────────────────────────────
function renderInline(text: string, keyPrefix: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
      : <span key={`${keyPrefix}-${i}`}>{part}</span>
  );
}

function renderReport(text: string) {
  const blocks: React.ReactNode[] = [];
  let listBuffer: string[] = [];
  function flushList(key: string) {
    if (listBuffer.length === 0) return;
    blocks.push(
      <ul key={key} className={styles.reportList}>
        {listBuffer.map((item, i) => <li key={i}>{renderInline(item, `${key}-li-${i}`)}</li>)}
      </ul>
    );
    listBuffer = [];
  }
  text.split('\n').forEach((line, idx) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      listBuffer.push(trimmed.slice(2));
      return;
    }
    flushList(`list-${idx}`);
    if (trimmed === '') return;
    blocks.push(<p key={`p-${idx}`}>{renderInline(trimmed, `p-${idx}`)}</p>);
  });
  flushList('list-end');
  return blocks;
}

// ── Number formatting helpers ──────────────────────────────────────────────────
function formatNum(val: number | null | undefined, decimals: number = 2, suffix: string = ''): string {
  if (val == null) return 'N/A';
  return val.toFixed(decimals) + suffix;
}

function formatConf(val: number | null | undefined): string {
  if (val == null) return 'N/A';
  return (val * 100).toFixed(0) + '%';
}

// ── Relationship badge ────────────────────────────────────────────────────────
function RelBadge({ rel }: { rel: string }) {
  const colours: Record<string, string> = {
    causes: 'var(--color-danger)',
    supports: '#f97316',
    precedes: 'var(--text-muted)',
  };
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 6px',
      borderRadius: '4px',
      fontSize: '0.7rem',
      fontWeight: 700,
      background: colours[rel] || colours.precedes,
      color: '#fff',
      textTransform: 'uppercase',
    }}>
      {rel}
    </span>
  );
}

// ── Method status pill ────────────────────────────────────────────────────────
function MethodPill({ name, status, nEdges }: { name: string; status: string; nEdges: number }) {
  const isOk = status === 'ok';
  const isSkipped = status === 'unavailable' || status === 'skipped';
  return (
    <div style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '3px 10px',
      borderRadius: 20,
      border: `1px solid ${isOk ? 'var(--color-active)' : isSkipped ? 'var(--text-muted)' : 'var(--color-danger)'}`,
      fontSize: '0.72rem',
      color: isOk ? 'var(--color-active)' : isSkipped ? 'var(--text-muted)' : 'var(--color-danger)',
      marginRight: 6,
      marginBottom: 4,
    }}>
      <span>{isOk ? '✓' : isSkipped ? '–' : '✗'}</span>
      <span>{name}</span>
      {isOk && <span style={{ opacity: 0.6 }}>({nEdges} edges)</span>}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function CausalPanel({ eventId, onSeekFrame }: { eventId: string; onSeekFrame?: (frame: number) => void }) {
  const [causal, setCausal] = useState<FullCausalResult | null>(null);
  const [sitrep, setSitrep] = useState<SitrepResult | null>(null);
  const [causalLoading, setCausalLoading] = useState(false);
  const [sitrepLoading, setSitrepLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initialLoad, setInitialLoad] = useState(true);
  const [hoveredEvent, setHoveredEvent] = useState<TemporalEvent | null>(null);
  const [activeTab, setActiveTab] = useState<'graph' | 'events' | 'methods' | 'sitrep'>('graph');

  useEffect(() => {
    async function load() {
      try {
        const [c, s] = await Promise.all([fetchCausalGraph(eventId), fetchSitrep(eventId)]);
        setCausal(c);
        setSitrep(s);
      } catch (err) {
        console.error(err);
      } finally {
        setInitialLoad(false);
      }
    }
    load();
  }, [eventId]);

  async function runCausal() {
    setCausalLoading(true);
    setError(null);
    try {
      const result = await analyzeCausal(eventId);
      setCausal(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Causal analysis failed');
    } finally {
      setCausalLoading(false);
    }
  }

  async function runSynthesis() {
    setSitrepLoading(true);
    setError(null);
    try {
      const result = await generateSitrep(eventId);
      setSitrep(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Situation report generation failed');
    } finally {
      setSitrepLoading(false);
    }
  }

  if (initialLoad) return null;

  const flaggedEntities = sitrep?.evidence?.entities.filter((e) => e.possible_collision) || [];
  const hasFullResult = causal && causal.status === 'ok';
  const primaryEvt = causal?.primary_event;

  return (
    <div className={styles.panel}>
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className={styles.panelHeader}>
        <h2 className={styles.sectionTitle}>Causal Reasoning — Why did this happen?</h2>
        <div className={styles.actions}>
          <button className={styles.actionBtn} onClick={runCausal} disabled={causalLoading}>
            {causalLoading ? 'Analyzing…' : causal ? 'Re-run Causal Analysis' : 'Run Causal Analysis'}
          </button>
          <button className={styles.actionBtn} onClick={runSynthesis} disabled={sitrepLoading}>
            {sitrepLoading ? 'Generating…' : sitrep ? 'Regenerate Situation Report' : 'Generate Situation Report'}
          </button>
        </div>
      </div>

      {error && <div className={styles.errorBox}>{error}</div>}

      {/* ── Primary event banner ─────────────────────────────────────────── */}
      {primaryEvt && (
        <div className={styles.incidentBox} style={{ marginBottom: 12, padding: '10px 16px' }}>
          <span className={styles.incidentBadge} style={{ marginRight: 10 }}>
            {primaryEvt.event_type.replace(/_/g, ' ')}
          </span>
          <span>
            Objects: <strong>{primaryEvt.object_ids.join(', ')}</strong>
            {' · '}Onset: <strong>{formatNum(primaryEvt.event_onset.timestamp, 2, 's')}</strong>
            {' · '}Confirmed: <strong>{formatNum(primaryEvt.confirmation.timestamp, 2, 's')}</strong>
            {' · '}Confidence: <strong>{formatConf(primaryEvt.confidence)}</strong>
          </span>
          {onSeekFrame && (
            <button
              className={styles.actionBtn}
              style={{ marginLeft: 12, padding: '2px 10px', fontSize: '0.75rem' }}
              onClick={() => onSeekFrame(primaryEvt.event_onset.frame)}
            >
              ⏭ Jump to onset
            </button>
          )}
        </div>
      )}

      {/* ── Explanation ───────────────────────────────────────────────────── */}
      {causal?.explanation && (
        <div className={styles.reportText} style={{ marginBottom: 16, padding: '10px 14px', background: 'var(--bg-card)', borderRadius: 8, borderLeft: '3px solid var(--color-active)' }}>
          <p style={{ margin: 0, lineHeight: 1.6, fontSize: '0.88rem' }}>{causal.explanation}</p>
        </div>
      )}

      {/* ── Tab bar ───────────────────────────────────────────────────────── */}
      {hasFullResult && (
        <div style={{ display: 'flex', gap: 4, marginBottom: 14, borderBottom: '1px solid var(--border)', paddingBottom: 4 }}>
          {(['graph', 'events', 'methods', 'sitrep'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: '4px 14px',
                borderRadius: '6px 6px 0 0',
                border: 'none',
                background: activeTab === tab ? 'var(--color-active)' : 'transparent',
                color: activeTab === tab ? '#fff' : 'var(--text-muted)',
                cursor: 'pointer',
                fontWeight: activeTab === tab ? 700 : 400,
                fontSize: '0.8rem',
              }}
            >
              {tab === 'graph' ? '🔗 Graph' : tab === 'events' ? '📋 Events' : tab === 'methods' ? '🔬 Methods' : '📄 SitRep'}
            </button>
          ))}
        </div>
      )}

      {/* ── No result state ───────────────────────────────────────────────── */}
      {!causal && !causalLoading && (
        <div className={styles.emptyState}>No causal analysis has been run yet for this event.</div>
      )}
      {causal && causal.status !== 'ok' && (
        <div className={styles.emptyState}>{causal.message || `Status: ${causal.status}`}</div>
      )}

      {/* ── GRAPH TAB ─────────────────────────────────────────────────────── */}
      {hasFullResult && activeTab === 'graph' && (
        <div className={styles.causalResult}>
          {/* Stats row */}
          <div className={styles.statRow}>
            {[
              ['Entities', causal.n_entities],
              ['Pairs', causal.n_pairs],
              ['Variables', causal.n_causal_vars],
              ['Timesteps', causal.n_timesteps],
              ['FPS', causal.fps],
              ['Events', causal.temporal_events?.length ?? 0],
              ['Consensus Edges', causal.consensus_edges?.length ?? 0],
              ['Confidence', formatConf(causal.confidence)],
            ].map(([label, value]) => (
              <div key={String(label)} className={styles.statCard}>
                <span className={styles.statLabel}>{label}</span>
                <span className={styles.statValue}>{value}</span>
              </div>
            ))}
          </div>

          {/* Causal graph */}
          <CausalGraphD3
            result={causal}
            onSeekFrame={onSeekFrame}
            onHoverEvent={setHoveredEvent}
          />

          {/* Hover tooltip */}
          {hoveredEvent && (
            <div className={styles.incidentBox} style={{ margin: '8px 0', padding: '8px 14px' }}>
              <strong>{hoveredEvent.event_type.replace(/_/g, ' ')}</strong>
              {' · '}Objects: {hoveredEvent.object_ids.join(', ')}
              {' · '}Onset: {formatNum(hoveredEvent.onset?.onset_timestamp ?? hoveredEvent.start_timestamp, 2, 's')}
              {' · '}Conf: {formatConf(hoveredEvent.confidence)}
              {hoveredEvent.onset?.onset_reason && (
                <span style={{ fontSize: '0.72rem', opacity: 0.7, marginLeft: 8 }}>
                  [{hoveredEvent.onset.onset_reason}]
                </span>
              )}
            </div>
          )}

          {/* Consensus edges table */}
          {causal.consensus_edges && causal.consensus_edges.length > 0 && (
            <>
              <h3 className={styles.subTitle} style={{ marginTop: 18 }}>Statistical Consensus Edges</h3>
              <table className={styles.driverTable}>
                <thead>
                  <tr>
                    <th>Source</th>
                    <th>→</th>
                    <th>Target</th>
                    <th>Relationship</th>
                    <th>Lag</th>
                    <th>Support</th>
                    <th>Confidence</th>
                    <th>p-value</th>
                  </tr>
                </thead>
                <tbody>
                  {causal.consensus_edges.map((e, i) => (
                    <tr key={i}>
                      <td style={{ fontSize: '0.72rem' }}>{e.source}</td>
                      <td>→</td>
                      <td style={{ fontSize: '0.72rem' }}>{e.target}</td>
                      <td><RelBadge rel={e.relationship} /></td>
                      <td>{e.lag_frames}f ({e.lag_seconds}s)</td>
                      <td>{e.support_count}/{e.available_methods}</td>
                      <td>{formatConf(e.final_confidence)}</td>
                      <td>{e.p_value?.toFixed(3) ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {causal.note && <p className={styles.caveat}>{causal.note}</p>}
        </div>
      )}

      {/* ── EVENTS TAB ────────────────────────────────────────────────────── */}
      {hasFullResult && activeTab === 'events' && (
        <div className={styles.causalResult}>
          <h3 className={styles.subTitle}>Temporal Event Timeline</h3>
          <table className={styles.driverTable}>
            <thead>
              <tr>
                <th>Event</th>
                <th>Type</th>
                <th>Objects</th>
                <th>Onset</th>
                <th>Confirmed</th>
                <th>Conf</th>
                <th>Seek</th>
              </tr>
            </thead>
            <tbody>
              {(causal.temporal_events ?? []).map((evt) => (
                <tr key={evt.event_id}>
                  <td style={{ fontSize: '0.7rem', opacity: 0.6 }}>{evt.event_id}</td>
                  <td>
                    <span style={{ color: EVENT_COLOURS[evt.event_type] || '#64748b', fontWeight: 600, fontSize: '0.8rem' }}>
                      {evt.event_type.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td>{evt.object_ids.join(', ')}</td>
                  <td>{formatNum(evt.onset?.onset_timestamp ?? evt.start_timestamp, 2, 's')}</td>
                  <td>{formatNum(evt.onset?.confirmation?.confirmation_timestamp ?? evt.end_timestamp, 2, 's')}</td>
                  <td>{formatConf(evt.confidence)}</td>
                  <td>
                    {onSeekFrame && (
                      <button
                        style={{ fontSize: '0.72rem', padding: '1px 6px', cursor: 'pointer', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 4, color: 'var(--text-primary)' }}
                        onClick={() => onSeekFrame(evt.onset?.onset_frame ?? evt.start_frame)}
                      >
                        ⏭
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Event causal edges */}
          {causal.temporal_event_edges && causal.temporal_event_edges.length > 0 && (
            <>
              <h3 className={styles.subTitle} style={{ marginTop: 18 }}>Event-Level Causal Edges</h3>
              <table className={styles.driverTable}>
                <thead>
                  <tr>
                    <th>From</th>
                    <th>→</th>
                    <th>To</th>
                    <th>Relationship</th>
                    <th>Lag</th>
                    <th>Confidence</th>
                    <th>Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {causal.temporal_event_edges.map((e, i) => (
                    <tr key={i}>
                      <td style={{ fontSize: '0.72rem' }}>{e.source_type.replace(/_/g, ' ')}</td>
                      <td>→</td>
                      <td style={{ fontSize: '0.72rem' }}>{e.target_type.replace(/_/g, ' ')}</td>
                      <td><RelBadge rel={e.relationship} /></td>
                      <td>{formatNum(e.lag_seconds, 2, 's')}</td>
                      <td>{formatConf(e.confidence)}</td>
                      <td style={{ fontSize: '0.7rem', opacity: 0.7 }}>{e.evidence.slice(0, 2).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      {/* ── METHODS TAB ───────────────────────────────────────────────────── */}
      {hasFullResult && activeTab === 'methods' && (
        <div className={styles.causalResult}>
          <h3 className={styles.subTitle}>Causal Discovery Methods</h3>
          <div style={{ marginBottom: 14 }}>
            {Object.entries(causal.method_results ?? {}).map(([name, res]) => (
              <MethodPill key={name} name={name} status={res.status} nEdges={res.n_edges} />
            ))}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            <p><strong>Available:</strong> {causal.available_methods?.join(', ') || 'none'}</p>
            <p><strong>Failed:</strong> {causal.failed_methods?.join(', ') || 'none'}</p>
            <p><strong>Skipped/Unavailable:</strong> {causal.skipped_methods?.join(', ') || 'none'}</p>
          </div>
          <div style={{ marginTop: 14, fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
            <p>PCMCI+ provides temporal causal evidence with lag information.</p>
            <p>LiNGAM provides directional causal evidence under non-Gaussian assumptions.</p>
            <p>GES provides score-based structural evidence (BIC).</p>
            <p>CausalForest estimates heterogeneous treatment effects (braking → outcome speed).</p>
            <p>CycleNet: not available as a pip package.</p>
            <p style={{ marginTop: 6, fontStyle: 'italic' }}>Results are ranked hypotheses supporting — not proving — physical causality.</p>
          </div>
        </div>
      )}

      {/* ── SITREP TAB ────────────────────────────────────────────────────── */}
      {activeTab === 'sitrep' && (
        <div className={styles.sitrepSection}>
          {!sitrep && (
            <div className={styles.emptyState}>No situation report generated yet.</div>
          )}
          {sitrep && (
            <>
              {sitrep.status === 'no_key' && (
                <div className={styles.emptyState}>
                  No LLM API key configured (<code>$LLM_API_KEY</code>) — showing the structured evidence packet only.
                </div>
              )}
              {sitrep.status === 'llm_error' && (
                <div className={styles.errorBox}>LLM call failed: {sitrep.message}</div>
              )}
              {sitrep.report && (
                <div className={styles.reportText}>{renderReport(sitrep.report)}</div>
              )}
              {flaggedEntities.length > 0 && (
                <div className={styles.incidentBox} style={{ marginTop: 14 }}>
                  <h4 className={styles.subTitle}>Incident Indicators</h4>
                  {flaggedEntities.map((en) => (
                    <div key={en.object_id} className={styles.incidentRow}>
                      <span className={styles.incidentBadge}>Possible collision</span>
                      <span>
                        {en.colour ? `${en.colour} ` : ''}{en.class} <strong>{en.object_id}</strong> decelerated
                        sharply then its track ended at {en.exit_s}s
                        {en.nearest_at_exit && (
                          <> — nearest entity at that instant: {en.nearest_at_exit.class} {en.nearest_at_exit.object_id} ({en.nearest_at_exit.distance_m} m away)</>
                        )}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {sitrep.evidence && (
                <table className={styles.driverTable} style={{ marginTop: 14 }}>
                  <thead>
                    <tr><th>Object</th><th>Class</th><th>Mean km/h</th><th>Peak km/h</th><th>Present</th></tr>
                  </thead>
                  <tbody>
                    {sitrep.evidence.entities.map((en) => (
                      <tr key={en.object_id} className={en.possible_collision ? styles.rowFlagged : undefined}>
                        <td>{en.object_id}</td>
                        <td>{en.colour ? `${en.colour} ` : ''}{en.class}</td>
                        <td>{en.speed_mean_kmh}</td>
                        <td>{en.speed_max_kmh}{en.speed_uncertain ? ' *' : ''}</td>
                        <td>{en.entry_s}s – {en.exit_s}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Local colour helper for event tab (avoid export)
const EVENT_COLOURS: Record<string, string> = {
  COLLISION: '#ef4444', CONTACT: '#f97316', NEAR_COLLISION: '#facc15',
  SUDDEN_BRAKING: '#3b82f6', SUDDEN_STOP: '#6366f1', APPROACHING: '#a3a3a3',
  CLOSING_DISTANCE: '#f59e0b', FALL: '#ec4899', TRAJECTORY_DEVIATION: '#8b5cf6',
  DEFAULT: '#64748b',
};
