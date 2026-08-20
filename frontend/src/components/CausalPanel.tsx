'use client';

import { useEffect, useState } from 'react';
import {
  analyzeCausal, fetchCausalGraph, generateSitrep, fetchSitrep,
  CausalResult, SitrepResult,
} from '@/lib/api';
import CausalGraph from './CausalGraph';
import styles from './CausalPanel.module.css';

function renderInline(text: string, keyPrefix: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
      : <span key={`${keyPrefix}-${i}`}>{part}</span>
  );
}

/** Minimal markdown renderer for LLM-written SitReps: **bold** + "- " bullet lists. */
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

const STATUS_COPY: Record<string, string> = {
  no_target: 'No object had enough valid near-field kinematics to analyze as a target.',
  insufficient: 'Not enough usable timesteps/variables after filtering — kinematics may be too sparse or out of the calibrated region.',
  error: 'Analysis failed.',
};

export default function CausalPanel({ eventId }: { eventId: string }) {
  const [causal, setCausal] = useState<CausalResult | null>(null);
  const [sitrep, setSitrep] = useState<SitrepResult | null>(null);
  const [causalLoading, setCausalLoading] = useState(false);
  const [sitrepLoading, setSitrepLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initialLoad, setInitialLoad] = useState(true);

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

  return (
    <div className={styles.panel}>
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

      {/* ── Causal graph (Track 2) ─────────────────────────────────────── */}
      {!causal && !causalLoading && (
        <div className={styles.emptyState}>No causal analysis has been run yet for this event.</div>
      )}

      {causal && causal.status !== 'ok' && (
        <div className={styles.emptyState}>
          {STATUS_COPY[causal.status] || causal.message || `Status: ${causal.status}`}
        </div>
      )}

      {causal && causal.status === 'ok' && (
        <div className={styles.causalResult}>
          {causal.episode && causal.episode.nodes.length > 0 && (
            <div className={styles.episodeBlock}>
              <h3 className={styles.subTitle}>Incident Episode — Stage Chain</h3>
              {causal.episode.root_cause?.primary_factor && (
                <p className={styles.caveat}>
                  Root cause (primary): {causal.episode.root_cause.primary_factor.kind} —{' '}
                  {causal.episode.root_cause.primary_factor.text}
                </p>
              )}
              <div className={styles.stageChain}>
                {causal.episode.nodes.map((node) => (
                  <div key={node.node_id} className={styles.stageNode}>
                    <div className={styles.stageHeader}>
                      <span className={styles.stageBadge}>{node.node_id}</span>
                      <span className={styles.stageName}>{node.state}</span>
                    </div>
                    <div className={styles.stageWindow}>
                      {node.window_s?.[0]}s – {node.window_s?.[1]}s
                    </div>
                    {node.evidence && node.evidence.length > 0 && (
                      <ul className={styles.stageEvidence}>
                        {node.evidence.map((line, j) => (
                          <li key={`${node.node_id}-ev-${j}`}>{line}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
              {causal.episode.relations.length > 0 && (
                <div className={styles.stageRelations}>
                  {causal.episode.relations.map((rel, i) => (
                    <span key={i} className={styles.stageRel}>
                      {rel.source_node} → {rel.target_node}: {rel.relation_type}
                      {rel.mechanism ? <span className={styles.stageRelMech}> — {rel.mechanism}</span> : null}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {(causal.targets?.length || 0) > 1 && (
            <p className={styles.caveat}>{causal.targets!.length} causal chains found in this scene.</p>
          )}
          {(causal.targets || []).map((target) => (
            <div key={target.target_object} className={styles.targetBlock}>
              <div className={styles.statRow}>
                <div className={styles.statCard}>
                  <span className={styles.statLabel}>Target</span>
                  <span className={styles.statValue}>{target.target_object} ({target.target_class})</span>
                </div>
                <div className={styles.statCard}>
                  <span className={styles.statLabel}>Speed Drop</span>
                  <span className={styles.statValue}>{target.target_speed_drop_mps} m/s</span>
                </div>
                <div className={styles.statCard}>
                  <span className={styles.statLabel}>Lead Fraction</span>
                  <span className={styles.statValue}>{target.target_lead_fraction}</span>
                </div>
                <div className={styles.statCard}>
                  <span className={styles.statLabel}>Timesteps</span>
                  <span className={styles.statValue}>{target.n_timesteps}</span>
                </div>
              </div>

              <CausalGraph result={target} />

              {(target.drivers_of_target_speed || []).length === 0 ? (
                <p className={styles.caveat}>
                  No causal link found between this vehicle and the others — its speed change is
                  explained by its own past only (own-past / non-causal result).
                </p>
              ) : (
                <table className={styles.driverTable}>
                  <thead>
                    <tr><th>Cause</th><th>Lag</th><th>Strength</th></tr>
                  </thead>
                  <tbody>
                    {(target.drivers_of_target_speed || [])
                      .filter((d) => d.cause !== 'tgt_speed')
                      .map((d, i) => (
                        <tr key={i}>
                          <td>{d.cause}</td>
                          <td>{d.lag}</td>
                          <td>{d.strength}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}
          {causal.note && <p className={styles.caveat}>{causal.note}</p>}
        </div>
      )}

      {/* ── Situation report (Track 4) ─────────────────────────────────── */}
      {sitrep && (
        <div className={styles.sitrepSection}>
          <h3 className={styles.subTitle}>Situation Report</h3>

          {sitrep.status === 'no_key' && (
            <div className={styles.emptyState}>
              No LLM API key configured (<code>$LLM_API_KEY</code>) — showing the structured evidence
              packet only; no narrative was generated.
            </div>
          )}
          {sitrep.status === 'llm_error' && (
            <div className={styles.errorBox}>LLM call failed: {sitrep.message}</div>
          )}
          {sitrep.report && (
            <div className={styles.reportText}>
              {renderReport(sitrep.report)}
            </div>
          )}

          {flaggedEntities.length > 0 && (
            <div className={styles.incidentBox}>
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
            <table className={styles.driverTable}>
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
        </div>
      )}
    </div>
  );
}
