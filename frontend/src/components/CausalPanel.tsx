'use client';

import { Fragment, useEffect, useState } from 'react';
import {
  analyzeCausal, fetchCausalGraph, generateSitrep, fetchSitrep,
  CausalEpisodeEntity, CausalResult, SitrepResult,
} from '@/lib/api';
import CausalEvidenceGraph from './CausalEvidenceGraph';
import TimeSeriesLink, { TsWindow } from './TimeSeriesLink';
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

const REL_TYPE_CLASS: Record<string, string> = {
  DIRECT_CAUSE: styles.stageRelDirect,
  TRIGGERED_RESPONSE: styles.stageRelTriggered,
  CONSEQUENCE: styles.stageRelConsequence,
};

/** Entity chips to show inside a stage node: named actors + collapsed counts. */
function chipsFor(node: { involved_entities?: string[] }, entities?: CausalEpisodeEntity[]): string[] {
  if (!entities || !node.involved_entities) return [];
  const ents = node.involved_entities
    .map((id) => entities.find((e) => e.id === id))
    .filter((e): e is CausalEpisodeEntity => !!e);
  const chips: string[] = [];
  for (const e of ents) {
    if (e.role === 'trailing' || e.role === 'aggregate') continue;
    chips.push(e.role && e.role !== 'vehicle' ? `${e.name} (${e.role})` : e.name);
  }
  const trailing = ents.filter((e) => e.role === 'trailing').length;
  if (trailing > 0) chips.push(`${trailing} trailing vehicles`);
  for (const e of ents) {
    if (e.role === 'aggregate') chips.push(`${e.name} — ${e.object_ids.length} vehicles`);
  }
  return chips;
}

export default function CausalPanel({
  eventId,
  csvData,
}: {
  eventId: string;
  csvData?: Record<string, string>[];
}) {
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
  const targets = causal?.targets || [];
  const entities = causal?.episode?.entities;

  const episodeWindows: TsWindow[] = (causal?.episode?.nodes || [])
    .filter((n) => n.node_id !== 'N1')
    .map((n) => ({
      label: `${n.node_id} ${n.state}`,
      start: n.window_s?.[0] ?? 0,
      end: n.window_s?.[1] ?? 0,
    }));

  const entityForOid = (oid: string): CausalEpisodeEntity | undefined =>
    entities?.find((e) => e.object_ids?.includes(oid));

  const unmappedHint = targets.some((t) =>
    (t.drivers_of_target_speed || []).some((d) => d.cause !== 'tgt_speed' && !d.cause_object));

  const linkedTargets = targets.filter((t) =>
    (t.drivers_of_target_speed || []).some((d) => d.cause !== 'tgt_speed' && !!d.cause_object));

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
              <h3 className={styles.subTitle}>Incident Episode — Timeline</h3>
              {causal.episode.root_cause?.primary_factor && (
                <p className={styles.rootCauseLine}>
                  Root cause (primary): {causal.episode.root_cause.primary_factor.kind} —{' '}
                  {causal.episode.root_cause.primary_factor.text}
                </p>
              )}
              <div className={styles.stageChain}>
                {causal.episode.nodes.map((node, i) => {
                  const outgoing = causal.episode!.relations.filter((r) => r.source_node === node.node_id);
                  const chips = chipsFor(node, entities);
                  return (
                    <Fragment key={node.node_id}>
                      <div className={styles.stageNode}>
                        <div className={styles.stageHeader}>
                          <span className={styles.stageBadge}>{node.node_id}</span>
                          <span className={styles.stageName}>{node.state}</span>
                        </div>
                        <div className={styles.stageWindow}>
                          {node.window_s?.[0]}s – {node.window_s?.[1]}s
                        </div>
                        {chips.length > 0 && (
                          <div className={styles.entityChips}>
                            {chips.map((c) => <span key={c} className={styles.entityChip}>{c}</span>)}
                          </div>
                        )}
                        {node.evidence && node.evidence.length > 0 && (
                          <p className={styles.stageEvidenceLine}>{node.evidence[0]}</p>
                        )}
                      </div>
                      {i < causal.episode!.nodes.length - 1 && (
                        <div className={styles.stageConnector}>
                          {outgoing.length === 0 && <span className={styles.stageConnMech}>relation not staged</span>}
                          {outgoing.map((rel) => (
                            <div key={rel.target_node} className={styles.stageConnItem}>
                              <span className={`${styles.stageConnLabel} ${REL_TYPE_CLASS[rel.relation_type] || ''}`}>
                                {rel.relation_type} → {rel.target_node}
                              </span>
                              {rel.mechanism && (
                                <span className={styles.stageConnMech}>{rel.mechanism}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </Fragment>
                  );
                })}
              </div>
              {causal.note && <p className={styles.caveat}>{causal.note}</p>}
            </div>
          )}

          {targets.length > 0 && (
            <div className={styles.evidenceBlock}>
              <h3 className={styles.subTitle}>Causal Evidence Graph</h3>
              <p className={styles.graphIntro}>
                Confirmed PCMCI+ links, mapped onto the physical vehicles. The episode initiator is
                shown at the top even when PCMCI+ did not analyze it directly.
              </p>
              <CausalEvidenceGraph targets={targets} entities={entities} />
              {unmappedHint && (
                <p className={styles.caveat}>
                  Some links lack a vehicle mapping (older analysis data) — press “Re-run Causal
                  Analysis” to populate the physical-vehicle mapping.
                </p>
              )}
            </div>
          )}

          {linkedTargets.length > 0 && csvData && csvData.length > 0 && (
            <div className={styles.evidenceBlock}>
              <h3 className={styles.subTitle}>Evidence Time Series</h3>
              <p className={styles.graphIntro}>
                Cause-vehicle speed (solid) and target speed (dashed) around the incident anchor —
                the lag arrow shows where the effect manifests.
              </p>
              {linkedTargets.map((target) => (
                <div key={target.target_object} className={styles.tsGroup}>
                  <h4 className={styles.tsGroupTitle}>
                    {target.target_object} ({target.target_class}) · speed drop {target.target_speed_drop_mps} m/s
                  </h4>
                  {(target.drivers_of_target_speed || [])
                    .filter((d) => d.cause !== 'tgt_speed' && !!d.cause_object)
                    .map((d, i) => {
                      const causeEnt = entityForOid(d.cause_object as string);
                      return (
                        <TimeSeriesLink
                          key={`${target.target_object}-${i}`}
                          causeId={d.cause_object as string}
                          causeName={causeEnt?.name}
                          effectId={target.target_object}
                          effectClass={target.target_class}
                          viaVar={d.cause}
                          lagSec={d.lag * 0.1}
                          strength={d.strength}
                          rows={csvData}
                          windows={episodeWindows}
                        />
                      );
                    })}
                </div>
              ))}
            </div>
          )}
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