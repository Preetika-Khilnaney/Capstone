'use client';

import { CausalTarget } from '@/lib/api';
import styles from './CausalPanel.module.css';

const TARGET_KEY = 'tgt_speed';

const VAR_LABELS: Record<string, string> = {
  tgt_speed: 'Target Speed',
  nn_speed: 'Nearest Speed',
  nn_gap: 'Nearest Gap',
  lead_speed: 'Lead Speed',
  lead_gap: 'Lead Gap',
};

function labelFor(v: string): string {
  return VAR_LABELS[v] || v;
}

/**
 * Node-link diagram of a causal result: external variable nodes with arrows
 * into the target, plus a self-loop for autoregression (target's own past).
 */
export default function CausalGraph({ result }: { result: CausalTarget }) {
  const variables = result.variables || [];
  const drivers = result.drivers_of_target_speed || [];
  const externalVars = variables.filter((v) => v !== TARGET_KEY);
  const selfLinks = drivers.filter((d) => d.cause === TARGET_KEY).sort((a, b) => a.lag - b.lag);
  const externalLinks = drivers.filter((d) => d.cause !== TARGET_KEY);

  const width = 560;
  const nodeR = 34;
  const targetX = width - 100;
  const targetY = 150;
  const rowGap = 76;
  const startY = 150 - ((externalVars.length - 1) * rowGap) / 2;

  const nodePos: Record<string, { x: number; y: number }> = {
    [TARGET_KEY]: { x: targetX, y: targetY },
  };
  externalVars.forEach((v, i) => {
    nodePos[v] = { x: 90, y: startY + i * rowGap };
  });

  const height = Math.max(300, externalVars.length * rowGap + 60);

  function strengthColor(s: number): string {
    return s >= 0 ? 'var(--color-danger)' : 'var(--color-active)';
  }

  return (
    <div className={styles.graphWrap}>
      <svg viewBox={`0 0 ${width} ${height}`} className={styles.graphSvg}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
          </marker>
        </defs>

        {/* external variable -> target edges */}
        {externalLinks.map((link, i) => {
          const from = nodePos[link.cause];
          const to = nodePos[TARGET_KEY];
          if (!from || !to) return null;
          const midX = (from.x + to.x) / 2;
          const midY = (from.y + to.y) / 2;
          return (
            <g key={`ext-${i}`}>
              <line
                x1={from.x + nodeR} y1={from.y}
                x2={to.x - nodeR - 6} y2={to.y}
                stroke={strengthColor(link.strength)}
                strokeWidth={Math.max(1.5, Math.min(5, Math.abs(link.strength) * 6))}
                markerEnd="url(#arrow)"
              />
              <text x={midX} y={midY - 8} textAnchor="middle" className={styles.edgeLabel}>
                lag {link.lag} · r={link.strength}
              </text>
            </g>
          );
        })}

        {/* external nodes */}
        {externalVars.map((v) => {
          const pos = nodePos[v];
          const active = externalLinks.some((l) => l.cause === v);
          return (
            <g key={v}>
              <circle
                cx={pos.x} cy={pos.y} r={nodeR}
                className={active ? styles.nodeActive : styles.nodeInactive}
              />
              <text x={pos.x} y={pos.y + 4} textAnchor="middle" className={styles.nodeLabel}>
                {labelFor(v)}
              </text>
            </g>
          );
        })}

        {/* target self-loop (autoregression) */}
        {selfLinks.length > 0 && (
          <g>
            <path
              d={`M ${targetX - 10} ${targetY - nodeR} C ${targetX - 60} ${targetY - nodeR - 60}, ${targetX + 60} ${targetY - nodeR - 60}, ${targetX + 10} ${targetY - nodeR}`}
              fill="none"
              stroke="var(--text-muted)"
              strokeWidth={2}
              markerEnd="url(#arrow)"
            />
            <text x={targetX} y={targetY - nodeR - 66} textAnchor="middle" className={styles.edgeLabel}>
              past self: {selfLinks.map((l) => `lag ${l.lag} (r=${l.strength})`).join(', ')}
            </text>
          </g>
        )}

        {/* target node */}
        <circle cx={targetX} cy={targetY} r={nodeR + 4} className={styles.nodeTarget} />
        <text x={targetX} y={targetY - 3} textAnchor="middle" className={styles.nodeLabelTarget}>
          {result.target_object}
        </text>
        <text x={targetX} y={targetY + 13} textAnchor="middle" className={styles.nodeSubLabelTarget}>
          {labelFor(TARGET_KEY)}
        </text>
      </svg>

      <div className={styles.graphLegend}>
        <span><i className={styles.dotDanger} /> speeds up / gap grows → drop (positive r)</span>
        <span><i className={styles.dotActive} /> slows down / gap shrinks → drop (negative r)</span>
        <span><i className={styles.dotMuted} /> tested, no significant link found</span>
      </div>
    </div>
  );
}
