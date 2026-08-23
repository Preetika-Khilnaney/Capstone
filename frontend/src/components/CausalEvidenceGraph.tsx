'use client';

import { CausalDriver, CausalEpisodeEntity, CausalTarget } from '@/lib/api';
import styles from './CausalPanel.module.css';

const LAG_STEP_S = 0.1;

const VAR_LABELS: Record<string, string> = {
  tgt_speed: 'target speed',
  nn_speed: 'nearest speed',
  nn_gap: 'nearest gap',
  rel_speed: 'rel. speed',
  lead_gap: 'lead gap',
};

const EDGE_MARKER_ID = 'evArrow';

interface MergedEdge {
  from: string;
  to: string;
  links: CausalDriver[];
}

interface StubEdge {
  fromVariable: string;
  to: string;
  links: CausalDriver[];
}

interface VehicleNode {
  id: string;
  name?: string;
  isInitiator: boolean;
  isTarget: boolean;
  target?: CausalTarget;
  x: number;
  y: number;
}

function entityFor(entities: CausalEpisodeEntity[] | undefined, oid: string): CausalEpisodeEntity | undefined {
  return entities?.find((e) => e.object_ids?.includes(oid));
}

function labelFor(v: string): string {
  return VAR_LABELS[v] || v;
}

function strengthColor(s: number): string {
  return s >= 0 ? 'var(--color-danger)' : 'var(--color-active)';
}

function spreadX(n: number, i: number, width: number, margin: number): number {
  if (n <= 1) return width / 2;
  return margin + ((width - 2 * margin) * i) / (n - 1);
}

/**
 * Unified vehicle-level causal graph: physical vehicles as nodes, confirmed
 * PCMCI+ links as labeled edges. The episode's initiator sits at the top even
 * when PCMCI+ never analyzed it directly. Targets with no significant links
 * are rendered as text badges instead of empty graphs.
 */
export default function CausalEvidenceGraph({
  targets,
  entities,
}: {
  targets: CausalTarget[];
  entities?: CausalEpisodeEntity[];
}) {
  const width = 820;
  const margin = 100;

  const initiatorOids = (entities || [])
    .filter((e) => e.role === 'initiator')
    .flatMap((e) => e.object_ids || []);

  const merged: MergedEdge[] = [];
  const stubs: StubEdge[] = [];
  const emptyTargets: CausalTarget[] = [];

  for (const target of targets) {
    const drivers = (target.drivers_of_target_speed || []).filter((d) => d.cause !== 'tgt_speed');
    if (drivers.length === 0) {
      emptyTargets.push(target);
      continue;
    }
    const withVehicle = drivers.filter((d) => !!d.cause_object);
    if (withVehicle.length === 0) {
      stubs.push({ fromVariable: drivers[0].cause, to: target.target_object, links: drivers });
      continue;
    }
    const bySrc = new Map<string, CausalDriver[]>();
    for (const d of withVehicle) {
      const src = d.cause_object as string;
      bySrc.set(src, [...(bySrc.get(src) || []), d]);
    }
    for (const [src, links] of bySrc) {
      merged.push({ from: src, to: target.target_object, links });
    }
    const unmapped = drivers.filter((d) => !d.cause_object);
    if (unmapped.length > 0) {
      stubs.push({ fromVariable: unmapped[0].cause, to: target.target_object, links: unmapped });
    }
  }

  const sourceOids = [...new Set(merged.map((e) => e.from))];
  const bottomOids = [...new Set(merged.map((e) => e.to))];
  // A vehicle that is both a cause and a target appears once, as a target
  // (own row); causes that are not targets go to the top row.
  const isTargetOid = new Set(bottomOids);
  const topOids = [...new Set([...initiatorOids, ...sourceOids])].filter((oid) => !isTargetOid.has(oid));
  const hasStubs = stubs.length > 0;

  const topY = 120;
  const bottomY = 330;
  const stubY = 510;
  const rowR = 38;
  const targetR = 44;
  const stubR = 30;
  const height = (hasStubs ? stubY : bottomY) + 140;

  const nodes: VehicleNode[] = topOids.map((oid, i) => {
    const isInitiator = initiatorOids.includes(oid);
    const ent = entityFor(entities, oid);
    return {
      id: oid,
      name: ent?.name,
      isInitiator,
      isTarget: false,
      x: spreadX(topOids.length, i, width, margin),
      y: topY,
    };
  });
  bottomOids.forEach((oid, i) => {
    const target = targets.find((t) => t.target_object === oid);
    if (!target) return;
    nodes.push({
      id: oid,
      name: entityFor(entities, oid)?.name,
      isInitiator: initiatorOids.includes(oid),
      isTarget: true,
      target,
      x: spreadX(bottomOids.length, i, width, margin),
      y: bottomY,
    });
  });

  const pos = (oid: string): { x: number; y: number } => {
    const n = nodes.find((nd) => nd.id === oid);
    return n ? { x: n.x, y: n.y } : { x: width / 2, y: bottomY };
  };

  function edgePath(fromX: number, fromY: number, toX: number, toY: number): string {
    if (fromY === toY) {
      const midY = fromY - rowR - 70;
      const midX = (fromX + toX) / 2;
      return `M ${fromX} ${fromY - rowR} C ${midX} ${midY}, ${midX} ${midY}, ${toX} ${toY - targetR}`;
    }
    const midY = (fromY + toY) / 2;
    return `M ${fromX} ${fromY + rowR} C ${fromX} ${midY}, ${toX} ${midY}, ${toX} ${toY - targetR}`;
  }

  function linkText(l: CausalDriver): string {
    return `via ${labelFor(l.cause)} · lag ${(l.lag * LAG_STEP_S).toFixed(1)} s · r=${l.strength}`;
  }

  return (
    <div className={styles.graphWrap}>
      <svg viewBox={`0 0 ${width} ${height}`} className={styles.graphSvg}>
        <defs>
          <marker id={EDGE_MARKER_ID} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
          </marker>
        </defs>

        {merged.map((edge, i) => {
          const from = pos(edge.from);
          const to = pos(edge.to);
          const dominant = edge.links[0];
          const midX = (from.x + to.x) / 2;
          const midY = from.y === to.y ? from.y - rowR - 96 : (from.y + to.y) / 2 - 12;
          return (
            <g key={`edge-${i}`}>
              <path
                d={edgePath(from.x, from.y, to.x, to.y)}
                fill="none"
                stroke={strengthColor(dominant.strength)}
                strokeWidth={Math.max(1.5, Math.min(5, Math.abs(dominant.strength) * 6))}
                markerEnd={`url(#${EDGE_MARKER_ID})`}
              />
              <text x={midX} y={midY} textAnchor="middle" className={styles.edgeLabel}>
                {edge.links.map((l, j) => (
                  <tspan key={j} x={midX} dy={j === 0 ? 0 : 11}>{linkText(l)}</tspan>
                ))}
              </text>
            </g>
          );
        })}

        {stubs.map((stub, i) => {
          const to = pos(stub.to);
          const fromX = spreadX(stubs.length, i, width, margin);
          const midY = (stubY + to.y) / 2;
          const dominant = stub.links[0];
          return (
            <g key={`stub-${i}`}>
              <path
                d={`M ${fromX} ${stubY + stubR} C ${fromX} ${midY}, ${to.x} ${midY}, ${to.x} ${to.y - targetR}`}
                fill="none"
                stroke="var(--text-muted)"
                strokeWidth={1.5}
                markerEnd={`url(#${EDGE_MARKER_ID})`}
              />
              <text x={(fromX + to.x) / 2} y={midY - 6} textAnchor="middle" className={styles.edgeLabel}>
                {linkText(dominant)}
              </text>
              <circle cx={fromX} cy={stubY} r={stubR} className={styles.nodeUnidentified} />
              <text x={fromX} y={stubY - 2} textAnchor="middle" className={styles.nodeLabel}>
                via
              </text>
              <text x={fromX} y={stubY + 13} textAnchor="middle" className={styles.nodeSubLabel}>
                {labelFor(stub.fromVariable)}
              </text>
            </g>
          );
        })}

        {nodes.map((node) => (
          <g key={node.id}>
            {node.isInitiator && (
              <text x={node.x} y={node.y - rowR - 12} textAnchor="middle" className={styles.tagInitiator}>
                INITIATOR
              </text>
            )}
            <circle
              cx={node.x}
              cy={node.y}
              r={node.isTarget ? targetR : rowR}
              className={
                node.isInitiator ? styles.nodeInitiator : node.isTarget ? styles.nodeTarget : styles.nodeVehicle
              }
            />
            <text x={node.x} y={node.y - (node.isTarget ? 4 : 2)} textAnchor="middle" className={styles.nodeLabelTarget}>
              {node.id}
            </text>
            {node.name && (
              <text x={node.x} y={node.y + 14} textAnchor="middle" className={styles.nodeSubLabel}>
                {node.name}
              </text>
            )}
            {node.target && (
              <text x={node.x} y={node.y + 14 + (node.name ? 11 : 0)} textAnchor="middle" className={styles.nodeSubLabel}>
                {node.target.target_class} · drop {node.target.target_speed_drop_mps} m/s
              </text>
            )}
          </g>
        ))}
      </svg>

      <div className={styles.graphLegend}>
        <span><i className={styles.dotDanger} /> initiator (episode root)</span>
        <span><i className={styles.dotBlue} /> analyzed target</span>
        <span><i className={styles.dotNeutral} /> other vehicle</span>
        <span><i className={styles.dotMuted} /> vehicle unidentified (old data)</span>
        <span><i className={styles.dotDanger} /> positive r: cause accelerates / gap grows → target drops</span>
        <span><i className={styles.dotActive} /> negative r: cause slows / gap shrinks → target drops</span>
      </div>

      {emptyTargets.length > 0 && (
        <div className={styles.badgeList}>
          {emptyTargets.map((t) => (
            <div key={t.target_object} className={styles.badgeCard}>
              <span className={styles.badgeTitle}>{t.target_object} ({t.target_class})</span>
              <span className={styles.badgeMeta}>
                Insufficient data ({t.n_timesteps} timesteps) — no significant links found.
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}