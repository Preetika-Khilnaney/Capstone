'use client';

import { useEffect, useRef, useMemo } from 'react';
import {
  FullCausalResult,
  CausalNode,
  CausalEdge,
  CausalEventEdge,
  TemporalEvent,
} from '@/lib/api';
import styles from './CausalPanel.module.css';

// ── Colour helpers ────────────────────────────────────────────────────────────
const EVENT_COLOURS: Record<string, string> = {
  COLLISION: '#ef4444',
  CONTACT: '#f97316',
  NEAR_COLLISION: '#facc15',
  SUDDEN_BRAKING: '#3b82f6',
  SUDDEN_STOP: '#6366f1',
  SUDDEN_ACCELERATION: '#22c55e',
  CLOSING_DISTANCE: '#f59e0b',
  APPROACHING: '#a3a3a3',
  FALL: '#ec4899',
  TRAJECTORY_DEVIATION: '#8b5cf6',
  SWERVE: '#0ea5e9',
  DEFAULT: '#64748b',
};

function eventColour(type: string): string {
  return EVENT_COLOURS[type] || EVENT_COLOURS.DEFAULT;
}

function relColour(rel: string): string {
  if (rel === 'causes') return '#ef4444';
  if (rel === 'supports') return '#f97316';
  return '#64748b';
}

function confOpacity(c: number): number {
  return 0.35 + c * 0.65;
}

// ── Layout: simple layered DAG ────────────────────────────────────────────────

interface LayoutNode {
  id: string;
  label: string;
  x: number;
  y: number;
  r: number;
  colour: string;
  type: string;
  node: CausalNode | null;
  event?: TemporalEvent;
}

function buildLayout(
  result: FullCausalResult,
  width: number,
  height: number,
): { nodes: LayoutNode[]; edgeData: Array<{ from: LayoutNode; to: LayoutNode; edge: CausalEventEdge | CausalEdge; kind: 'event' | 'stat' }> } {
  // Sort temporal events by onset timestamp
  const events = [...(result.temporal_events || [])].sort(
    (a, b) => (a.onset?.onset_timestamp ?? a.start_timestamp) - (b.onset?.onset_timestamp ?? b.start_timestamp),
  );

  const nodeMap = new Map<string, LayoutNode>();

  // Entity nodes at the top
  const entities = result.entities || [];
  entities.forEach((ent, i) => {
    const x = width * ((i + 1) / (entities.length + 1));
    const y = 60;
    const id = `entity_${ent.object_id}`;
    nodeMap.set(id, {
      id,
      label: `${ent.class ?? 'Vehicle'} ${ent.object_id}`,
      x, y, r: 36,
      colour: '#2563eb',
      type: 'object',
      node: result.nodes?.find((n) => n.id === id) ?? null,
    });
  });

  // Event nodes in temporal order, layered below entities
  const layerHeight = Math.max(90, (height - 120) / Math.max(events.length, 1));
  events.forEach((evt, i) => {
    const x = width / 2 + ((i % 2 === 0 ? -1 : 1) * width * 0.22);
    const y = 130 + i * Math.min(layerHeight, 100);
    nodeMap.set(evt.event_id, {
      id: evt.event_id,
      label: evt.event_type.replace(/_/g, ' '),
      x, y, r: 32,
      colour: eventColour(evt.event_type),
      type: 'event',
      node: result.nodes?.find((n) => n.id === evt.event_id) ?? null,
      event: evt,
    });
  });

  // Build edge list
  const edgeData: Array<{ from: LayoutNode; to: LayoutNode; edge: CausalEventEdge | CausalEdge; kind: 'event' | 'stat' }> = [];

  for (const edge of result.temporal_event_edges || []) {
    const from = nodeMap.get(edge.source_event_id);
    const to = nodeMap.get(edge.target_event_id);
    if (from && to) edgeData.push({ from, to, edge, kind: 'event' });
  }

  return { nodes: Array.from(nodeMap.values()), edgeData };
}

// ── Arrow helper ──────────────────────────────────────────────────────────────
function arrowPath(from: LayoutNode, to: LayoutNode): string {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return '';
  const ux = dx / dist;
  const uy = dy / dist;
  const x1 = from.x + ux * from.r;
  const y1 = from.y + uy * from.r;
  const x2 = to.x - ux * (to.r + 8);
  const y2 = to.y - uy * (to.r + 8);
  // Slight curve
  const mx = (x1 + x2) / 2 - uy * 20;
  const my = (y1 + y2) / 2 + ux * 20;
  return `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  result: FullCausalResult;
  onSeekFrame?: (frame: number) => void;
  onHoverEvent?: (evt: TemporalEvent | null) => void;
}

export default function CausalGraphD3({ result, onSeekFrame, onHoverEvent }: Props) {
  const WIDTH = 700;
  const HEIGHT = Math.max(400, 130 + (result.temporal_events?.length ?? 0) * 95);

  const { nodes, edgeData } = useMemo(() => buildLayout(result, WIDTH, HEIGHT), [result]);

  return (
    <div className={styles.graphWrap}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className={styles.graphSvg}
        style={{ width: '100%', height: `${HEIGHT}px`, maxHeight: '600px' }}
      >
        <defs>
          {/* Arrow markers per relationship type */}
          {(['causes', 'supports', 'precedes'] as const).map((rel) => (
            <marker
              key={rel}
              id={`arrow-${rel}`}
              viewBox="0 0 10 10"
              refX={9}
              refY={5}
              markerWidth={6}
              markerHeight={6}
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill={relColour(rel)} />
            </marker>
          ))}
          {/* Glow filter */}
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Edges */}
        {edgeData.map((e, i) => {
          const rel = ('relationship' in e.edge ? e.edge.relationship : 'precedes') as string;
          const conf = 'confidence' in e.edge ? (e.edge as CausalEventEdge).confidence : (e.edge as CausalEdge).final_confidence;
          const d = arrowPath(e.from, e.to);
          if (!d) return null;
          return (
            <g key={`edge-${i}`}>
              <path
                d={d}
                fill="none"
                stroke={relColour(rel)}
                strokeWidth={rel === 'causes' ? 2.5 : 1.5}
                strokeDasharray={rel === 'precedes' ? '5 4' : undefined}
                opacity={confOpacity(conf)}
                markerEnd={`url(#arrow-${rel})`}
              />
              {/* Confidence label */}
              <text
                x={(e.from.x + e.to.x) / 2 - (e.to.y - e.from.y) * 0.025}
                y={(e.from.y + e.to.y) / 2 + (e.to.x - e.from.x) * 0.025}
                textAnchor="middle"
                fill={relColour(rel)}
                fontSize={10}
                opacity={0.85}
              >
                {rel} · {conf.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Nodes */}
        {nodes.map((n) => {
          const isCollision = n.event?.event_type === 'COLLISION' || n.event?.event_type === 'CONTACT';
          const onsetTs = n.event?.onset?.onset_timestamp;
          const confirmTs = n.event?.onset?.confirmation?.confirmation_timestamp;

          return (
            <g
              key={n.id}
              style={{ cursor: n.event && onSeekFrame ? 'pointer' : 'default' }}
              onClick={() => {
                if (n.event && onSeekFrame) {
                  onSeekFrame(n.event.onset?.onset_frame ?? n.event.start_frame);
                }
              }}
              onMouseEnter={() => n.event && onHoverEvent?.(n.event)}
              onMouseLeave={() => onHoverEvent?.(null)}
            >
              <circle
                cx={n.x}
                cy={n.y}
                r={n.r}
                fill={n.colour}
                fillOpacity={0.18}
                stroke={n.colour}
                strokeWidth={isCollision ? 2.5 : 1.5}
                filter={isCollision ? 'url(#glow)' : undefined}
              />
              {/* Node label */}
              <text x={n.x} y={n.y - 3} textAnchor="middle" fill={n.colour} fontSize={10} fontWeight={600}>
                {n.label}
              </text>
              {/* Timestamp under event nodes */}
              {onsetTs != null && (
                <text x={n.x} y={n.y + 12} textAnchor="middle" fill={n.colour} fontSize={9} opacity={0.8}>
                  onset {onsetTs.toFixed(1)}s
                </text>
              )}
              {/* Confidence dot */}
              <text x={n.x} y={n.y + (onsetTs != null ? 22 : 13)} textAnchor="middle" fill={n.colour} fontSize={9} opacity={0.7}>
                conf {((n.event?.confidence ?? 1) * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div className={styles.graphLegend}>
        <span style={{ color: '#ef4444' }}>● causes</span>
        <span style={{ color: '#f97316' }}>● supports</span>
        <span style={{ color: '#64748b' }}>- - precedes</span>
        <span style={{ color: '#2563eb' }}>● entity</span>
        <span style={{ color: '#ef4444' }}>● collision/contact</span>
        <span style={{ color: '#3b82f6' }}>● braking</span>
        <span style={{ color: '#f59e0b' }}>● closing distance</span>
      </div>
    </div>
  );
}
