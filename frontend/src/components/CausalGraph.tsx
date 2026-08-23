'use client';

import { useState, useRef, useMemo, useCallback } from 'react';
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
  const events = [...(result.temporal_events || [])].sort(
    (a, b) => (a.onset?.onset_timestamp ?? a.start_timestamp) - (b.onset?.onset_timestamp ?? b.start_timestamp),
  );

  const nodeMap = new Map<string, LayoutNode>();

  // Event nodes in temporal order, placed first
  const layerHeight = Math.max(100, (height - 140) / Math.max(events.length, 1));
  events.forEach((evt, i) => {
    const x = width / 2 + ((i % 2 === 0 ? -1 : 1) * width * 0.24);
    const y = 150 + i * Math.min(layerHeight, 110);
    nodeMap.set(evt.event_id, {
      id: evt.event_id,
      label: evt.event_type.replace(/_/g, ' '),
      x, y, r: 36,
      colour: eventColour(evt.event_type),
      type: 'event',
      node: result.nodes?.find((n) => n.id === evt.event_id) ?? null,
      event: evt,
    });
  });

  // Collect entity IDs that appear in event edges (connected to events)
  const connectedEntityIds = new Set<string>();
  for (const edge of result.temporal_event_edges || []) {
    // Entity IDs in edges reference object_ids like "V_02" — match against entity objects
    const srcObj = edge.source_event_id.replace('entity_', '');
    const tgtObj = edge.target_event_id.replace('entity_', '');
    if (nodeMap.has(`entity_${srcObj}`)) connectedEntityIds.add(srcObj);
    if (nodeMap.has(`entity_${tgtObj}`)) connectedEntityIds.add(tgtObj);
  }

  // Also collect entity IDs from the temporal_event.object_ids
  for (const evt of events) {
    for (const oid of evt.object_ids || []) {
      connectedEntityIds.add(oid);
    }
  }

  // Only render entity nodes that are connected to events
  const entities = (result.entities || []).filter(
    (ent) => connectedEntityIds.has(ent.object_id),
  );

  // Entity nodes at the top, spaced evenly
  entities.forEach((ent, i) => {
    const x = entities.length === 1
      ? width / 2
      : width * ((i + 1) / (entities.length + 1));
    const y = 70;
    const id = `entity_${ent.object_id}`;
    nodeMap.set(id, {
      id,
      label: `${ent.class ?? 'Vehicle'} ${ent.object_id}`,
      x, y, r: 40,
      colour: '#2563eb',
      type: 'object',
      node: result.nodes?.find((n) => n.id === id) ?? null,
    });
  });

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
  const x2 = to.x - ux * (to.r + 10);
  const y2 = to.y - uy * (to.r + 10);
  const mx = (x1 + x2) / 2 - uy * 24;
  const my = (y1 + y2) / 2 + ux * 24;
  return `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  result: FullCausalResult;
  onSeekFrame?: (frame: number) => void;
  onHoverEvent?: (evt: TemporalEvent | null) => void;
}

export default function CausalGraphD3({ result, onSeekFrame, onHoverEvent }: Props) {
  const WIDTH = 900;
  const HEIGHT = Math.max(450, 150 + (result.temporal_events?.length ?? 0) * 105);

  const { nodes, edgeData } = useMemo(() => buildLayout(result, WIDTH, HEIGHT), [result]);

  // ── Zoom / Pan state ──────────────────────────────────────────────────────
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panRef = useRef({ startX: 0, startY: 0, panX: 0, panY: 0, active: false });
  const svgRef = useRef<SVGSVGElement>(null);

  const handleZoomSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setZoom(Math.round(parseFloat(e.target.value) * 10) / 10);
  }, []);

  const handleReset = useCallback(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, []);

  // Drag to pan
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    // Only start drag on left click directly on the SVG background
    if (e.button !== 0) return;
    panRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y, active: true };
    e.preventDefault();
  }, [pan.x, pan.y]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!panRef.current.active) return;
    const dx = e.clientX - panRef.current.startX;
    const dy = e.clientY - panRef.current.startY;
    setPan({ x: panRef.current.panX + dx, y: panRef.current.panY + dy });
  }, []);

  const onMouseUp = useCallback(() => {
    panRef.current.active = false;
  }, []);

  // Mouse wheel zoom
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.round(Math.max(0.3, Math.min(3.0, z + delta)) * 10) / 10);
  }, []);

  const svgHeight = Math.min(HEIGHT, 600);

  return (
    <div className={styles.graphWrap}>
      {/* ── Zoom toolbar ─────────────────────────────────────────────────── */}
      <div className={styles.graphToolbar}>
        <span className={styles.graphToolbarLabel}>Zoom</span>
        <input
          type="range"
          className={styles.graphZoomSlider}
          min="0.3"
          max="3.0"
          step="0.1"
          value={zoom}
          onChange={handleZoomSlider}
        />
        <span className={styles.graphZoomValue}>{Math.round(zoom * 100)}%</span>
        <button className={styles.graphResetBtn} onClick={handleReset}>Reset</button>
      </div>

      {/* ── SVG with zoom/pan ───────────────────────────────────────────── */}
      <div style={{ overflow: 'hidden', borderRadius: 'var(--radius-md)', cursor: panRef.current.active ? 'grabbing' : 'grab' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className={styles.graphSvg}
          style={{ width: '100%', height: `${svgHeight}px`, display: 'block' }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onWheel={onWheel}
        >
          <defs>
            {(['causes', 'supports', 'precedes'] as const).map((rel) => (
              <marker
                key={rel}
                id={`arrow-${rel}`}
                viewBox="0 0 10 10"
                refX={9}
                refY={5}
                markerWidth={7}
                markerHeight={7}
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill={relColour(rel)} />
              </marker>
            ))}
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* All content wrapped in a transform group for zoom/pan */}
          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`} style={{ transformOrigin: '0 0' }}>
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
                    strokeWidth={rel === 'causes' ? 3 : 1.8}
                    strokeDasharray={rel === 'precedes' ? '6 5' : undefined}
                    opacity={confOpacity(conf)}
                    markerEnd={`url(#arrow-${rel})`}
                  />
                  <text
                    x={(e.from.x + e.to.x) / 2 - (e.to.y - e.from.y) * 0.025}
                    y={(e.from.y + e.to.y) / 2 + (e.to.x - e.from.x) * 0.025}
                    textAnchor="middle"
                    fill={relColour(rel)}
                    fontSize={12}
                    opacity={0.85}
                  >
                    {rel} · {conf != null ? conf.toFixed(2) : 'N/A'}
                  </text>
                </g>
              );
            })}

            {/* Nodes */}
            {nodes.map((n) => {
              const isCollision = n.event?.event_type === 'COLLISION' || n.event?.event_type === 'CONTACT';
              const onsetTs = n.event?.onset?.onset_timestamp;

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
                    strokeWidth={isCollision ? 3 : 1.8}
                    filter={isCollision ? 'url(#glow)' : undefined}
                  />
                  {/* Node label */}
                  <text x={n.x} y={n.y - 4} textAnchor="middle" fill={n.colour} fontSize={13} fontWeight={600}>
                    {n.label}
                  </text>
                  {/* Timestamp under event nodes */}
                  {onsetTs != null && (
                    <text x={n.x} y={n.y + 13} textAnchor="middle" fill={n.colour} fontSize={11} opacity={0.8}>
                      onset {onsetTs.toFixed(1)}s
                    </text>
                  )}
                  {/* Confidence */}
                  <text x={n.x} y={n.y + (onsetTs != null ? 25 : 14)} textAnchor="middle" fill={n.colour} fontSize={11} opacity={0.7}>
                    conf {n.event?.confidence != null ? (n.event.confidence * 100).toFixed(0) + '%' : 'N/A'}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>

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
