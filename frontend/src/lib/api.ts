/**
 * API client for the Capstone Video Surveillance Backend
 * Connects to FastAPI backend running on http://localhost:8000
 */

// API base URL - can be configured via environment variable
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * EventDetail interface matching the backend Pydantic model
 */
export interface EventDetail {
  Event_ID: string;
  Trigger_Time: number;
  Raw_Video_Path: string;
  Causal_CSV_Path: string;
  Crops_Dir_Path: string;
  Duration_s: number | null;
  Status: string;
  Source_Video_Path: string | null;
  Video_ID: string | null;
}

/**
 * VideoSource interface matching the backend Pydantic model
 */
export interface VideoSource {
  Video_ID: string;
  Label: string;
  File_Path: string;
  Added_At: number;
}

/**
 * Result from a RAG semantic search
 */
export interface RAGSearchResult {
  event_id: string;
  object_id: string;
  image_path: string;
  distance: number;
}

/**
 * Fetch all events from the backend
 */
export async function fetchEvents(): Promise<EventDetail[]> {
  const res = await fetch(`${API_BASE}/api/events`);
  if (!res.ok) {
    throw new Error(`Failed to fetch events: ${res.statusText}`);
  }
  const data = await res.json();
  return data.events;
}

/**
 * Fetch details for a specific event
 */
export async function fetchEventDetail(eventId: string): Promise<EventDetail> {
  const res = await fetch(`${API_BASE}/api/events/${eventId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch event ${eventId}: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch list of crop filenames for an event
 */
export async function fetchCrops(eventId: string): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/events/${eventId}/crops`);
  if (!res.ok) {
    throw new Error(`Failed to fetch crops for ${eventId}: ${res.statusText}`);
  }
  const data = await res.json();
  return data.crops || [];
}

/**
 * Build URL for a specific crop image
 */
export function getCropUrl(eventId: string, filename: string): string {
  return `${API_BASE}/api/events/${eventId}/crops/${filename}`;
}

/**
 * Build URL for the event's CSV file
 */
export function getCsvUrl(eventId: string): string {
  return `${API_BASE}/api/events/${eventId}/csv`;
}

/**
 * Build URL for the event's extracted video clip
 */
export function getVideoUrl(eventId: string): string {
  return `${API_BASE}/api/events/${eventId}/video`;
}

/**
 * Build URL for the event's source video file
 */
export function getSourceVideoUrl(eventId: string): string {
  return `${API_BASE}/api/events/${eventId}/source-video`;
}

/**
 * Fetch all registered video sources
 */
export async function fetchSources(): Promise<VideoSource[]> {
  const res = await fetch(`${API_BASE}/api/sources`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sources: ${res.statusText}`);
  }
  const data = await res.json();
  return data.sources;
}

/**
 * Build URL for streaming a source video
 */
export function getSourceStreamUrl(videoId: string): string {
  return `${API_BASE}/api/sources/${videoId}/stream`;
}

/**
 * Trigger the pipeline on a video file
 */
export async function triggerPipeline(videoPath: string, cameraId?: string, srcPts?: number[][]): Promise<{ event_id: string | null; status: string; message: string }> {
  const payload: any = { video_path: videoPath };
  if (cameraId) payload.camera_id = cameraId;
  if (srcPts) payload.src_pts = srcPts;

  const res = await fetch(`${API_BASE}/api/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Failed to trigger pipeline: ${detail}`);
  }
  return res.json();
}

/**
 * Upload a video file from the browser and trigger the pipeline on it
 */
export async function uploadPipeline(file: File, cameraId?: string, srcPts?: number[][]): Promise<{ event_id: string | null; status: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  if (cameraId) formData.append('camera_id', cameraId);
  if (srcPts) formData.append('src_pts', JSON.stringify(srcPts));

  const res = await fetch(`${API_BASE}/api/pipeline/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Failed to upload pipeline: ${detail}`);
  }
  return res.json();
}

/**
 * Index an event's crops into LanceDB RAG pipeline
 */
export async function ragIngest(eventId: string): Promise<{ status: string; event_id: string; ingested_count: number }> {
  const res = await fetch(`${API_BASE}/api/rag/ingest/${eventId}`, {
    method: 'POST',
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Failed to ingest event: ${detail}`);
  }
  return res.json();
}

/**
 * Perform semantic search for entities
 */
export async function ragSearch(query: string, limit: number = 5): Promise<RAGSearchResult[]> {
  const res = await fetch(`${API_BASE}/api/rag/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Search failed: ${detail}`);
  }
  const data = await res.json();
  return data.results || [];
}

// ── Causal Engine API (Track 2) ──────────────────────────────────────────────

export interface CausalDriver {
  cause: string;
  lag: number;
  strength: number;
  cause_object?: string | null;
  cause_object_frac?: number | null;
}

export interface CausalTarget {
  target_object: string;
  target_class: string;
  target_speed_drop_mps: number;
  target_lead_fraction: number;
  variables: string[];
  n_timesteps: number;
  tau_max: number;
  pc_alpha_used?: number;
  drivers_of_target_speed: CausalDriver[];
}

export interface CausalEpisodeNode {
  node_id: string;
  state: string;
  window_s: [number, number];
  involved_entities?: string[];
  evidence?: string[];
}

export interface CausalEpisodeRelation {
  source_node: string;
  target_node: string;
  relation_type: string;
  mechanism?: string;
}

export interface CausalEpisodeEntity {
  id: string;
  name: string;
  kind: string;
  role: string;
  object_ids: string[];
  class?: string | null;
  colour?: string | null;
}

export interface CausalEpisode {
  entities?: CausalEpisodeEntity[];
  nodes: CausalEpisodeNode[];
  relations: CausalEpisodeRelation[];
  root_cause?: {
    primary_factor?: { kind?: string; text?: string };
    contributing_factors?: string[];
    mitigating_factors?: string[];
    summary?: string;
  };
}

export interface CausalResult {
  status: string;
  message?: string;
  event_id?: string;
  targets?: CausalTarget[];
  episode?: CausalEpisode | null;
  note?: string;
}

/**
 * Run PCMCI+ causal discovery on an event's kinematics (Track 2)
 */
export async function analyzeCausal(eventId: string): Promise<CausalResult> {
  const res = await fetch(`${API_BASE}/api/causal/analyze/${eventId}`, { method: 'POST' });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Causal analysis failed: ${detail}`);
  }
  return res.json();
}

/**
 * Fetch a previously-computed causal graph for an event, if one exists
 */
export async function fetchCausalGraph(eventId: string): Promise<CausalResult | null> {
  const res = await fetch(`${API_BASE}/api/causal/${eventId}`);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`Failed to fetch causal graph: ${res.statusText}`);
  }
  return res.json();
}

// ── Synthesis API (Track 4 — Situation Reports) ──────────────────────────────

export interface SitrepEntity {
  object_id: string;
  class: string;
  frames_tracked: number;
  speed_max_kmh: number;
  speed_mean_kmh: number;
  decelerated: boolean;
  speed_uncertain: boolean;
  entry_s: number;
  exit_s: number;
  colour?: string;
  possible_collision: boolean;
  nearest_at_exit?: { object_id: string; class: string; distance_m: number };
}

export interface SitrepEvidence {
  event: { event_id: string; source: string | null; trigger_time_s: number | null; duration_s: number | null };
  scene: { vehicles_tracked: number; persons_tracked: number; class_counts: Record<string, number>; window_s: [number, number] };
  entities: SitrepEntity[];
  causal: {
    target_object: string;
    target_class: string;
    target_speed_drop_mps: number;
    external_drivers: CausalDriver[];
    interpretation: string;
  }[] | null;
}

export interface SitrepResult {
  status: string;
  message?: string;
  event_id?: string;
  report: string | null;
  evidence?: SitrepEvidence;
}

/**
 * Build the evidence packet and (if $LLM_API_KEY is set) generate the SitRep (Track 4)
 */
export async function generateSitrep(eventId: string): Promise<SitrepResult> {
  const res = await fetch(`${API_BASE}/api/synthesis/${eventId}`, { method: 'POST' });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch { /* use statusText fallback */ }
    throw new Error(`Synthesis failed: ${detail}`);
  }
  return res.json();
}

/**
 * Fetch a previously-generated SitRep for an event, if one exists
 */
export async function fetchSitrep(eventId: string): Promise<SitrepResult | null> {
  const res = await fetch(`${API_BASE}/api/synthesis/${eventId}`);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`Failed to fetch SitRep: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch system configuration
 */
export async function fetchConfig(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) {
    throw new Error(`Failed to fetch config: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch system logs (last 100 lines)
 */
export async function fetchLogs(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/logs`);
  if (!res.ok) {
    throw new Error(`Failed to fetch logs: ${res.statusText}`);
  }
  const data = await res.json();
  return data.logs || [];
}


// ── Privacy & Audit API ──────────────────────────────────────────────────────

/**
 * Privacy status interface
 */
export interface PrivacyStatus {
  face_blur_enabled: boolean;
  plate_redaction_enabled: boolean;
  edge_processing: boolean;
  encryption_enabled: boolean;
  encryption_key_configured: boolean;
  faces_blurred: number;
  plates_redacted: number;
  frames_processed: number;
  crops_processed: number;
}

/**
 * Audit log entry interface
 */
export interface AuditEntry {
  ID: number;
  Timestamp: number;
  Action: string;
  Actor: string;
  Resource: string | null;
  Details: string | null;
  Checksum: string | null;
}

/**
 * Audit log response with pagination
 */
export interface AuditLogResponse {
  entries: AuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Fetch current privacy configuration and statistics
 */
export async function fetchPrivacyStatus(): Promise<PrivacyStatus> {
  const res = await fetch(`${API_BASE}/api/privacy/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch privacy status: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch audit log entries with optional filters
 */
export async function fetchAuditLog(
  limit: number = 50,
  offset: number = 0,
  action?: string,
  actor?: string,
): Promise<AuditLogResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (action) params.set('action', action);
  if (actor) params.set('actor', actor);

  const res = await fetch(`${API_BASE}/api/privacy/audit?${params}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch audit log: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Build URL for exporting audit log as CSV
 */
export function getAuditExportUrl(): string {
  return `${API_BASE}/api/privacy/audit/export`;
}
