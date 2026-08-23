'use client';

import { useEffect, useState, use } from 'react';
import { fetchEventDetail, fetchCrops, EventDetail, getCropUrl, getCsvUrl, getSourceVideoUrl, getVideoUrl } from '@/lib/api';
import VideoAnnotator from '@/components/VideoAnnotator';
import CausalPanel from '@/components/CausalPanel';
import styles from './event.module.css';
import Link from 'next/link';

export default function EventDetailView({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [event, setEvent] = useState<EventDetail | null>(null);
  const [crops, setCrops] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [csvData, setCsvData] = useState<any[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchEventDetail(id);
        setEvent(data);
        
        if (data.Status === 'Extracted') {
          const fetchedCrops = await fetchCrops(id);
          setCrops(fetchedCrops);
          
          // Fetch and parse CSV
          const csvRes = await fetch(getCsvUrl(id));
          if (csvRes.ok) {
            const csvText = await csvRes.text();
            const lines = csvText.trim().split('\n');
            const headers = lines[0].split(',');
            const parsedRows = lines.slice(1).map(line => {
              const values = line.split(',');
              const row: any = {};
              headers.forEach((h, i) => { row[h.trim()] = values[i]; });
              return row;
            });
            setCsvData(parsedRows);
          }
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading) return <div className={styles.container}>Loading Event Data...</div>;
  if (!event) return <div className={styles.container}>Event Not Found.</div>;

  // Use source video if available, otherwise fall back to extracted clip
  const hasSourceVideo = !!event.Source_Video_Path;
  const videoUrl = hasSourceVideo ? getSourceVideoUrl(event.Event_ID) : getVideoUrl(event.Event_ID);

  // Event clip boundaries in the source-video timeline. The CSV timestamps are
  // anchor-relative — frame 0 sits at -pre_buffer_seconds (the incident anchor is
  // t=0) — so the clip's source-time start/end = Trigger_Time + min/max(Timestamp).
  // Deriving the window from the CSV (instead of a hardcoded pre-buffer) keeps the
  // overlay boxes in sync with source playback. Falls back to a 4s pre-buffer
  // estimate when no CSV is available yet.
  const csvTimestamps = csvData
    .map(row => parseFloat(row.Timestamp))
    .filter(n => Number.isFinite(n));
  const windowStart = csvTimestamps.length ? Math.min(...csvTimestamps) : null;
  const windowEnd = csvTimestamps.length ? Math.max(...csvTimestamps) : null;
  const eventStartSec = hasSourceVideo
    ? (windowStart !== null ? event.Trigger_Time + windowStart : Math.max(0, event.Trigger_Time - 4.0))
    : 0;
  const eventEndSec = hasSourceVideo
    ? (windowEnd !== null ? event.Trigger_Time + windowEnd : eventStartSec + (event.Duration_s || 10.0))
    : (event.Duration_s || 10.0);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.titleGroup}>
          <Link href="/" className={styles.backBtn}>← Back</Link>
          <h1 className={styles.title}>Event Analysis: {event.Event_ID}</h1>
        </div>
        <div className={styles.metaCards}>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Status</span>
            <span className={styles.metaValue}>{event.Status}</span>
          </div>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Trigger Time</span>
            <span className={styles.metaValue}>{event.Trigger_Time.toFixed(1)}s into source</span>
          </div>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Event Window</span>
            <span className={styles.metaValue}>{eventStartSec.toFixed(1)}s – {eventEndSec.toFixed(1)}s</span>
          </div>
          <div className={styles.metaCard}>
            <span className={styles.metaLabel}>Total Tracks</span>
            <span className={styles.metaValue}>{crops.length} entities</span>
          </div>
        </div>
      </header>

      {event.Status === 'Extracted' && (
        <div className={styles.mediaGrid}>
          {/* Main Video & Analytics Pane */}
          <div className={styles.videoSection}>
            <div className={styles.playerCard}>
              <VideoAnnotator 
                videoUrl={videoUrl}
                csvData={csvData}
                eventStartSec={hasSourceVideo ? eventStartSec : 0}
                eventEndSec={hasSourceVideo ? eventEndSec : (event.Duration_s || 10)}
                isSourceVideo={hasSourceVideo}
              />
            </div>
          </div>
          
          {/* Entity Crops Gallery */}
          <div className={styles.cropsSection}>
            <h2 className={styles.sectionTitle}>Detected Entities</h2>
            <div className={styles.cropsGrid}>
              {crops.length > 0 ? crops.map(filename => (
                <div key={filename} className={styles.cropCard}>
                  <img src={getCropUrl(event.Event_ID, filename)} alt={filename} className={styles.cropImg} />
                  <div className={styles.cropLabel}>{filename.replace(`${event.Event_ID}_`, '').replace('_crop.jpg', '')}</div>
                </div>
              )) : (
                <div className={styles.emptyCrops}>No entities detected during event.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {event.Status === 'Extracted' && <CausalPanel eventId={event.Event_ID} csvData={csvData} />}

      {event.Status === 'Processing' && (
        <div className={styles.processingPane}>
          <div className={styles.spinner}></div>
          <p>This event is currently being processed by the pipeline. Please refresh to check status.</p>
        </div>
      )}
    </div>
  );
}
