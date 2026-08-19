'use client';

import { useState } from 'react';
import { triggerPipeline } from '@/lib/api';
import { useRouter } from 'next/navigation';
import styles from './upload.module.css';

export default function UploadEvent() {
  const [videoPath, setVideoPath] = useState('');
  const [cameraId, setCameraId] = useState('');
  const [srcPtsStr, setSrcPtsStr] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoPath.trim()) return;
    
    let parsedPts: number[][] | undefined = undefined;
    if (srcPtsStr.trim()) {
      try {
        parsedPts = JSON.parse(srcPtsStr);
        if (!Array.isArray(parsedPts) || parsedPts.length !== 4) {
          throw new Error('src_pts must be a 4x2 array');
        }
      } catch (err: any) {
        setError('Invalid JSON for Source Points: ' + err.message);
        return;
      }
    }

    setLoading(true);
    setError(null);
    try {
      const result = await triggerPipeline(videoPath, cameraId.trim() || undefined, parsedPts);
      router.push(`/events/${result.event_id}`);
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Ingest Video Event</h1>
        <p className={styles.subtitle}>Trigger the Track 1 data engineering pipeline manually from local video assets.</p>
      </header>

      <div className={styles.card}>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="videoPath" className={styles.label}>
              Absolute Path to Video File (.mp4)
            </label>
            <input
              id="videoPath"
              type="text"
              value={videoPath}
              onChange={(e) => setVideoPath(e.target.value)}
              placeholder="e.g. C:/videos/intersection.mp4 or ./dataset/test.mp4"
              className={styles.input}
              disabled={loading}
            />
            <p className={styles.hint}>
              The backend will run Phase 0 (heuristic scanning) followed by Perception and Data Storage sequentially.
            </p>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="cameraId" className={styles.label}>
              Camera ID (Optional)
            </label>
            <input
              id="cameraId"
              type="text"
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
              placeholder="e.g. cam_01"
              className={styles.input}
              disabled={loading}
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="srcPts" className={styles.label}>
              Source Points JSON (Optional)
            </label>
            <textarea
              id="srcPts"
              value={srcPtsStr}
              onChange={(e) => setSrcPtsStr(e.target.value)}
              placeholder="[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
              className={styles.input}
              disabled={loading}
              rows={3}
            />
            <p className={styles.hint}>
              4x2 matrix of pixel coordinates for dynamic homography calibration.
            </p>
          </div>

          {error && <div className={styles.errorAlert}>{error}</div>}

          <button 
            type="submit" 
            className={styles.submitBtn} 
            disabled={loading || !videoPath.trim()}
          >
            {loading ? 'Initializing Pipeline...' : 'Trigger Pipeline Analysis'}
          </button>
        </form>
      </div>
    </div>
  );
}
