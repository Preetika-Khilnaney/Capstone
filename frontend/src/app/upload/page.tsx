'use client';

import { useState } from 'react';
import { uploadPipeline } from '@/lib/api';
import { useRouter } from 'next/navigation';
import styles from './upload.module.css';

export default function UploadEvent() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [cameraId, setCameraId] = useState('');
  const [srcPtsStr, setSrcPtsStr] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setVideoFile(file);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoFile) return;

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
      const result = await uploadPipeline(videoFile, cameraId.trim() || undefined, parsedPts);
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
        <p className={styles.subtitle}>Upload a video from this machine and trigger the Track 1 data engineering pipeline.</p>
      </header>

      <div className={styles.card}>
        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.formGroup}>
            <label htmlFor="videoFile" className={styles.label}>
              Select Video File (.mp4, .avi, .mov, .mkv)
            </label>
            <input
              id="videoFile"
              type="file"
              accept="video/*,.mp4,.avi,.mov,.mkv,.webm,.m4v"
              onChange={handleFileChange}
              className={styles.input}
              disabled={loading}
            />
            {videoFile && (
              <p className={styles.hint}>
                Selected: {videoFile.name} ({Math.round(videoFile.size / 1048576 * 100) / 100} MB)
              </p>
            )}
            <p className={styles.hint}>
              The file is uploaded to the server, then Phase 0 (heuristic scanning) followed by Perception and Data Storage runs sequentially.
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
            disabled={loading || !videoFile}
          >
            {loading ? 'Uploading & Initializing Pipeline...' : 'Upload & Trigger Pipeline Analysis'}
          </button>
        </form>
      </div>
    </div>
  );
}