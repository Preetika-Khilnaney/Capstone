'use client';

import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import styles from './annotator.module.css';

export interface VideoAnnotatorHandle {
  seek: (time: number) => void;
}

interface VideoAnnotatorProps {
  videoUrl: string;
  csvData: any[];
  eventStartSec: number;
  eventEndSec: number;
  isSourceVideo: boolean;
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const VideoAnnotator = forwardRef<VideoAnnotatorHandle, VideoAnnotatorProps>(
  function VideoAnnotator({ videoUrl, csvData, eventStartSec, eventEndSec, isSourceVideo }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const timelineRef = useRef<HTMLDivElement>(null);
    const [showAnnotations, setShowAnnotations] = useState(true);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    const framesData = useRef<Map<number, any[]>>(new Map());

    useEffect(() => {
      const map = new Map<number, any[]>();
      csvData.forEach(row => {
        const frameId = parseInt(row.Frame_ID, 10);
        if (!isNaN(frameId)) {
          if (!map.has(frameId)) map.set(frameId, []);
          map.get(frameId)!.push(row);
        }
      });
      framesData.current = map;
    }, [csvData]);

    const isInEventWindow = useCallback((time: number) => {
      return time >= eventStartSec && time <= eventEndSec;
    }, [eventStartSec, eventEndSec]);

    const drawAnnotations = useCallback(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth || 1920;
        canvas.height = video.videoHeight || 1080;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (showAnnotations && isInEventWindow(video.currentTime)) {
        const relativeTime = isSourceVideo ? video.currentTime - eventStartSec : video.currentTime;
        const currentFrame = Math.max(0, Math.floor(relativeTime * 10));
        const rows = framesData.current.get(currentFrame) || [];
        rows.forEach(row => {
          const x1 = parseFloat(row.BBox_X1);
          const y1 = parseFloat(row.BBox_Y1);
          const x2 = parseFloat(row.BBox_X2);
          const y2 = parseFloat(row.BBox_Y2);
          if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) return;
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = 'rgba(59, 130, 246, 0.85)';
          const velocity = parseFloat(row.Velocity_mps || '0');
          const label = `${row.Class} ${row.Object_ID} | ${velocity.toFixed(1)}m/s`;
          ctx.font = 'bold 14px monospace';
          const textWidth = ctx.measureText(label).width;
          ctx.fillRect(x1, y1 - 24, textWidth + 10, 24);
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, x1 + 5, y1 - 7);
        });
      }
    }, [showAnnotations, isSourceVideo, eventStartSec, isInEventWindow]);

    // Run the render loop only while playing; the loop stops on its own when the
    // media pauses/ends, and the effect cleanup cancels a pending frame.
    useEffect(() => {
      if (!isPlaying) return;
      let raf: number = 0;
      function tick() {
        drawAnnotations();
        const v = videoRef.current;
        if (v && !v.paused && !v.ended) {
          raf = requestAnimationFrame(tick);
        }
      }
      raf = requestAnimationFrame(tick);
      return () => cancelAnimationFrame(raf);
    }, [isPlaying, drawAnnotations]);

    // Keep the playhead/timer state in step with the media clock without re-deriving
    // it from canvas frames: `timeupdate` (~4Hz) is enough for a 10fps overlay and is
    // robust even when requestAnimationFrame is throttled (background/battery saver).
    const syncTime = useCallback(() => {
      if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
    }, []);

    const handlePlay = () => { setIsPlaying(true); };
    const handlePause = () => { setIsPlaying(false); syncTime(); drawAnnotations(); };
    const handleEnded = () => { setIsPlaying(false); syncTime(); drawAnnotations(); };
    const handleSeek = () => { syncTime(); drawAnnotations(); };
    const handleTimeUpdate = () => { syncTime(); };
    const handleLoadedMetadata = () => { if (videoRef.current) setDuration(videoRef.current.duration); };

    useEffect(() => { drawAnnotations(); }, [showAnnotations, csvData, drawAnnotations]);

    // Expose seek() to the parent. Defined after drawAnnotations so the handle
    // always captures the latest render closure.
    useImperativeHandle(ref, () => ({
      seek(time: number) {
        const video = videoRef.current;
        if (!video) return;
        // If metadata not loaded yet, wait for it then seek
        if (video.readyState >= 1) {
          video.currentTime = time;
          drawAnnotations();
        } else {
          const onLoaded = () => {
            video.currentTime = time;
            drawAnnotations();
            video.removeEventListener('loadedmetadata', onLoaded);
          };
          video.addEventListener('loadedmetadata', onLoaded);
        }
      },
    }));

    const jumpToEvent = () => {
      if (videoRef.current) { videoRef.current.currentTime = eventStartSec; syncTime(); drawAnnotations(); }
    };

    const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
      if (!timelineRef.current || !videoRef.current || duration === 0) return;
      const rect = timelineRef.current.getBoundingClientRect();
      videoRef.current.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
      syncTime();
      drawAnnotations();
    };

    const eventStartPct = duration > 0 ? (eventStartSec / duration) * 100 : 0;
    const eventWidthPct = duration > 0 ? ((eventEndSec - eventStartSec) / duration) * 100 : 0;
    const playheadPct = duration > 0 ? (currentTime / duration) * 100 : 0;
    const inEvent = isInEventWindow(currentTime);

    return (
      <div className={styles.wrapper}>
        <div className={styles.videoContainer}>
          <video
            ref={videoRef}
            src={videoUrl}
            className={styles.video}
            onPlay={handlePlay}
            onPause={handlePause}
            onEnded={handleEnded}
            onSeeked={handleSeek}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            crossOrigin="anonymous"
          />
          <canvas ref={canvasRef} className={styles.canvas} />
        </div>

        <div className={styles.controlsBar}>
          <button className={styles.playBtn} onClick={() => {
            if (videoRef.current) { if (videoRef.current.paused) videoRef.current.play(); else videoRef.current.pause(); }
          }}>
            {isPlaying ? '⏸' : '▶'}
          </button>
          <span className={styles.timeDisplay}>{formatTime(currentTime)} / {formatTime(duration)}</span>
          <div className={styles.timeline} ref={timelineRef} onClick={handleTimelineClick}>
            <div className={styles.timelineTrack}></div>
            {isSourceVideo && (
              <div className={styles.eventHighlight} style={{ left: `${eventStartPct}%`, width: `${eventWidthPct}%` }}>
                <div className={styles.eventHighlightLabel}>EVENT</div>
              </div>
            )}
            <div className={styles.playhead} style={{ left: `${playheadPct}%` }} />
          </div>
          <div className={styles.rightControls}>
            {isSourceVideo && (
              <button className={styles.jumpBtn} onClick={jumpToEvent} title="Jump to extracted event">
                Go to Event
              </button>
            )}
            <button className={`${styles.toggleBtn} ${showAnnotations ? styles.active : ''}`} onClick={() => setShowAnnotations(!showAnnotations)}>
              {showAnnotations ? '◉ Overlay ON' : '○ Overlay OFF'}
            </button>
          </div>
        </div>

        {isSourceVideo && (
          <div className={`${styles.eventIndicator} ${inEvent ? styles.eventActive : ''}`}>
            {inEvent ? '● Currently viewing extracted event window' : '○ Outside event window — annotations inactive'}
          </div>
        )}
      </div>
    );
  }
);

export default VideoAnnotator;
