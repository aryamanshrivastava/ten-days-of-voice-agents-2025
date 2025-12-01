'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { Track } from 'livekit-client';
import { AnimatePresence, motion } from 'motion/react';
import {
  type TrackReference,
  VideoTrack,
  useLocalParticipant,
  useTracks,
  useVoiceAssistant,
} from '@livekit/components-react';
import { cn } from '@/lib/utils';

const MotionContainer = motion.create('div');

const TILE_SPRING = {
  type: 'spring' as const,
  stiffness: 720,
  damping: 48,
  mass: 1,
};

const layoutClasses = {
  grid: [
    'h-full w-full',
    'grid gap-x-4 place-content-center',
    'grid-cols-[1fr_1fr] grid-rows-[60px_1fr_60px]',
  ],
  agentChatOpenWithSecondTile: ['col-start-1 row-start-1', 'self-center justify-self-end'],
  agentChatOpenWithoutSecondTile: ['col-start-1 row-start-1', 'col-span-2', 'place-content-center'],
  agentChatClosed: ['col-start-1 row-start-1', 'col-span-2 row-span-3', 'place-content-center'],
  secondTileChatOpen: ['col-start-2 row-start-1', 'self-center justify-self-start'],
  secondTileChatClosed: ['col-start-2 row-start-3', 'place-content-end'],
};

export function useLocalTrackRef(source: Track.Source) {
  const { localParticipant } = useLocalParticipant();
  const publication = localParticipant.getTrackPublication(source);

  const trackRef = useMemo<TrackReference | undefined>(
    () =>
      publication
        ? {
            source,
            participant: localParticipant,
            publication,
          }
        : undefined,
    [source, publication, localParticipant],
  );

  return trackRef;
}

/**
 * Audio waveform / ECG-style visualizer for the agent's audio track.
 */
const AudioWaveform = ({
  trackRef,
  className,
}: {
  trackRef?: TrackReference;
  className?: string;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas || !trackRef?.publication?.track) return;
    const mediaTrack = trackRef.publication.track.mediaStreamTrack;
    if (!mediaTrack) return;
    if (typeof window === 'undefined') return;

    // Web Audio plumbing
    const stream = new MediaStream([mediaTrack]);
    const AudioCtx = (window as any).AudioContext || (window as any).webkitAudioContext;
    const audioContext = new AudioCtx();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);

    analyser.fftSize = 2048;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let animationId: number;

    const renderWaveform = () => {
      animationId = requestAnimationFrame(renderWaveform);
      analyser.getByteTimeDomainData(dataArray);

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const { width, height } = canvas;

      ctx.clearRect(0, 0, width, height);

      // Waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#22d3ee'; // cyan-400
      ctx.shadowBlur = 6;
      ctx.shadowColor = '#22d3ee';

      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const value = dataArray[i] / 128.0; // [0, 2], 1 = center
        const y = (value * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    renderWaveform();

    return () => {
      cancelAnimationFrame(animationId);
      try {
        source.disconnect();
        analyser.disconnect();
      } catch {
        // ignore teardown errors
      }
      audioContext.close();
    };
  }, [trackRef]);

  return <canvas ref={canvasRef} className={className} width={300} height={150} />;
};

interface TileLayoutProps {
  chatOpen: boolean;
}

export function TileLayout({ chatOpen }: TileLayoutProps) {
  const {
    state: agentState, // currently unused but kept for future logic
    audioTrack: agentAudioTrack,
    videoTrack: agentVideoTrack,
  } = useVoiceAssistant();

  const [screenShareTrack] = useTracks([Track.Source.ScreenShare]);
  const cameraTrack: TrackReference | undefined = useLocalTrackRef(Track.Source.Camera);

  const isCameraEnabled = cameraTrack && !cameraTrack.publication.isMuted;
  const isScreenShareEnabled = screenShareTrack && !screenShareTrack.publication.isMuted;
  const hasSecondaryTile = isCameraEnabled || isScreenShareEnabled;

  const animationDelay = chatOpen ? 0 : 0.12;
  const isAvatarEnabled = agentVideoTrack !== undefined;
  const videoWidth = agentVideoTrack?.publication.dimensions?.width ?? 0;
  const videoHeight = agentVideoTrack?.publication.dimensions?.height ?? 0;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-8 bottom-32 z-50 md:top-12 md:bottom-40">
      <div className="relative mx-auto h-full max-w-4xl px-4 md:px-0">
        <div className={cn(layoutClasses.grid)}>
          {/* Agent tile */}
          <div
            className={cn([
              'grid transition-all duration-500 ease-spring',
              !chatOpen && layoutClasses.agentChatClosed,
              chatOpen && hasSecondaryTile && layoutClasses.agentChatOpenWithSecondTile,
              chatOpen && !hasSecondaryTile && layoutClasses.agentChatOpenWithoutSecondTile,
            ])}
          >
            <AnimatePresence mode="popLayout">
              {!isAvatarEnabled && (
                // Voice-only agent (ECG / waveform tile)
                <MotionContainer
                  key="agent-audio"
                  layoutId="agent"
                  initial={{ opacity: 0, scale: 0.8, filter: 'blur(10px)' }}
                  animate={{
                    opacity: 1,
                    scale: chatOpen ? 1 : 1.18,
                    filter: 'blur(0px)',
                  }}
                  transition={{ ...TILE_SPRING, delay: animationDelay }}
                  className={cn(
                    'relative overflow-hidden',
                    'bg-black/95 backdrop-blur-md',
                    'border border-cyan-400/40',
                    'shadow-[0_0_16px_-4px_rgba(34,211,238,0.45)]',
                    chatOpen
                      ? 'h-[60px] w-[60px] rounded-lg'
                      : 'h-[120px] w-[120px] rounded-xl',
                  )}
                >
                  <div
                    className="absolute inset-0 z-0 opacity-15"
                    style={{
                      backgroundImage:
                        'linear-gradient(rgba(34,211,238,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.4) 1px, transparent 1px)',
                      backgroundSize: '10px 10px',
                    }}
                  />
                  <AudioWaveform
                    trackRef={agentAudioTrack}
                    className="relative z-10 h-full w-full"
                  />
                </MotionContainer>
              )}

              {isAvatarEnabled && (
                // Avatar video agent
                <MotionContainer
                  key="agent-avatar"
                  layoutId="avatar"
                  initial={{
                    scale: 1,
                    opacity: 1,
                    maskImage: 'radial-gradient(circle, black 0%, transparent 0%)',
                  }}
                  animate={{
                    maskImage: chatOpen
                      ? 'radial-gradient(circle, black 100%, transparent 100%)'
                      : 'radial-gradient(circle, black 65%, transparent 80%)',
                    borderRadius: chatOpen ? 8 : 12,
                  }}
                  transition={{
                    ...TILE_SPRING,
                    delay: animationDelay,
                    maskImage: { duration: 0.8 },
                  }}
                  className={cn(
                    'relative overflow-hidden bg-black',
                    'border border-cyan-400/30 shadow-[0_0_16px_-4px_rgba(34,211,238,0.35)]',
                    chatOpen
                      ? 'h-[60px] w-[60px]'
                      : 'aspect-video h-auto w-full max-w-[400px]',
                  )}
                >
                  <VideoTrack
                    width={videoWidth}
                    height={videoHeight}
                    trackRef={agentVideoTrack}
                    className={cn(
                      'h-full w-full object-cover opacity-95',
                      chatOpen ? 'scale-110' : 'scale-100',
                    )}
                  />
                </MotionContainer>
              )}
            </AnimatePresence>
          </div>

          {/* Secondary tile: camera or screen-share */}
          <div
            className={cn([
              'grid transition-all duration-500',
              chatOpen && layoutClasses.secondTileChatOpen,
              !chatOpen && layoutClasses.secondTileChatClosed,
            ])}
          >
            <AnimatePresence>
              {(isCameraEnabled || isScreenShareEnabled) && (
                <MotionContainer
                  key="camera-tile"
                  layout="position"
                  layoutId="camera"
                  initial={{ opacity: 0, scale: 0.82, y: 18 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.82, y: 18 }}
                  transition={{ ...TILE_SPRING, delay: animationDelay }}
                  className={cn(
                    'relative overflow-hidden',
                    'h-[60px] w-[60px] rounded-lg',
                    'border border-neutral-800 bg-neutral-900',
                    'shadow-lg shadow-black/40',
                  )}
                >
                  <VideoTrack
                    trackRef={cameraTrack || screenShareTrack}
                    width={
                      (cameraTrack || screenShareTrack)?.publication.dimensions?.width ??
                      0
                    }
                    height={
                      (cameraTrack || screenShareTrack)?.publication.dimensions?.height ??
                      0
                    }
                    className="h-full w-full object-cover grayscale-[0.08]"
                  />
                  <div className="absolute bottom-1 h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_5px_rgba(34,211,238,1)]" />
                </MotionContainer>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}