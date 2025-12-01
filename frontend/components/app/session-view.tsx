'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { useLocalParticipant } from '@livekit/components-react';
import { ParticipantEvent, type LocalParticipant } from 'livekit-client';
import type { AppConfig } from '@/app-config';
import { ChatTranscript } from '@/components/app/chat-transcript';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import { TileLayout } from '@/components/app/tile-layout';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { useChatMessages } from '@/hooks/useChatMessages';
import { useConnectionTimeout } from '@/hooks/useConnectionTimout';
import { useDebugMode } from '@/hooks/useDebug';
import { cn } from '@/lib/utils';
import { ScrollArea } from '../livekit/scroll-area/scroll-area';

const MotionBottom = motion.create('div');

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

const BOTTOM_VIEW_MOTION_PROPS = {
  variants: {
    visible: { opacity: 1, translateY: '0%' },
    hidden: { opacity: 0, translateY: '100%' },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.28,
    delay: 0.4,
    ease: 'easeOut',
  },
};

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

/**
 * Simple edge-fade overlay to soften scroll boundaries.
 */
export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'pointer-events-none h-4',
        top && 'bg-gradient-to-b from-black/15 to-transparent',
        bottom && 'bg-gradient-to-t from-black/15 to-transparent',
        className,
      )}
    />
  );
}

/* === BACKGROUND ICONS === */
const FloatingIcon = ({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) => (
  <div
    className={cn(
      'pointer-events-none absolute select-none text-white/80',
      className,
    )}
  >
    {children}
  </div>
);

const TvIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" className="h-full w-full">
    <path d="M21 6h-7.59l3.29-3.29L16 2l-4 4-4-4-.71.71L10.59 6H3a2 2 0 0 0-2 2v12c0 1.1.9 2 2 2h18a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2Zm0 14H3V8h18v12ZM9 10v8l7-4-7-4Z" />
  </svg>
);

const TeddyIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" className="h-full w-full">
    <path d="M12 2C9 2 7 3.5 6 5 5 5 2 6 2 9c0 2 1.5 3.5 3 3.8V14c0 3 4 4 7 4s7-1 7-4v-1.2c1.5-.3 3-1.8 3-3.8 0-3-3-4-4-4-1-1.5-3-3-6-3Zm-3 8c.8 0 1.5.7 1.5 1.5S9.8 13 9 13s-1.5-.7-1.5-1.5S8.2 10 9 10Zm6 0c.8 0 1.5.7 1.5 1.5S15.8 13 15 13s-1.5-.7-1.5-1.5S14.2 10 15 10Z" />
  </svg>
);

const NoteIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" className="h-full w-full">
    <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55A4 4 0 1 0 14 17V7h4V3h-6Z" />
  </svg>
);

const StarIcon = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" className="h-full w-full">
    <path d="M12 17.27 18.18 21 16.54 13.97 22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27Z" />
  </svg>
);

/* === PLAYER BADGE (top-left) === */
function PlayerBadge({ participant }: { participant?: LocalParticipant }) {
  const [displayName, setDisplayName] = useState('Player One');

  useEffect(() => {
    if (!participant) return;

    const syncName = () => {
      let name = participant.name || '';

      // Optional metadata override if the default name is too generic
      if ((!name || name === 'user' || name === 'identity') && participant.metadata) {
        try {
          const meta = JSON.parse(participant.metadata);
          if (meta.displayName) name = meta.displayName;
          else if (meta.name) name = meta.name;
        } catch {
          // metadata not parseable, ignore
        }
      }

      const cleaned =
        name === 'user' || name === 'identity' || name.trim() === ''
          ? 'Player One'
          : name;

      setDisplayName(cleaned);
    };

    syncName();

    participant.on(ParticipantEvent.NameChanged, syncName);
    participant.on(ParticipantEvent.MetadataChanged, syncName);

    return () => {
      participant.off(ParticipantEvent.NameChanged, syncName);
      participant.off(ParticipantEvent.MetadataChanged, syncName);
    };
  }, [participant]);

  return (
    <motion.div
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.4 }}
      className="absolute left-4 top-4 z-40 flex items-center gap-3 rounded-2xl border border-white/30 bg-white/15 p-2 pr-4 shadow-xl ring-1 ring-white/20 backdrop-blur-2xl"
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-amber-300 to-orange-500 text-purple-900 shadow-sm">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          className="size-6"
        >
          <path
            fillRule="evenodd"
            d="M7.5 6a4.5 4.5 0 1 1 9 0 4.5 4.5 0 0 1-9 0ZM3.751 20.105a8.25 8.25 0 0 1 16.498 0 .75.75 0 0 1-.437.695A18.683 18.683 0 0 1 12 22.5c-2.786 0-5.433-.608-7.812-1.7a.75.75 0 0 1-.437-.695Z"
            clipRule="evenodd"
          />
        </svg>
      </div>
      <div className="flex flex-col">
        <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-white/70">
          Contestant
        </span>
        <span className="leading-none text-sm font-extrabold tracking-wide text-white">
          {displayName}
        </span>
      </div>
    </motion.div>
  );
}

interface SessionViewProps {
  appConfig: AppConfig;
}

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  useConnectionTimeout(200_000);
  useDebugMode({ enabled: IN_DEVELOPMENT });

  const { localParticipant } = useLocalParticipant();
  const messages = useChatMessages();

  const [isChatOpen, setIsChatOpen] = useState(false);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);

  const controlConfig: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: appConfig.supportsChatInput,
    camera: appConfig.supportsVideoInput,
    screenShare: appConfig.supportsVideoInput,
  };

  // Auto-scroll chat when the local user sends a new message
  useEffect(() => {
    const lastMessage = messages.at(-1);
    const isLastMessageLocal = lastMessage?.from?.isLocal === true;

    if (transcriptScrollRef.current && isLastMessageLocal) {
      transcriptScrollRef.current.scrollTop =
        transcriptScrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <section
      className="relative z-10 h-full w-full overflow-hidden"
      {...props}
    >
      {/* Background gradient and subtle floating icons */}
      <div
        aria-hidden="true"
        className="absolute inset-0 -z-10 select-none overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-indigo-900 to-fuchsia-700" />

        <FloatingIcon className="left-[8%] top-[12%] h-12 w-12 -rotate-12 opacity-15">
          <TvIcon />
        </FloatingIcon>
        <FloatingIcon className="right-[12%] top-[24%] h-10 w-10 rotate-6 opacity-15">
          <TeddyIcon />
        </FloatingIcon>
        <FloatingIcon className="left-[6%] bottom-[18%] h-14 w-14 -rotate-3 opacity-10">
          <NoteIcon />
        </FloatingIcon>
        <FloatingIcon className="right-[8%] bottom-[26%] h-9 w-9 rotate-45 opacity-20">
          <StarIcon />
        </FloatingIcon>
        <FloatingIcon className="left-[26%] top-[45%] h-6 w-6 rotate-12 opacity-15">
          <StarIcon />
        </FloatingIcon>
        <FloatingIcon className="right-[30%] bottom-[40%] h-11 w-11 -rotate-[18deg] opacity-10">
          <TvIcon />
        </FloatingIcon>
      </div>

      {/* Participant badge */}
      <PlayerBadge participant={localParticipant} />

      {/* Chat transcript overlay (right side focus) */}
      <div
        className={cn(
          'fixed inset-0 grid grid-cols-1 grid-rows-1',
          !isChatOpen && 'pointer-events-none',
        )}
      >
        <Fade top className="absolute inset-x-4 top-0 h-40" />

        <ScrollArea
          ref={transcriptScrollRef}
          className="px-4 pb-[150px] pt-32 md:px-6 md:pb-[180px]"
        >
          <ChatTranscript
            hidden={!isChatOpen}
            messages={messages}
            className="ml-auto mr-0 max-w-lg space-y-3 transition-opacity duration-300 ease-out md:mr-12"
          />
        </ScrollArea>
      </div>

      {/* Main visual area (agent tiles, etc.) */}
      <TileLayout chatOpen={isChatOpen} />

      {/* Bottom control bar, aligned to the right */}
      <MotionBottom
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="fixed inset-x-3 bottom-0 z-50 md:inset-x-10"
      >
        {appConfig.isPreConnectBufferEnabled && (
          <PreConnectMessage messages={messages} className="pb-4" />
        )}

        <div className="relative ml-auto mr-0 max-w-lg pb-3 md:mr-4 md:pb-10">
          <Fade bottom className="absolute inset-x-0 top-0 h-4 -translate-y-full" />
          <AgentControlBar
            controls={controlConfig}
            onChatOpenChange={setIsChatOpen}
          />
        </div>
      </MotionBottom>
    </section>
  );
};