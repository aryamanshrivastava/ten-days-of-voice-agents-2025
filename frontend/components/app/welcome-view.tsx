'use client';

import React, { useState } from 'react';
import { Button } from '@/components/livekit/button';

function EqualizerIcon() {
  return (
    <svg
      width="64"
      height="64"
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="mb-6 size-20 text-white drop-shadow-[0_0_16px_rgba(255,255,255,0.5)] relative z-10"
    >
      <path
        d="M15 24V40C15 40.8 14.7 41.6 14.1 42.1C13.6 42.7 12.8 43 12 43C11.2 43 10.4 42.7 9.9 42.1C9.3 41.6 9 40.8 9 40V24C9 23.2 9.3 22.4 9.9 21.9C10.4 21.3 11.2 21 12 21C12.8 21 13.6 21.3 14.1 21.9C14.7 22.4 15 23.2 15 24ZM22 5C21.2 5 20.4 5.3 19.9 5.9C19.3 6.4 19 7.2 19 8V56C19 56.8 19.3 57.6 19.9 58.1C20.4 58.7 21.2 59 22 59C22.8 59 23.6 58.7 24.1 58.1C24.7 57.6 25 56.8 25 56V8C25 7.2 24.7 6.4 24.1 5.9C23.6 5.3 22.8 5 22 5ZM32 13C31.2 13 30.4 13.3 29.9 13.9C29.3 14.4 29 15.2 29 16V48C29 48.8 29.3 49.6 29.9 50.1C30.4 50.7 31.2 51 32 51C32.8 51 33.6 50.7 34.1 50.1C34.7 49.6 35 48.8 35 48V16C35 15.2 34.7 14.4 34.1 13.9C33.6 13.3 32.8 13 32 13ZM42 21C41.2 21 40.4 21.3 39.9 21.9C39.3 22.4 39 23.2 39 24V40C39 40.8 39.3 41.6 39.9 42.1C40.4 42.7 41.2 43 42 43C42.8 43 43.6 42.7 44.1 42.1C44.7 41.6 45 40.8 45 40V24C45 23.2 44.7 22.4 44.1 21.9C43.6 21.3 42.8 21 42 21ZM52 17C51.2 17 50.4 17.3 49.9 17.9C49.3 18.4 49 19.2 49 20V44C49 44.8 49.3 45.6 49.9 46.1C50.4 46.7 51.2 47 52 47C52.8 47 53.6 46.7 54.1 46.1C54.7 45.6 55 44.8 55 44V20C55 19.2 54.7 18.4 54.1 17.9C53.6 17.3 52.8 17 52 17Z"
        fill="currentColor"
      />
    </svg>
  );
}

function RetroTvIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <rect x="2" y="7" width="20" height="15" rx="2" ry="2" />
      <polyline points="17 2 12 7 7 2" />
    </svg>
  );
}

export const WelcomeView = React.forwardRef<HTMLDivElement, any>(
  ({ startButtonText, onStartCall }, ref) => {
    const [name, setName] = useState('');
    const [started, setStarted] = useState(false);

    const handleStart = async () => {
      setStarted(true);
      onStartCall?.(name.trim());
    };

    return (
      <div
        ref={ref}
        className="flex min-h-screen w-full flex-col items-center justify-center bg-transparent text-white"
      >
        {!started && (
          <section
            className="
              relative flex w-full max-w-sm flex-col items-center gap-3 rounded-3xl
              bg-[#111111] p-8 text-center
              shadow-[0_0_50px_rgba(147,51,234,0.45),12px_12px_28px_#050505,-12px_-12px_28px_#1f1f1f]
              border border-white/10 mx-4 md:mx-0
            "
          >
            {/* Decorative icons - UPDATED EMOJIS */}
            <RetroTvIcon className="pointer-events-none absolute right-4 top-4 h-9 w-9 -rotate-12 text-cyan-400 drop-shadow-[0_0_10px_rgba(34,211,238,0.7)] opacity-90" />
            <RetroTvIcon className="pointer-events-none absolute bottom-16 left-3 h-7 w-7 rotate-6 text-pink-500 drop-shadow-[0_0_10px_rgba(236,72,153,0.7)] opacity-90" />

            {/* 🎬 New emoji pack */}
            <div className="absolute left-6 top-10 text-3xl rotate-[-15deg] drop-shadow-[0_0_14px_rgba(59,130,246,0.6)] pointer-events-none select-none">
              🎬
            </div>

            <div className="absolute right-6 bottom-4 text-2xl rotate-[10deg] drop-shadow-[0_0_14px_rgba(168,85,247,0.6)] pointer-events-none select-none">
              🌟
            </div>

            <div className="absolute left-3 top-1/2 text-xl -rotate-[18deg] drop-shadow-[0_0_14px_rgba(252,211,77,0.6)] pointer-events-none select-none">
              🪄
            </div>

            <EqualizerIcon />

            <h2 className="relative z-10 mb-1 text-2xl font-bold tracking-tight text-white drop-shadow-md">
              Step Into Improv Battle
            </h2>
            <p className="relative z-10 max-w-prose text-sm font-medium leading-6 text-white/80">
              Pick your stage name and jump into the spotlight.
            </p>

            <div className="relative z-10 mt-7 w-full">
              <label className="mb-2 ml-1 block text-left text-[11px] font-semibold uppercase tracking-[0.18em] text-white/60">
                Stage name
              </label>

              <div className="flex w-full items-stretch gap-2">
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleStart()}
                  placeholder="e.g. Aryaman the Great"
                  className="
                    flex-1 rounded-xl border-none bg-[#151515] px-4 py-3
                    font-semibold text-white placeholder:text-white/40
                    shadow-[inset_4px_4px_8px_#050505,inset_-4px_-4px_8px_#262626]
                    focus:outline-none focus:ring-2 focus:ring-violet-400/60
                    transition-all duration-200
                  "
                />

                <Button
                  variant="primary"
                  size="default"
                  onClick={handleStart}
                  className="
                  h-12
                    flex items-center justify-center rounded-xl px-5 font-bold uppercase
                    bg-gradient-to-r from-yellow-400 to-orange-500 text-purple-900
                    hover:from-yellow-300 hover:to-orange-400
                    shadow-lg transition-all duration-150 hover:scale-[1.03] border-none
                  "
                >
                  {startButtonText ?? 'Start'}
                </Button>
              </div>
            </div>

            <div className="relative z-10 mt-4 font-mono text-[10px] uppercase tracking-[0.22em] text-white/35">
              Press Enter to jump in
            </div>
          </section>
        )}

        {started && (
          <div className="text-center text-2xl font-bold text-white drop-shadow-lg animate-pulse">
            🎤 Warming up the mic…
          </div>
        )}
      </div>
    );
  }
);

WelcomeView.displayName = 'WelcomeView';