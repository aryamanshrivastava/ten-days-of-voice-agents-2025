'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { Monitor, Moon, Sun } from '@phosphor-icons/react';
import { cn } from '@/lib/utils';

// ---------------------------------------------------------------------------
// Theme config
// ---------------------------------------------------------------------------
const THEME_STORAGE_KEY = 'improv-battle-theme';
const THEME_MEDIA_QUERY = '(prefers-color-scheme: dark)';

// Small inline script to set the initial theme BEFORE React renders.
// This avoids a flash of the wrong theme on first paint.
const THEME_SCRIPT = `
  (function() {
    try {
      var doc = document.documentElement;
      var stored = localStorage.getItem("${THEME_STORAGE_KEY}");
      var systemPrefersDark = window.matchMedia("${THEME_MEDIA_QUERY}").matches;
      var effective = stored || (systemPrefersDark ? "dark" : "light");

      if (stored === "system") {
        effective = systemPrefersDark ? "dark" : "light";
      }

      doc.classList.remove("light", "dark");
      doc.classList.add(effective === "dark" ? "dark" : "light");
    } catch (e) {}
  })();
`
  .replace(/\n/g, '')
  .replace(/\s+/g, ' ');

export type ThemeMode = 'dark' | 'light' | 'system';

// Injects the inline theme script into <head>
export function ApplyThemeScript() {
  return (
    <script
      id="theme-script"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }}
    />
  );
}

interface ThemeToggleProps {
  className?: string;
}

export function ThemeToggle({ className }: ThemeToggleProps) {
  const [theme, setTheme] = useState<ThemeMode | undefined>();

  // 1. Read persisted preference on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode | null;
      setTheme(stored ?? 'system');
    } catch {
      setTheme('system');
    }
  }, []);

  // 2. When using 'system', react to OS-level theme changes
  useEffect(() => {
    if (theme !== 'system') return;

    const mediaQuery = window.matchMedia(THEME_MEDIA_QUERY);

    const applySystemTheme = () => {
      const doc = document.documentElement;
      doc.classList.remove('light', 'dark');
      doc.classList.add(mediaQuery.matches ? 'dark' : 'light');
    };

    applySystemTheme();
    mediaQuery.addEventListener('change', applySystemTheme);

    return () => mediaQuery.removeEventListener('change', applySystemTheme);
  }, [theme]);

  const applyTheme = (next: ThemeMode) => {
    const doc = document.documentElement;

    // persist preference
    localStorage.setItem(THEME_STORAGE_KEY, next);
    setTheme(next);

    // update document class list
    doc.classList.remove('light', 'dark');

    if (next === 'system') {
      const prefersDark = window.matchMedia(THEME_MEDIA_QUERY).matches;
      doc.classList.add(prefersDark ? 'dark' : 'light');
    } else {
      doc.classList.add(next);
    }
  };

  // While mounting, render a neutral placeholder to avoid hydration mismatch
  if (!theme) {
    return (
      <div
        className={cn(
          'h-9 w-24 rounded-full bg-neutral-200/40 dark:bg-neutral-800/40',
          className,
        )}
      />
    );
  }

  return (
    <div
      className={cn(
        'group relative flex items-center justify-center gap-1 rounded-full border border-neutral-200 bg-neutral-100 p-1 dark:border-neutral-800 dark:bg-neutral-900',
        className,
      )}
      role="radiogroup"
      aria-label="Select color theme"
    >
      <ThemeButton
        mode="light"
        current={theme}
        onClick={() => applyTheme('light')}
        icon={<Sun weight={theme === 'light' ? 'fill' : 'bold'} />}
        label="Light theme"
      />
      <ThemeButton
        mode="system"
        current={theme}
        onClick={() => applyTheme('system')}
        icon={<Monitor weight={theme === 'system' ? 'fill' : 'bold'} />}
        label="Use system theme"
      />
      <ThemeButton
        mode="dark"
        current={theme}
        onClick={() => applyTheme('dark')}
        icon={<Moon weight={theme === 'dark' ? 'fill' : 'bold'} />}
        label="Dark theme"
      />
    </div>
  );
}

function ThemeButton({
  mode,
  current,
  onClick,
  icon,
  label,
}: {
  mode: ThemeMode;
  current: ThemeMode;
  onClick: () => void;
  icon: ReactNode;
  label: string;
}) {
  const isActive = current === mode;

  return (
    <button
      type="button"
      role="radio"
      aria-checked={isActive}
      aria-label={label}
      onClick={onClick}
      className={cn(
        'relative flex h-7 w-7 items-center justify-center rounded-full text-sm font-medium outline-none transition-all duration-200 focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
        isActive
          ? 'bg-white text-neutral-950 shadow-sm hover:bg-neutral-50 dark:bg-neutral-800 dark:text-neutral-50 dark:hover:bg-neutral-700'
          : 'text-neutral-500 hover:bg-neutral-200/50 hover:text-neutral-900 dark:text-neutral-400 dark:hover:bg-neutral-800/50 dark:hover:text-neutral-100',
      )}
    >
      <span className="z-10 text-base">{icon}</span>
    </button>
  );
}