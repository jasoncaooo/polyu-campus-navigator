import { Mic, RotateCcw, Volume2, VolumeX } from 'lucide-react';
import campusGuide from '../assets/polyu-campus-guide-2026-11-18.png';

export type AvatarState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'error';

const AVATAR_LABELS: Record<AvatarState, string> = {
  idle: 'idle',
  listening: 'listening…',
  thinking: 'thinking…',
  speaking: 'speaking…',
  error: 'no connection',
};

export interface AvatarDockProps {
  state: AvatarState;
  voiceOn: boolean;
  lang: string;
  micAvailable: boolean;
  micRecording: boolean;
  micLabel: string;
  canReplay: boolean;
  languageLocked: boolean;
  onReplay(): void;
  hint: string;
  onMic(): void;
  onToggleVoice(): void;
  onLangChange(lang: string): void;
}

/* The approved campus guide stays fully visible; the dock retains its voice controls. */
function CampusGuide({ state }: { state: AvatarState }) {
  return (
    <div className={'campus-guide ' + state}>
      <img
        src={campusGuide}
        alt="PolyU campus guide holding a map and pointing the way"
        className="campus-guide-image"
        draggable={false}
      />
    </div>
  );
}

const VOICE_BTN =
  'inline-flex cursor-pointer items-center gap-[7px] rounded-[11px] border border-line bg-panel-soft ' +
  'px-[13px] py-2 text-[13px] text-ink transition hover:border-line-strong ' +
  'disabled:cursor-not-allowed disabled:opacity-45';

export function AvatarDock(props: AvatarDockProps) {
  const { state, voiceOn, lang, micAvailable, micRecording, hint } = props;
  return (
    <div className="relative z-[2] mx-4 my-2 flex items-center gap-3 rounded-[14px] border border-line bg-panel px-3.5 py-2.5 shadow-[0_8px_20px_var(--accent-ring)]">
      <CampusGuide state={state} />
      <div className="flex min-w-0 flex-1 flex-col gap-[7px]">
        <div role="status" className="self-start rounded-full border border-line bg-panel-soft px-[9px] py-0.5 text-[10px] uppercase tracking-[0.14em] text-muted">
          {AVATAR_LABELS[state]}
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            id="mic" title="Dictate a question" aria-pressed={micRecording}
            disabled={!micAvailable}
            onClick={props.onMic}
            className={
              VOICE_BTN +
              (micRecording
                ? ' animate-[pulseRing_1.1s_infinite] border-danger bg-danger-soft text-danger'
                : '')
            }
          >
            <Mic size={16} strokeWidth={1.9} aria-hidden="true" />
            <span>{props.micLabel}</span>
          </button>

          <button
            type="button"
            id="speaker" title="Automatically read new replies" aria-pressed={voiceOn} disabled={!window.speechSynthesis}
            onClick={props.onToggleVoice}
            className={VOICE_BTN + (voiceOn ? ' border-brand bg-accent-soft text-brand' : '')}
          >
            {voiceOn ? (
              <Volume2 size={16} strokeWidth={1.9} aria-hidden="true" />
            ) : (
              <VolumeX size={16} strokeWidth={1.9} aria-hidden="true" />
            )}
            <span>Auto-read</span>
          </button>

          <button id="replay" type="button" onClick={props.onReplay} disabled={!props.canReplay}
            className={VOICE_BTN}>
            <RotateCcw size={15} aria-hidden="true" />
            {state === 'speaking' ? 'Stop' : 'Replay'}
          </button>

          <label title="Speech language">
            <select
              id="lang" aria-label="Speech language" disabled={props.languageLocked}
              value={lang}
              onChange={(event) => props.onLangChange(event.target.value)}
              className="cursor-pointer rounded-[11px] border border-line bg-panel-soft px-2.5 py-2 font-sans text-[13px] text-ink"
            >
              <option value="en-US">English</option>
              <option value="zh-HK">廣東話</option>
              <option value="zh-CN">普通话</option>
            </select>
          </label>
        </div>

        <p id="voice-hint" role="status" className="m-0 text-[11px] leading-snug text-muted">{hint}</p>
      </div>
    </div>
  );
}
