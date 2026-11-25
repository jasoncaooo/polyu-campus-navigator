import { useEffect, useRef, type ReactNode } from 'react';
import { RotateCcw, Send } from 'lucide-react';
import type { ChatMessage } from '../types';
import { AvatarDock, type AvatarState } from './AvatarDock';
import { Message } from './Message';
import { BTN_GHOST, PANEL, PANEL_HEAD, PANEL_TITLE } from './ui';

export interface ChatPanelProps {
  messages: ChatMessage[];
  draft: string;
  avatar: AvatarState;
  voiceOn: boolean;
  lang: string;
  micAvailable: boolean;
  micRecording: boolean;
  hint: string;
  micLabel: string;
  languageLocked: boolean;
  canReplay: boolean;
  busy: boolean;
  sendBlocked: boolean;
  editor: ReactNode;
  onReplay(): void;
  /** Increments when something outside the panel wants the composer focused. */
  focusSignal: number;
  /** The message whose route list is live on the map, and which option. */
  routeMessageId: number | null;
  routeSelection: number;
  onDraftChange(value: string): void;
  onSend(text: string): void;
  onReset(): void;
  onFacilityClick(name: string): void;
  onSelectRoute(messageId: number, index: number): void;
  onMic(): void;
  onToggleVoice(): void;
  onLangChange(lang: string): void;
}

export function ChatPanel(props: ChatPanelProps) {
  const chat = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const node = chat.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [props.messages]);

  useEffect(() => {
    if (props.focusSignal > 0) composer.current?.focus();
  }, [props.focusSignal]);

  return (
    <section className={PANEL}>
      <div className={PANEL_HEAD}>
        <h2 className={PANEL_TITLE}>Chat</h2>
        <button id="reset" type="button" disabled={props.busy} onClick={props.onReset} className={BTN_GHOST + ' inline-flex items-center gap-1.5'}>
          <RotateCcw size={13} aria-hidden="true" />
          Reset
        </button>
      </div>

      <div ref={chat} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-3.5">
        {props.messages.map((message) => (
          <Message
            key={message.id}
            message={message}
            selection={props.routeMessageId === message.id ? props.routeSelection : null}
            onFacilityClick={props.onFacilityClick}
            onSelectRoute={(index) => props.onSelectRoute(message.id, index)}
          />
        ))}
      </div>

      <AvatarDock
        state={props.avatar}
        voiceOn={props.voiceOn}
        lang={props.lang}
        micAvailable={props.micAvailable}
        micRecording={props.micRecording}
        hint={props.hint}
        micLabel={props.micLabel}
        languageLocked={props.languageLocked}
        canReplay={props.canReplay}
        onReplay={props.onReplay}
        onMic={props.onMic}
        onToggleVoice={props.onToggleVoice}
        onLangChange={props.onLangChange}
      />

      {props.editor}
      <form
        className="flex gap-2 px-4 py-3"
        data-composer-form
        onSubmit={(event) => {
          event.preventDefault();
          if (!props.sendBlocked) props.onSend(props.draft);
        }}
      >
        <input
          id="input" ref={composer}
          type="text"
          data-composer
          autoComplete="off"
          placeholder="Ask about a route, a facility, or opening hours…"
          value={props.draft}
          onChange={(event) => props.onDraftChange(event.target.value)}
          className="min-w-0 flex-1 rounded-[11px] border border-line bg-panel-soft px-[13px] py-[11px] text-sm text-ink placeholder:text-muted/80 focus:border-brand focus:outline-none"
        />
        <button
          type="submit" id="send" disabled={props.sendBlocked}
          className="inline-flex cursor-pointer items-center gap-1.5 rounded-[11px] bg-brand px-4 py-[11px] text-[13px] font-medium text-onaccent transition hover:brightness-110 disabled:opacity-50"
        >
          <Send size={14} aria-hidden="true" />
          Send
        </button>
      </form>
    </section>
  );
}
