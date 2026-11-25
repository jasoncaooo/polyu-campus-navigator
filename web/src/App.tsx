import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchGraph, fetchStatus, resetSession, sendChat } from './api';
import { useRouteDraft } from './components/RouteDraft';
import { ChatPanel } from './components/ChatPanel';
import { MapPanel } from './components/MapPanel';
import { SettingsDialog } from './components/SettingsDialog';
import { TopBar } from './components/TopBar';
import type { AvatarState } from './components/AvatarDock';
import type { CampusMap } from './lib/campusMap';
import { cancelSpeech, speak } from './lib/speech';
import { useSpeechRecognition } from './hooks/useSpeechRecognition';
import type {
  ChatMessage,
  ChatPayload,
  FacilityCard,
  Graph,
  RouteOption,
  Status,
} from './types';

const GREETING =
  'Hi! I am PolyU Campus Navigator Chatbot. I am here to help you navigate the PolyU. ' +
  'Whether you need walking directions or an essential amenity, I have your back! ';

/** Bumped when the shape below changes, so an old transcript is dropped not misread. */
const STORAGE_KEY = 'campus-navigator/transcript/2';

interface Saved {
  messages: ChatMessage[];
  routes: RouteOption[];
  facilities: FacilityCard[];
  routeMessageId: number | null;
}

function greeting(): ChatMessage {
  return { id: 0, role: 'bot', payload: { text: GREETING, route: null, facilities: [] } };
}

/** A route as this build draws it: the numbers the map and the receipt both read. */
function isRoute(value: unknown): boolean {
  const route = value as Record<string, unknown> | null;
  if (!route || typeof route !== 'object') return false;
  if (typeof route.metres !== 'number') return false;
  if (!Array.isArray(route.path) || !Array.isArray(route.segments)) return false;
  return true;
}

function isMessage(value: unknown): value is ChatMessage {
  const message = value as ChatMessage;
  if (!message || typeof message.id !== 'number') return false;
  if (message.role !== 'user' && message.role !== 'bot') return false;
  if (typeof message.payload === 'string') return true;
  if (!message.payload || typeof message.payload.text !== 'string') return false;
  // A saved reply carries a route the map will draw and a list the chat will offer. A
  // payload written by an older build has the same idea and a different shape, and
  // rendering it throws -- which is a white page, not a missing detail. The version in the
  // key is the first guard; this is the second.
  if (message.payload.route != null && !isRoute(message.payload.route)) return false;
  const routes = message.payload.routes;
  if (routes != null && (!Array.isArray(routes) || !routes.every(isRoute))) return false;
  if (message.payload.facilities != null && !Array.isArray(message.payload.facilities)) return false;
  return true;
}

/**
 * What the browser kept from last time.
 *
 * The engine remembers the conversation itself (one Dialogue per browser cookie); this is
 * the part that survives a refresh, a reload and a closed tab, so the words on screen and
 * the line on the map still belong to the same conversation after F5. Anything unexpected
 * in storage means starting clean rather than throwing on the first render.
 */
function load(): Saved {
  const fresh: Saved = { messages: [greeting()], routes: [], facilities: [], routeMessageId: null };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return fresh;
    const saved = JSON.parse(raw) as { v?: number; messages?: unknown };
    if (saved?.v !== 1 || !Array.isArray(saved.messages)) return fresh;
    const messages = saved.messages.filter(isMessage);
    if (!messages.length) return fresh;
    // Only the last reply can still be on the map, and live behaviour replaces the two
    // panels wholesale each turn, so reading one reply restores the state exactly.
    const last = [...messages].reverse().find((message) => message.role === 'bot');
    const payload = last && typeof last.payload !== 'string' ? last.payload : null;
    const routes = payload?.routes ?? [];
    return {
      messages,
      routes,
      facilities: payload?.facilities ?? [],
      routeMessageId: routes.length && last ? last.id : null,
    };
  } catch {
    return fresh;
  }
}

const SAVED = load();

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [engineOffline, setEngineOffline] = useState(false);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [routes, setRoutes] = useState<RouteOption[]>(SAVED.routes);
  const [routeSelection, setRouteSelection] = useState(0);
  const [routeMessageId, setRouteMessageId] = useState<number | null>(SAVED.routeMessageId);
  const [facilities, setFacilities] = useState<FacilityCard[]>(SAVED.facilities);
  const [messages, setMessages] = useState<ChatMessage[]>(SAVED.messages);
  const [draft, setDraft] = useState('');
  const [focusSignal, setFocusSignal] = useState(0);
  const [avatar, setAvatarState] = useState<AvatarState>('idle');
  const [voiceOn, setVoiceOn] = useState(true);
  const [lang, setLang] = useState('en-US');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const mapRef = useRef<CampusMap | null>(null);
  const nextId = useRef(SAVED.messages.reduce((n, message) => Math.max(n, message.id + 1), 1));
  const [busy, setBusy] = useState(false);
  const pending = useRef(false);
  const [dictated, setDictated] = useState(false);
  const [lastReply, setLastReply] = useState<{ text: string; lang: string } | null>(null);
  const [playHint, setPlayHint] = useState('');
  const autoRead = useRef(voiceOn);
  const recognition = useSpeechRecognition((text, final) => {
    setDraft(text);
    setDictated(final);
  });
  const routeDraft = useRouteDraft(draft, lang, graph, dictated);
  const micActive = recognition.phase !== 'idle';
  const micLabel = !micActive ? 'Speak' : recognition.phase === 'listening' ? 'Finish' : 'Cancel';

  const push = useCallback((role: ChatMessage['role'], payload: ChatPayload | string) => {
    const id = nextId.current;
    nextId.current += 1;
    setMessages((previous) => [...previous, { id, role, payload }]);
    return id;
  }, []);

  const stopPlayback = () => {
    cancelSpeech();
    setAvatarState('idle');
  };

  const play = (reply: {text: string; lang: string}) => {
    setPlayHint('');
    setAvatarState('speaking');
    const started = speak(reply.text, reply.lang, (error) => {
      setAvatarState('idle');
      if (error) setPlayHint(error);
    });
    if (!started) setAvatarState('idle');
  };

  const send = async (text: string, fromComposer = false) => {
    const trimmed = text.trim();
    if (!trimmed || pending.current || (fromComposer && (micActive || routeDraft.blocked))) return;
    pending.current = true;
    setBusy(true);
    recognition.cancel();
    stopPlayback();
    const endpoints = fromComposer ? routeDraft.endpoints : undefined;
    const confirmation = fromComposer ? routeDraft.confirmation : undefined;
    push('user', confirmation ? trimmed + '\n' + confirmation : trimmed);
    setDraft('');
    setDictated(false);
    setAvatarState('thinking');
    setPlayHint('');
    try {
      const payload = await sendChat(trimmed, lang, endpoints);
      if (payload.error) throw new Error(payload.error);
      const id = push('bot', payload);
      const options = payload.routes || [];
      setRoutes(options);
      setRouteSelection(0);
      setRouteMessageId(options.length ? id : null);
      setFacilities(payload.facilities || []);
      const reply = payload.speech ? { text: payload.speech, lang: payload.speech_lang || lang } : null;
      setLastReply(reply);
      if (reply && autoRead.current) play(reply);
      else setAvatarState('idle');
    } catch (error) {
      push('bot', { text: String(error), route: null, facilities: [] });
      setAvatarState('error');
    } finally {
      pending.current = false;
      setBusy(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const next = await fetchStatus();
        if (!cancelled) setStatus(next);
      } catch {
        if (!cancelled) setEngineOffline(true);
      }
      try {
        const loaded = await fetchGraph();
        if (!cancelled) setGraph(loaded);
      } catch {
        // the map is decoration around a working chat: a failure here is survivable
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ v: 1, messages }));
    } catch {
      // private mode or a full quota: the transcript is a convenience, not a dependency
    }
  }, [messages]);

  useEffect(() => {
    // Some browsers populate their voice list only after the first request.
    window.speechSynthesis?.getVoices();
    return cancelSpeech;
  }, []);

  const reset = async () => {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    recognition.cancel();
    stopPlayback();
    setDraft('');
    setDictated(false);
    setLastReply(null);
    setPlayHint('');
    try {
      await resetSession();
      try {
        window.localStorage.removeItem(STORAGE_KEY);
      } catch {
        // nothing was stored, so there is nothing to clear
      }
      setMessages([]);
      setRoutes([]);
      setRouteSelection(0);
      setRouteMessageId(null);
      setFacilities([]);
      mapRef.current?.clearOverlays();
      push('bot', { text: 'Conversation cleared. Where would you like to go?', route: null, facilities: [] });
    } catch {
      push('bot', { text: 'Could not reset the conversation. Try again.', route: null, facilities: [] });
    } finally {
      pending.current = false;
      setBusy(false);
    }
  };

  return (
    <>
      <TopBar status={status} engineOffline={engineOffline} onOpenSettings={() => setSettingsOpen(true)} />

      <SettingsDialog
        open={settingsOpen}
        status={status}
        onClose={() => setSettingsOpen(false)}
        onSaved={setStatus}
      />

      <main className="grid h-[calc(100vh-66px)] grid-cols-[minmax(430px,1fr)_minmax(420px,1.1fr)] gap-3.5 p-3.5 max-[1180px]:h-auto max-[1180px]:grid-cols-1">
        <ChatPanel
          messages={messages}
          draft={draft}
          avatar={recognition.phase === 'listening' ? 'listening' : recognition.phase === 'transcribing' ? 'thinking' : avatar}
          voiceOn={voiceOn}
          lang={lang}
          micAvailable={recognition.supported && !busy}
          micRecording={micActive}
          micLabel={micLabel}
          languageLocked={micActive || busy}
          busy={busy}
          canReplay={Boolean(window.speechSynthesis && lastReply && !micActive && !busy)}
          onReplay={() => {
            if (avatar === 'speaking') stopPlayback();
            else if (lastReply) play(lastReply);
          }}
          hint={playHint || recognition.hint}
          editor={routeDraft.editor}
          sendBlocked={busy || micActive || routeDraft.blocked || !draft.trim()}
          focusSignal={focusSignal}
          routeMessageId={routeMessageId}
          routeSelection={routeSelection}
          onDraftChange={(text) => {
            if (micActive) recognition.cancel();
            if (!text.trim()) setDictated(false);
            setDraft(text);
          }}
          onSend={(text) => void send(text, true)}
          onReset={() => void reset()}
          onFacilityClick={(name) => void send('how do I get to ' + name + '?')}
          onSelectRoute={(messageId, index) => {
            setRouteMessageId(messageId);
            setRouteSelection(index);
          }}
          onMic={() => {
            if (!micActive) {
              setDictated(false);
              stopPlayback();
              setPlayHint('');
            }
            recognition.toggle(lang, draft);
          }}
          onToggleVoice={() => {
            autoRead.current = !voiceOn;
            setVoiceOn(!voiceOn);
            if (voiceOn && avatar === 'speaking') stopPlayback();
          }}
          onLangChange={(language) => {
            stopPlayback();
            setLang(language);
            setPlayHint('');
          }}
        />

        <MapPanel
          graph={graph}
          routes={routes}
          routeSelection={routeSelection}
          facilities={facilities}
          mapRef={mapRef}
          onPrefill={(text) => {
            recognition.cancel();
            setDictated(false);
            setDraft(text);
            setFocusSignal((n) => n + 1);
          }}
          onSend={(text) => void send(text)}
        />
      </main>
    </>
  );
}
