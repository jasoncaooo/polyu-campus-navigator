import { useEffect, useRef, useState } from 'react';
import { LocalSpeechInput } from '../lib/localSpeech';

type Phase = 'idle' | 'opening' | 'listening' | 'transcribing';
interface Input { start(): void; stop(): void; abort(): void }

const errors: Record<string, string> = {
  'not-allowed': 'Allow microphone access in your browser or type your question.',
  'service-not-allowed': 'Speech input is blocked in this browser. Type your question.',
  'no-speech': 'No speech detected. Try Speak again or type your question.',
  'audio-capture': 'Check your microphone and try again.',
  'network': 'The browser speech service could not connect. Retry or type your question.',
  'language-not-supported': 'This recognition language is unavailable in this browser.',
};

export function useSpeechRecognition(onTranscript: (text: string, final: boolean) => void) {
  const [local, setLocal] = useState<boolean | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [feedback, setFeedback] = useState('');
  const input = useRef<Input | null>(null);
  const Constructor = window.SpeechRecognition || window.webkitSpeechRecognition;
  const supported = local === true || (local === false && Boolean(Constructor));
  const transcript = useRef(onTranscript);
  transcript.current = onTranscript;

  useEffect(() => {
    const request = new AbortController();
    async function probe() {
      let available = false;
      if (LocalSpeechInput.supported()) {
        try {
          const res = await fetch('/api/speech-status', { signal: request.signal });
          available = res.ok && (await res.json()).available === true;
        } catch { /* Fall back to browser recognition. */ }
      }
      if (!request.signal.aborted) setLocal(available);
    }
    void probe();
    return () => {
      request.abort();
      const previous = input.current;
      input.current = null;
      previous?.abort();
    };
  }, []);

  const cancel = () => {
    const previous = input.current;
    input.current = null;
    previous?.abort();
    setPhase('idle');
    setFeedback('');
  };

  const toggle = (lang: string, prefix: string) => {
    if (input.current) {
      if (phase === 'listening') input.current.stop();
      else cancel();
      return;
    }
    if (!supported) return;
    let rec: Input;
    let received = false;
    let failed = false;
    const isCurrent = () => input.current === rec;
    const events = {
      listening() {
        if (!isCurrent()) return;
        setPhase('listening');
        setFeedback('Listening. Select Finish when you are done.');
      },
      transcribing() {
        if (!isCurrent()) return;
        setPhase('transcribing');
        setFeedback('Transcribing your question…');
      },
      result(text: string, final = true) {
        if (!isCurrent() || !text.trim()) return;
        received = true;
        transcript.current([prefix.trimEnd(), text.trim()].filter(Boolean).join(' '), final);
      },
      error(code: string, detail?: string) {
        if (!isCurrent()) return;
        failed = true;
        setFeedback(detail || errors[code] || 'Speech input stopped. Try again or type your question.');
      },
      end() {
        if (!isCurrent()) return;
        input.current = null;
        setPhase('idle');
        if (!failed) setFeedback(received ? 'Review your question, then select Send.' : errors['no-speech']);
      },
    };
    if (local) {
      rec = new LocalSpeechInput(lang, events);
    } else if (Constructor) {
      const browser = new Constructor();
      rec = browser;
      browser.lang = lang;
      browser.continuous = false;
      browser.interimResults = true;
      browser.onstart = events.listening;
      browser.onend = events.end;
      browser.onerror = (event) => events.error(event.error);
      browser.onresult = (event) => {
        const results = Array.from(event.results);
        events.result(results.map((item) => item[0].transcript).join(' '), results.every((item) => item.isFinal));
      };
    } else return;
    input.current = rec;
    setFeedback('Opening microphone…');
    setPhase('opening');
    try { rec.start(); }
    catch { events.error('audio-capture'); events.end(); }
  };

  const readyHint = local === null ? 'Checking speech input…' : local ? 'Offline speech ready.' :
    Constructor ? 'Browser speech recognition ready.' : 'Speech input is unavailable here. Type your question.';
  return { supported, phase, hint: feedback || readyHint, toggle, cancel };
}
