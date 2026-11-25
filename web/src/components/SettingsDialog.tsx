import { useEffect, useRef, useState, type RefObject } from 'react';
import { ArrowUpRight, X } from 'lucide-react';
import { saveSettings } from '../api';
import type { Status } from '../types';
import { BTN_GHOST, BTN_PRIMARY, FIELD_INPUT } from './ui';

export interface SettingsDialogProps {
  open: boolean;
  status: Status | null;
  onClose(): void;
  onSaved(status: Status): void;
}

/* Where each key comes from. The dialog is the only place a key can be entered, so the
   link that produces one sits next to the field that wants it. */
const LINKS = {
  llm: 'https://platform.deepseek.com/api_keys',
  embeddings: 'https://bailian.console.alibabacloud.com/',
  vectors: 'https://app.pinecone.io/',
};

function KeyRow(props: {
  label: string;
  link: string;
  linkLabel: string;
  hint: string;
  placeholder: string;
  value: string;
  onChange(value: string): void;
  inputRef?: RefObject<HTMLInputElement>;
  onEnter?: () => void;
}) {
  return (
    <label className="mb-3 block">
      <span className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[11.5px] text-muted">{props.label}</span>
        <a
          href={props.link}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-0.5 text-[11px] text-brand underline decoration-dotted underline-offset-2 hover:decoration-solid"
        >
          {props.linkLabel}
          <ArrowUpRight size={10} aria-hidden="true" />
        </a>
      </span>
      <input
        ref={props.inputRef}
        type="password"
        autoComplete="off"
        spellCheck={false}
        placeholder={props.placeholder}
        value={props.value}
        onChange={(event) => props.onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') props.onEnter?.();
        }}
        className={FIELD_INPUT}
      />
      <span className="mt-1 block text-[11px] leading-snug text-muted">{props.hint}</span>
    </label>
  );
}

export function SettingsDialog({ open, status, onClose, onSaved }: SettingsDialogProps) {
  const [key, setKey] = useState('');
  const [model, setModel] = useState('');
  const [embeddingsKey, setEmbeddingsKey] = useState('');
  const [pineconeKey, setPineconeKey] = useState('');
  const [message, setMessage] = useState('');
  const [tone, setTone] = useState<'idle' | 'ok' | 'err'>('idle');
  const keyField = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setKey('');
    setModel('');
    setEmbeddingsKey('');
    setPineconeKey('');
    setTone('idle');
    const stored = [
      status?.api_key_present ? 'LLM key set' : 'no LLM key (grammar only)',
      status?.embeddings_key_present ? 'embeddings key set' : 'no embeddings key',
      status?.pinecone_key_present ? 'vector-store key set' : 'no vector-store key',
    ];
    setMessage('Stored: ' + stored.join(' · ') + '. Leave a field blank to keep what is there.');
    keyField.current?.focus();
  }, [open, status]);

  if (!open) return null;

  const mode = status?.places?.mode;

  const save = async () => {
    const payload = {
      apiKey: key.trim() || null,
      model: model.trim() || null,
      embeddingsApiKey: embeddingsKey.trim() || null,
      pineconeApiKey: pineconeKey.trim() || null,
    };
    if (!payload.apiKey && !payload.model && !payload.embeddingsApiKey && !payload.pineconeApiKey) {
      setMessage('Nothing to save — every field is empty.');
      setTone('err');
      return;
    }
    setMessage('Saving…');
    setTone('idle');
    try {
      const result = await saveSettings(payload);
      onSaved(result.status);
      const places = result.status.places;
      setMessage(
        'Saved to ' +
          result.env +
          '. Place names: ' +
          (places?.mode === 'dense'
            ? 'dense (' + places.vectors + ' vectors).'
            : 'built-in scorer — ' + (places?.warning || 'the vector index is not ready') + '.'),
      );
      setTone('ok');
      setTimeout(onClose, 1400);
    } catch (error) {
      setMessage('Could not reach the engine: ' + String(error));
      setTone('err');
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] grid place-items-center bg-[rgba(28,18,21,0.45)] backdrop-blur-[3px]"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        className="max-h-[calc(100vh-32px)] w-[min(486px,calc(100vw-32px))] overflow-y-auto rounded-panel border border-line bg-panel px-5 pb-4 pt-[18px] shadow-2xl"
      >
        <header className="flex items-center justify-between">
          <h2 id="settings-title" className="m-0 text-[15px] font-semibold text-ink">
            Keys and models
          </h2>
          <button type="button" aria-label="Close" onClick={onClose} className={BTN_GHOST}>
            <X size={15} aria-hidden="true" />
          </button>
        </header>

        <p className="mb-3.5 mt-2 text-xs leading-relaxed text-muted">
          Saved to <code className="font-mono text-[11.5px]">.env</code> in the project folder.
          Every key is optional — each one switches a layer from its fallback to its
          model-backed path.
        </p>

        <KeyRow
          label="DeepSeek API key — understanding and wording"
          link={LINKS.llm}
          linkLabel="get a key"
          hint="Without it the deterministic grammar parses, so fewer phrasings are understood."
          placeholder="sk-… (leave blank to keep the current one)"
          value={key}
          onChange={setKey}
          inputRef={keyField}
          onEnter={() => void save()}
        />

        <label className="mb-3 block">
          <span className="mb-1 block text-[11.5px] text-muted">Model id</span>
          <input
            type="text"
            autoComplete="off"
            spellCheck={false}
            placeholder="deepseek-chat"
            value={model}
            onChange={(event) => setModel(event.target.value)}
            className={FIELD_INPUT}
          />
        </label>

        <KeyRow
          label="Embeddings API key — place-name matching"
          link={LINKS.embeddings}
          linkLabel="register"
          hint={
            'OpenAI-compatible /embeddings endpoint. Now using ' +
            (status?.embeddings_model || 'text-embedding-v4') +
            ' at ' +
            (status?.embeddings_base_url || 'dashscope-intl.aliyuncs.com') +
            '.'
          }
          placeholder="DashScope / Model Studio API key"
          value={embeddingsKey}
          onChange={setEmbeddingsKey}
        />

        <KeyRow
          label="Pinecone API key — the vector index"
          link={LINKS.vectors}
          linkLabel="register"
          hint="Holds one vector per place. After changing models, refill it: python tools/build.py embed --recreate"
          placeholder="pcsk_… (leave blank to keep the current one)"
          value={pineconeKey}
          onChange={setPineconeKey}
        />

        <p
          className={
            'mb-3 mt-1 min-h-4 text-xs leading-relaxed ' +
            (tone === 'ok' ? 'text-ok' : tone === 'err' ? 'text-danger' : 'text-muted')
          }
        >
          {message}
        </p>

        <footer className="flex items-center justify-between gap-2">
          <span className="font-mono text-[10.5px] text-muted">
            place names: {mode || '—'}
          </span>
          <span className="flex gap-2">
            <button type="button" onClick={() => void save()} className={BTN_PRIMARY}>
              Save to .env
            </button>
            <button type="button" onClick={onClose} className={BTN_GHOST}>
              Cancel
            </button>
          </span>
        </footer>
      </div>
    </div>
  );
}
