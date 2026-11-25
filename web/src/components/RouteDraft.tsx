import { useEffect, useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import type { Graph, GraphNode } from '../types';

const roles = ['origin', 'destination'] as const;
type Role = (typeof roles)[number];
type ParsedPlace = { text: string | null; id: string | null; label: string | null };
type ParsedRoute = Record<Role, ParsedPlace>;
type Preview = { is_route: false } | ({ is_route: true } & ParsedRoute);
type Field = ParsedPlace & { edit?: { id: string | null; query: string } };
type Fields = Record<Role, Field>;

const wording = {
  'en-US': { origin: 'From', destination: 'To', search: 'Search places', clear: 'Clear', none: 'No matching place', choose: 'Choose a place from the list.', failed: 'Place preview unavailable. Review the text before sending.', heard: 'Heard:', change: 'change' },
  'zh-CN': { origin: '起点', destination: '终点', search: '搜索地点', clear: '清除', none: '找不到匹配地点', choose: '请从列表中选择地点。', failed: '暂时无法预览地点，请在发送前核对文字。', heard: '听到：', change: '修改' },
  'zh-HK': { origin: '起點', destination: '終點', search: '搜尋地點', clear: '清除', none: '找不到相符地點', choose: '請從列表中選擇地點。', failed: '暫時無法預覽地點，請在傳送前核對文字。', heard: '聽到：', change: '修改' },
};

function copyFor(lang: string) {
  return wording[lang as keyof typeof wording] || wording['en-US'];
}

function fold(value: string) {
  return value.normalize('NFKC').toLowerCase().replace(/[\s_-]+/g, '');
}

function choiceId(field: Field) {
  return field.edit ? field.edit.id : field.id;
}

function display(field: Field, nodes: Map<string, GraphNode>, lang: string) {
  const place = nodes.get(choiceId(field) || '');
  if (place) return place.names?.[lang] || place.name;
  return field.edit ? field.edit.query : field.label || field.text || '';
}

function EndpointInput({ role, value, invalid, places, lang, onEdit }: {
  role: Role;
  value: string;
  invalid: boolean;
  places: GraphNode[];
  lang: string;
  onEdit(edit: NonNullable<Field['edit']>): void;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const input = useRef<HTMLInputElement>(null);
  const copy = copyFor(lang);
  const id = 'voice-' + role;
  const label = (node: GraphNode) => node.names?.[lang] || node.name;

  const hits = useMemo(() => {
    if (!open) return [];
    const query = fold(value);
    const matches: { place: GraphNode; rank: number }[] = [];
    for (const place of places) {
      const names = [place.id, place.name, ...Object.values(place.names || {}), ...(place.aliases || [])].map(fold);
      const rank = names.includes(query) ? 0 : names.some(n => n.startsWith(query)) ? 1 : names.some(n => n.includes(query)) ? 2 : 3;
      if (rank < 3) matches.push({ place, rank });
    }
    const name = (node: GraphNode) => node.names?.[lang] || node.name;
    matches.sort((a, b) => a.rank - b.rank || name(a.place).length - name(b.place).length || name(a.place).localeCompare(name(b.place)));
    return matches.slice(0, 6).map(({ place }) => place);
  }, [places, open, value, lang]);

  const choose = (node: GraphNode) => {
    onEdit({ id: node.id, query: label(node) });
    setOpen(false);
    setActive(-1);
  };
  const change = (query: string) => {
    onEdit({ id: null, query });
    setOpen(true);
    setActive(-1);
  };
  return (
    <div className="relative min-w-0">
      <label htmlFor={id} className="mb-1 block text-[11px] font-medium text-muted">{copy[role]}</label>
      <div className="relative">
        <input id={id} ref={input} type="text" role="combobox" autoComplete="off" spellCheck={false}
          aria-expanded={open} aria-controls={id + '-options'} aria-autocomplete="list" aria-invalid={invalid}
          aria-activedescendant={open && hits[active] ? id + '-option-' + active : undefined}
          value={value} placeholder={copy.search}
          onChange={e => change(e.target.value)}
          onFocus={e => { e.target.select(); setOpen(true); setActive(-1); }}
          onBlur={() => setOpen(false)}
          onKeyDown={e => {
            if (e.nativeEvent.isComposing) return;
            if (e.key === 'Escape') { setOpen(false); return; }
            if (e.key === 'Enter' && open) {
              e.preventDefault();
              if (hits[active]) choose(hits[active]);
              else if (hits.length === 1) choose(hits[0]);
            }
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
              e.preventDefault();
              setOpen(true);
              if (!hits.length) { setActive(0); return; }
              const next = e.key === 'ArrowDown' ? (active + 1) % hits.length : active <= 0 ? hits.length - 1 : active - 1;
              setActive(next);
              document.getElementById(id + '-option-' + next)?.scrollIntoView({ block: 'nearest' });
            }
          }}
          className="w-full rounded-[9px] border border-line bg-panel-soft py-2 pl-2.5 pr-7 text-[13px] text-ink focus:border-brand focus:outline-none aria-invalid:border-warn"
        />
        {value && <button type="button" aria-label={copy.clear + ' ' + copy[role]}
          onClick={() => { change(''); input.current?.focus(); }}
          className="absolute inset-y-0 right-0 rounded-r-[9px] px-2 text-muted hover:text-brand"><X size={13} /></button>}
      </div>
      <div hidden={!open} id={id + '-options'} role="listbox" aria-label={copy[role]}
        className="absolute bottom-full z-20 mb-1 max-h-48 w-full overflow-auto rounded-[10px] border border-line bg-panel shadow-lg">
        {hits.map((node, index) => {
          const duplicate = hits.some(other => other.id !== node.id && label(other) === label(node));
          return <div key={node.id} id={id + '-option-' + index} role="option" aria-selected={index === active}
            onMouseDown={e => e.preventDefault()} onClick={() => choose(node)}
            className="cursor-pointer px-2.5 py-1.5 text-[12px] text-ink hover:bg-accent-soft aria-selected:bg-accent-soft">
            {label(node)}{duplicate && <small className="ml-1 text-muted">{[node.building, node.floor].filter(Boolean).join(' · ') || node.id}</small>}
          </div>;
        })}
        {!hits.length && <div className="px-2.5 py-1.5 text-xs text-muted">{copy.none}</div>}
      </div>
    </div>
  );
}

interface Result {
  text: string;
  lang: string;
  fields?: Fields;
  failed?: boolean;
}

/** Preview a dictated question without advancing the dialogue. Only Send commits the choices. */
export function useRouteDraft(text: string, lang: string, graph: Graph | null, enabled: boolean) {
  const [result, setResult] = useState<Result>();
  const [openFields, setOpenFields] = useState(false);
  const places = useMemo(() => (graph?.nodes || []).filter(n => n.kind !== 'waypoint' && n.name), [graph]);
  const nodes = useMemo(() => new Map(places.map(n => [n.id, n])), [places]);
  const wanted = enabled && Boolean(text.trim());
  useEffect(() => {
    if (!wanted) { setResult(undefined); return; }
    const controller = new AbortController();
    let cancelled = false;
    let timeout: number;
    const timer = window.setTimeout(async () => {
      try {
        timeout = window.setTimeout(() => controller.abort(), 5000);
        const response = await fetch('/api/route-draft', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, lang }), signal: controller.signal,
        });
        if (!response.ok) throw new Error('preview failed');
        const draft: Preview = await response.json();
        if (cancelled) return;
        setResult(previous => {
          if (!draft.is_route) return { text, lang };
          const field = (role: Role): Field => {
            const prev = previous?.fields?.[role];
            return { ...draft[role], edit: prev?.text === draft[role].text ? prev?.edit : undefined };
          };
          return { text, lang, fields: { origin: field('origin'), destination: field('destination') } };
        });
      } catch {
        if (!cancelled) setResult({ text, lang, failed: true });
      } finally {
        window.clearTimeout(timeout);
      }
    }, 150);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [wanted, text, lang]);

  const current = wanted && result?.text === text && result.lang === lang ? result : undefined;
  const fields = current?.fields;
  if (!fields || !places.length) {
    const failed = current?.failed || (fields && !places.length);
    return { blocked: wanted && !current, editor: failed
      ? <p role="status" className="px-4 text-xs text-muted">{copyFor(lang).failed}</p> : null };
  }
  const invalid = (role: Role) => Boolean(fields[role].edit && !choiceId(fields[role]));
  const blocked = roles.some(invalid);
  const labels = roles.map(role => display(fields[role], nodes, lang));
  const endpoints = { origin: choiceId(fields.origin) || undefined, destination: choiceId(fields.destination) || undefined };
  // What the app heard is worth one quiet line -- speech recognition mishears building
  // codes ("Block Y" and "Core Y" are one syllable apart) and this is the only place the
  // correction can be made. The two input fields with their suggestion lists are behind
  // "change": they used to open on *every* dictation, which read as the app handing a
  // voice command back for manual processing.
  const edited = roles.some(role => fields[role].edit);
  const confirmation = edited && labels.every(Boolean) ? labels.join(' \u2192 ') : undefined;
  const showFields = edited || blocked || openFields;
  return {
    blocked, endpoints: endpoints.origin || endpoints.destination ? endpoints : undefined, confirmation,
    editor: <div className="px-4 pt-2">
      {showFields ? (
        <div className="grid grid-cols-2 gap-2">
          {roles.map((role, index) => <EndpointInput key={role} role={role} lang={lang} value={labels[index]}
            invalid={invalid(role)} places={places}
            onEdit={edit => setResult(prev => prev?.fields && prev === current ? {
              ...prev, fields: { ...prev.fields, [role]: { ...prev.fields[role], edit } },
            } : prev)} />)}
        </div>
      ) : (
        <div className="flex items-baseline gap-2 text-[11.5px] text-muted">
          <span className="min-w-0 truncate">{copyFor(lang).heard + ' ' + labels.join(' \u2192 ')}</span>
          <button type="button" onClick={() => setOpenFields(true)}
            className="flex-none underline decoration-dotted hover:text-brand">{copyFor(lang).change}</button>
        </div>
      )}
      {blocked && <p role="status" className="mt-1 text-xs text-muted">{copyFor(lang).choose}</p>}
    </div>,
  };
}
