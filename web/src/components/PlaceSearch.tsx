import { useEffect, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { legendKeyOf } from '../lib/geo';
import type { GraphNode } from '../types';

export interface PlaceSearchProps {
  nodes: GraphNode[];
  onChoose(node: GraphNode): void;
}

/** Does `needle` appear at the start of a word, not inside one?
 *
 *  "pao" must find Pao Yue-Kong Library, and must not find the Institutional Planning
 *  and Analytics Office, whose acronym "IPAO" happens to contain the letters p-a-o. */
function startsAWord(text: string, needle: string): boolean {
  if (!needle) return false;
  let from = 0;
  for (;;) {
    const at = text.indexOf(needle, from);
    if (at < 0) return false;
    if (at === 0 || !/[a-z0-9]/.test(text[at - 1])) return true;
    from = at + 1;
  }
}

/** What a place IS, with its location bracket removed: "Block L (Pao Yue-Kong Library)"
 *  and "Pao Yue-Kong Library" are two entries, "Anywhere (Block L)" twice is one. */
function headOf(name: string): string {
  return name
    .replace(/[（(][^()（）]*[)）]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

/* Type a place, arrow through the hits, Enter to route. Searches the same node list
   the map is drawn from, so anything visible on the map is findable here.

   Two rules keep the list honest, and they mirror the server-side resolver:
   a match must start a word, and one answer per name -- the campus map publishes the
   library as a main location, as the building it sits in, and as an office record, and
   three rows that read the same are worse than one. */
export function PlaceSearch({ nodes, onChoose }: PlaceSearchProps) {
  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<GraphNode[]>([]);
  const [active, setActive] = useState(-1);
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  const places = nodes.filter((node) => node.kind !== 'waypoint');

  useEffect(() => {
    const onDocumentClick = (event: MouseEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('click', onDocumentClick);
    return () => document.removeEventListener('click', onDocumentClick);
  }, []);

  const search = (value: string) => {
    const needle = value.trim().toLowerCase();
    if (needle.length < 2) {
      setHits([]);
      setOpen(false);
      setActive(-1);
      return;
    }
    const matches = places
      .filter(
        (node) =>
          startsAWord(node.name.toLowerCase(), needle) ||
          startsAWord((node.service || '').toLowerCase(), needle),
      )
      .sort((a, b) => {
        const ai = a.name.toLowerCase().indexOf(needle);
        const bi = b.name.toLowerCase().indexOf(needle);
        return ai - bi || a.name.length - b.name.length;
      });

    // One row per name, preferring the place over a facility published under the same
    // name. The duplicate rows stay reachable by typing their bracket ("LIB", "Block L").
    const seen = new Map<string, GraphNode>();
    for (const node of matches) {
      const head = headOf(node.name);
      const kept = seen.get(head);
      if (!kept) {
        seen.set(head, node);
        continue;
      }
      if (kept.kind === 'facility' && node.kind !== 'facility') seen.set(head, node);
    }
    const found = Array.from(seen.values()).slice(0, 7);

    setHits(found);
    setOpen(true);
    setActive(found.length ? 0 : -1);
  };

  const choose = (index: number) => {
    const hit = hits[index];
    if (!hit) return;
    setOpen(false);
    setQuery('');
    setHits([]);
    onChoose(hit);
  };

  return (
    <div className="absolute left-3 right-[62px] top-3 z-[6]" ref={wrap}>
      <div className="relative">
        <Search
          size={14}
          aria-hidden="true"
          className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted"
        />
        <input
          type="search"
          autoComplete="off"
          spellCheck={false}
          placeholder="Search a place — library, Core D, AED…"
          aria-label="Search for a place on campus"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            search(event.target.value);
          }}
          onFocus={() => {
            if (query.trim().length >= 2) search(query);
          }}
          onKeyDown={(event) => {
            if (event.key === 'ArrowDown' && hits.length) {
              setActive((active + 1) % hits.length);
              event.preventDefault();
            } else if (event.key === 'ArrowUp' && hits.length) {
              setActive((active - 1 + hits.length) % hits.length);
              event.preventDefault();
            } else if (event.key === 'Enter') {
              choose(active >= 0 ? active : 0);
              event.preventDefault();
            } else if (event.key === 'Escape') {
              setOpen(false);
            }
          }}
          className="w-full rounded-[10px] border border-line bg-panel py-2.5 pl-9 pr-3.5 text-[13px] text-ink shadow-[0_6px_18px_var(--accent-ring)] placeholder:text-muted/80 focus:border-brand focus:outline-none"
        />
      </div>

      {open ? (
        <div className="mt-1.5 max-h-[262px] overflow-y-auto rounded-[10px] border border-line bg-panel shadow-[0_14px_30px_var(--accent-ring)]">
          {hits.length ? (
            hits.map((hit, index) => (
              <div
                key={hit.id}
                onMouseDown={(event) => {
                  event.preventDefault();
                  choose(index);
                }}
                className={
                  'flex cursor-pointer items-center gap-[9px] border-b border-panel-soft px-3 py-[9px] text-[12.5px] last:border-b-0 ' +
                  (index === active ? 'bg-accent-soft' : 'hover:bg-accent-soft')
                }
              >
                <i
                  className="h-2 w-2 flex-none rounded-full bg-brand"
                  style={
                    hit.service
                      ? { background: 'var(--cat-' + legendKeyOf(hit) + ')' }
                      : undefined
                  }
                />
                <span>{hit.name}</span>
                <span className="ml-auto whitespace-nowrap text-[11px] text-muted">
                  {hit.kind === 'facility' ? hit.service || 'facility' : 'place'}
                </span>
              </div>
            ))
          ) : (
            <div className="px-3 py-2.5 text-xs text-muted">No place matches that name.</div>
          )}
        </div>
      ) : null}
    </div>
  );
}
