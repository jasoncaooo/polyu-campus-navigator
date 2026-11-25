import type { ChatMessage, ChatPayload, FacilityCard, RouteOption } from '../types';

function Readable({ text }: { text: string }) {
  // Keep the engineer-readable part (segment table, search stats) visually quieter.
  return (
    <span>
      {text.split('\n').map((line, index) => {
        const quiet = /^\s*[-…]/.test(line) || /^(Search|Total|Route:)/.test(line) || !line.trim();
        return (
          <span key={index} className={quiet ? 'text-[12.5px] text-muted' : undefined}>
            {(index ? '\n' : '') + line}
          </span>
        );
      })}
    </span>
  );
}

interface TraceStep {
  node?: string;
  name?: string;
  g?: number;
  h?: number | null;
  f?: number;
}

/**
 * The A* run that produced this route, in the order the nodes left the frontier.
 *
 * It costs eight short lines and it is the one place in the UI where the search is not a
 * claim: g is the walking distance so far, h the straight line still to go, f = g + h, and
 * the list is how far the run had looked by the time it reached the goal. A dropdown
 * cannot show this because a dropdown has already decided which nodes exist.
 */
function Trace({ route }: { route: ChatPayload['route'] }) {
  const steps = (route?.explain ?? []) as TraceStep[];
  if (!steps.length) return null;
  // Dijkstra has no heuristic, and a made-up h would be worse than an absent one.
  const guided = steps.some((step) => typeof step.h === 'number');
  return (
    <div className="mt-1.5 rounded-[8px] border border-line bg-panel-soft px-2 py-1.5">
      <div className="text-[9.5px] uppercase tracking-[0.09em] text-muted">
        {guided ? 'A* search order \u00b7 g walked, h straight-line left, f = g + h' : 'search order \u00b7 g walked'}
      </div>
      <ol className="mt-1 flex flex-col font-mono text-[10.5px] leading-snug text-muted">
        {steps.map((step, index) => (
          <li key={(step.node ?? '') + index} className="truncate" title={step.name}>
            <span className="text-ink">
              {index + 1}. {step.name ?? step.node}
            </span>{' '}
            g {step.g}
            {guided ? ' h ' + (step.h ?? 0) + ' f ' + step.f : null}
          </li>
        ))}
      </ol>
    </div>
  );
}

function Summary({ payload }: { payload: ChatPayload }) {
  const bits: string[] = [];
  const route = payload.route;
  if (route && route.reachable) {
    // The friendly number first. The count of graph links is what the search actually
    // walked, but "26 segments" is a fact about the data structure, not about the walk,
    // and it was the only thing this line said.
    bits.push(Math.round(route.metres) + ' m');
    if (route.stats) {
      bits.push(
        route.stats.algorithm.toUpperCase() +
          ' expanded ' +
          route.stats.nodes_expanded +
          ' nodes in ' +
          route.stats.ms.toFixed(1) +
          ' ms',
      );
    }
  }
  if ((payload.facilities || []).length > 1) bits.push(payload.facilities.length + ' places');
  // Say what opens. It was a bare "26 segments · ASTAR expanded 253 nodes" with a small
  // triangle, and both readers who saw it assumed the details were broken.
  return (
    <summary className="summary text-[11.5px] tracking-[0.04em] text-muted hover:text-ink">
      {'route breakdown \u00b7 ' + (bits.length ? bits.join(' \u00b7 ') : 'the numbers')}
    </summary>
  );
}

const TAG =
  'rounded-full border border-line px-2 py-0.5 text-[10.5px] uppercase tracking-[0.06em] text-muted';

function FacilityCards({
  facilities,
  lang,
  onPick,
}: {
  facilities: FacilityCard[];
  lang: string;
  onPick(name: string): void;
}) {
  const chinese = lang === 'zh-HK' || lang === 'zh-CN';
  return (
    <div className="mt-1 flex flex-col gap-1.5">
      {facilities.map((facility, index) => {
        const name = facility.display_name || facility.name;
        const where = [facility.display_building || facility.building, facility.display_floor || facility.floor]
          .filter(Boolean).join(chinese ? ' · ' : ', ');
        const bits: string[] = [];
        if (where) bits.push(where);
        if (facility.metres != null) bits.push(facility.metres + (chinese ? ' 米' : ' m'));
        const hours = facility.display_hours || facility.hours_today;
        if (hours) {
          const open = lang === 'zh-HK' ? '開放中' : chinese ? '开放中' : 'open';
          const closed = lang === 'zh-HK' ? '未開放' : chinese ? '未开放' : 'closed';
          const status = typeof facility.open_now === 'boolean' ? (facility.open_now ? open : closed) : '';
          bits.push(status && facility.hours_today?.toLowerCase() !== 'closed' ? status + ' · ' + hours : hours);
        }
        return (
          <button
            type="button"
            key={facility.name + index}
            className="cursor-pointer text-left rounded-[10px] border border-line bg-panel-soft px-2.5 py-2 text-[12.5px] transition hover:border-brand"
            title={(chinese ? '前往' : 'Ask for a route to ') + name}
            onClick={() => onPick(facility.name)}
          >
            <b className="font-semibold text-brand-dark">{name}</b>
            {bits.length ? <span className="mt-0.5 block text-[11.5px] text-muted">{bits.join(' · ')}</span> : null}
          </button>
        );
      })}
    </div>
  );
}

function RouteOptions({
  routes,
  selection,
  onSelect,
}: {
  routes: RouteOption[];
  selection: number | null;
  onSelect(index: number): void;
}) {
  if (routes.length < 2) return null;
  return (
    <div className="mt-2 flex flex-col gap-1">
      {routes.map((route, index) => {
        const facts = [
          route.legs + (route.legs === 1 ? ' leg' : ' legs'),
          route.flights
            ? route.flights + (route.flights === 1 ? ' flight of steps' : ' flights of steps')
            : 'no steps',
          route.via ? 'via ' + route.via : '',
        ].filter(Boolean);
        return (
          <button
            key={index}
            type="button"
            title={'Show route ' + (index + 1) + ' on the map'}
            onClick={() => onSelect(index)}
            data-route-option={index}
            data-selected={selection === index ? 'true' : 'false'}
            className={
              'flex w-full cursor-pointer items-center gap-2 rounded-[10px] border px-2.5 py-[7px] ' +
              'text-left text-[12.5px] transition ' +
              (selection === index
                ? 'border-brand bg-accent-soft shadow-[inset_2px_0_0_var(--accent)]'
                : 'border-line bg-panel-soft hover:border-brand')
            }
          >
            <i
              className="h-[3px] w-3.5 flex-none rounded-[2px]"
              style={{ background: 'var(--route-' + (index + 1) + ')' }}
            />

            <span className="font-semibold">{Math.round(route.metres)} m</span>
            <span className="ml-auto text-right text-muted">{facts.join(' · ')}</span>
          </button>
        );
      })}
    </div>
  );
}

export interface MessageProps {
  message: ChatMessage;
  /** Which option of THIS message is currently on the map, or null. */
  selection: number | null;
  onFacilityClick(name: string): void;
  onSelectRoute(index: number): void;
}

export function Message({ message, selection, onFacilityClick, onSelectRoute }: MessageProps) {
  const { role, payload } = message;

  if (role === 'user' || typeof payload === 'string') {
    return (
      <div className={'flex max-w-full flex-col gap-1.5 ' + (role === 'user' ? 'items-end' : '')}>
        <div className="text-[11px] uppercase tracking-[0.1em] text-muted">
          {role === 'user' ? 'you' : 'navigator'}
        </div>
        <div
          className={
            'whitespace-pre-wrap break-words rounded-[13px] px-[13px] py-[11px] text-sm leading-relaxed ' +
            (role === 'user'
              ? 'rounded-tr-[4px] bg-brand text-onaccent'
              : 'rounded-tl-[4px] border border-line bg-panel-soft text-ink')
          }
        >
          {String(payload)}
        </div>
      </div>
    );
  }

  const tags = payload.notes ?? [];

  return (
    <div className="flex max-w-full flex-col gap-1.5">
      <div className="text-[11px] uppercase tracking-[0.1em] text-muted">navigator</div>
      <div className="whitespace-pre-wrap break-words rounded-[13px] rounded-tl-[4px] border border-line bg-panel-soft px-[13px] py-[11px] text-sm leading-relaxed text-ink">
        {payload.lead ? (
          <>
            <div className="text-[14.5px] leading-relaxed">{payload.lead}</div>
            <details className="mt-2 border-t border-dashed border-line pt-1.5">
              <Summary payload={payload} />
              <Trace route={payload.route} />
              <span className="mt-1.5 block">
                <Readable text={payload.text} />
              </span>
            </details>
          </>
        ) : (
          <Readable text={payload.text} />
        )}

        {payload.routes && payload.routes.length ? (
          <RouteOptions routes={payload.routes} selection={selection} onSelect={onSelectRoute} />
        ) : null}

        {payload.narrated || payload.parser || tags.length ? (
          <div className="flex flex-wrap gap-1.5">
            {payload.narrated ? <span className={TAG + ' border-line-strong text-brand'}>prose: model · numbers: A*</span> : null}
            {payload.parser ? (
              <span className="text-[10px] italic text-muted opacity-75">understood by {payload.parser}</span>
            ) : null}
            {tags.map((note, index) => (
              <span key={note + index} className={TAG}>
                {note}
              </span>
            ))}
          </div>
        ) : null}

        {payload.facilities && payload.facilities.length ? (
          <FacilityCards facilities={payload.facilities} lang={payload.speech_lang || 'en-US'} onPick={onFacilityClick} />
        ) : null}
      </div>
    </div>
  );
}
