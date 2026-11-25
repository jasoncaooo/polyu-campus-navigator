import { Compass, Settings } from 'lucide-react';
import type { Status } from '../types';

export interface TopBarProps {
  status: Status | null;
  engineOffline: boolean;
  onOpenSettings(): void;
}

function pillText(
  status: Status | null,
  engineOffline: boolean,
): { text: string; ok: boolean; title: string } {
  if (engineOffline) {
    return { text: 'engine offline', ok: false, title: 'the engine is not answering' };
  }
  const connected = Boolean(status && status.model);
  if (connected && status) {
    return {
      text: 'model: ' + status.model.split('/').pop()!.replace(':free', ''),
      ok: true,
      title: status.mode + '\nkey: present',
    };
  }
  return {
    text: 'no model connected',
    ok: false,
    title:
      (status ? status.mode : 'connecting…') +
      '\nUse the gear to add a DeepSeek key; every feature still works without one',
  };
}

const ROUND_BTN =
  'grid h-7 w-7 cursor-pointer place-items-center rounded-full bg-onaccent text-brand ' +
  'transition hover:bg-accent-soft focus-visible:outline focus-visible:outline-2 ' +
  'focus-visible:outline-offset-2 focus-visible:outline-onaccent';

export function TopBar({ status, engineOffline, onOpenSettings }: TopBarProps) {
  const pill = pillText(status, engineOffline);
  const counts = status && status.knowledge
    ? 'Loaded: ' +
      status.nodes +
      ' places, ' +
      status.edges +
      ' links, ' +
      status.knowledge.documents +
      ' knowledge records (' +
      status.knowledge.with_opening_hours +
      ' with published hours).'
    : '';

  return (
    <header className="flex items-center justify-between gap-4 bg-gradient-to-br from-brand to-brand-dark px-[22px] py-3.5 text-onaccent">
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-[11px] bg-onaccent text-brand">
          <Compass size={20} strokeWidth={2.2} aria-hidden="true" />
        </span>
        <div>
          <h1 className="m-0 text-[17px] tracking-[0.2px]">PolyU Campus Navigator</h1>
          <p className="m-0 mt-0.5 text-xs text-onaccent-dim">
            COMP4431 Artificial Intelligence · Group 21
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span
          title={pill.title}
          className={
            'whitespace-nowrap rounded-full border border-transparent bg-panel px-[11px] py-[5px] text-xs ' +
            (pill.ok ? 'text-ok' : 'text-warn')
          }
        >
          {pill.text}
        </span>

        {/* (i) hover/focus card: where the data comes from. */}
        <div className="group relative inline-flex">
          <button type="button" aria-label="Where does this data come from?" aria-describedby="info-pop" className={ROUND_BTN}>
            i
          </button>
          <div
            id="info-pop"
            role="tooltip"
            className="invisible absolute right-0 top-[calc(100%+10px)] z-40 w-[330px] -translate-y-1 rounded-xl border border-line bg-panel px-4 py-3.5 text-xs leading-relaxed text-ink opacity-0 shadow-2xl transition group-focus-within:visible group-focus-within:translate-y-0 group-focus-within:opacity-100 group-hover:visible group-hover:translate-y-0 group-hover:opacity-100"
          >
            <h3 className="mb-1.5 text-[13px] font-semibold">Where this data comes from</h3>
            <p className="mb-2 text-muted">Scraped once from the official PolyU campus map. </p>
            <ul className="mb-2 list-disc pl-4 text-ink">
              <li className="mb-1.5">
                <b>Campus Map</b> — 561 POIs with waypoints and trilingual names
                <br />
                <span className="font-mono text-[11.5px] text-brand">polyu.edu.hk/campus-map</span>
              </li>
            </ul>
            <p className="mb-0 text-[11.5px] text-muted">
              Walkway links are <b>hand-authored</b> on top of published coordinates and may contain
              errors. Stairs and lifts are not modelled. Opening hours are quoted only where PolyU
              publishes them.
            </p>
            <p className="mb-0 mt-2 text-[11.5px] text-muted">{counts}</p>
          </div>
        </div>

        <button
          type="button"
          aria-label="Model settings"
          title="Model settings (API key)"
          onClick={onOpenSettings}
          className={ROUND_BTN}
        >
          <Settings size={15} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
