/* Shared utility strings. Tailwind's answer to repetition is a component, and for a
   handful of buttons and fields written once each a string constant is clearer than
   a component that takes three boolean props to say the same thing. */
export const BTN_PRIMARY =
  'rounded-xl bg-brand px-4 py-2.5 text-[13px] font-medium text-onaccent transition hover:brightness-110';

export const BTN_GHOST =
  'rounded-xl border border-line px-3 py-1.5 text-[13px] text-muted transition hover:text-ink hover:border-line-strong';

export const FIELD_INPUT =
  'w-full rounded-[9px] border border-line bg-panel-soft px-3 py-2.5 text-[13px] text-ink placeholder:text-muted/70 focus:border-brand focus:outline-none';

export const PANEL = 'flex min-h-0 flex-col overflow-hidden rounded-panel border border-line bg-panel';

export const PANEL_HEAD =
  'flex items-center justify-between border-b border-line px-4 py-3';

export const PANEL_TITLE = 'm-0 text-[13px] uppercase tracking-[0.09em] text-muted';
