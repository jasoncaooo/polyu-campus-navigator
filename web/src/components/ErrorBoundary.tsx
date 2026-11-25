import { Component, type ErrorInfo, type ReactNode } from 'react';

interface State {
  error: Error | null;
}

/**
 * Anything that throws while rendering shows this instead of an empty page.
 *
 * A blank screen is the worst possible failure: the app is running, the server is up, and
 * the only thing the user can tell is that nothing appears. It happened once from a saved
 * conversation written by an older build, which is exactly the kind of mismatch that turns
 * a stale browser into a mystery. Clearing the saved transcript is the recovery, so the
 * panel offers it directly.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // the console is where a developer looks; the panel is where a user looks
    console.error('Campus Navigator failed to render', error, info.componentStack);
  }

  clear = (): void => {
    try {
      Object.keys(window.localStorage)
        .filter((key) => key.startsWith('campus-navigator/'))
        .forEach((key) => window.localStorage.removeItem(key));
    } catch {
      // storage unavailable: reloading is still the right next step
    }
    window.location.reload();
  };

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div className="mx-auto mt-16 max-w-xl rounded-[13px] border border-line bg-panel-soft p-6 text-sm text-ink">
        <h1 className="text-base font-semibold">The interface hit an error</h1>
        <p className="mt-2 text-muted">
          The page could not be drawn. If this browser opened the app before, its saved
          conversation may be from an older version — clearing it usually fixes this.
        </p>
        <p className="mt-2 font-mono text-[11.5px] text-danger">{String(this.state.error.message)}</p>
        <div className="mt-4 flex gap-2">
          <button type="button" onClick={this.clear}
            className="cursor-pointer rounded-[9px] bg-brand px-3 py-1.5 text-[13px] text-onaccent">
            Clear the saved conversation and reload
          </button>
          <button type="button" onClick={() => window.location.reload()}
            className="cursor-pointer rounded-[9px] border border-line px-3 py-1.5 text-[13px] text-muted hover:border-brand">
            Just reload
          </button>
        </div>
      </div>
    );
  }
}
