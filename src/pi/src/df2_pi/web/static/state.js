// Server state, held in module-level signals. The poller writes them and
// components read them; a component re-renders only when a signal it reads
// changes, so a poll never re-renders (or steals focus from) an input
// someone is editing.
//
// Only state lives here. Anything at frame rate - the preview - bypasses
// signals entirely and draws through a ref.

import { signal } from "@preact/signals";

export const POLL_INTERVAL_MS = 500; // 2 Hz
const REQUEST_TIMEOUT_MS = 2000;

/** The last `GET /api/state` snapshot (RunnerState as JSON), or null before the first. */
export const runnerState = signal(null);

/** "connecting" until the first answer, then "ok" or "lost". */
export const connection = signal("connecting");

/** Poll `/api/state` forever. The next request is scheduled only after the
 *  previous one settles, so a slow server never has requests piling up. */
export function startPolling() {
  async function poll() {
    try {
      const response = await fetch("api/state", {
        cache: "no-store",
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      runnerState.value = await response.json();
      connection.value = "ok";
    } catch {
      connection.value = "lost";
    }
    setTimeout(poll, POLL_INTERVAL_MS);
  }
  poll();
}
