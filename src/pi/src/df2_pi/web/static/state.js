// Server state, held in module-level signals. The poller writes them and
// components read them; a component re-renders only when a signal it reads
// changes, so a poll never re-renders (or steals focus from) an input
// someone is editing.
//
// Only state lives here. Anything at frame rate - the preview - bypasses
// signals entirely and draws through a ref.

import { computed, signal } from "@preact/signals";

export const POLL_INTERVAL_MS = 500; // 2 Hz
const AFTER_COMMAND_MS = 100; // a queued command applies at the next frame (~33 ms)
const REQUEST_TIMEOUT_MS = 2000;

/** The last `GET /api/state` snapshot (RunnerState as JSON), or null before the first. */
export const runnerState = signal(null);

/** The id of whatever is playing now. Changes only when the animation does,
 *  so a component reading this is not re-rendered by every poll. */
export const currentAnimationId = computed(() => runnerState.value?.animation?.[0] ?? null);

/** "connecting" until the first answer, then "ok" or "lost". */
export const connection = signal("connecting");

/** Why the last transport command was refused, or null. */
export const commandError = signal(null);

let timer = null;
let inFlight = false;
let pollAgain = false;

async function poll() {
  if (inFlight) {
    pollAgain = true;
    return;
  }
  clearTimeout(timer);
  inFlight = true;
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
  } finally {
    inFlight = false;
    // The next request is scheduled only after this one settles, so a slow
    // server never has requests piling up.
    if (pollAgain) {
      pollAgain = false;
      poll();
    } else {
      timer = setTimeout(poll, POLL_INTERVAL_MS);
    }
  }
}

/** Poll `/api/state` forever. */
export function startPolling() {
  poll();
}

/** Poll shortly, rather than at the next 2 Hz tick. */
function pollSoon() {
  if (inFlight) {
    pollAgain = true;
  } else {
    clearTimeout(timer);
    timer = setTimeout(poll, AFTER_COMMAND_MS);
  }
}

async function refusal(response) {
  try {
    const body = await response.json();
    // FastAPI: a string from HTTPException, a list from body validation.
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch {
    // not JSON
  }
  return `HTTP ${response.status}`;
}

/** POST `/api/transport/<name>`. Commands are queued and applied at the
 *  next frame boundary, so the change shows on a following poll - which
 *  this brings forward - never in the response. Resolves true if queued. */
export async function command(name, body) {
  try {
    const response = await fetch(`api/transport/${name}`, {
      method: "POST",
      headers: body === undefined ? {} : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!response.ok) {
      commandError.value = `${name}: ${await refusal(response)}`;
      return false;
    }
    commandError.value = null;
    return true;
  } catch (exc) {
    commandError.value = `${name}: ${exc.message}`;
    return false;
  } finally {
    pollSoon();
  }
}

// ---- animations ----------------------------------------------------------
// The registry changes only on a reload, so this is fetched when needed,
// not polled.

/** The last `GET /api/animations` ({animations, errors}), or null before the first. */
export const animationList = signal(null);

/** The last reload's result ({changed, errors}), or null. */
export const lastReload = signal(null);

/** Why the last animations request failed, or null. */
export const animationsError = signal(null);

export async function loadAnimations() {
  try {
    const response = await fetch("api/animations", {
      cache: "no-store",
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!response.ok) throw new Error(await refusal(response));
    animationList.value = await response.json();
    animationsError.value = null;
  } catch (exc) {
    animationsError.value = `animations: ${exc.message}`;
  }
}

/** Re-import what changed on disk, then refresh the list. */
export async function reloadAnimations() {
  try {
    const response = await fetch("api/animations/reload", {
      method: "POST",
      signal: AbortSignal.timeout(10 * REQUEST_TIMEOUT_MS), // imports every changed file
    });
    if (!response.ok) throw new Error(await refusal(response));
    lastReload.value = await response.json();
    animationsError.value = null;
  } catch (exc) {
    animationsError.value = `reload: ${exc.message}`;
  }
  await loadAnimations();
}
