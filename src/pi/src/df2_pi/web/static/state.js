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

/** The id of the animation running as a layer, or null. */
export const layerAnimationId = computed(() => runnerState.value?.layer?.animation?.[0] ?? null);

/** "connecting" until the first answer, then "ok" or "lost". */
export const connection = signal("connecting");

/** The tab on show: "playlists", "animations", "external" or "diagnostics". Here, not
 *  in app.js, so a panel can tell whether it is on screen. */
export const activeTab = signal("playlists");

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

/** PATCH `/api/settings`. A setting the floor applies live (brightness,
 *  rotation) shows on a following poll, which this brings forward.
 *  Resolves true if accepted; a refusal lands in `commandError`. */
export async function changeSettings(changes) {
  try {
    const response = await fetch("api/settings", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(changes),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!response.ok) {
      commandError.value = `settings: ${await refusal(response)}`;
      return false;
    }
    commandError.value = null;
    return true;
  } catch (exc) {
    commandError.value = `settings: ${exc.message}`;
    return false;
  } finally {
    pollSoon();
  }
}

// ---- streamed commands -----------------------------------------------------

const STREAM_INTERVAL_MS = 66; // ~15 Hz per command, like live params

const streamLatest = new Map(); // command -> the latest body not yet sent
const streamSentAt = new Map(); // command -> when it was last sent
const streamBusy = new Set(); // commands with a send scheduled or in flight

/** `command(name, body)` for a control that is dragged: coalesced per
 *  command, at most one POST per STREAM_INTERVAL_MS and one at a time, always
 *  carrying the latest body - so the end of a drag is what lands. */
export function streamCommand(name, body) {
  streamLatest.set(name, body);
  if (!streamBusy.has(name)) scheduleStream(name);
}

function scheduleStream(name) {
  streamBusy.add(name);
  const wait = Math.max(0, (streamSentAt.get(name) ?? -Infinity) + STREAM_INTERVAL_MS - performance.now());
  setTimeout(async () => {
    const body = streamLatest.get(name);
    streamLatest.delete(name);
    streamSentAt.set(name, performance.now());
    await command(name, body);
    streamBusy.delete(name);
    if (streamLatest.has(name)) scheduleStream(name);
  }, wait);
}

// ---- live params ---------------------------------------------------------

const PARAM_INTERVAL_MS = 66; // ~15 Hz per param: a dragged slider streams, the server is not flooded

/** Why the running animation last refused a param, by name. */
export const paramErrors = signal({});

const pendingParams = new Map(); // name -> the latest value not yet sent
const lastSent = new Map(); // name -> when it was last sent (performance.now)
const busy = new Set(); // names with a send scheduled or in flight

/** Set one param of the running animation. Calls are coalesced per param:
 *  at most one POST per PARAM_INTERVAL_MS, one at a time (so they cannot
 *  arrive out of order), always carrying the latest value - the end of a
 *  drag is always what lands. */
export function setLiveParam(name, value) {
  pendingParams.set(name, value);
  if (!busy.has(name)) scheduleParam(name);
}

function scheduleParam(name) {
  busy.add(name);
  const wait = Math.max(0, (lastSent.get(name) ?? -Infinity) + PARAM_INTERVAL_MS - performance.now());
  setTimeout(() => sendParam(name), wait);
}

async function sendParam(name) {
  const value = pendingParams.get(name);
  pendingParams.delete(name);
  lastSent.set(name, performance.now());
  try {
    const response = await fetch("api/transport/params", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ [name]: value }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (response.ok) {
      if (name in paramErrors.value) {
        const { [name]: _, ...rest } = paramErrors.value;
        paramErrors.value = rest;
      }
    } else {
      const body = await response.json().catch(() => ({}));
      const detail = body.detail;
      if (detail && typeof detail === "object" && "param" in detail) {
        paramErrors.value = { ...paramErrors.value, [detail.param]: detail.message };
      } else {
        commandError.value = `params: ${typeof detail === "string" ? detail : `HTTP ${response.status}`}`;
      }
    }
  } catch (exc) {
    commandError.value = `params: ${exc.message}`;
  }
  busy.delete(name);
  if (pendingParams.has(name)) scheduleParam(name); // moved again while this was in flight
  else pollSoon();
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
