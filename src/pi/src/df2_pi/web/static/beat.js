// Beat sync (#125): the External tab's section - source, tempo, tap, resync,
// nudge, multiplier, bar length, latency offset, launch quantize - and the
// transport bar's beat readout. Settings are a PATCH to /api/beat, applied
// and stored at once.

import { html } from "htm/preact";
import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import { NumberField } from "./params.js";
import { activeTab, runnerState } from "./state.js";

const POLL_MS = 1000;
const SOURCES = [
  ["off", "Off", "No beat: animations run on their own clocks"],
  ["link", "Ableton Link", "Tempo and bar from Resolume, Ableton, DJ software on this network"],
  ["tap", "Tap tempo", "Tap along: the mean of the last four taps, the last one beat 1"],
];
const MULTIPLIERS = [[0.5, "½×"], [1.0, "1×"], [2.0, "2×"]];
const QUANTA = [
  ["off", "Off", "Launches happen at once"],
  ["beat", "Beat", "Next, previous, go-to, loads and one-offs wait for the next beat"],
  ["bar", "Bar", "…for the next bar line (the next beat while the bar is unknown)"],
];
const LAUNCH_WAIT = { beat: "waiting for the beat", bar: "waiting for the bar" };

/** The beat settings and status, shared by the section and the transport readout. */
export const beatInfo = signal(null);

async function call(method, path, body) {
  const response = await fetch(`api/beat${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const json = await response.json().catch(() => null);
  if (response.status === 503) return { unavailable: true };
  if (!response.ok) {
    const detail = json?.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : `HTTP ${response.status}`);
  }
  return json;
}

async function refreshBeat() {
  try {
    beatInfo.value = await call("GET", "");
  } catch {
    // the next poll tries again
  }
}

/** Load once at startup (the transport needs the source), then poll while the External tab is open. */
export function startBeatPolling() {
  refreshBeat();
  setInterval(() => {
    if (activeTab.value === "external" && !document.hidden) refreshBeat();
  }, POLL_MS);
}

function RadioRow({ label, options, value, onPick }) {
  return html`
    <div class="floor-buttons" role="radiogroup" aria-label=${label}>
      ${options.map(([v, text, help]) => html`
        <button role="radio" aria-checked=${value === v} class=${value === v ? "active" : ""} title=${help ?? ""} onClick=${() => onPick(v)}>${text}</button>`)}
    </div>`;
}

function BeatHealth({ s, status }) {
  if (s.source === "off") return html`<div class="health idle">Off</div>`;
  if (status.error) return html`<div class="health bad">${status.error}</div>`;
  if (status.active) {
    const peers = s.source === "link" ? ` · ${status.peers} peer${status.peers === 1 ? "" : "s"}` : "";
    const bar = status.bar_known ? "" : " · bar unknown (resync to set it)";
    return html`<div class="health ok">${status.tempo.toFixed(1)} BPM${peers}${bar}</div>`;
  }
  const waiting = s.source === "link" ? "No Link peers on the network" : `Tap at least twice (${status.taps} so far)`;
  return html`<div class="health idle">${waiting}</div>`;
}

export function BeatSection() {
  const [error, setError] = useState(null);
  const info = beatInfo.value;
  const act = async (method, path, body) => {
    try {
      beatInfo.value = await call(method, path, body);
      setError(null);
    } catch (exc) {
      setError(exc.message);
      refreshBeat();
    }
  };
  const change = (changes) => act("PATCH", "", changes);

  if (info == null) return null;
  if (info.unavailable) return html`<section class="diag-section"><h3>Beat sync</h3><p class="muted">Beat sync is not running.</p></section>`;
  const { settings: s, status } = info;
  return html`
    <section class="diag-section">
      <h3>Beat sync <span class="diag-aside">the floor's ctx.beat, and launches on the beat</span></h3>
      <${RadioRow} label="Beat source" options=${SOURCES} value=${s.source} onPick=${(v) => change({ source: v })} />
      <${BeatHealth} s=${s} status=${status} />
      ${s.source !== "off" && html`
        <div class="ext-row beat-controls">
          ${s.source === "tap" && html`<button class="beat-tap" onClick=${() => act("POST", "/tap")}>Tap</button>`}
          <button onClick=${() => act("POST", "/resync")} title="Make the next beat a downbeat">Resync</button>
          <button onClick=${() => act("POST", "/nudge", { ms: -10 })} title="The beat 10 ms earlier on the floor">−10 ms</button>
          <button onClick=${() => act("POST", "/nudge", { ms: 10 })} title="The beat 10 ms later on the floor">+10 ms</button>
          <${RadioRow} label="Multiplier" options=${MULTIPLIERS} value=${s.multiplier} onPick=${(v) => change({ multiplier: v })} />
        </div>
        <div class="ext-row">
          <${NumberField} label="Beats per bar" value=${s.beats_per_bar} min="1" max="16" onCommit=${(v) => change({ beats_per_bar: v })} />
          <${NumberField} label="Latency offset (ms)" value=${s.offset_ms} min="-500" max="500" onCommit=${(v) => change({ offset_ms: v })} />
        </div>`}
      <div class="ext-field">
        <span class="label">Quantize launches</span>
        <${RadioRow} label="Quantize launches" options=${QUANTA} value=${s.launch_quantum} onPick=${(v) => change({ launch_quantum: v })} />
      </div>
      <p class="ext-note">
        The latency offset reads the music that much later than each frame goes out, for the time it takes to
        reach the LEDs: tune it by eye until the floor lands on the kick. Beats per bar is also Link's quantum.
      </p>
      ${error && html`<div class="command-error" role="alert">${error}</div>`}
    </section>`;
}

// ---- the transport readout ---------------------------------------------------

/** Tempo and a lamp per beat of the bar. The state arrives at 2 Hz, far too
 *  slow to show beats, so the lamp runs locally from the last snapshot's
 *  position and tempo and is corrected at every poll. */
export function BeatIndicator({ live }) {
  const state = runnerState.value;
  const beat = state?.beat;
  const anchor = useRef(null);
  const [lit, setLit] = useState(-1);

  useEffect(() => {
    anchor.current = beat ? { at: performance.now(), bar: beat.bar_phase * beat.beats_per_bar, tempo: beat.tempo, per: beat.beats_per_bar } : null;
  }, [beat]);

  useEffect(() => {
    let frame;
    const tick = () => {
      const a = anchor.current;
      if (a) {
        const position = (a.bar + ((performance.now() - a.at) / 1000) * (a.tempo / 60)) % a.per;
        const index = Math.floor(position);
        setLit((current) => (current === index ? current : index));
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, []);

  const tap = beatInfo.value?.settings?.source === "tap";
  if (!beat && !tap) return null;
  const waiting = state?.launch_pending ? LAUNCH_WAIT[state.launch_quantum] : null;
  const tapNow = async () => { beatInfo.value = await call("POST", "/tap").catch(() => beatInfo.value); };
  return html`
    <div class="beat-indicator" aria-label="Beat">
      <span class="label">Beat</span>
      <span class="beat-readout">
        <span class="num">${beat ? beat.tempo.toFixed(1) : "—"}</span>
        ${beat && html`
          <span class="beat-lamps" aria-hidden="true">
            ${Array.from({ length: beat.beats_per_bar }, (_, i) => html`<span class=${`beat-lamp${i === lit ? " on" : ""}${i === 0 ? " one" : ""}`}></span>`)}
          </span>`}
        ${tap && html`<button class="beat-tap" disabled=${!live} onClick=${tapNow}>Tap</button>`}
        ${waiting && html`<span class="tag">${waiting}</span>`}
      </span>
    </div>`;
}
