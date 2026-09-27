// The page shell: transport bar (top), preview (left), tabs (right).
// Later issues fill the remaining tabs.

import { html, render } from "htm/preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { signal } from "@preact/signals";
import { AnimationsPanel } from "./animations.js";
import { createPreview } from "./preview.js";
import { connectPreview, previewStatus } from "./preview-stream.js";
import { ParamControls } from "./params.js";
import {
  animationList,
  command,
  commandError,
  connection,
  currentAnimationId,
  paramErrors,
  runnerState,
  setLiveParam,
  startPolling,
} from "./state.js";

const DASH = "—";
const BRIGHTNESS_SETTLE_MS = 2000; // how long a released slider waits for the server to agree

/** Seconds as m:ss; a dash for null. */
function clock(seconds) {
  if (seconds == null) return DASH;
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function Field({ label, children, wide }) {
  return html`
    <div class=${wide ? "field wide" : "field"}>
      <span class="label">${label}</span>
      <span class="value">${children}</span>
    </div>`;
}

const CONNECTION_TEXT = { connecting: "Connecting", ok: "Live", lost: "No state" };

function ConnectionBadge() {
  const status = connection.value;
  return html`<span class=${`badge badge-${status}`} role="status">${CONNECTION_TEXT[status]}</span>`;
}

// ---- controls ------------------------------------------------------------

const ICONS = {
  previous: "M6 5h2v14H6zM20 5v14L9 12z",
  next: "M16 5h2v14h-2zM4 5v14l11-7z",
  play: "M7 4v16l13-8z",
  pause: "M6 5h4v14H6zM14 5h4v14h-4z",
};

function Icon({ name }) {
  return html`<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d=${ICONS[name]} fill="currentColor" /></svg>`;
}

function PlaybackButtons({ state, live }) {
  const running = state?.playing && !state?.paused;
  // Paused resumes where it was; stopped (ended, or nothing loaded) plays from the top.
  const toggle = () => command(running ? "pause" : state?.paused ? "resume" : "play");
  return html`
    <div class="buttons">
      <button class="icon" onClick=${() => command("previous")} disabled=${!live} aria-label="Previous" title="Previous">
        <${Icon} name="previous" />
      </button>
      <button class="icon primary" onClick=${toggle} disabled=${!live} aria-label=${running ? "Pause" : "Play"} title=${running ? "Pause" : "Play"}>
        <${Icon} name=${running ? "pause" : "play"} />
      </button>
      <button class="icon" onClick=${() => command("next")} disabled=${!live} aria-label="Next" title="Next">
        <${Icon} name="next" />
      </button>
    </div>`;
}

/** Posts on release, not on every step of a drag. While dragging, and after
 *  release until the server reports the new value, the slider shows its
 *  own value rather than the polled one - a poll never yanks it back. */
function BrightnessSlider({ state, live }) {
  const server = state?.brightness ?? 255;
  const [draft, setDraft] = useState(null);
  const releasedAt = useRef(0);

  useEffect(() => {
    if (draft === null || !releasedAt.current) return;
    if (server === draft || Date.now() - releasedAt.current > BRIGHTNESS_SETTLE_MS) {
      setDraft(null);
      releasedAt.current = 0;
    }
  }, [state]);

  const shown = draft ?? server;
  return html`
    <label class="brightness">
      <span class="label">Brightness</span>
      <input
        type="range" min="0" max="255" step="1"
        value=${shown}
        disabled=${!live}
        onInput=${(e) => { releasedAt.current = 0; setDraft(e.currentTarget.valueAsNumber); }}
        onChange=${(e) => {
          const value = e.currentTarget.valueAsNumber;
          setDraft(value);
          releasedAt.current = Date.now();
          command("brightness", { value });
        }}
      />
      <span class="num">${Math.round((shown / 255) * 100)}%</span>
    </label>`;
}

function BlackoutButton({ state, live }) {
  const on = !!state?.blacked_out;
  return html`
    <button
      class=${on ? "blackout active" : "blackout"}
      aria-pressed=${on}
      disabled=${!live}
      onClick=${() => command("blackout", { on: !on })}
      title=${on ? "The floor is blacked out: click to restore" : "Black out the floor"}
    >${on ? "Blacked out" : "Blackout"}</button>`;
}

// ---- transport bar -------------------------------------------------------

/** Controls for whatever is playing, generated from its Param specs, each
 *  edit streamed to the runner. Keyed by animation, so the controls (and any
 *  half-finished edit) swap cleanly when the runner moves on. */
function LiveParams() {
  const id = currentAnimationId.value;
  const specs = animationList.value?.animations.find((a) => a.id === id)?.params;
  useEffect(() => { paramErrors.value = {}; }, [id]);
  if (!specs || Object.keys(specs).length === 0) return null;
  return html`
    <div class="live-params" aria-label="Animation parameters">
      <${ParamControls}
        key=${id}
        specs=${specs}
        values=${runnerState.value?.params ?? {}}
        errors=${paramErrors.value}
        onChange=${setLiveParam}
      />
    </div>`;
}

function Progress({ state }) {
  const elapsed = state?.elapsed_s;
  const remaining = state?.remaining_s;
  if (elapsed == null || remaining == null) return html`<div class="progress" aria-hidden="true"></div>`;
  const total = elapsed + remaining;
  const fraction = total > 0 ? Math.min(1, elapsed / total) : 0;
  return html`
    <div class="progress" role="progressbar" aria-label="Current entry" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(fraction * 100)}>
      <div class="progress-fill" style=${{ width: `${fraction * 100}%` }}></div>
    </div>`;
}

function TransportBar() {
  const state = runnerState.value;
  const live = connection.value === "ok" && state != null;
  // `playlist` and `animation` are [id, name] pairs, or null.
  const playlist = state?.playlist?.[1] ?? DASH;
  const animation = state?.animation?.[1] ?? DASH;
  const position = state?.entry_index != null ? `${state.entry_index + 1} / ${state.entry_count}` : DASH;
  const error = commandError.value;
  return html`
    <header class=${state?.blacked_out ? "transport blacked-out" : "transport"}>
      <div class="transport-row">
        <${PlaybackButtons} state=${state} live=${live} />
        <div class="readout">
          <${Field} label="Playlist" wide>${playlist}<//>
          <${Field} label="Entry"><span class="num">${position}</span><//>
          <${Field} label="Animation" wide>${animation}${state?.one_off ? html` <span class="tag">one-off</span>` : ""}<//>
          <${Field} label="Elapsed"><span class="num">${clock(state?.elapsed_s)}</span><//>
          <${Field} label="Remaining"><span class="num">${clock(state?.remaining_s)}</span><//>
          <${Field} label="Frame"><span class="num">${state?.frame ?? DASH}</span><//>
        </div>
        <${BrightnessSlider} state=${state} live=${live} />
        <${BlackoutButton} state=${state} live=${live} />
        <${ConnectionBadge} />
      </div>
      <${LiveParams} />
      <${Progress} state=${state} />
      ${error && html`<div class="command-error" role="alert">${error}</div>`}
    </header>`;
}

// ---- preview and tabs ----------------------------------------------------

const GEOMETRY_RETRY_MS = 2000;

function PreviewStatus() {
  const { connection, format, fps, missed } = previewStatus.value;
  const detail = connection === "live" ? ` · ${format} · ${fps} fps${missed ? ` · ${missed} missed` : ""}` : "";
  return html`<div class="preview-status num">${connection}${detail}</div>`;
}

/** The floor, drawn by preview.js. Re-renders only when the geometry
 *  arrives: the stream hands each record to the renderer's draw() directly,
 *  so a frame never goes through Preact. */
function Preview() {
  const canvas = useRef(null);
  const draw = useRef(() => {});
  const [geometry, setGeometry] = useState(null);
  const [failure, setFailure] = useState(null);

  useEffect(() => {
    let timer = null;
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch("api/preview/geometry");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = await response.json();
        if (!cancelled) setGeometry(body);
      } catch {
        if (!cancelled) timer = setTimeout(load, GEOMETRY_RETRY_MS);
      }
    }
    load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!geometry) return undefined;
    let preview;
    try {
      preview = createPreview(canvas.current, geometry);
    } catch (exc) {
      setFailure(`The floor preview needs WebGL: ${exc.message}.`);
      return undefined;
    }
    draw.current = (record) => preview.draw(record);
    const observer = new ResizeObserver(() => preview.resize());
    observer.observe(canvas.current);
    return () => {
      observer.disconnect();
      draw.current = () => {};
    };
  }, [geometry]);

  useEffect(() => {
    if (failure) return undefined; // nothing to draw into: do not load the Pi
    const stream = connectPreview((record) => draw.current(record));
    return () => stream.close();
  }, [failure]);

  return html`
    <section class="preview" aria-label="Floor preview">
      <canvas class="floor" ref=${canvas} hidden=${!!failure}></canvas>
      ${failure ? html`<p class="preview-failure">${failure}</p>` : html`<${PreviewStatus} />`}
    </section>`;
}

// [id, label, panel component or null while the tab is still empty]
const TABS = [
  ["playlists", "Playlists", null],
  ["animations", "Animations", AnimationsPanel],
  ["diagnostics", "Diagnostics", null],
];
const activeTab = signal(TABS[0][0]);

function Tabs() {
  const active = activeTab.value;
  return html`
    <section class="tabs">
      <div class="tablist" role="tablist">
        ${TABS.map(([id, label]) => html`
          <button
            role="tab"
            id=${`tab-${id}`}
            aria-selected=${id === active}
            aria-controls=${`panel-${id}`}
            onClick=${() => { activeTab.value = id; }}
          >${label}</button>`)}
      </div>
      ${TABS.map(([id, , Panel]) => html`
        <div
          class="panel"
          role="tabpanel"
          id=${`panel-${id}`}
          aria-labelledby=${`tab-${id}`}
          hidden=${id !== active}
        >${Panel && html`<${Panel} />`}</div>`)}
    </section>`;
}

function App() {
  return html`
    <${TransportBar} />
    <main>
      <${Preview} />
      <${Tabs} />
    </main>`;
}

render(html`<${App} />`, document.getElementById("app"));
startPolling();
