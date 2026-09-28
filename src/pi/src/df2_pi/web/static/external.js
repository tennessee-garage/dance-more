// The External tab: Art-Net / sACN from a media server. Signal status, the
// source switch (internal / external / mix), and the receiver's settings.
// Every change is a PATCH to /api/external, applied and stored at once.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { useDraft } from "./params.js";
import { activeTab } from "./state.js";

const POLL_MS = 1000;
const SOURCES = [
  ["internal", "Internal", "Ignore Art-Net; the playlist plays"],
  ["external", "External", "Art-Net takes over while there is signal"],
  ["mix", "Mix", "Art-Net over the playlist, by the mix amount"],
];
const MODES = [
  ["tile", "Tile - 64 tile colours (1 universe)"],
  ["grid", "Grid - a W x H image, sampled at each LED"],
  ["raw", "Raw - every LED in chain order (23 universes)"],
];

async function patch(changes) {
  const response = await fetch("api/external", {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(changes),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    throw new Error(typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : `HTTP ${response.status}`);
  }
  return body;
}

function Signal({ status }) {
  if (status.live) {
    const origin = [status.protocol === "sacn" ? "sACN" : "Art-Net", status.sender].filter(Boolean).join(" · ");
    return html`<div class="health ok">Live: ${origin} · ${status.fps} fps</div>`;
  }
  const since = status.age_s != null ? `last frame ${status.age_s.toFixed(1)} s ago` : "nothing received yet";
  return html`<div class="health idle">No signal (${since})</div>`;
}

function NumberField({ label, value, min, max, step = 1, onCommit, disabled }) {
  const [shown, edit, release] = useDraft(value);
  return html`
    <label class="ext-field">
      <span class="label">${label}</span>
      <input
        type="number" min=${min} max=${max} step=${step} value=${String(shown)} disabled=${disabled}
        onInput=${(e) => edit(e.currentTarget.value)}
        onChange=${(e) => { const v = Number(e.currentTarget.value); release(); if (!Number.isNaN(v)) onCommit(v); }}
      />
    </label>`;
}

export function ExternalPanel() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  const refresh = async () => {
    try {
      const response = await fetch("api/external");
      if (response.status === 503) {
        setInfo({ unavailable: true });
        return;
      }
      if (response.ok) setInfo(await response.json());
    } catch {
      // the next poll tries again
    }
  };

  useEffect(() => {
    refresh();
    const timer = setInterval(() => {
      if (activeTab.value === "external" && !document.hidden) refresh();
    }, POLL_MS);
    return () => clearInterval(timer);
  }, []);

  const change = async (changes) => {
    try {
      setInfo(await patch(changes));
      setError(null);
    } catch (exc) {
      setError(exc.message);
      refresh();
    }
  };

  if (info == null) return html`<p class="muted">Loading…</p>`;
  if (info.unavailable) {
    return html`<p class="muted">Art-Net / sACN input is not running (the server was started with <code>--no-external</code>).</p>`;
  }
  const { settings: s, status } = info;
  const listening = status.listening.map((p) => `${p === "sacn" ? "sACN" : "Art-Net"} :${status.ports[p]}`).join(", ") || "nothing";

  return html`
    <div class="diagnostics external">
      <section class="diag-section">
        <h3>Signal</h3>
        <${Signal} status=${status} />
        <dl class="diag-counters">
          <div><dt>Showing</dt><dd>${status.applied}</dd></div>
          <div><dt>Frames</dt><dd class="num">${status.frames}</dd></div>
          <div><dt>Packets</dt><dd class="num">${status.packets}</dd></div>
          <div><dt>Polls answered</dt><dd class="num">${status.polls}</dd></div>
          <div><dt>Listening</dt><dd>${listening}</dd></div>
        </dl>
        ${Object.entries(status.errors).map(([p, e]) => html`<div class="sink-error">${p}: ${e}</div>`)}
      </section>

      <section class="diag-section">
        <h3>Source <span class="diag-aside">what the floor shows while there is signal</span></h3>
        <div class="floor-buttons" role="radiogroup" aria-label="Source">
          ${SOURCES.map(([value, label, help]) => html`
            <button
              role="radio" aria-checked=${s.source === value} class=${s.source === value ? "active" : ""}
              title=${help} onClick=${() => change({ source: value })}
            >${label}</button>`)}
        </div>
        ${s.source === "mix" && html`
          <label class="ext-field ext-mix">
            <span class="label">Mix</span>
            <${MixSlider} value=${s.mix} onCommit=${(v) => change({ mix: v })} />
          </label>`}
        <${NumberField}
          label="Timeout (s)" value=${s.timeout_s} min="0.2" max="60" step="0.1"
          onCommit=${(v) => change({ timeout_s: v })}
        />
      </section>

      <section class="diag-section">
        <h3>Input <span class="diag-aside">${status.universes} universe${status.universes === 1 ? "" : "s"}</span></h3>
        <label class="ext-field">
          <span class="label">Mode</span>
          <select value=${s.mode} onChange=${(e) => change({ mode: e.currentTarget.value })}>
            ${MODES.map(([value, label]) => html`<option value=${value}>${label}</option>`)}
          </select>
        </label>
        ${s.mode === "grid" && html`
          <div class="ext-row">
            <${NumberField} label="Grid width" value=${s.grid_width} min="1" max="136" onCommit=${(v) => change({ grid_width: v })} />
            <${NumberField} label="Grid height" value=${s.grid_height} min="1" max="136" onCommit=${(v) => change({ grid_height: v })} />
          </div>`}
        <div class="ext-row">
          <label class="ext-check">
            <input type="checkbox" checked=${s.artnet_enabled} onChange=${(e) => change({ artnet_enabled: e.currentTarget.checked })} />
            Art-Net
          </label>
          <${NumberField}
            label="First universe (0-based)" value=${s.artnet_universe} min="0" max="32767"
            onCommit=${(v) => change({ artnet_universe: v })} disabled=${!s.artnet_enabled}
          />
        </div>
        <div class="ext-row">
          <label class="ext-check">
            <input type="checkbox" checked=${s.sacn_enabled} onChange=${(e) => change({ sacn_enabled: e.currentTarget.checked })} />
            sACN
          </label>
          <${NumberField}
            label="First universe (1-based)" value=${s.sacn_universe} min="1" max="63999"
            onCommit=${(v) => change({ sacn_universe: v })} disabled=${!s.sacn_enabled}
          />
        </div>
        <p class="ext-note">
          Colour values are sent as they are - leave the sender's output gamma at 1.0.
          Tile and grid are raster order from the top-left of the floor as the preview shows it.
          For raw mode, <a href="api/floor/leds?format=csv" download>download every LED's position and address (CSV)</a>.
        </p>
      </section>
      ${error && html`<div class="command-error" role="alert">${error}</div>`}
    </div>`;
}

function MixSlider({ value, onCommit }) {
  const [shown, edit, release] = useDraft(value);
  return html`
    <span class="slider">
      <input
        type="range" min="0" max="1" step="0.01" value=${String(shown)} aria-label="Mix"
        onInput=${(e) => edit(e.currentTarget.valueAsNumber)}
        onChange=${(e) => { release(); onCommit(e.currentTarget.valueAsNumber); }}
      />
      <span class="num">${Math.round(shown * 100)}%</span>
    </span>`;
}
