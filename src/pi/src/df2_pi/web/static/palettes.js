// The floor palette (#128): a chip row in the transport bar for picking the
// active palette, and the Palettes tab for the library and your own.
// The active one comes from the runner's state, so a desk changing it over
// DMX shows here too.

import { html } from "htm/preact";
import { signal } from "@preact/signals";
import { useState } from "preact/hooks";
import { animationList, runnerState } from "./state.js";

const MIN_STOPS = 2;
const MAX_STOPS = 8;
const FLOOR = "floor"; // a palette param's value for "the floor's palette" (palette.py)

/** {active, palettes: [{name, stops, builtin}]} from /api/palettes, or {unavailable}. */
export const paletteList = signal(null);
const paletteError = signal(null);

async function call(method, path = "", body) {
  const response = await fetch(`api/palettes${path}`, {
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

async function act(method, path, body) {
  try {
    paletteList.value = await call(method, path, body);
    paletteError.value = null;
    return true;
  } catch (exc) {
    paletteError.value = exc.message;
    return false;
  }
}

export function loadPalettes() {
  act("GET");
}

/** The active palette's name: the runner's, live; the list's until the first poll. */
function activeName() {
  return runnerState.value?.palette ?? paletteList.value?.active;
}

/** A CSS background for a palette's loop: the stops evenly spaced, back to the
 *  first at the end, blended in linear light as the floor blends them (with a
 *  plain gradient first, for a browser without srgb-linear interpolation). */
function gradient(stops) {
  const list = [...stops, stops[0]].map((s) => `#${s}`).join(", ");
  return `background: linear-gradient(to right, ${list}); background: linear-gradient(in srgb-linear to right, ${list});`;
}

/** Whether `id`, playing with `params`, takes its colours from the floor's
 *  palette: it has a palette param (one offering "floor") set to "floor". */
function followsFloor(id, params) {
  const specs = animationList.value?.animations.find((a) => a.id === id)?.params;
  if (!specs || !params) return false;
  return Object.entries(specs).some(([name, spec]) => spec.choices?.includes(FLOOR) && params[name] === FLOOR);
}

/** The floor palette's chips, shown only while what is playing (or the layer
 *  over it) follows the floor palette - otherwise they would change nothing. */
export function PaletteBar({ live }) {
  const list = paletteList.value;
  const state = runnerState.value;
  if (!list || list.unavailable) return null;
  if (!followsFloor(state?.animation?.[0], state?.params) && !followsFloor(state?.layer?.animation?.[0], state?.layer?.params)) return null;
  const active = activeName();
  return html`
    <div class="live-params palette-bar" aria-label="Palette">
      <span class="row-label" title="The floor's colour scheme, for animations that follow it">Palette</span>
      <div class="palette-chips" role="radiogroup" aria-label="Floor palette">
        ${list.palettes.map((p) => html`
          <button
            role="radio" aria-checked=${p.name === active} title=${p.name} disabled=${!live}
            class=${p.name === active ? "palette-chip active" : "palette-chip"}
            style=${gradient(p.stops)}
            onClick=${() => act("POST", "/active", { name: p.name })}
          ><span class="palette-chip-name">${p.name}</span></button>`)}
      </div>
    </div>`;
}

function Editor({ start, onDone }) {
  const [name, setName] = useState(start.name);
  const [stops, setStops] = useState(start.stops);
  const set = (i, value) => setStops(stops.map((s, j) => (j === i ? value.replace("#", "") : s)));
  const save = async () => {
    if (await act("PUT", `/${encodeURIComponent(name.trim())}`, { stops })) onDone();
  };
  return html`
    <div class="palette-editor">
      <div class="palette-preview" style=${gradient(stops)}></div>
      <label class="ext-field">
        <span class="label">Name</span>
        <input type="text" value=${name} disabled=${start.existing} maxlength="32" placeholder="a-z, 0-9, - and _"
          onInput=${(e) => setName(e.currentTarget.value.toLowerCase())} />
      </label>
      <div class="palette-stops">
        ${stops.map((s, i) => html`
          <span class="palette-stop">
            <input type="color" value=${`#${s}`} aria-label=${`Stop ${i + 1}`} onInput=${(e) => set(i, e.currentTarget.value)} />
            <button class="icon-small" disabled=${stops.length <= MIN_STOPS} title="Remove this stop"
              onClick=${() => setStops(stops.filter((_, j) => j !== i))}>×</button>
          </span>`)}
        <button disabled=${stops.length >= MAX_STOPS} onClick=${() => setStops([...stops, stops[stops.length - 1]])}>Add stop</button>
      </div>
      <div class="palette-actions">
        <button class="primary" disabled=${!name.trim()} onClick=${save}>Save</button>
        <button onClick=${onDone}>Cancel</button>
      </div>
    </div>`;
}

export function PalettesPanel() {
  const [editing, setEditing] = useState(null); // {name, stops, existing}
  const list = paletteList.value;
  if (list == null) return html`<p class="muted">Loading…</p>`;
  if (list.unavailable) return html`<p class="muted">Palettes are not available.</p>`;
  const active = activeName();
  const error = paletteError.value;
  return html`
    <div class="diagnostics palettes">
      <section class="diag-section">
        <h3>Palettes <span class="diag-aside">the floor's colour scheme, for animations that follow it</span></h3>
        <p class="ext-note">
          Animations with a Palette set to <em>floor</em> take their colours from the active one. A desk can switch it
          on DMX control channel 18, counting the list below from 0.
        </p>
        ${editing && html`<${Editor} key=${editing.name || "new"} start=${editing} onDone=${() => setEditing(null)} />`}
        <ul class="palette-list">
          ${list.palettes.map((p, i) => html`
            <li class=${p.name === active ? "palette-row active" : "palette-row"}>
              <span class="num palette-index" title="Its DMX channel 18 value">${i}</span>
              <span class="palette-preview" style=${gradient(p.stops)}></span>
              <span class="palette-name">${p.name}${p.builtin ? "" : html` <span class="tag">yours</span>`}</span>
              <span class="palette-actions">
                <button disabled=${p.name === active} onClick=${() => act("POST", "/active", { name: p.name })}>${p.name === active ? "Active" : "Use"}</button>
                ${p.builtin
                  ? html`<button onClick=${() => setEditing({ name: `${p.name}-copy`, stops: p.stops, existing: false })}>Copy</button>`
                  : html`
                    <button onClick=${() => setEditing({ name: p.name, stops: p.stops, existing: true })}>Edit</button>
                    <button class="danger" onClick=${() => act("DELETE", `/${encodeURIComponent(p.name)}`)}>Delete</button>`}
              </span>
            </li>`)}
        </ul>
        <button onClick=${() => setEditing({ name: "", stops: ["ff0000", "0000ff"], existing: false })}>New palette</button>
        ${error && html`<div class="command-error" role="alert">${error}</div>`}
      </section>
    </div>`;
}
