// The page shell: transport bar (top), preview (left), tabs (right).
// Later issues fill the regions; this one fills the transport readout.

import { html, render } from "htm/preact";
import { signal } from "@preact/signals";
import { connection, runnerState, startPolling } from "./state.js";

const DASH = "—";

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

function TransportBar() {
  const state = runnerState.value;
  // `playlist` and `animation` are [id, name] pairs, or null.
  const playlist = state?.playlist?.[1] ?? DASH;
  const animation = state?.animation?.[1] ?? DASH;
  return html`
    <header class="transport">
      <div class="readout">
        <${Field} label="Playlist" wide>${playlist}<//>
        <${Field} label="Animation" wide>${animation}<//>
        <${Field} label="Elapsed"><span class="num">${clock(state?.elapsed_s)}</span><//>
        <${Field} label="Remaining"><span class="num">${clock(state?.remaining_s)}</span><//>
        <${Field} label="Frame"><span class="num">${state?.frame ?? DASH}</span><//>
      </div>
      <${ConnectionBadge} />
    </header>`;
}

function Preview() {
  return html`<section class="preview" aria-label="Floor preview"><div class="floor"></div></section>`;
}

const TABS = [
  ["playlists", "Playlists"],
  ["animations", "Animations"],
  ["diagnostics", "Diagnostics"],
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
      ${TABS.map(([id]) => html`
        <div
          class="panel"
          role="tabpanel"
          id=${`panel-${id}`}
          aria-labelledby=${`tab-${id}`}
          hidden=${id !== active}
        ></div>`)}
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
