// The External tab's MIDI section (#131): the ports listening, MIDI learn,
// a monitor of what arrives, and the mapping. Every change to the mapping is
// written to its YAML file at once.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { activeTab } from "./state.js";

const POLL_MS = 1000;
// Suggestions for the learn field: any /floor/... address works.
const ADDRESSES = [
  "/floor/next", "/floor/previous", "/floor/restart", "/floor/goto/0", "/floor/play/lightning",
  "/floor/palette/fire", "/floor/palette/ocean", "/floor/bump", "/floor/blackout", "/floor/freeze", "/floor/hold",
  "/floor/brightness", "/floor/speed", "/floor/strobe", "/floor/mix", "/floor/hue", "/floor/saturation",
  "/floor/macro/1", "/floor/macro/2", "/floor/macro/3", "/floor/macro/4", "/floor/trigger/0",
  "/floor/tempo/tap", "/floor/tempo/resync", "/floor/reset",
];

async function call(method, path = "", body) {
  const response = await fetch(`api/midi${path}`, {
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

function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : `${Math.floor(s / 60)} min`;
}

function LearnForm({ learning, act }) {
  const [to, setTo] = useState("/floor/");
  const [toggle, setToggle] = useState(false);
  const [value, setValue] = useState("");
  const [thisPort, setThisPort] = useState(false);
  if (learning) {
    return html`
      <div class="health ok midi-learning">
        Move a control on your MIDI device to bind it to <code>${learning.to}</code>…
        <button onClick=${() => act("DELETE", "/learn")}>Cancel</button>
      </div>`;
  }
  const start = () => act("POST", "/learn", {
    to: to.trim(), toggle, this_port_only: thisPort, value: value.trim() === "" ? null : Number(value),
  });
  return html`
    <div class="ext-row midi-learn">
      <label class="ext-field">
        <span class="label">Bind a control to</span>
        <input type="text" list="midi-addresses" value=${to} onInput=${(e) => setTo(e.currentTarget.value)} />
        <datalist id="midi-addresses">${ADDRESSES.map((a) => html`<option value=${a} />`)}</datalist>
      </label>
      <label class="ext-check" title="Each press flips it on or off: blackout, freeze, hold">
        <input type="checkbox" checked=${toggle} onChange=${(e) => setToggle(e.currentTarget.checked)} /> Toggle
      </label>
      <label class="ext-field midi-value" title="Send this on a press instead of how hard it was hit">
        <span class="label">Fixed value</span>
        <input type="text" inputmode="decimal" placeholder="—" value=${value} onInput=${(e) => setValue(e.currentTarget.value)} />
      </label>
      <label class="ext-check" title="Only for the device it is learned on">
        <input type="checkbox" checked=${thisPort} onChange=${(e) => setThisPort(e.currentTarget.checked)} /> This device only
      </label>
      <button disabled=${!to.trim().startsWith("/floor/") || to.trim() === "/floor/"} onClick=${start}>Learn</button>
    </div>`;
}

export function MidiSection() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const refresh = async () => {
    try {
      setInfo(await call("GET"));
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
  const act = async (method, path, body) => {
    try {
      setInfo(await call(method, path, body));
      setError(null);
    } catch (exc) {
      setError(exc.message);
      refresh();
    }
  };

  if (info == null) return null;
  if (info.unavailable) return html`<section class="diag-section"><h3>MIDI</h3><p class="muted">MIDI is not running.</p></section>`;
  const { status: st, map } = info;
  const problems = Object.entries(st.port_errors);
  const health = !st.enabled
    ? html`<div class="health idle">Off</div>`
    : st.ports.length === 0
      ? html`<div class="health idle">No MIDI devices found${problems.length ? ` - ${problems.map(([p, e]) => `${p}: ${e}`).join("; ")}` : ""}</div>`
      : html`<div class="health ok">Listening to ${st.ports.join(", ")} · ${st.received} received${st.errors ? ` · ${st.errors} not understood` : ""}${st.clock_received ? ` · clock` : ""}</div>`;
  return html`
    <section class="diag-section">
      <h3>MIDI <span class="diag-aside">controllers, and anything else that sends MIDI</span></h3>
      <label class="ext-check">
        <input type="checkbox" checked=${st.enabled} onChange=${(e) => act("PATCH", "", { enabled: e.currentTarget.checked })} />
        Listen for MIDI
      </label>
      ${health}
      ${st.map_error && html`<div class="health bad">${st.map_error}</div>`}
      ${st.enabled && html`<${LearnForm} learning=${st.learning} act=${act} />`}
      ${st.recent.length > 0 && html`
        <table class="diag-table osc-recent">
          <thead><tr><th>Ago</th><th>Device</th><th>Message</th><th>Did</th></tr></thead>
          <tbody>
            ${st.recent.map((m) => html`
              <tr class=${m.error ? "warn" : ""}>
                <td class="num">${ago(m.t)}</td>
                <td>${m.port}</td>
                <td class="num">${m.message}</td>
                <td class="osc-address">${m.learned ? "learned" : m.error ? html`<span class="tag">${m.error}</span>` : (m.sent ?? []).join(", ") || "—"}</td>
              </tr>`)}
          </tbody>
        </table>`}
      <details class="midi-map">
        <summary>Mapping: ${map.bindings.length} bindings</summary>
        <div class="ext-row">
          <label class="ext-check" title="Program Change, with Bank Select CC0/CC32: bank = playlist (name order, from 0), program = entry">
            <input type="checkbox" checked=${map.program_change} onChange=${(e) => act("PATCH", "/map", { program_change: e.currentTarget.checked })} /> Program change picks entries
          </label>
          <label class="ext-check" title="Pass MIDI clock to beat sync (choose MIDI clock as the beat source)">
            <input type="checkbox" checked=${map.clock} onChange=${(e) => act("PATCH", "/map", { clock: e.currentTarget.checked })} /> MIDI clock to beat sync
          </label>
          <button onClick=${() => confirm("Replace the whole mapping with the APC mini default?") && act("POST", "/default")}>Reset to APC mini default</button>
        </div>
        <table class="diag-table midi-bindings">
          <thead><tr><th>Control</th><th>Does</th><th></th></tr></thead>
          <tbody>
            ${map.bindings.map((b, i) => html`
              <tr>
                <td class="num">${b.control}</td>
                <td class="osc-address">${b.to}${b.toggle ? " (toggle)" : ""}${b.value != null ? ` = ${b.value}` : ""}</td>
                <td><button class="icon-small" title="Remove" onClick=${() => act("DELETE", `/bindings/${i}`)}>×</button></td>
              </tr>`)}
          </tbody>
        </table>
        <p class="ext-note">Saved to <code>${st.map_path}</code>; edit it by hand and the floor picks the change up.</p>
      </details>
      ${error && html`<div class="command-error" role="alert">${error}</div>`}
    </section>`;
}
