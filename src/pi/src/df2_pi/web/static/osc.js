// The External tab's OSC section (#132): on/off, the port, and a monitor of
// the last messages received - what TouchDesigner, Resolume or a phone app
// is actually sending, with any error the floor had with it.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { NumberField } from "./params.js";
import { activeTab } from "./state.js";

const POLL_MS = 1000;

async function call(method, body) {
  const response = await fetch("api/osc", {
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

/** OSC floats are single precision: 0.4 arrives as 0.4000000059604645. */
function shown(value) {
  return typeof value === "number" && !Number.isInteger(value) ? +value.toFixed(4) : value;
}

function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : `${Math.floor(s / 60)} min`;
}

export function OscSection() {
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
  const change = async (changes) => {
    try {
      setInfo(await call("PATCH", changes));
      setError(null);
    } catch (exc) {
      setError(exc.message);
      refresh();
    }
  };

  if (info == null) return null;
  if (info.unavailable) return html`<section class="diag-section"><h3>OSC</h3><p class="muted">OSC is not running.</p></section>`;
  const { settings: s, status } = info;
  const health = !s.enabled
    ? html`<div class="health idle">Off</div>`
    : status.error
      ? html`<div class="health bad">${status.error}</div>`
      : html`<div class="health ok">Listening on UDP ${status.port} · ${status.received} received${status.errors ? ` · ${status.errors} not understood` : ""}${status.subscribers.length ? ` · feedback to ${status.subscribers.join(", ")}` : ""}</div>`;
  return html`
    <section class="diag-section">
      <h3>OSC <span class="diag-aside">TouchDesigner, Resolume, phone apps: /floor/... addresses</span></h3>
      <label class="ext-check">
        <input type="checkbox" checked=${s.enabled} onChange=${(e) => change({ enabled: e.currentTarget.checked })} />
        Listen for OSC
      </label>
      ${s.enabled && html`
        <div class="ext-row">
          <${NumberField} label="UDP port" value=${s.port} min="1" max="65535" onCommit=${(v) => change({ port: v })} />
        </div>`}
      ${health}
      ${status.recent.length > 0 && html`
        <table class="diag-table osc-recent">
          <thead><tr><th>Ago</th><th>From</th><th>Address</th><th>Values</th></tr></thead>
          <tbody>
            ${status.recent.map((m) => html`
              <tr class=${m.error ? "warn" : ""} title=${m.error ?? ""}>
                <td class="num">${ago(m.t)}</td>
                <td>${m.from ?? ""}</td>
                <td class="osc-address">${m.address}${m.error ? html` <span class="tag">${m.error}</span>` : ""}</td>
                <td class="num">${m.args.map(shown).join(" ")}</td>
              </tr>`)}
          </tbody>
        </table>`}
      <p class="ext-note">
        Send to this machine's address on the port above. The addresses are in
        <a href="https://github.com/tennessee-garage/dance-more/blob/main/docs/external-input.md#osc">docs/external-input.md</a>;
        from Resolume, set the outgoing address of a clip or a fader to one of them.
      </p>
      ${error && html`<div class="command-error" role="alert">${error}</div>`}
    </section>`;
}
