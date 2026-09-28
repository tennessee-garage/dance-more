// The Diagnostics tab: where "the floor looks stuttery" becomes *which*
// phase, and "a row is dark" becomes *which* row.
//
// Everything but the floor section comes from GET /api/state at the 2 Hz
// poll. The floor section's queries are Row Bus admin requests, sent only
// when a button is pressed.

import { html } from "htm/preact";
import { useState } from "preact/hooks";
import { runnerState } from "./state.js";

const DASH = "—";
const WARN_AT = 0.5; // of the frame budget
const BAD_AT = 0.9;

function ms(value, digits = 2) {
  return value == null ? DASH : value.toFixed(digits);
}

/** "", "warn" or "bad" for `used` ms of a `budget` ms frame. */
function level(used, budget) {
  if (used == null || !budget) return "";
  const share = used / budget;
  return share > BAD_AT ? "bad" : share > WARN_AT ? "warn" : "";
}

function Section({ title, children, aside }) {
  return html`
    <section class="diag-section">
      <h3>${title}${aside && html`<span class="diag-aside">${aside}</span>`}</h3>
      ${children}
    </section>`;
}

// ---- timing ------------------------------------------------------------------

function Timing({ timing, budget }) {
  const rows = [
    ...Object.entries(timing.phases).map(([name, p]) => [name, p, (v) => v]),
    ["jitter", timing.jitter_ms, (v) => v],
    // Slack is what is left of the frame: what it leaves unused is what counts against the budget.
    ["slack", timing.slack_ms, (v) => (v == null ? null : budget - v)],
    ["sleep overshoot", timing.sleep_overshoot_ms, (v) => v],
  ];
  return html`
    <table class="diag-table num">
      <thead><tr><th></th><th>p50</th><th>p95</th><th>max</th></tr></thead>
      <tbody>
        ${rows.map(([name, p, used]) => html`
          <tr>
            <th>${name}</th>
            ${["p50", "p95", "max"].map((k) => html`<td class=${level(used(p[k]), budget)}>${ms(p[k])}</td>`)}
          </tr>`)}
      </tbody>
    </table>`;
}

function Counters({ timing }) {
  const items = [
    ["frames", timing.frames],
    ["dropped", timing.dropped],
    ["re-anchors", timing.reanchors],
    ["spin margin", `${ms(timing.spin_margin_ms)} ms`],
  ];
  return html`
    <dl class="diag-counters">
      ${items.map(([k, v]) => html`<div><dt>${k}</dt><dd class=${k === "dropped" && v > 0 ? "num warn" : "num"}>${v}</dd></div>`)}
    </dl>`;
}

// ---- the Row Bus ---------------------------------------------------------------

function Bar({ value, budget }) {
  const share = Math.min(1, (value ?? 0) / budget);
  return html`<span class="bar"><span class=${`bar-fill ${level(value, budget)}`} style=${{ width: `${share * 100}%` }}></span></span>`;
}

function RowBus({ stats, budget }) {
  if (!stats) return html`<p class="muted">No floor attached (started with --no-hardware).</p>`;
  return html`
    <div class="rowbus">
      ${stats.chains.map((chain) => html`
        <div class="chain">
          <div class="chain-head num">
            chain ${chain.chain}: ${ms(chain.wire_ms, 1)} ms
            <${Bar} value=${chain.wire_ms} budget=${budget} />
          </div>
          ${chain.rows.map((row) => html`
            <div class="row-bar num">
              <span class="row-label">row ${row}</span>
              <${Bar} value=${stats.row_wire_ms[row]} budget=${budget} />
              <span class="row-figures">${ms(stats.row_wire_ms[row], 1)} ms · ${stats.row_bytes[row]} B</span>
            </div>`)}
        </div>`)}
      <p class="muted">Estimated from the last frame's payloads; the slowest chain is the Row Bus phase.</p>
    </div>`;
}

// ---- sinks and problems ---------------------------------------------------------

const COUNTERS = ["frames", "frames_handled", "dropped", "failures", "subscriber_count"];

function Sinks({ sinks }) {
  const hardware = sinks.hardware;
  return html`
    ${hardware && html`
      <div class=${hardware.attached && hardware.healthy !== false ? "health ok" : "health bad"}>
        Hardware: ${!hardware.attached ? `detached — ${hardware.reason}` : hardware.healthy === false ? "UNHEALTHY" : "healthy"}
        ${hardware.muted && html` · blacked out`}
      </div>`}
    <ul class="sink-list">
      ${Object.entries(sinks).map(([name, s]) => html`
        <li class=${s.attached ? "" : "detached"}>
          <span class="sink-name">${name}</span>
          ${s.attached
            ? html`<span class="sink-counters num">
                ${COUNTERS.filter((k) => s[k] != null).map((k) => `${k.replace("_", " ")} ${s[k]}`).join(" · ")}
              </span>`
            : html`<span class="sink-reason">detached: ${s.reason}</span>`}
          ${s.last_error && html`<div class="sink-error">${s.last_error}</div>`}
        </li>`)}
    </ul>`;
}

function Problems({ state }) {
  const loadErrors = Object.entries(state.load_errors);
  if (state.warnings.length === 0 && state.disabled_entries.length === 0 && loadErrors.length === 0) {
    return html`<p class="muted">None.</p>`;
  }
  return html`
    <ul class="problems">
      ${state.warnings.map((w) => html`<li class="warn">${w}</li>`)}
      ${state.disabled_entries.length > 0 && html`
        <li class="warn">Entries disabled after repeated failures: ${state.disabled_entries.join(", ")}</li>`}
      ${loadErrors.map(([id, message]) => html`<li class="bad"><strong>${id}</strong> failed to load: ${message}</li>`)}
    </ul>`;
}

// ---- the floor itself -------------------------------------------------------------

function uptime(seconds) {
  if (seconds == null) return DASH;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function volts(mV) {
  return mV == null ? DASH : `${(mV / 1000).toFixed(2)} V`;
}

function amps(mA) {
  return mA == null ? DASH : `${(mA / 1000).toFixed(2)} A`;
}

function StatusTable({ rows }) {
  return html`
    <table class="diag-table num">
      <thead><tr><th>row</th><th>chain</th><th>state</th><th>tiles</th><th>uptime</th><th>voltage</th><th>current</th></tr></thead>
      <tbody>
        ${rows.map((r) => html`
          <tr class=${!r.responding ? "bad" : r.state !== "running" ? "warn" : ""}>
            <th>${r.row}</th><td>${r.chain}</td>
            ${r.responding
              ? html`<td>${r.state ?? DASH}</td><td>${r.tiles_found ?? DASH}</td><td>${uptime(r.uptime_s)}</td>
                     <td>${volts(r.voltage_mV)}</td><td>${amps(r.current_mA)}</td>`
              : html`<td colspan="5">not responding</td>`}
          </tr>`)}
      </tbody>
    </table>`;
}

function VersionTable({ report }) {
  return html`
    <div class=${report.ok ? "health ok" : "health bad"}>${report.ok ? "Every row and tile in step" : "Out of step (marked)"}</div>
    <p class="muted versions-against">
      In step means matching what most of the floor runs —
      rows: <strong>${report.row_version?.text ?? DASH}</strong>,
      tiles: <strong>${report.tile_version?.text ?? DASH}</strong>
    </p>
    <table class="diag-table num versions">
      <thead><tr><th>row</th><th>firmware</th><th>tiles</th></tr></thead>
      <tbody>
        ${report.rows.map((r) => {
          const inStep = r.tiles.filter((t) => t.version && !t.out_of_step).length;
          const empty = r.tiles.filter((t) => t.state === "empty").length;
          const problems = r.tiles.filter((t) => t.out_of_step);
          return html`
            <tr>
              <th>${r.row}</th>
              <td class=${r.out_of_step ? "bad" : ""}>${r.responding ? r.version.text : "not responding"}</td>
              <td>
                ${!r.responding ? DASH : html`
                  <span>${inStep} in step${empty ? ` · ${empty} empty` : ""}</span>
                  ${problems.map((t) => html`<div class="bad">slot ${t.slot}: ${t.version ? t.version.text : t.state}</div>`)}`}
              </td>
            </tr>`;
        })}
      </tbody>
    </table>`;
}

function Floor({ hardware }) {
  const [busy, setBusy] = useState(null);
  const [result, setResult] = useState(null); // {kind, data} or {kind: "error", message}
  if (!hardware) return html`<p class="muted">No floor attached (started with --no-hardware).</p>`;
  const run = async (kind, method, path) => {
    setBusy(kind);
    try {
      const response = await fetch(path, { method });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        setResult({ kind: "error", message: body.detail ?? `HTTP ${response.status}` });
      } else {
        setResult({ kind, data: response.status === 204 ? null : await response.json(), at: new Date() });
      }
    } catch (exc) {
      setResult({ kind: "error", message: exc.message });
    }
    setBusy(null);
  };
  return html`
    <div class="floor-admin">
      <div class="floor-buttons">
        <button disabled=${busy} onClick=${() => run("status", "GET", "api/floor/status")}>${busy === "status" ? "Asking…" : "Row status"}</button>
        <button disabled=${busy} onClick=${() => run("version", "GET", "api/floor/version")}>${busy === "version" ? "Asking…" : "Firmware versions"}</button>
        <button
          disabled=${busy} class="danger"
          title="Clears every tile's pixels and effect. It does not mute: while the floor is playing, the next frame lights it again."
          onClick=${() => run("blackout", "POST", "api/floor/blackout")}
        >Send BLACKOUT</button>
      </div>
      <p class="muted">Admin requests on the Row Bus, sent between frames, one row a frame.</p>
      ${result?.kind === "error" && html`<div class="command-error" role="alert">${result.message}</div>`}
      ${result?.kind === "status" && html`<${StatusTable} rows=${result.data.rows} />`}
      ${result?.kind === "version" && html`<${VersionTable} report=${result.data} />`}
      ${result?.kind === "blackout" && html`<p class="muted">BLACKOUT sent at ${result.at.toLocaleTimeString()}.</p>`}
    </div>`;
}

// ---- the tab ---------------------------------------------------------------------

export function DiagnosticsPanel() {
  const state = runnerState.value;
  if (!state) return html`<p class="muted">Waiting for the runner…</p>`;
  const { timing } = state;
  const budget = 1000 / timing.fps;
  const hardware = state.sinks.hardware?.attached ? state.sinks.hardware : null;
  return html`
    <div class="diagnostics">
      <${Section} title="Frame timing" aside=${`ms · budget ${budget.toFixed(1)} ms · last ${timing.window} frames`}>
        <${Timing} timing=${timing} budget=${budget} />
        <${Counters} timing=${timing} />
      <//>
      <${Section} title="Row Bus"><${RowBus} stats=${hardware?.encode_stats} budget=${budget} /><//>
      <${Section} title="Sinks"><${Sinks} sinks=${state.sinks} /><//>
      <${Section} title="Problems"><${Problems} state=${state} /><//>
      <${Section} title="The floor"><${Floor} hardware=${hardware} /><//>
    </div>`;
}
