// Show controls: speed, strobe, colour, freeze and bump, acting on whatever
// is playing (engine/overlays.py). Sliders stream while dragged; nothing here
// is stored - a restart comes back with every control at rest.

import { html } from "htm/preact";
import { useDraft } from "./params.js";
import { command, streamCommand } from "./state.js";

/** A slider that streams `send(value)` while dragged and shows the server's
 *  value otherwise. Values are strings, as the DOM's are, so a poll never
 *  rewrites the thumb under a drag. */
function ShowSlider({ label, value, min, max, step, format, send, live, title }) {
  const [shown, edit, release] = useDraft(value);
  return html`
    <div class="param" title=${title ?? ""}>
      <div class="param-head"><span class="label">${label}</span></div>
      <div class="slider">
        <div class="track">
          <input
            type="range" min=${min} max=${max} step=${step} value=${String(shown)}
            disabled=${!live}
            aria-label=${label}
            onInput=${(e) => { const v = e.currentTarget.valueAsNumber; edit(v); send(v); }}
            onChange=${release}
          />
        </div>
        <span class="num param-value">${format(shown)}</span>
      </div>
    </div>`;
}

const hex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const rgbOf = (text) => [1, 3, 5].map((i) => parseInt(text.slice(i, i + 2), 16));

function TintControl({ show, live }) {
  const [amount, edit, release] = useDraft(show.tint_amount);
  const [colour, editColour, releaseColour] = useDraft(hex(show.tint));
  const send = (rgbText, a) => {
    const [r, g, b] = rgbOf(rgbText);
    streamCommand("tint", { r, g, b, amount: a });
  };
  return html`
    <div class="param" title="Colourise toward a colour; black stays black">
      <div class="param-head"><span class="label">Tint</span></div>
      <div class="slider">
        <input
          type="color" class="tint-colour" value=${colour} disabled=${!live} aria-label="Tint colour"
          onInput=${(e) => { const c = e.currentTarget.value; editColour(c); send(c, amount); }}
          onChange=${releaseColour}
        />
        <div class="track">
          <input
            type="range" min="0" max="1" step="0.01" value=${String(amount)} disabled=${!live}
            aria-label="Tint amount"
            onInput=${(e) => { const a = e.currentTarget.valueAsNumber; edit(a); send(colour, a); }}
            onChange=${release}
          />
        </div>
        <span class="num param-value">${Math.round(amount * 100)}%</span>
      </div>
    </div>`;
}

// On pointer-down, for a hit that lands with the finger; a keyboard press
// arrives as a click with no pointer detail.
const bump = () => command("bump", { level: 1, decay_s: 0.3 });

export function ShowControls({ state, live }) {
  const show = state?.show;
  if (!show) return null;
  const frozen = show.frozen;
  return html`
    <div class="live-params show-controls" aria-label="Show controls">
      <span class="row-label" title="These act on whatever is playing">Show</span>
      <div class="param-controls">
        <${ShowSlider}
          label="Speed" value=${show.speed} min="0" max="4" step="0.05" live=${live}
          format=${(v) => `${v.toFixed(2)}×`} send=${(v) => streamCommand("speed", { value: v })}
          title="How fast animations run; 1× is as written"
        />
        <${ShowSlider}
          label="Strobe" value=${show.strobe_hz} min="0" max=${show.strobe_max_hz} step="0.5" live=${live}
          format=${(v) => (v > 0 ? `${v} Hz` : "off")} send=${(v) => streamCommand("strobe", { rate_hz: v })}
          title=${`Shutter the picture; capped at ${show.strobe_max_hz} Hz in settings`}
        />
        <${ShowSlider}
          label="Hue shift" value=${show.hue_shift} min="0" max="1" step="0.005" live=${live}
          format=${(v) => `${Math.round(v * 360)}°`} send=${(v) => streamCommand("hue_shift", { value: v })}
        />
        <${ShowSlider}
          label="Saturation" value=${show.saturation} min="0" max="2" step="0.05" live=${live}
          format=${(v) => `${Math.round(v * 100)}%`} send=${(v) => streamCommand("saturation", { value: v })}
        />
        <${TintControl} show=${show} live=${live} />
        <div class="show-buttons">
          <button
            class=${frozen ? "active" : ""} aria-pressed=${frozen} disabled=${!live}
            onClick=${() => command("freeze", { on: !frozen })}
            title="Hold the picture; animations keep running underneath"
          >${frozen ? "Frozen" : "Freeze"}</button>
          <button
            disabled=${!live}
            onPointerDown=${bump}
            onClick=${(e) => { if (e.detail === 0) bump(); }}
            title="A flash of white, fading over 0.3 s - full-white current for an instant"
          >Bump</button>
          <button disabled=${!live} onClick=${() => command("reset_show")} title="Every show control back to rest">Reset</button>
        </div>
      </div>
    </div>`;
}
