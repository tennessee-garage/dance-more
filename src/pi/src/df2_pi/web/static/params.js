// Controls generated from an animation's Param specs (GET /api/animations),
// so declaring a parameter is all an author does to get a UI for it.
//
//   <ParamControls specs values errors onChange />
//
// `onChange(name, value)` fires as the user moves a control - continuously
// for a slider, on commit (blur / Enter) for typed input. What happens then
// is the caller's: the transport bar streams it to the running animation;
// the playlist editor (#90) stores it as an override.

import { html } from "htm/preact";
import { useEffect, useRef, useState } from "preact/hooks";

/** Which control a spec gets. Plain, so the mapping reads on its own:
 *
 *    choices                      -> select
 *    bool                         -> switch
 *    int / float with min and max -> slider (int steps by 1)
 *    int / float otherwise        -> number
 *    str                          -> text
 */
export function controlFor(spec) {
  if (spec.choices != null) return "select";
  if (spec.type === "bool") return "switch";
  if (spec.type === "int" || spec.type === "float") {
    return spec.min != null && spec.max != null ? "slider" : "number";
  }
  return "text";
}

/** A float slider's step: about a hundredth of its range, rounded to 1, 2 or 5. */
export function sliderStep(spec) {
  if (spec.type === "int") return 1;
  const raw = (spec.max - spec.min) / 100;
  const power = 10 ** Math.floor(Math.log10(raw));
  const nice = [1, 2, 5, 10].find((m) => m * power >= raw);
  return nice * power;
}

/** Where `v` sits along a bounded numeric spec's range, 0..1, and back -
 *  Param.to_unit() / from_unit() for sliders, so a slider on a "log" param
 *  moves the way a knob mapped to it does. */
export function toUnit(spec, v) {
  const clamped = Math.min(spec.max, Math.max(spec.min, v));
  if (spec.max === spec.min) return 0;
  if (spec.curve === "log") return Math.log(clamped / spec.min) / Math.log(spec.max / spec.min);
  return (clamped - spec.min) / (spec.max - spec.min);
}

export function fromUnit(spec, u) {
  const v = spec.curve === "log"
    ? spec.min * (spec.max / spec.min) ** u
    : spec.min + u * (spec.max - spec.min);
  if (spec.type === "int") return Math.round(v);
  return Math.min(spec.max, Math.max(spec.min, Number(v.toPrecision(3))));
}

function decimals(step) {
  return step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step)));
}

function same(a, b) {
  return typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-9 : a === b;
}

const SETTLE_MS = 2000;

/** The value a control shows. While the user is editing it is their own;
 *  after they let go it stays theirs until `value` agrees or SETTLE_MS
 *  pass - so a poll arriving mid-gesture, or with the old value just after,
 *  never yanks the control. Returns [shown, edit(v), release()]. */
export function useDraft(value) {
  const [draft, setDraft] = useState(null); // {v} while holding; null otherwise
  const released = useRef(false);
  const timer = useRef(null);
  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (draft !== null && released.current && same(draft.v, value)) {
      clearTimeout(timer.current);
      setDraft(null);
    }
  }, [value, draft]);
  const edit = (v) => {
    released.current = false;
    clearTimeout(timer.current);
    setDraft({ v });
  };
  const release = () => {
    released.current = true;
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setDraft(null), SETTLE_MS);
  };
  return [draft !== null ? draft.v : value, edit, release];
}

// ---- one component per control type ---------------------------------------
// Each gets the shown value and the draft's edit/release from ParamControl,
// which owns them so that a reset moves every kind of control at once.

function Slider({ name, spec, value, edit, release, onChange }) {
  const step = sliderStep(spec);
  const log = spec.curve === "log";
  const parse = (text) => {
    if (log) return fromUnit(spec, parseFloat(text));
    return spec.type === "int" ? parseInt(text, 10) : parseFloat(text);
  };
  const defaultAt = toUnit(spec, spec.default) * 100;
  // A log slider runs over 0..1 and maps through the curve; a linear one is the range itself.
  const range = log
    ? { min: 0, max: 1, step: 0.001, value: toUnit(spec, value) }
    : { min: spec.min, max: spec.max, step, value };
  return html`
    <div class="slider">
      <div class="track">
        <input
          type="range" min=${range.min} max=${range.max} step=${range.step} value=${range.value}
          aria-label=${spec.label ?? name}
          onInput=${(e) => { const v = parse(e.currentTarget.value); edit(v); onChange(name, v); }}
          onChange=${release}
        />
        <span class="default-mark" style=${{ left: `${defaultAt}%` }} title=${`default ${spec.default}`}></span>
      </div>
      <span class="num param-value">${log ? String(Number(Number(value).toPrecision(3))) : Number(value).toFixed(decimals(step))}</span>
    </div>`;
}

function NumberInput({ name, spec, value, edit, release, onChange }) {
  const parse = (text) => (spec.type === "int" ? parseInt(text, 10) : parseFloat(text));
  return html`
    <input
      type="number" step=${spec.type === "int" ? 1 : "any"} value=${value}
      aria-label=${spec.label ?? name}
      onInput=${(e) => edit(e.currentTarget.value)}
      onChange=${(e) => {
        const v = parse(e.currentTarget.value);
        if (Number.isNaN(v)) return;
        edit(v);
        release();
        onChange(name, v);
      }}
    />`;
}

function Select({ name, spec, value, edit, release, onChange }) {
  return html`
    <select
      value=${String(value)}
      aria-label=${spec.label ?? name}
      onChange=${(e) => {
        const v = spec.choices.find((c) => String(c) === e.currentTarget.value);
        edit(v);
        release();
        onChange(name, v);
      }}
    >
      ${spec.choices.map((c) => html`<option value=${String(c)}>${String(c)}</option>`)}
    </select>`;
}

function Switch({ name, spec, value, edit, release, onChange }) {
  return html`
    <label class="switch">
      <input
        type="checkbox" role="switch" checked=${!!value}
        aria-label=${spec.label ?? name}
        onChange=${(e) => { const v = e.currentTarget.checked; edit(v); release(); onChange(name, v); }}
      />
      <span class="switch-track" aria-hidden="true"></span>
    </label>`;
}

function TextInput({ name, spec, value, edit, release, onChange }) {
  return html`
    <input
      type="text" value=${value}
      aria-label=${spec.label ?? name}
      onInput=${(e) => edit(e.currentTarget.value)}
      onChange=${(e) => { release(); onChange(name, e.currentTarget.value); }}
    />`;
}

const CONTROLS = { slider: Slider, number: NumberInput, select: Select, switch: Switch, text: TextInput };

function ParamControl({ name, spec, value, error, onChange }) {
  const [shown, edit, release] = useDraft(value);
  const Control = CONTROLS[controlFor(spec)];
  const reset = () => {
    edit(spec.default);
    release();
    onChange(name, spec.default);
  };
  return html`
    <div class=${error ? "param has-error" : "param"} title=${spec.help ?? ""}>
      <div class="param-head">
        <span class="label">${spec.label ?? name}</span>
        ${spec.macro != null && html`<span class="macro" title=${`External controls reach this as macro ${spec.macro}`}>M${spec.macro}</span>`}
        <button
          class="reset" disabled=${same(shown, spec.default)} onClick=${reset}
          title=${`Reset to ${spec.default}`} aria-label=${`Reset ${spec.label ?? name}`}
        >↺</button>
      </div>
      <${Control} name=${name} spec=${spec} value=${shown} edit=${edit} release=${release} onChange=${onChange} />
      ${error && html`<div class="param-error" role="alert">${error}</div>`}
    </div>`;
}

/** One control per spec, in declaration order. `values` fills each (the
 *  default where a name is missing); `errors` is {name: message}. */
export function ParamControls({ specs, values = {}, errors = {}, onChange }) {
  return html`
    <div class="param-controls">
      ${Object.entries(specs).map(([name, spec]) => html`
        <${ParamControl}
          key=${name} name=${name} spec=${spec}
          value=${values[name] ?? spec.default}
          error=${errors[name]}
          onChange=${onChange}
        />`)}
    </div>`;
}

/** A number input that keeps its draft while edited and commits on change. */
export function NumberField({ label, value, min, max, step = 1, onCommit, disabled }) {
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
