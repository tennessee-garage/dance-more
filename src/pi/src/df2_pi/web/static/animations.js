// The Animations tab: what the registry loaded, what failed and why, a
// one-off Play and a Layer toggle per animation, and a Reload button.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import {
  animationList,
  animationsError,
  command,
  currentAnimationId,
  lastReload,
  layerAnimationId,
  loadAnimations,
  reloadAnimations,
} from "./state.js";

// A one-off plays this long, then the runner returns to the playlist.
export const PREVIEW_HOLD_S = 30;

function LoadError({ error }) {
  return html`
    <div class="load-error">
      <span class="stage">${error.stage}</span>
      <span class="message">${error.message}</span>
    </div>`;
}

function AnimationRow({ animation, playing, layered }) {
  const period = animation.period != null ? `${animation.period} s` : null;
  return html`
    <li class=${playing ? "animation playing" : "animation"}>
      <div class="animation-main">
        <div class="animation-title">
          <span class="animation-name">${animation.name}</span>
          <span class=${`format format-${animation.format}`}>${animation.format}</span>
          ${period && html`<span class="period num" title="Period">${period}</span>`}
        </div>
        ${animation.tags.length > 0 && html`
          <div class="tags">${animation.tags.map((tag) => html`<span class="chip">${tag}</span>`)}</div>`}
        ${animation.description && html`<div class="description">${animation.description}</div>`}
        ${animation.error && html`
          <div class="stale">The last edit failed to load; this is the previous version.</div>
          <${LoadError} error=${animation.error} />`}
      </div>
      <button
        class="play"
        onClick=${() => command("animation", { id: animation.id, hold: PREVIEW_HOLD_S })}
        title=${`Play ${animation.name} for ${PREVIEW_HOLD_S} s, then back to the playlist`}
      >${playing ? "Playing" : "Play"}</button>
      <button
        class=${layered ? "layer active" : "layer"}
        aria-pressed=${layered}
        onClick=${() => (layered ? command("clear_layer") : command("layer", { id: animation.id }))}
        title=${layered ? "Remove this layer" : `Run ${animation.name} as a layer over whatever plays`}
      >${layered ? "Layered" : "Layer"}</button>
    </li>`;
}

function ReloadStatus() {
  const result = lastReload.value;
  if (result == null) return null;
  const n = result.changed.length;
  return html`<span class="reload-status">${n === 0 ? "No changes" : `${n} changed: ${result.changed.join(", ")}`}</span>`;
}

export function AnimationsPanel() {
  const [reloading, setReloading] = useState(false);
  useEffect(() => { loadAnimations(); }, []);

  const list = animationList.value;
  const current = currentAnimationId.value;
  const layered = layerAnimationId.value;
  const error = animationsError.value;
  // Files that failed and have no last good version to fall back on.
  const failedOnly = list ? Object.entries(list.errors).filter(([id]) => !list.animations.some((a) => a.id === id)) : [];

  const reload = async () => {
    setReloading(true);
    await reloadAnimations();
    setReloading(false);
  };

  return html`
    <div class="animations">
      <div class="panel-header">
        <span class="count">${list ? `${list.animations.length} animations` : "Loading…"}</span>
        <${ReloadStatus} />
        <button class="reload" onClick=${reload} disabled=${reloading}>${reloading ? "Reloading…" : "Reload"}</button>
      </div>
      ${error && html`<div class="command-error" role="alert">${error}</div>`}
      ${failedOnly.length > 0 && html`
        <ul class="failed">
          ${failedOnly.map(([id, e]) => html`
            <li class="failed-file">
              <div class="animation-title"><span class="animation-name">${e.path}</span></div>
              <${LoadError} error=${e} />
            </li>`)}
        </ul>`}
      ${list && html`
        <ul class="animation-list">
          ${list.animations.map((a) => html`<${AnimationRow} key=${a.id} animation=${a} playing=${a.id === current} layered=${a.id === layered} />`)}
        </ul>`}
    </div>`;
}
