// The Playlists tab: the playlists on one side, the selected one's entries
// on the other (stacked when the panel is narrow). Every write answers with
// the whole playlist, which replaces what is shown.
//
// Editing the playlist the runner has loaded changes the database only; the
// runner keeps its own copy. When a write lands on it the tab offers to load
// it again.

import { html } from "htm/preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { computed, signal } from "@preact/signals";
import { ParamControls } from "./params.js";
import { animationList, command, runnerState } from "./state.js";

const PARAMS_SAVE_MS = 400; // a param edit is saved once the control has been still this long
// Durations offered for an animation with a `period`: the whole multiple of
// it nearest each of these, so an entry ends on a loop boundary.
const DURATION_TARGETS_S = [15, 30, 45, 60, 90, 120, 180, 240, 300, 600];

const summaries = signal(null); // GET /api/playlists
const selectedId = signal(null);
const selected = signal(null); // GET /api/playlists/{id}
const settings = signal(null);
const problem = signal(null); // why the last request failed
const reloadOffered = signal(false); // an edit landed on the loaded playlist
const loadedId = computed(() => runnerState.value?.playlist?.[0] ?? null);
// The entry the runner is on, by id: its position in the runner's loaded copy
// stops matching this list once the playlist is reordered here.
const playingEntryId = computed(() => runnerState.value?.entry_id ?? null);
const oneOff = computed(() => !!runnerState.value?.one_off);

// ---- requests ----------------------------------------------------------------

async function api(method, path, body) {
  try {
    const response = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (response.status === 204) return {};
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = data?.detail;
      problem.value = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : `HTTP ${response.status}`;
      return null;
    }
    problem.value = null;
    return data;
  } catch (exc) {
    problem.value = exc.message;
    return null;
  }
}

async function refreshList() {
  const data = await api("GET", "api/playlists");
  if (data) summaries.value = data;
}

async function select(id) {
  selectedId.value = id;
  reloadOffered.value = false;
  if (id == null) {
    selected.value = null;
    return;
  }
  const data = await api("GET", `api/playlists/${id}`);
  if (data) selected.value = data;
}

/** A write answered with the playlist as it now stands. */
function landed(detail) {
  if (!detail) return;
  selected.value = detail;
  if (detail.loaded) reloadOffered.value = true;
  refreshList();
}

async function loadSettings() {
  const data = await api("GET", "api/settings");
  if (data) settings.value = data;
}

// ---- helpers -----------------------------------------------------------------

function clock(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function specsFor(animationId) {
  return animationList.value?.animations.find((a) => a.id === animationId)?.params ?? null;
}

function periodMultiples(period) {
  if (!period) return [];
  const out = new Set(DURATION_TARGETS_S.map((t) => Math.max(1, Math.round(t / period)) * period));
  return [...out].sort((a, b) => a - b);
}

function paramSummary(overrides) {
  const pairs = Object.entries(overrides);
  return pairs.length === 0 ? "defaults" : pairs.map(([k, v]) => `${k}=${v}`).join(", ");
}

/** A duration field: commits on change; offers the period's multiples. */
function DurationInput({ value, period, onCommit, listId }) {
  const multiples = periodMultiples(period);
  return html`
    <span class="duration">
      <input
        type="number" min="0.1" step="any" value=${value} list=${multiples.length ? listId : undefined}
        aria-label="Duration in seconds"
        onChange=${(e) => { const v = parseFloat(e.currentTarget.value); if (v > 0) onCommit(v); }}
      />
      <span class="unit">s</span>
      ${multiples.length > 0 && html`
        <datalist id=${listId}>${multiples.map((m) => html`<option value=${m}>${m / period} × ${period} s</option>`)}</datalist>`}
    </span>`;
}

function Switch({ checked, label, onChange }) {
  return html`
    <label class="switch" title=${label}>
      <input type="checkbox" role="switch" checked=${checked} aria-label=${label} onChange=${(e) => onChange(e.currentTarget.checked)} />
      <span class="switch-track" aria-hidden="true"></span>
    </label>`;
}

// ---- the list ----------------------------------------------------------------

function NewPlaylist() {
  const [name, setName] = useState("");
  const create = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    const created = await api("POST", "api/playlists", { name: name.trim() });
    if (created) {
      setName("");
      await refreshList();
      select(created.id);
    }
  };
  return html`
    <form class="new-playlist" onSubmit=${create}>
      <input type="text" placeholder="New playlist" value=${name} onInput=${(e) => setName(e.currentTarget.value)} aria-label="New playlist name" />
      <button type="submit" disabled=${!name.trim()}>Add</button>
    </form>`;
}

function PlaylistList() {
  const list = summaries.value;
  return html`
    <div class="playlist-list">
      <${NewPlaylist} />
      ${list == null ? html`<p class="muted">Loading…</p>` : html`
        <ul>
          ${list.map((p) => html`
            <li key=${p.id}>
              <button
                class=${p.id === selectedId.value ? "playlist-item selected" : "playlist-item"}
                onClick=${() => select(p.id)}
              >
                <span class="playlist-name">
                  ${p.name}
                  ${p.startup && html`<span class="marker" title="Loaded when the floor starts">★</span>`}
                  ${p.id === loadedId.value && html`<span class="marker playing" title="Loaded in the runner now">●</span>`}
                </span>
                <span class="playlist-meta num">
                  ${p.entry_count} · ${clock(p.total_duration_s)}${p.loop ? " · loop" : ""}${p.shuffle ? " · shuffle" : ""}
                </span>
              </button>
            </li>`)}
        </ul>`}
    </div>`;
}

// ---- one playlist --------------------------------------------------------------

function PlaylistHeader({ playlist }) {
  const patch = async (fields) => landed(await api("PATCH", `api/playlists/${playlist.id}`, fields));
  const remove = async () => {
    if (!confirm(`Delete "${playlist.name}" and its ${playlist.entry_count} entries?`)) return;
    if (await api("DELETE", `api/playlists/${playlist.id}`)) {
      await select(null);
      refreshList();
    }
  };
  const startup = async () => landed(await api("POST", `api/playlists/${playlist.id}/startup`));
  const load = async () => {
    if (await command("load", { playlist: playlist.id })) reloadOffered.value = false;
  };
  return html`
    <div class="playlist-header">
      <input
        class="playlist-title" type="text" value=${playlist.name} aria-label="Playlist name"
        onChange=${(e) => { const v = e.currentTarget.value.trim(); if (v && v !== playlist.name) patch({ name: v }); }}
      />
      <div class="playlist-settings">
        <span class="setting"><${Switch} checked=${playlist.loop} label="Loop" onChange=${(v) => patch({ loop: v })} /> Loop</span>
        <span class="setting"><${Switch} checked=${playlist.shuffle} label="Shuffle" onChange=${(v) => patch({ shuffle: v })} /> Shuffle</span>
        <span class="setting">
          Crossfade
          <input
            type="number" min="0" step="0.5" value=${playlist.crossfade_s} aria-label="Crossfade in seconds"
            onChange=${(e) => { const v = parseFloat(e.currentTarget.value); if (v >= 0) patch({ crossfade_s: v }); }}
          /> s
        </span>
      </div>
      <div class="playlist-actions">
        <button onClick=${load} title="Play this playlist from its first entry">${playlist.loaded ? "Reload into runner" : "Load into runner"}</button>
        <button onClick=${startup} disabled=${playlist.startup} title="Load this playlist when the floor starts">
          ${playlist.startup ? "★ Startup" : "Set as startup"}
        </button>
        <button class="danger" onClick=${remove}>Delete</button>
      </div>
      ${reloadOffered.value && playlist.loaded && html`
        <div class="reload-offer" role="status">
          This playlist is playing; the runner keeps the version it loaded until you <button class="link" onClick=${load}>reload it</button>.
        </div>`}
    </div>`;
}

/** An entry's params, edited with the animation's own controls and saved a
 *  moment after the last change (not on every step of a drag). */
function EntryParams({ playlistId, entry }) {
  const specs = specsFor(entry.animation_id);
  const [values, setValues] = useState(entry.resolved_params);
  const timer = useRef(null);
  useEffect(() => setValues(entry.resolved_params), [entry]);
  useEffect(() => () => clearTimeout(timer.current), []);
  if (!specs || Object.keys(specs).length === 0) return html`<p class="muted">No parameters.</p>`;
  const change = (name, value) => {
    const next = { ...values, [name]: value };
    setValues(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(
      async () => landed(await api("PATCH", `api/playlists/${playlistId}/entries/${entry.id}`, { params: next })),
      PARAMS_SAVE_MS,
    );
  };
  return html`<${ParamControls} specs=${specs} values=${values} onChange=${change} />`;
}

/** Elapsed and remaining on the playing row: the one thing here that
 *  changes at every poll, so the only thing re-rendered by it. */
function EntryProgress() {
  const state = runnerState.value;
  if (!state) return null;
  const remaining = state.remaining_s == null ? "" : ` \u00b7 ${clock(state.remaining_s)} left`;
  return html`<span class="entry-progress num">${clock(state.elapsed_s)}${remaining}</span>`;
}

function EntryRow({ playlistId, entry, index, drag, loaded }) {
  const [open, setOpen] = useState(false);
  const current = loaded && entry.id === playingEntryId.value;
  const playing = current && !oneOff.value;
  const resumes = current && oneOff.value;
  const patch = async (fields) => landed(await api("PATCH", `api/playlists/${playlistId}/entries/${entry.id}`, fields));
  const remove = async () => landed(await api("DELETE", `api/playlists/${playlistId}/entries/${entry.id}`));
  const row = useRef(null);
  const classes = ["entry"];
  if (playing) classes.push("playing");
  if (resumes) classes.push("resumes");
  if (!entry.enabled) classes.push("disabled");
  if (drag.dropIndex === index) classes.push("drop-before");
  if (drag.dropIndex === index + 1 && drag.isLast(index)) classes.push("drop-after");
  if (drag.draggingId === entry.id) classes.push("dragging");
  return html`
    <li
      ref=${row} class=${classes.join(" ")}
      onDragOver=${(e) => drag.over(e, index)}
      onDrop=${drag.drop}
    >
      <div class="entry-line">
        <span
          class="handle" draggable="true" title="Drag to reorder" aria-label=${`Move entry ${index + 1}`}
          onDragStart=${(e) => drag.start(e, entry.id, row.current)} onDragEnd=${drag.end}
        >⠇</span>
        <span class="entry-position num">${index + 1}</span>
        ${playing && html`<span class="now-playing" title="Playing now" aria-label="Playing now">\u25b6</span>`}
        ${resumes && html`<span class="now-playing resumes" title="A one-off is playing; the playlist resumes here" aria-label="Resumes here">\u21a9</span>`}
        <span class="entry-name">
          ${entry.unresolved
            ? html`<span class="unresolved" title=${entry.error}>${entry.animation_id}</span>`
            : entry.animation.name}
          ${entry.warnings.length > 0 && html`<span class="marker warn" title=${entry.warnings.join("\n")}>!</span>`}
        </span>
        <${DurationInput}
          value=${entry.duration_s} period=${entry.animation?.period}
          listId=${`periods-${entry.id}`} onCommit=${(v) => patch({ duration_s: v })}
        />
        <${Switch} checked=${entry.enabled} label="Enabled" onChange=${(v) => patch({ enabled: v })} />
        <button class="icon-small danger" onClick=${remove} title="Remove entry" aria-label=${`Remove entry ${index + 1}`}>×</button>
      </div>
      ${playing && html`<div class="entry-sub"><${EntryProgress} /></div>`}
      ${!entry.unresolved && html`
        <button class="link params-summary" onClick=${() => setOpen(!open)} aria-expanded=${open}>
          ${paramSummary(entry.params)}
        </button>`}
      ${open && html`<div class="entry-params"><${EntryParams} playlistId=${playlistId} entry=${entry} /></div>`}
    </li>`;
}

/** Native HTML5 drag and drop: the handle drags the row, and dropping posts
 *  `move` with the row's new position. */
function useReorder(playlist) {
  const [draggingId, setDraggingId] = useState(null);
  const [dropIndex, setDropIndex] = useState(null);
  const count = playlist.entries.length;
  return {
    draggingId,
    dropIndex,
    isLast: (index) => index === count - 1,
    start(e, id, rowElement) {
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", String(id));
      if (rowElement) e.dataTransfer.setDragImage(rowElement, 16, 16);
      setDraggingId(id);
    },
    over(e, index) {
      if (draggingId == null) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      const box = e.currentTarget.getBoundingClientRect();
      setDropIndex(e.clientY > box.top + box.height / 2 ? index + 1 : index);
    },
    async drop(e) {
      e.preventDefault();
      const from = playlist.entries.findIndex((x) => x.id === draggingId);
      const to = dropIndex;
      setDraggingId(null);
      setDropIndex(null);
      if (from < 0 || to == null) return;
      const position = to > from ? to - 1 : to; // the dragged row leaves its own slot first
      if (position === from) return;
      landed(await api("POST", `api/playlists/${playlist.id}/entries/${draggingId}/move`, { position }));
    },
    end() {
      setDraggingId(null);
      setDropIndex(null);
    },
  };
}

function AddEntry({ playlist }) {
  const animations = animationList.value?.animations ?? [];
  const [animationId, setAnimationId] = useState("");
  const [duration, setDuration] = useState(null);
  const [values, setValues] = useState({});
  const animation = animations.find((a) => a.id === animationId);
  const choose = (id) => {
    setAnimationId(id);
    setValues({});
  };
  const add = async () => {
    const body = { animation_id: animationId, params: values };
    if (duration) body.duration_s = duration;
    const detail = await api("POST", `api/playlists/${playlist.id}/entries`, body);
    if (detail) {
      landed(detail);
      setValues({});
      setDuration(null);
    }
  };
  return html`
    <div class="add-entry">
      <div class="entry-line">
        <select value=${animationId} onChange=${(e) => choose(e.currentTarget.value)} aria-label="Animation to add">
          <option value="">Add an animation…</option>
          ${animations.map((a) => html`<option value=${a.id}>${a.name}</option>`)}
        </select>
        <${DurationInput}
          value=${duration ?? settings.value?.default_entry_duration ?? ""} period=${animation?.period}
          listId="periods-new" onCommit=${setDuration}
        />
        <button onClick=${add} disabled=${!animationId}>Add</button>
      </div>
      ${animation && Object.keys(animation.params).length > 0 && html`
        <div class="entry-params">
          <${ParamControls} specs=${animation.params} values=${values} onChange=${(name, v) => setValues({ ...values, [name]: v })} />
        </div>`}
    </div>`;
}

function PlaylistView({ playlist }) {
  const drag = useReorder(playlist);
  return html`
    <div class="playlist-view">
      <${PlaylistHeader} playlist=${playlist} />
      ${playlist.entries.length === 0
        ? html`<p class="muted">No entries yet.</p>`
        : html`
          <ol class="entries" onDragLeave=${(e) => { if (!e.currentTarget.contains(e.relatedTarget)) drag.end(); }}>
            ${playlist.entries.map((entry, index) => html`
              <${EntryRow}
                key=${entry.id} playlistId=${playlist.id} entry=${entry} index=${index} drag=${drag}
                loaded=${playlist.id === loadedId.value}
              />`)}
          </ol>`}
      <${AddEntry} playlist=${playlist} />
    </div>`;
}

// ---- the tab -------------------------------------------------------------------

export function PlaylistsPanel() {
  useEffect(() => {
    refreshList();
    loadSettings();
  }, []);
  // The runner loading a playlist (here or anywhere) moves the "playing" marker.
  const loaded = loadedId.value;
  useEffect(() => {
    if (summaries.value) refreshList();
    // Nothing chosen yet: show what is playing, which is usually why the tab was opened.
    if (selectedId.value == null && loaded != null) select(loaded);
    else if (selectedId.value != null) select(selectedId.value);
  }, [loaded]);

  const playlist = selected.value;
  return html`
    <div class="playlists">
      ${problem.value && html`<div class="command-error" role="alert">${problem.value}</div>`}
      <${PlaylistList} />
      ${playlist
        ? html`<${PlaylistView} key=${playlist.id} playlist=${playlist} />`
        : html`<p class="muted playlist-view">Choose a playlist, or add one.</p>`}
    </div>`;
}
