// The preview stream: one WebSocket to /ws/preview, binary records in the
// wire format output/preview.py defines, handed straight to a callback.
// Frames never touch signals or component state - at 30 fps that would
// diff the page on every frame. Only a once-a-second status goes to a signal.
//
// Latest wins here too, and the server only sends what has been asked for.
// Only the newest record is kept and handed over once per animation frame,
// and each hand-over sends `{"ack": FRAME_NO}`, covering every record up to
// it. The server keeps at most two records unacknowledged, so a page or a
// link slower than the stream gets fewer, current frames - never a backlog
// sitting in socket buffers, seconds behind the floor.
//
// Starts on `full` at every frame. If FRAME_NO gaps show this browser is
// not keeping up for FALLBACK_AFTER_S seconds in a row, it asks for
// `tiles` at 10 fps instead: 192 bytes a frame, not 11,520. Closes while
// the tab is hidden and reconnects (with backoff) when it has to.

import { signal } from "@preact/signals";

const HEADER_BYTES = 7; // VER FORMAT FLAGS FRAME_NO(u32 BE)
const VERSION = 0x01;
const FORMATS = { 1: "full", 2: "tiles" };
const FLAG_TILE_SOURCE = 0x01;

const FALLBACK = { format: "tiles", fps: 10 };
const FALLBACK_AFTER_S = 3; // consecutive lagging seconds
const LAGGING_MISSED = 0.2; // a second is lagging when this share of frames never arrived
const RECONNECT_MIN_MS = 1000;
const RECONNECT_MAX_MS = 10000;

/** Once a second: {connection, format, fps, missed} - frames received and
 *  frames skipped (FRAME_NO gaps) over the last second. */
export const previewStatus = signal({ connection: "connecting", format: null, fps: 0, missed: 0 });

/** Parse one binary record. The payload is a view into `buffer`, chain
 *  order: `full` is led_count RGB triples, `tiles` one RGB per tile. */
export function parseRecord(buffer) {
  const view = new DataView(buffer);
  const version = view.getUint8(0);
  if (version !== VERSION) throw new Error(`unsupported preview version ${version}`);
  return {
    format: FORMATS[view.getUint8(1)] ?? "unknown",
    tileSource: (view.getUint8(2) & FLAG_TILE_SOURCE) !== 0,
    frameNo: view.getUint32(3, false),
    payload: new Uint8Array(buffer, HEADER_BYTES),
  };
}

function socketUrl(format, fps) {
  const url = new URL("ws/preview", document.baseURI);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("format", format);
  if (fps != null) url.searchParams.set("fps", String(fps));
  return url;
}

/** Stream records to `onRecord(record)` until `close()`. */
export function connectPreview(onRecord) {
  let socket = null;
  let settings = { format: "full", fps: null };
  let fellBack = false;
  let retryMs = RECONNECT_MIN_MS;
  let retryTimer = null;
  let closed = false;

  // newest record not yet handed over, and whether a hand-over is scheduled
  let latest = null;
  let scheduled = false;

  function deliver() {
    scheduled = false;
    const record = latest;
    latest = null;
    if (!record) return;
    onRecord(record);
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ ack: record.frameNo }));
  }

  // per-second counters
  let lastFrameNo = null;
  let received = 0;
  let missed = 0;
  let laggingSeconds = 0;

  function open() {
    if (closed || document.hidden || socket) return;
    const ws = new WebSocket(socketUrl(settings.format, settings.fps));
    ws.binaryType = "arraybuffer";
    socket = ws;
    lastFrameNo = null;
    latest = null; // from the previous connection: never acked on this one
    ws.onopen = () => { retryMs = RECONNECT_MIN_MS; };
    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        console.warn("preview:", event.data); // a refused control message
        return;
      }
      const record = parseRecord(event.data);
      // Gaps only mean lag while every frame was asked for.
      if (lastFrameNo !== null && settings.fps == null && record.frameNo > lastFrameNo + 1) {
        missed += record.frameNo - lastFrameNo - 1;
      }
      lastFrameNo = record.frameNo;
      received += 1;
      latest = record;
      if (!scheduled) {
        scheduled = true;
        requestAnimationFrame(deliver);
      }
    };
    ws.onclose = () => {
      if (socket === ws) socket = null;
      if (closed || document.hidden) return;
      retryTimer = setTimeout(open, retryMs);
      retryMs = Math.min(RECONNECT_MAX_MS, retryMs * 2);
    };
  }

  function drop() {
    clearTimeout(retryTimer);
    if (socket) {
      const ws = socket;
      socket = null;
      ws.close();
    }
  }

  function tick() {
    const lagging = received + missed > 0 && missed / (received + missed) > LAGGING_MISSED;
    laggingSeconds = lagging ? laggingSeconds + 1 : 0;
    if (!fellBack && laggingSeconds >= FALLBACK_AFTER_S && socket?.readyState === WebSocket.OPEN) {
      fellBack = true;
      settings = { ...FALLBACK };
      socket.send(JSON.stringify(FALLBACK));
      console.info(`preview: missed ${missed} of ${received + missed} frames; falling back to tiles at ${FALLBACK.fps} fps`);
    }
    const connection = socket?.readyState === WebSocket.OPEN ? "live" : document.hidden ? "paused" : "connecting";
    previewStatus.value = { connection, format: settings.format, fps: received, missed };
    received = 0;
    missed = 0;
  }

  function onVisibility() {
    if (document.hidden) drop();
    else open();
  }

  const ticker = setInterval(tick, 1000);
  document.addEventListener("visibilitychange", onVisibility);
  open();

  return {
    close() {
      closed = true;
      clearInterval(ticker);
      document.removeEventListener("visibilitychange", onVisibility);
      drop();
    },
  };
}
