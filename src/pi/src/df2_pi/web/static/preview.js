// The floor, drawn: every LED a point of light with one fixed bloom,
// summed additively on black; strictly top-down, row 0 along the bottom.
// Plain JS, no Preact: the stream hands records to draw() directly.
//
//   const preview = createPreview(canvas, geometry)   // throws without WebGL
//   preview.draw(record)   // a parsed preview record (preview-stream.js)
//   preview.resize()       // after the canvas's container changed size
//
// The look is render_bloom's in output/dev.py, evaluated per pixel rather
// than blurred: a core a third of a cell wide at the LED's full colour,
// plus a gaussian glow of sigma 1.6 cells peaking at 1.4/pi of it. The same
// radius and falloff for every LED; no controls.
//
// WebGL only: one draw call of led_count points, each shaded by that
// profile, blended ONE+ONE - about 0.1 ms a frame. A 2D canvas cannot keep
// up (sprites per LED channel with "lighter" compositing took 206 ms for
// an all-lit frame on a laptop, blocking the page's controls while it
// drew), so there is no 2D fallback.

const SIGMA_CELLS = 1.6; // render_bloom's `radius`
const GLOW_PEAK = 1.4 / Math.PI; // its `gain`, as the peak of the blurred point
const CORE_HALF_CELLS = 1 / 6; // its core: a third of a cell across
const REACH_SIGMAS = 3; // the glow is drawn out to here (0.5% of peak)

/** Per-LED canvas position in cells, display-oriented: canonical y=0 (next
 *  to the Pi) at the bottom. Returns [x, y] pairs, y down, chain order. */
function ledCentres(geometry) {
  const { led_count: n, led_to_cell: cells, height } = geometry;
  const centres = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const y = cells[2 * i];
    const x = cells[2 * i + 1];
    centres[2 * i] = x + 0.5;
    centres[2 * i + 1] = height - 1 - y + 0.5; // the display flip
  }
  return centres;
}

/** A record's colours as one RGB triple per LED, chain order. A `tiles`
 *  record paints each tile's LEDs with the tile's colour. */
function ledColours(record, geometry, out) {
  const { payload, format } = record;
  if (format === "full") {
    out.set(payload.subarray(0, out.length));
    return out;
  }
  const per = geometry.leds_per_tile;
  for (let tile = 0; tile < geometry.tiles; tile++) {
    const r = payload[3 * tile];
    const g = payload[3 * tile + 1];
    const b = payload[3 * tile + 2];
    for (let led = 0, o = 3 * tile * per; led < per; led++, o += 3) {
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
    }
  }
  return out;
}

// ---- WebGL -----------------------------------------------------------------

const VERTEX = `
attribute vec2 a_centre;   // cells, display-oriented, y down
attribute vec3 a_colour;   // 0..1
uniform vec2 u_floor;      // floor size in cells
uniform float u_point;     // point size in device pixels
varying vec3 v_colour;
void main() {
  vec2 unit = a_centre / u_floor;
  gl_Position = vec4(unit.x * 2.0 - 1.0, 1.0 - unit.y * 2.0, 0.0, 1.0);
  gl_PointSize = u_point;
  v_colour = a_colour;
}`;

const FRAGMENT = `
precision mediump float;
uniform float u_reach;     // cells from the centre to the point's edge
uniform float u_sigma;
uniform float u_peak;
uniform float u_core;
varying vec3 v_colour;
void main() {
  vec2 d = (gl_PointCoord - 0.5) * 2.0 * u_reach;   // cells from the LED
  float core = step(abs(d.x), u_core) * step(abs(d.y), u_core);
  float glow = u_peak * exp(-dot(d, d) / (2.0 * u_sigma * u_sigma));
  gl_FragColor = vec4(v_colour * (core + glow), 1.0);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}

function webglRenderer(canvas, geometry) {
  // preserveDrawingBuffer: so the canvas can be captured (a screenshot, or
  // the side-by-side against render_bloom).
  const gl = canvas.getContext("webgl", { alpha: false, antialias: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);

  const n = geometry.led_count;
  const at = (name) => gl.getAttribLocation(program, name);
  const uniform = (name) => gl.getUniformLocation(program, name);

  const centres = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, centres);
  gl.bufferData(gl.ARRAY_BUFFER, ledCentres(geometry), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(at("a_centre"));
  gl.vertexAttribPointer(at("a_centre"), 2, gl.FLOAT, false, 0, 0);

  const colourData = new Uint8Array(n * 3);
  const colours = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, colours);
  gl.bufferData(gl.ARRAY_BUFFER, colourData, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(at("a_colour"));
  gl.vertexAttribPointer(at("a_colour"), 3, gl.UNSIGNED_BYTE, true, 0, 0);

  gl.uniform2f(uniform("u_floor"), geometry.width, geometry.height);
  gl.uniform1f(uniform("u_sigma"), SIGMA_CELLS);
  gl.uniform1f(uniform("u_peak"), GLOW_PEAK);
  gl.uniform1f(uniform("u_core"), CORE_HALF_CELLS);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE); // additive; the framebuffer clips at 1
  const maxPoint = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];

  return {
    resize() {
      gl.viewport(0, 0, canvas.width, canvas.height);
      const cellPx = canvas.width / geometry.width;
      // The glow reaches REACH_SIGMAS out, unless the GPU caps point size.
      const point = Math.min(2 * REACH_SIGMAS * SIGMA_CELLS * cellPx, maxPoint);
      gl.uniform1f(uniform("u_point"), point);
      gl.uniform1f(uniform("u_reach"), point / cellPx / 2);
    },
    draw(record) {
      ledColours(record, geometry, colourData);
      gl.bindBuffer(gl.ARRAY_BUFFER, colours);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, colourData);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.POINTS, 0, n);
    },
  };
}

// ---- entry point -----------------------------------------------------------

/** Throws if the browser has no WebGL. */
export function createPreview(canvas, geometry) {
  const impl = webglRenderer(canvas, geometry);
  if (impl == null) throw new Error("this browser has WebGL turned off or unavailable");

  let last = null;

  function fit() {
    // Square, the size of the canvas's box, in device pixels.
    const side = Math.max(1, Math.round(Math.min(canvas.clientWidth, canvas.clientHeight) * (window.devicePixelRatio || 1)));
    if (canvas.width !== side || canvas.height !== side) {
      canvas.width = canvas.height = side;
    }
    impl.resize();
  }
  fit();

  return {
    draw(record) {
      last = record;
      impl.draw(record);
    },
    resize() {
      fit();
      if (last) impl.draw(last);
    },
  };
}
