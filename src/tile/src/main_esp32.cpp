#include <Arduino.h>
#include <math.h>
#include "protocol.h"
#include "pixel_buffer.h"
#include "esp32/pins_rc.h"

// ---------------------------------------------------------------------------
// RS-485 frame TX
// ---------------------------------------------------------------------------

static void rc_send(uint8_t addr, uint8_t cmd,
                    const uint8_t *payload = nullptr, uint8_t len = 0)
{
    Frame f{};
    f.addr = addr;
    f.cmd  = cmd;
    f.len  = len;
    if (payload && len)
        memcpy(f.payload, payload, len);

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(f, buf, sizeof(buf));
    if (n <= 0) return;

    digitalWrite(PIN_RC_DE, HIGH);
    Serial1.write(buf, (size_t)n);
    Serial1.flush();                // wait for TX FIFO empty + last stop bit
    digitalWrite(PIN_RC_DE, LOW);
}

static void push_solid(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t payload[3] = {r, g, b};
    rc_send(ADDR_BROADCAST, (uint8_t)Cmd::SET_COLOR, payload, 3);
    rc_send(ADDR_BROADCAST, (uint8_t)Cmd::LATCH);
}

static void push_leds(const Pixel leds[PixelBuffer::NUM_LEDS])
{
    uint8_t payload[PixelBuffer::NUM_LEDS * 3];
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        payload[i * 3 + 0] = leds[i].r;
        payload[i * 3 + 1] = leds[i].g;
        payload[i * 3 + 2] = leds[i].b;
    }
    rc_send(ADDR_BROADCAST, (uint8_t)Cmd::SET_LEDS,
            payload, sizeof(payload));
    rc_send(ADDR_BROADCAST, (uint8_t)Cmd::LATCH);
}

// ---------------------------------------------------------------------------
// Animation helpers
// ---------------------------------------------------------------------------

static void hsv_to_rgb(float h, float s, float v,
                        uint8_t &r, uint8_t &g, uint8_t &b)
{
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r1, g1, b1;
    int   sector = (int)(h * 6.0f);
    switch (sector % 6) {
        case 0: r1=c; g1=x; b1=0; break;
        case 1: r1=x; g1=c; b1=0; break;
        case 2: r1=0; g1=c; b1=x; break;
        case 3: r1=0; g1=x; b1=c; break;
        case 4: r1=x; g1=0; b1=c; break;
        default:r1=c; g1=0; b1=x; break;
    }
    r = (uint8_t)((r1 + m) * 255.0f);
    g = (uint8_t)((g1 + m) * 255.0f);
    b = (uint8_t)((b1 + m) * 255.0f);
}

static Pixel solid_pixel(uint8_t r, uint8_t g, uint8_t b)
{
    return {r, g, b};
}

static void frame_chase(Pixel out[PixelBuffer::NUM_LEDS],
                        int tick, uint8_t r, uint8_t g, uint8_t b,
                        int trail = 4)
{
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
        out[i] = {0, 0, 0};
    for (int i = 0; i <= trail; i++) {
        int   idx   = ((tick - i) % PixelBuffer::NUM_LEDS
                       + PixelBuffer::NUM_LEDS) % PixelBuffer::NUM_LEDS;
        float fade  = 1.0f - (float)i / (trail + 1);
        out[idx]    = {(uint8_t)(r * fade),
                       (uint8_t)(g * fade),
                       (uint8_t)(b * fade)};
    }
}

static void frame_rainbow(Pixel out[PixelBuffer::NUM_LEDS], int tick)
{
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        float hue = fmodf((float)(i + tick) / PixelBuffer::NUM_LEDS, 1.0f);
        hsv_to_rgb(hue, 1.0f, 1.0f, out[i].r, out[i].g, out[i].b);
    }
}

static void frame_pulse(Pixel out[PixelBuffer::NUM_LEDS],
                        int tick, uint8_t r, uint8_t g, uint8_t b)
{
    float brightness = (sinf(tick * (float)M_PI / 20.0f) + 1.0f) / 2.0f;
    Pixel c = {(uint8_t)(r * brightness),
               (uint8_t)(g * brightness),
               (uint8_t)(b * brightness)};
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
        out[i] = c;
}

// ---------------------------------------------------------------------------
// Test pattern (same sequence as tools/tile-test.py --test)
// ---------------------------------------------------------------------------

static void run_test_pattern()
{
    static const uint32_t FRAME_MS = 40;   // ~25 fps
    Pixel leds[PixelBuffer::NUM_LEDS];

    Serial.println("[test] solid colors");

    const struct { const char *name; uint8_t r, g, b; } solids[] = {
        {"red",   255,   0,   0},
        {"green",   0, 255,   0},
        {"blue",    0,   0, 255},
        {"amber", 255, 128,   0},
        {"white", 255, 255, 255},
        {"off",     0,   0,   0},
    };
    for (auto &s : solids) {
        Serial.printf("[test]   %s\n", s.name);
        for (int i = 0; i < 25; i++) {   // ~1 s
            push_solid(s.r, s.g, s.b);
            delay(FRAME_MS);
        }
    }

    Serial.println("[test] chase (white, 3 laps)");
    for (int t = 0; t < PixelBuffer::NUM_LEDS * 3; t++) {
        frame_chase(leds, t, 255, 255, 255);
        push_leds(leds);
        delay(FRAME_MS);
    }

    const struct { uint8_t r, g, b; } chase_colors[] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
    };
    for (auto &c : chase_colors) {
        for (int t = 0; t < PixelBuffer::NUM_LEDS; t++) {
            frame_chase(leds, t, c.r, c.g, c.b);
            push_leds(leds);
            delay(FRAME_MS);
        }
    }

    Serial.println("[test] rainbow (3 rotations)");
    for (int t = 0; t < PixelBuffer::NUM_LEDS * 3; t++) {
        frame_rainbow(leds, t);
        push_leds(leds);
        delay(FRAME_MS);
    }

    Serial.println("[test] pulse cyan (3 cycles)");
    for (int t = 0; t < 40 * 3; t++) {
        frame_pulse(leds, t, 0, 200, 255);
        push_leds(leds);
        delay(FRAME_MS);
    }

    Serial.println("[test] done");
    push_solid(0, 0, 0);
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------

void setup()
{
    Serial.begin(115200);
    delay(500);   // wait for USB CDC to enumerate

    Serial1.begin(1000000, SERIAL_8N1, PIN_RC_RX, PIN_RC_TX);

    digitalWrite(PIN_RC_DE, LOW);
    pinMode(PIN_RC_DE, OUTPUT);

    Serial.println("[rc] DF2 tile tester ready");
    Serial.println("[rc] commands: r g b w 0 t");
    Serial.println("[rc]   r/g/b/w  = solid colour");
    Serial.println("[rc]   0        = all off");
    Serial.println("[rc]   t        = full test pattern");

    // Clear any stale tile state.
    push_solid(0, 0, 0);
}

static uint32_t last_heartbeat = 0;

void loop()
{
    if (Serial.available()) {
        char ch = (char)Serial.read();
        switch (ch) {
            case 'r': Serial.println("[rc] red");   push_solid(255,   0,   0); break;
            case 'g': Serial.println("[rc] green"); push_solid(  0, 255,   0); break;
            case 'b': Serial.println("[rc] blue");  push_solid(  0,   0, 255); break;
            case 'w': Serial.println("[rc] white"); push_solid(255, 255, 255); break;
            case '0': Serial.println("[rc] off");   push_solid(  0,   0,   0); break;
            case 't': run_test_pattern(); break;
            default: break;
        }
    }

    uint32_t now = millis();
    if (now - last_heartbeat >= 5000) {
        Serial.println("[rc] alive");
        last_heartbeat = now;
    }
}
