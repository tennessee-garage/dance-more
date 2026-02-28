# LED Acrylic Simulator - Native Python Version

High-performance OpenGL-based visualization of LED-lit acrylic sheets using ModernGL.

## Features

- Real-time 3D rendering with OpenGL 3.3+
- Interactive controls for LED configuration and acrylic properties
- Smooth Perlin noise-based surface texture
- Physically-based lighting with subsurface scattering
- 60+ FPS performance on most hardware

## Installation

**Requirements:** Python 3.9+ (including Python 3.13)

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulator:
```bash
python main.py
```

### Controls

**Mouse:**
- Click and drag to rotate camera
- Scroll to zoom in/out

**Keyboard:**
- Q/W: Decrease/Increase LEDs per side
- A/S: Decrease/Increase LED brightness
- Z/X: Decrease/Increase haziness
- C/V: Decrease/Increase scattering strength
- H: Show help
- ESC: Exit

Current parameters are displayed in the console every second.

## Architecture

- `main.py` - Main application with ModernGL renderer
- Built-in vertex/fragment shaders for acrylic material
- Keyboard-based parameter controls
- Custom Perlin noise generation for realistic surface texture

## Performance

This native implementation is 10-100x faster than browser-based Three.js, achieving:
- 60+ FPS with complex lighting
- Instant parameter response
- Low CPU/GPU usage

## Parameters

Current simulation models:
- Sheet size: 12" × 12" × 3/8" acrylic
- LEDs positioned around edges
- Physically-based transmission, scattering, and fresnel effects
- Real-time subsurface scattering approximation
