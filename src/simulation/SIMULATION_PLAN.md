# LED Acrylic Sheet Simulation - Project Plan

## Project Overview
Simulate a light-up dancefloor acrylic sheet with edge-mounted LEDs to visualize light propagation, scattering, and color effects through a frosted acrylic material.

**LED Specification**: 5050 SMD LEDs (WS2815/WS2812B or SK6812/SK6819 family) at 12V

---

## Technology Stack Recommendation

### **Primary: Three.js + TypeScript/React (Recommended)**
- **Why**: Real-time WebGL rendering, excellent for iterative parameter adjustments
- **Strengths**: 
  - Smooth interactive visualization with live parameter updates
  - Rich ecosystem for 3D graphics
  - Can create custom shaders for light scattering effects
  - Cross-platform (browser-based, deployable anywhere)
  - Good for demonstrating results to others

### **Alternative: Python + Pygame/Moderngl**
- **Why**: Best for physics-based simulation and experimentation
- **Use if**: Need more precise light propagation calculations or want to export physically accurate data

---

## Architecture Overview

```
simulation/
├── src/
│   ├── components/
│   │   ├── Scene.tsx               # Main 3D scene container
│   │   ├── AcrylicSheet.tsx        # 3D acrylic mesh with materials
│   │   ├── LEDStrips.tsx           # LED strip geometry and positions
│   │   ├── Lighting.tsx            # Light sources from LEDs
│   │   └── Controls.tsx            # UI for parameter adjustment
│   ├── utils/
│   │   ├── geometry.ts             # Acrylic sheet geometry generation
│   │   ├── ledPositioning.ts       # Calculate LED positions around edges
│   │   ├── lightPhysics.ts         # Light propagation calculations
│   │   └── materials.ts            # Shader materials for scattering
│   ├── shaders/
│   │   ├── acrylic.vert            # Vertex shader
│   │   └── acrylic.frag            # Fragment shader (scattering effects)
│   └── App.tsx
├── public/
└── package.json
```

---

## Implementation Phases

### **Phase 1: Foundation**
**Goal**: Get basic 3D visualization working

- [x] Set up Three.js scene with basic lighting
- [x] Create 3D acrylic sheet geometry (12"×12"×3/8", centered view from above)
- [x] Position and visualize LEDs around sheet edges
- [x] Create basic interactive controls for LED count and colors
- [x] Implement simple point lights at LED positions

**Deliverable**: Basic scene showing acrylic sheet from above with colored LED lights

---

### **Phase 2: Light Physics & Scattering**
**Goal**: Realistic light propagation and frosted glass effect

#### 2a: Material System
- [ ] Create custom shader material for acrylic
- [ ] Implement opacity-based haziness (0 = clear, 1 = fully frosted)
- [ ] Add subsurface scattering approximation for light glow

#### 2b: Light Cone Simulation
- [ ] Model LED light cones (cone angle, intensity falloff)
- [ ] Implement attenuation with depth into acrylic
- [ ] Add color temperature/spectrum simulation

#### 2c: Edge-Mounted Light Entry
- [ ] Simulate light refracting into acrylic from edges
- [ ] Model total internal reflection at acrylic-air boundaries
- [ ] Create "glow" effect where light enters

#### 2d: Backing Material Interaction
- [ ] Add rear-surface backing layer (vinyl) with selectable reflectance
- [ ] Support white (diffuse reflector), silver (specular reflector), black (absorber)
- [ ] Model how backing changes internal reflections and overall brightness

**Key Considerations**:
- LEDs mounted flush with edge (0° angle, face directly aligned)
- Light bounces inside acrylic due to haziness
- Edge scattering creates diffuse internal reflection
- Backing material changes reflected vs absorbed light at the rear surface

---

### **Phase 3: Advanced Rendering**
**Goal**: High-quality visualization matching physical reality

- [ ] Implement normal maps for acrylic surface texture
- [ ] Add ambient occlusion for depth perception
- [ ] Create glow/bloom post-processing effect
- [ ] Implement soft shadows from acrylic thickness
- [ ] Add chromatic aberration if using RGB LEDs

---

### **Phase 4: Interactive Tuning & UI**
**Goal**: Make it easy to experiment with parameters

- [x] Create slider controls for all parameters
- [x] Add real-time visualization updates
- [ ] Export/save configurations
- [ ] Add preset configurations (different haziness levels, LED patterns)
- [x] Add toggle view: Single sheet vs. 10×10 grid layout
- [ ] Mobile-responsive design

---

## Key Parameters to Expose

### **LED Configuration**
- `ledsPerSide`: 1-20 (adjusts spacing around edges, using 5050 SMD WS28xx/SK68xx family)
- `ledBrightness`: 0-100 (intensity multiplier)
- `ledColor`: RGB or presets (red, green, blue, white, RGB cycle)
- `ledConeAngle`: 30-120° (spread of light cone)
- `ledVoltage`: 12V (power specification)

### **Acrylic Properties**
- `haziness`: 0-1 (0=crystal clear, 1=fully frosted)
- `scatteringStrength`: 0-5 (how much light bounces internally)
- `thickness`: 3/8" (for depth calculations, could adjust)
- `refractionIndex`: 1.49 (acrylic, could expose to tinker)
- `backingMaterial`: white | silver | black (rear-surface vinyl behavior)

### **Visualization**
- `viewAngle`: Elevation angle for viewing
- `exposure`: Brightness overall
- `bloomStrength`: Glow effect intensity

---

## Technical Deep Dives

### **1. Custom Shader for Scattering**
```glsl
// Pseudo-code for acrylic material
fragColor = ledLightColor * haziness * scattering;
fragColor += subsurfaceScatteringGlow(lightDepth, haziness);
```

Use a **volumetric froxel grid** or **screen-space ambient occlusion** approach to simulate light diffusion through acrylic.

### **2. LED Positioning Algorithm**
```
For a sheet with 4 sides:
- Top side: LEDs spaced evenly along top edge
- Right side: LEDs along right edge
- Bottom side: LEDs along bottom edge
- Left side: LEDs along left edge
All positioned flush with the acrylic edge
```

### **3. Light Cone Math**
- Model each LED as a **cone light** with falloff
- Distance attenuation: `intensity = 1 / (1 + distance²)`
- Cone falloff based on angle from LED direction
- Composite all LED contributions

### **4. Haziness Implementation**
- Use noise texture (Perlin noise) modulated by haziness parameter
- **Clear (haziness=0)**: Sharp LED points visible, high contrast
- **Frosted (haziness=1)**: Light diffuses, soft glowing cloud
- Intermediate: Gradual transition

---

## Alternative Simulation Approaches

### **Option A: Rasterization (Recommended)**
- Each LED = point light or cone light
- Shader-based scattering and diffusion
- Real-time performance, good visual quality
- **Best for**: Interactive visualization, parameter tuning

### **Option B: Photon Mapping**
- Trace photons from LEDs through acrylic
- Expensive compute but physically accurate
- **Best for**: High-quality renders, studying light patterns

### **Option C: Monte Carlo Ray Tracing**
- Trace rays from LEDs, account for refraction/reflection
- Very accurate but slow (pre-baked or GPU-accelerated)
- **Best for**: Understanding physics, validation

---

## Tools & Libraries

### **Graphics & Rendering**
- `three.js` - 3D graphics engine
- `glsl-shader-loader` - Load custom shaders
- `postprocessing` - Bloom, glow effects

### **UI & State Management**
- `React` - Component framework
- `zustand` or `redux` - State management for parameters
- `react-three-fiber` - React bindings for Three.js (optional)

### **Utilities**
- `lil-gui` - Parameter sliders and controls
- `stats.js` - Performance monitoring (FPS)

### **Math & Physics**
- `three.js` math utilities (Vector3, Matrix4, etc.)
- Hand-rolled light propagation calculations

---

## Development Roadmap

### **Phase 1 - Foundation**
- Project setup and scaffolding
- Basic 3D scene and acrylic sheet mesh
- LED strip positioning and visualization
- Simple lighting

### **Phase 2 - Physics**
- Custom acrylic material shader
- Scattering and frosted glass effect
- Light cone calculations
- Edge light entry simulation

### **Phase 3 - Polish**
- Advanced visual effects (bloom, glow)
- Performance optimization
- Realistic material properties

### **Phase 4 - Interaction & Deployment**
- Interactive parameter controls
- Save/load configurations
- Documentation and examples
- Ready for experimentation

---

## Success Criteria

- ✅ Real-time interactive visualization
- ✅ Adjustable LED count (per side) and color
- ✅ Visible scattering effects at different haziness levels
- ✅ Light glow visible at acrylic edges
- ✅ Smooth parameter updates (no lag)
- ✅ Physically plausible (light behaves reasonably)
- ✅ Easy to save/compare different configurations
- ✅ Single sheet view for detailed experimentation
- ✅ 10×10 grid view to visualize many sheets together

---

## Next Steps

1. **Choose tech stack** (recommend Three.js + React)
2. **Set up project scaffolding**
3. **Create basic scene** with acrylic sheet and LEDs
4. **Implement custom shader** for scattering
5. **Add interactive controls**
6. **Iterate on visual quality**

Would you like me to proceed with implementation using Three.js/React?
