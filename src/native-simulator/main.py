#!/usr/bin/env python3
"""
LED Acrylic Simulator - Native Python Implementation
High-performance OpenGL-based visualization of LED-lit acrylic sheets
"""

import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import numpy as np
from pathlib import Path
import sys

# Setup logging to file
logfile = open('/tmp/simulator_debug.log', 'w')
def log(msg):
    print(msg, file=logfile, flush=True)
    print(msg)


class AcrylicSimulator(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "LED Acrylic Simulator"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    
    def __init__(self, **kwargs):
        log("AcrylicSimulator.__init__ starting")
        super().__init__(**kwargs)
        log("super().__init__() complete")
        
        # Simulation parameters
        self.params = {
            'leds_per_side': 4,
            'led_brightness': 1.0,
            'led_color': np.array([1.0, 0.0, 0.0], dtype='f4'),  # Red
            'haziness': 0.5,
            'scattering_strength': 1.0,
            'view_angle': 45.0,
            'exposure': 1.0,
        }
        log("Parameters initialized")
        
        # Camera setup
        self.camera_distance = 20.0
        self.camera_rotation = 0.0
        self.camera_pitch = 30.0
        
        # Mouse control
        self.mouse_pressed = False
        self.last_mouse_pos = None
        
        log("Loading shaders...")
        # Load shaders and create program
        self.load_shaders()
        log("Shaders loaded")
        
        log("Creating geometry...")
        self.create_geometry()
        log("Geometry created")
        
        log("Setting up textures...")
        self.setup_textures()
        log("Textures set up")
        
        log("Setting up UI...")
        self.setup_ui()
        log("UI set up, __init__ complete")
        
    def load_shaders(self):
        """Load and compile shaders for acrylic material"""
        vertex_shader = """
            #version 330
            
            in vec3 in_position;
            
            uniform mat4 m_proj;
            uniform mat4 m_view;
            uniform mat4 m_model;
            
            out vec3 frag_position;
            
            void main() {
                vec4 world_pos = m_model * vec4(in_position, 1.0);
                frag_position = world_pos.xyz;
                vec4 view_pos = m_view * world_pos;
                gl_Position = m_proj * view_pos;
            }
        """
        
        fragment_shader = """
            #version 330
            
            in vec3 frag_position;
            out vec4 fragColor;
            
            const int MAX_LEDS = 80;
            
            uniform int num_leds;
            uniform vec3 led_positions[MAX_LEDS];
            uniform vec3 led_color;
            uniform float led_brightness;
            uniform float haziness;
            uniform float scattering_strength;
            uniform sampler2D noise_texture;
            
            void main() {
                // Frosted acrylic base color
                vec3 color = vec3(0.95, 0.95, 0.98);
                
                // Accumulate LED lighting
                float total_light = 0.0;
                for(int i = 0; i < min(num_leds, MAX_LEDS); i++) {
                    float dist = distance(frag_position, led_positions[i]);
                    float intensity = led_brightness / (1.0 + dist * dist * 0.5);
                    total_light += intensity;
                }
                
                // Apply haziness
                float haziness_factor = mix(1.0, 0.5, haziness);
                total_light *= haziness_factor;
                total_light = max(0.1, total_light);
                
                // Blend with LED color
                vec3 led_component = led_color * min(total_light * scattering_strength, 2.0);
                color = mix(color, color * led_component, total_light * 0.3);
                
                fragColor = vec4(color, 0.85);
            }
        """
        
        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            log(f"Shader compilation error: {e}")
            raise
        
        # Print shader compilation status
        log("Shader program compiled successfully")
        log(f"Program uniforms: {list(self.program)}")
        
    def create_geometry(self):
        """Create the acrylic sheet geometry (12" x 12")"""
        # Create a flat quad in world units (inches)
        width, height = 12.0, 12.0
        
        vertices = np.array([
            # Triangle 1
            -width/2, 0, -height/2,
             width/2, 0, -height/2,
             width/2, 0,  height/2,
            # Triangle 2
            -width/2, 0, -height/2,
             width/2, 0,  height/2,
            -width/2, 0,  height/2,
        ], dtype='f4')
        
        log(f"Creating acrylic sheet: {width}x{height} inches")
        log(f"Vertex count: {len(vertices)//3}")
        
        vbo = self.ctx.buffer(vertices.tobytes())
        log(f"VBO created")
        
        try:
            self.vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '3f', 'in_position')],
            )
            log(f"VAO created successfully")
        except Exception as e:
            log(f"ERROR creating VAO: {e}")
            import traceback
            log(traceback.format_exc())
            raise
        
    def setup_textures(self):
        """Create noise texture for acrylic surface"""
        size = 512
        noise_data = self.generate_noise_texture(size)
        
        self.noise_texture = self.ctx.texture(
            (size, size), 1,
            noise_data.tobytes(),
            dtype='f1'
        )
        self.noise_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.noise_texture.repeat_x = True
        self.noise_texture.repeat_y = True
        
    def generate_noise_texture(self, size):
        """Generate smooth Perlin-like noise"""
        grid_size = 16
        grid = np.random.RandomState(12345).rand(grid_size + 1, grid_size + 1)
        
        # Create output array
        noise = np.zeros((size, size), dtype='f4')
        
        for y in range(size):
            for x in range(size):
                # Normalized coordinates
                xn = (x / size) * grid_size
                yn = (y / size) * grid_size
                
                xi = int(xn)
                yi = int(yn)
                xf = xn - xi
                yf = yn - yi
                
                # Clamp to grid
                x0, x1 = min(xi, grid_size - 1), min(xi + 1, grid_size)
                y0, y1 = min(yi, grid_size - 1), min(yi + 1, grid_size)
                
                # Bilinear interpolation with smoothstep
                u = xf * xf * (3 - 2 * xf)
                v = yf * yf * (3 - 2 * yf)
                
                v00, v10 = grid[y0, x0], grid[y0, x1]
                v01, v11 = grid[y1, x0], grid[y1, x1]
                
                nx0 = v00 * (1 - u) + v10 * u
                nx1 = v01 * (1 - u) + v11 * u
                noise[y, x] = nx0 * (1 - v) + nx1 * v
        
        return (noise * 255).astype('u1')
        
    def calculate_led_positions(self):
        """Calculate LED positions around the sheet edges (avoiding corners)"""
        leds_per_side = self.params['leds_per_side']
        sheet_size = 12.0
        positions = []
        
        # Space LEDs with LED-to-corner gap = 0.5 * LED-to-LED gap
        # Position formula: t = (2*i + 1) / (2*N)
        
        # Bottom edge (left to right)
        for i in range(leds_per_side):
            t = (2 * i + 1) / (2 * leds_per_side)
            x = -sheet_size/2 + t * sheet_size
            positions.append([x, -0.2, -sheet_size/2])
        
        # Right edge (bottom to top)
        for i in range(leds_per_side):
            t = (2 * i + 1) / (2 * leds_per_side)
            z = -sheet_size/2 + t * sheet_size
            positions.append([sheet_size/2, -0.2, z])
        
        # Top edge (right to left)
        for i in range(leds_per_side):
            t = (2 * i + 1) / (2 * leds_per_side)
            x = sheet_size/2 - t * sheet_size
            positions.append([x, -0.2, sheet_size/2])
        
        # Left edge (top to bottom)
        for i in range(leds_per_side):
            t = (2 * i + 1) / (2 * leds_per_side)
            z = sheet_size/2 - t * sheet_size
            positions.append([-sheet_size/2, -0.2, z])
        
        return np.array(positions, dtype='f4').flatten()
    
    def setup_ui(self):
        """Setup 2D UI rendering for sliders and text"""
        # Create 2D shader for rectangles
        ui_vertex = """
            #version 330
            in vec2 in_position;
            in vec4 in_color;
            out vec4 v_color;
            uniform mat4 m_proj;
            
            void main() {
                v_color = in_color;
                gl_Position = m_proj * vec4(in_position, 0.0, 1.0);
            }
        """
        
        ui_fragment = """
            #version 330
            in vec4 v_color;
            out vec4 fragColor;
            
            void main() {
                fragColor = v_color;
            }
        """
        
        self.ui_program = self.ctx.program(
            vertex_shader=ui_vertex,
            fragment_shader=ui_fragment
        )
    
    def create_ui_ortho(self, width, height):
        """Create orthographic projection matrix for UI (0,0 at top-left)"""
        result = np.eye(4, dtype='f4')
        result[0, 0] = 2.0 / width
        result[1, 1] = -2.0 / height  # Flip Y
        result[0, 3] = -1.0
        result[1, 3] = 1.0
        result[2, 2] = -1.0
        return result
    
    def draw_rect(self, x, y, width, height, color):
        """Draw a filled rectangle in screen coordinates"""
        vertices = np.array([
            x, y,
            x + width, y,
            x, y + height,
            x + width, y,
            x + width, y + height,
            x, y + height,
        ], dtype='f4')
        
        colors = np.array([color] * 6, dtype='f4')
        
        vbo_vert = self.ctx.buffer(vertices.tobytes())
        vbo_color = self.ctx.buffer(colors.tobytes())
        vao = self.ctx.vertex_array(
            self.ui_program,
            [(vbo_vert, '2f', 'in_position'), (vbo_color, '4f', 'in_color')]
        )
        vao.render(moderngl.TRIANGLES)
        vbo_vert.release()
        vbo_color.release()
        vao.release()
    
    def render_ui(self):
        """Render on-screen UI with parameter sliders"""
        # Update UI projection matrix for rectangles
        width, height = self.wnd.size
        ui_proj = self.create_ui_ortho(width, height)
        self.ui_program['m_proj'].write(ui_proj.T.tobytes())
        
        # UI positioning
        margin = 20
        bar_width = 200
        bar_height = 20
        bar_spacing = 45
        start_y = margin
        
        # Define parameters to display with colors
        params_display = [
            ('LEDs/Side', 'Q/W', self.params['leds_per_side'], 1, 20, (0.2, 0.8, 1.0)),  # cyan
            ('Brightness', 'A/S', self.params['led_brightness'], 0.0, 2.0, (1.0, 0.8, 0.2)),  # yellow
            ('Haziness', 'Z/X', self.params['haziness'], 0.0, 1.0, (0.8, 0.2, 1.0)),  # purple  
            ('Scattering', 'C/V', self.params['scattering_strength'], 0.0, 5.0, (1.0, 0.4, 0.2)),  # orange
        ]
        
        # Calculate bar position
        label_width = 140
        bar_x = margin + label_width
        
        for i, (label, keys, value, min_val, max_val, bar_color) in enumerate(params_display):
            y = start_y + i * bar_spacing
            
            # Draw background bar (dark gray)
            self.draw_rect(bar_x, y, bar_width, bar_height, (0.2, 0.2, 0.2, 0.8))
            
            # Draw filled portion with parameter-specific color
            fill_ratio = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
            fill_width = bar_width * fill_ratio
            self.draw_rect(bar_x, y, fill_width, bar_height, (*bar_color, 0.9))
            
            # Draw border around bar
            border_thickness = 2
            border_color = (0.8, 0.8, 0.8, 1.0)
            # Top border
            self.draw_rect(bar_x, y, bar_width, border_thickness, border_color)
            # Bottom border
            self.draw_rect(bar_x, y + bar_height - border_thickness, bar_width, border_thickness, border_color)
            # Left border
            self.draw_rect(bar_x, y, border_thickness, bar_height, border_color)
            # Right border
            self.draw_rect(bar_x + bar_width - border_thickness, y, border_thickness, bar_height, border_color)
        
        # Console output for current values (since text rendering needs more work)
        if not hasattr(self, '_last_ui_print') or hasattr(self, '_frametime') and self._frametime:
            # Print once every 30 frames to avoid spam
            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            self._frame_counter += 1
            
            if self._frame_counter % 30 == 0:
                print(f"\\rLEDs:{self.params['leds_per_side']:2d} Bright:{self.params['led_brightness']:.2f} Haze:{self.params['haziness']:.2f} Scatter:{self.params['scattering_strength']:.2f}  Keys: Q/W A/S Z/X C/V", end='', flush=True)
        
    def on_render(self, time: float, frametime: float):
        """Main render loop"""
        # Store frametime for FPS calculation
        self._frametime = frametime
        
        # DEBUG: Clear to bright blue
        self.ctx.clear(0.0, 0.5, 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Calculate camera position (proper spherical orbit)
        yaw_rad = np.radians(self.camera_rotation)
        pitch_rad = np.radians(self.camera_pitch)
        cos_pitch = np.cos(pitch_rad)

        # Prevent singularity at exact +/-90 pitch while preserving top-down behavior
        if abs(cos_pitch) < 1e-4:
            cos_pitch = 1e-4

        cam_x = np.sin(yaw_rad) * cos_pitch * self.camera_distance
        cam_z = np.cos(yaw_rad) * cos_pitch * self.camera_distance
        cam_y = np.sin(pitch_rad) * self.camera_distance
        camera_pos = np.array([cam_x, cam_y, cam_z], dtype='f4')
        
        # Matrices
        view = self.create_look_at(camera_pos, np.array([0, 0, 0], dtype='f4'))
        proj = self.create_perspective(45.0, self.aspect_ratio, 0.1, 100.0)
        model = np.eye(4, dtype='f4')
        
        # Set matrices (transposed for OpenGL column-major)
        self.program['m_proj'].write(proj.T.tobytes())
        self.program['m_view'].write(view.T.tobytes())
        self.program['m_model'].write(model.T.tobytes())
        
        # Set LED parameters
        if 'num_leds' in self.program and 'led_positions' in self.program:
            led_positions = self.calculate_led_positions()
            num_leds = len(led_positions) // 3
            max_leds = 80
            if num_leds < max_leds:
                padding = np.zeros((max_leds - num_leds) * 3, dtype='f4')
                led_positions = np.concatenate([led_positions, padding])
            self.program['num_leds'].value = num_leds
            self.program['led_positions'].write(led_positions.tobytes())
        
        if 'led_color' in self.program:
            self.program['led_color'].write(self.params['led_color'].tobytes())
        if 'led_brightness' in self.program:
            self.program['led_brightness'].value = self.params['led_brightness']
        if 'haziness' in self.program:
            self.program['haziness'].value = self.params['haziness']
        if 'scattering_strength' in self.program:
            self.program['scattering_strength'].value = self.params['scattering_strength']
        
        # Render
        try:
            self.vao.render(moderngl.TRIANGLES)
            if not hasattr(self, '_render_success'):
                self._render_success = True
                log("Successfully rendered acrylic sheet!")
        except Exception as e:
            log(f"ERROR rendering: {e}")
            import traceback
            log(traceback.format_exc())
        
        # Render UI overlay
        self.render_ui()
        
        # Print controls (less frequently now that we have on-screen UI)
        if not hasattr(self, '_last_console_print') or time - self._last_console_print > 5.0:
            self._last_console_print = time
            fps = 1.0 / frametime if frametime > 0 else 0
            print(f"FPS: {fps:.1f}")
        
    def print_controls(self, time):
        """Print controls overlay to console"""
        # Print parameters periodically
        if not hasattr(self, '_last_print') or time - self._last_print > 1.0:
            self._last_print = time
            fps = 1.0 / self._frametime if hasattr(self, '_frametime') and self._frametime > 0 else 0
            print("\n" + "="*50)
            print(f"FPS: {fps:.1f}")
            print(f"LEDs Per Side: {self.params['leds_per_side']} (Q/W to adjust)")
            print(f"LED Brightness: {self.params['led_brightness']:.2f} (A/S to adjust)")
            print(f"Haziness: {self.params['haziness']:.2f} (Z/X to adjust)")
            print(f"Scattering: {self.params['scattering_strength']:.2f} (C/V to adjust)")
            print("="*50)
            print("Controls: Mouse drag = rotate, Scroll = zoom")
            print("Press H for help")
    
    def on_key_event(self, key, action, modifiers):
        """Handle keyboard events"""
        keys = self.wnd.keys
        
        if action == keys.ACTION_PRESS:
            # LEDs per side
            if key == keys.Q:
                self.params['leds_per_side'] = max(1, self.params['leds_per_side'] - 1)
                print(f"LEDs Per Side: {self.params['leds_per_side']}")
            elif key == keys.W:
                self.params['leds_per_side'] = min(20, self.params['leds_per_side'] + 1)
                print(f"LEDs Per Side: {self.params['leds_per_side']}")
            
            # LED brightness
            elif key == keys.A:
                self.params['led_brightness'] = max(0.0, self.params['led_brightness'] - 0.1)
                print(f"LED Brightness: {self.params['led_brightness']:.2f}")
            elif key == keys.S:
                self.params['led_brightness'] = min(2.0, self.params['led_brightness'] + 0.1)
                print(f"LED Brightness: {self.params['led_brightness']:.2f}")
            
            # Haziness
            elif key == keys.Z:
                self.params['haziness'] = max(0.0, self.params['haziness'] - 0.05)
                print(f"Haziness: {self.params['haziness']:.2f}")
            elif key == keys.X:
                self.params['haziness'] = min(1.0, self.params['haziness'] + 0.05)
                print(f"Haziness: {self.params['haziness']:.2f}")
            
            # Scattering
            elif key == keys.C:
                self.params['scattering_strength'] = max(0.0, self.params['scattering_strength'] - 0.2)
                print(f"Scattering: {self.params['scattering_strength']:.2f}")
            elif key == keys.V:
                self.params['scattering_strength'] = min(5.0, self.params['scattering_strength'] + 0.2)
                print(f"Scattering: {self.params['scattering_strength']:.2f}")
            
            # Help
            elif key == keys.H:
                print("\n" + "="*50)
                print("KEYBOARD CONTROLS")
                print("="*50)
                print("Q/W: Decrease/Increase LEDs per side")
                print("A/S: Decrease/Increase LED brightness")
                print("Z/X: Decrease/Increase haziness")
                print("C/V: Decrease/Increase scattering")
                print("H: Show this help")
                print("ESC: Exit")
                print("="*50)
    
    def create_look_at(self, eye, target):
        """Create stable view matrix for orbit camera, including top-down pole handling"""
        f = target - eye
        f = f / np.linalg.norm(f)

        up = np.array([0, 1, 0], dtype='f4')

        # Near poles, world-up becomes parallel to forward; choose yaw-aligned up instead
        if abs(np.dot(f, up)) > 0.999:
            yaw_rad = np.radians(self.camera_rotation)
            up = np.array([np.sin(yaw_rad), 0.0, np.cos(yaw_rad)], dtype='f4')

        s = np.cross(f, up)
        s_len = np.linalg.norm(s)
        if s_len < 1e-6:
            up = np.array([1, 0, 0], dtype='f4')
            s = np.cross(f, up)
            s_len = np.linalg.norm(s)

        s = s / s_len
        u = np.cross(s, f)
        
        result = np.eye(4, dtype='f4')
        result[0, :3] = s
        result[1, :3] = u
        result[2, :3] = -f
        result[0, 3] = -np.dot(s, eye)
        result[1, 3] = -np.dot(u, eye)
        result[2, 3] = np.dot(f, eye)
        
        return result
        
    def create_perspective(self, fovy, aspect, near, far):
        """Create perspective projection matrix"""
        f = 1.0 / np.tan(np.radians(fovy) / 2.0)
        result = np.zeros((4, 4), dtype='f4')
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2.0 * far * near) / (near - far)
        result[3, 2] = -1.0
        
        # Debug first time
        if not hasattr(self, '_proj_matrix_debug'):
            self._proj_matrix_debug = True
            log(f"Projection matrix created:")
            log(f"  FOV: {fovy}, Aspect: {aspect}, Near: {near}, Far: {far}")
            log(f"  Projection matrix:\\n{result}")
        
        return result
        
    def on_mouse_press_event(self, x, y, button):
        """Handle mouse button press"""
        log(f"Mouse press: ({x}, {y}) button={button}")
        self.mouse_pressed = True
        self.last_mouse_pos = (x, y)
        
    def on_mouse_release_event(self, x, y, button):
        """Handle mouse button release"""
        log(f"Mouse release: ({x}, {y}) button={button}")
        self.mouse_pressed = False
        
    def on_mouse_position_event(self, x, y, dx, dy):
        """Handle mouse movement (non-drag)"""
        pass  # Dragging is handled by on_mouse_drag_event
        
    def on_mouse_scroll_event(self, x_offset, y_offset):
        """Handle mouse scroll for zoom"""
        log(f"Mouse scroll: x_offset={x_offset}, y_offset={y_offset}")
        self.camera_distance = np.clip(self.camera_distance - y_offset * 2, 5, 50)
    
    def on_mouse_drag_event(self, x, y, dx, dy):
        """Handle mouse drag for camera rotation"""
        log(f"Mouse drag: ({x}, {y}) dx={dx}, dy={dy}")
        # Reverse controls: drag left rotates right, drag up rotates down
        self.camera_rotation -= dx * 0.5
        self.camera_pitch = np.clip(self.camera_pitch + dy * 0.5, -90, 90)


if __name__ == '__main__':
    mglw.run_window_config(AcrylicSimulator)
