import * as THREE from 'three'

// Sheet dimensions: 12" x 12" x 3/8"
// Convert to world units (1 unit = 1 inch for simplicity)
const SHEET_WIDTH = 12
const SHEET_HEIGHT = 12
const SHEET_DEPTH = 0.375

export function createAcrylicSheetGeometry(): THREE.BufferGeometry {
  // Create box: width (X), depth/thickness (Y), height (Z)
  // This creates a flat sheet in the XZ plane with thin Y dimension
  const geometry = new THREE.BoxGeometry(SHEET_WIDTH, SHEET_DEPTH, SHEET_HEIGHT)
  return geometry
}

export function getSheetDimensions() {
  return {
    width: SHEET_WIDTH,
    height: SHEET_HEIGHT,
    depth: SHEET_DEPTH,
  }
}

// Generate a grid of sheet positions for grid view
export function generateGridPositions(
  gridSize: number = 10,
  spacing: number = 1.5,
): THREE.Vector3[] {
  const positions: THREE.Vector3[] = []
  const offset = (gridSize - 1) / 2

  for (let x = 0; x < gridSize; x++) {
    for (let z = 0; z < gridSize; z++) {
      const posX = (x - offset) * SHEET_WIDTH * spacing
      const posZ = (z - offset) * SHEET_HEIGHT * spacing
      positions.push(new THREE.Vector3(posX, 0, posZ))
    }
  }

  return positions
}

//Noise texture for haziness - smoothly varying without banding
export function createNoiseTexture(size: number = 512): THREE.Texture {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size

  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get canvas context')

  const imageData = ctx.createImageData(size, size)
  const data = imageData.data

  // Generate smooth Perlin-like noise with proper interpolation
  // Use a grid of random values and interpolate between them
  const gridSize = 16
  const grid: number[][] = []

  // Initialize grid with seeded random values
  let seed = 12345
  const seededRandom = () => {
    seed = Math.sin(seed) * 10000
    return seed - Math.floor(seed)
  }

  for (let i = 0; i <= gridSize; i++) {
    grid[i] = []
    for (let j = 0; j <= gridSize; j++) {
      grid[i][j] = seededRandom()
    }
  }

  // Smoothstep interpolation function
  const smoothstep = (t: number) => t * t * (3 - 2 * t)

  // Interpolate between two values
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t

  // Generate texture by interpolating the grid
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const xNorm = (x / size) * gridSize
      const yNorm = (y / size) * gridSize

      const xi = Math.floor(xNorm)
      const yi = Math.floor(yNorm)
      const xf = xNorm - xi
      const yf = yNorm - yi

      // Clamp to grid bounds
      const x0 = Math.min(xi, gridSize - 1)
      const x1 = Math.min(xi + 1, gridSize)
      const y0 = Math.min(yi, gridSize - 1)
      const y1 = Math.min(yi + 1, gridSize)

      // Get corner values
      const v00 = grid[x0][y0]
      const v10 = grid[x1][y0]
      const v01 = grid[x0][y1]
      const v11 = grid[x1][y1]

      // Smooth interpolation
      const u = smoothstep(xf)
      const v = smoothstep(yf)

      const nx0 = lerp(v00, v10, u)
      const nx1 = lerp(v01, v11, u)
      const value = lerp(nx0, nx1, v)

      // Add subtle second octave for variation
      const xNorm2 = (x / size) * gridSize * 2
      const yNorm2 = (y / size) * gridSize * 2
      const xi2 = Math.floor(xNorm2) % gridSize
      const yi2 = Math.floor(yNorm2) % gridSize
      const xf2 = xNorm2 - Math.floor(xNorm2)
      const yf2 = yNorm2 - Math.floor(yNorm2)

      const v002 = grid[xi2][yi2]
      const v102 = grid[(xi2 + 1) % gridSize][yi2]
      const v012 = grid[xi2][(yi2 + 1) % gridSize]
      const v112 = grid[(xi2 + 1) % gridSize][(yi2 + 1) % gridSize]

      const u2 = smoothstep(xf2)
      const v2 = smoothstep(yf2)

      const nx02 = lerp(v002, v102, u2)
      const nx12 = lerp(v012, v112, u2)
      const value2 = lerp(nx02, nx12, v2)

      // Combine with second octave
      const finalValue = value * 0.7 + value2 * 0.3
      const brightness = Math.round(finalValue * 255)

      const pixelIndex = (y * size + x) * 4
      data[pixelIndex] = brightness
      data[pixelIndex + 1] = brightness
      data[pixelIndex + 2] = brightness
      data[pixelIndex + 3] = 255
    }
  }

  ctx.putImageData(imageData, 0, 0)

  const texture = new THREE.CanvasTexture(canvas)
  texture.minFilter = THREE.LinearFilter
  texture.magFilter = THREE.LinearFilter
  texture.wrapS = THREE.RepeatWrapping
  texture.wrapT = THREE.RepeatWrapping

  return texture
}
