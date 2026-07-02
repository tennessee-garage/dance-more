import * as THREE from 'three'
import { getSheetDimensions } from './geometry'

export interface LEDPosition {
  position: THREE.Vector3
  direction: THREE.Vector3
  sideIndex: number
}

export function calculateLEDPositions(ledsPerSide: number): LEDPosition[] {
  const { width, height, depth } = getSheetDimensions()
  const positions: LEDPosition[] = []

  // Top side (positive Z)
  for (let i = 0; i < ledsPerSide; i++) {
    const x = (-width / 2) + (width / ledsPerSide) * (i + 0.5)
    positions.push({
      position: new THREE.Vector3(x, 0, height / 2),
      direction: new THREE.Vector3(0, 1, 0), // Shining inward
      sideIndex: 0,
    })
  }

  // Right side (positive X)
  for (let i = 0; i < ledsPerSide; i++) {
    const z = (-height / 2) + (height / ledsPerSide) * (i + 0.5)
    positions.push({
      position: new THREE.Vector3(width / 2, 0, z),
      direction: new THREE.Vector3(-1, 0, 0), // Shining inward
      sideIndex: 1,
    })
  }

  // Bottom side (negative Z)
  for (let i = 0; i < ledsPerSide; i++) {
    const x = (width / 2) - (width / ledsPerSide) * (i + 0.5)
    positions.push({
      position: new THREE.Vector3(x, 0, -height / 2),
      direction: new THREE.Vector3(0, 1, 0), // Shining inward
      sideIndex: 2,
    })
  }

  // Left side (negative X)
  for (let i = 0; i < ledsPerSide; i++) {
    const z = (height / 2) - (height / ledsPerSide) * (i + 0.5)
    positions.push({
      position: new THREE.Vector3(-width / 2, 0, z),
      direction: new THREE.Vector3(1, 0, 0), // Shining inward
      sideIndex: 3,
    })
  }

  return positions
}

// Get total LED count for given ledsPerSide
export function getTotalLEDCount(ledsPerSide: number): number {
  return ledsPerSide * 4 // 4 sides
}

// Helper to visualize LED positions (for debugging)
export function createLEDVisualization(positions: LEDPosition[]): THREE.Group {
  const group = new THREE.Group()

  positions.forEach(led => {
    const geometry = new THREE.IcosahedronGeometry(0.2, 2)
    const material = new THREE.MeshBasicMaterial({ color: 0xff0000 })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.position.copy(led.position)
    group.add(mesh)
  })

  return group
}
