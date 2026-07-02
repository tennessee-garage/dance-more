import * as THREE from 'three'
import { createNoiseTexture } from './geometry'

export function createAcrylicMaterial(
  haziness: number,
  scatteringStrength: number,
): THREE.MeshPhysicalMaterial {
  const noiseTexture = createNoiseTexture(256)

  const material = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0,
    roughness: 0.1 + haziness * 0.6,
    transmission: 0.3 + (1 - haziness) * 0.4, // Less transparent when hazy, more when clear
    thickness: 0.5 + scatteringStrength * 0.5,
    ior: 1.49,
    clearcoat: 0.8,
    clearcoatRoughness: 0.05 + haziness * 0.3,
    transparent: true,
    opacity: 0.85 + haziness * 0.15, // More opaque when hazy
    alphaMap: noiseTexture,
    normalMap: noiseTexture,
    normalScale: new THREE.Vector2(0.2 * haziness, 0.2 * haziness),
    side: THREE.DoubleSide,
    envMapIntensity: 1.2,
  })

  return material
}

