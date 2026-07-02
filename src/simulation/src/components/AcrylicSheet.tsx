import React, { useEffect, useMemo } from 'react'
import { createAcrylicSheetGeometry } from '../utils/geometry'
import { createAcrylicMaterial } from '../utils/materials'
import type { SimulationParams } from '../App'

interface AcrylicSheetProps {
  params: SimulationParams
  position: [number, number, number]
}

const AcrylicSheet: React.FC<AcrylicSheetProps> = ({ params, position }) => {
  const geometry = useMemo(() => createAcrylicSheetGeometry(), [])
  const material = useMemo(
    () => createAcrylicMaterial(params.haziness, params.scatteringStrength),
    [],
  )

  useEffect(() => {
    material.roughness = 0.1 + params.haziness * 0.6
    material.transmission = 0.3 + (1 - params.haziness) * 0.4
    material.thickness = 0.5 + params.scatteringStrength * 0.5
    material.clearcoatRoughness = 0.05 + params.haziness * 0.3
    material.opacity = 0.85 + params.haziness * 0.15
    material.normalScale.set(0.2 * params.haziness, 0.2 * params.haziness)
  }, [material, params.haziness, params.scatteringStrength])

  return (
    <mesh
      geometry={geometry}
      material={material}
      position={position}
      castShadow
      receiveShadow
    />
  )
}

export default AcrylicSheet
