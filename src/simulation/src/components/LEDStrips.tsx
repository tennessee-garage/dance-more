import React, { useMemo } from 'react'
import { calculateLEDPositions } from '../utils/ledPositioning'
import type { SimulationParams } from '../App'

interface LEDStripsProps {
  params: SimulationParams
  position: [number, number, number]
}

const LEDStrips: React.FC<LEDStripsProps> = ({ params, position }) => {
  const ledPositions = useMemo(
    () => calculateLEDPositions(params.ledsPerSide),
    [params.ledsPerSide],
  )

  const ledColorHex =
    (Math.round(params.ledColor[0] * 255) << 16) |
    (Math.round(params.ledColor[1] * 255) << 8) |
    Math.round(params.ledColor[2] * 255)

  const lightIntensity = (params.ledBrightness / 100) * params.exposure

  return (
    <group position={position}>
      {ledPositions.map((led, idx) => (
        <group key={`led-${idx}`}>
          <mesh position={[led.position.x, led.position.y, led.position.z]}>
            <icosahedronGeometry args={[0.15, 2]} />
            <meshBasicMaterial color={ledColorHex} />
          </mesh>

          <pointLight
            position={[led.position.x, led.position.y, led.position.z]}
            color={ledColorHex}
            intensity={lightIntensity * 2}
            distance={50}
            decay={2}
            castShadow
          />
        </group>
      ))}
    </group>
  )
}

export default LEDStrips
