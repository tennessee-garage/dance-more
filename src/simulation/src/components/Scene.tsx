import React from 'react'
import { useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import AcrylicSheet from './AcrylicSheet'
import LEDStrips from './LEDStrips'
import type { SimulationParams } from '../App'
import { generateGridPositions } from '../utils/geometry'

interface SceneProps {
  params: SimulationParams
}

const Scene: React.FC<SceneProps> = ({ params }) => {
  const { camera } = useThree()

  React.useEffect(() => {
    if (params.viewMode === 'single') {
      camera.position.set(0, 15, 20)
    } else {
      camera.position.set(0, 80, 100)
    }
    camera.updateProjectionMatrix()
  }, [params.viewMode, camera])

  return (
    <>
      <ambientLight intensity={0.5 * params.exposure} />
      <directionalLight
        position={[30, 50, 30]}
        intensity={0.3 * params.exposure}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />

      {params.viewMode === 'single' ? (
        <group>
          <AcrylicSheet params={params} position={[0, 0, 0]} />
          <LEDStrips params={params} position={[0, 0, 0]} />
        </group>
      ) : (
        <group>
          {generateGridPositions(10, 1.5).map((pos, idx) => (
            <group key={idx} position={[pos.x, pos.y, pos.z]}>
              <AcrylicSheet params={params} position={[0, 0, 0]} />
              <LEDStrips params={params} position={[0, 0, 0]} />
            </group>
          ))}
        </group>
      )}

      <OrbitControls enableDamping dampingFactor={0.05} />
    </>
  )
}

export default Scene
