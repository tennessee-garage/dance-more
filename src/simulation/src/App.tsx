import React, { useState } from 'react'
import { Canvas } from '@react-three/fiber'
import Scene from './components/Scene'
import Controls from './components/Controls'
import './App.css'

export interface SimulationParams {
  ledsPerSide: number
  ledBrightness: number
  ledColor: [number, number, number]
  ledConeAngle: number
  haziness: number
  scatteringStrength: number
  viewAngle: number
  exposure: number
  bloomStrength: number
  viewMode: 'single' | 'grid'
}

const defaultParams: SimulationParams = {
  ledsPerSide: 4,
  ledBrightness: 100,
  ledColor: [1, 0, 0],
  ledConeAngle: 60,
  haziness: 0.5,
  scatteringStrength: 1,
  viewAngle: 45,
  exposure: 1,
  bloomStrength: 0.5,
  viewMode: 'single',
}

function App() {
  const [params, setParams] = useState<SimulationParams>(defaultParams)

  const handleParamChange = (key: keyof SimulationParams, value: any) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="app-container">
      <div className="canvas-container">
        <Canvas
          camera={{
            position: [0, 15, 20],
            fov: 50,
            near: 0.1,
            far: 1000,
          }}
        >
          <color attach="background" args={[0.015, 0.015, 0.02]} />
          <Scene params={params} />
        </Canvas>
      </div>
      <Controls params={params} onParamChange={handleParamChange} />
    </div>
  )
}

export default App
