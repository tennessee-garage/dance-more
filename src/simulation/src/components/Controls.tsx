import React from 'react'
import type { SimulationParams } from '../App'

interface ControlsProps {
  params: SimulationParams
  onParamChange: (key: keyof SimulationParams, value: any) => void
}

const Controls: React.FC<ControlsProps> = ({ params, onParamChange }) => {
  const handleColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hex = e.target.value
    const r = parseInt(hex.slice(1, 3), 16) / 255
    const g = parseInt(hex.slice(3, 5), 16) / 255
    const b = parseInt(hex.slice(5, 7), 16) / 255
    onParamChange('ledColor', [r, g, b])
  }

  const colorHex = `#${Math.round(params.ledColor[0] * 255)
    .toString(16)
    .padStart(2, '0')}${Math.round(params.ledColor[1] * 255)
    .toString(16)
    .padStart(2, '0')}${Math.round(params.ledColor[2] * 255)
    .toString(16)
    .padStart(2, '0')}`

  return (
    <div className="controls-panel">
      <h2>LED Acrylic Simulator</h2>

      <div className="control-group">
        <label>View Mode</label>
        <select
          value={params.viewMode}
          onChange={e =>
            onParamChange('viewMode', e.target.value as 'single' | 'grid')
          }
        >
          <option value="single">Single Sheet</option>
          <option value="grid">10x10 Grid</option>
        </select>
      </div>

      <h3>LED Configuration</h3>

      <div className="control-group">
        <label>LEDs Per Side: {params.ledsPerSide}</label>
        <input
          type="range"
          min="1"
          max="20"
          value={params.ledsPerSide}
          onChange={e => onParamChange('ledsPerSide', parseInt(e.target.value))}
        />
        <div className="value-display">Total: {params.ledsPerSide * 4} LEDs</div>
      </div>

      <div className="control-group">
        <label>LED Color</label>
        <input
          type="color"
          value={colorHex}
          onChange={handleColorChange}
        />
      </div>

      <div className="control-group">
        <label>LED Brightness: {Math.round(params.ledBrightness)}%</label>
        <input
          type="range"
          min="0"
          max="200"
          value={params.ledBrightness}
          onChange={e =>
            onParamChange('ledBrightness', parseInt(e.target.value))
          }
        />
      </div>

      <div className="control-group">
        <label>LED Cone Angle: {Math.round(params.ledConeAngle)}°</label>
        <input
          type="range"
          min="30"
          max="120"
          value={params.ledConeAngle}
          onChange={e =>
            onParamChange('ledConeAngle', parseInt(e.target.value))
          }
        />
      </div>

      <h3>Acrylic Properties</h3>

      <div className="control-group">
        <label>Haziness: {(params.haziness * 100).toFixed(0)}%</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={params.haziness}
          onChange={e =>
            onParamChange('haziness', parseFloat(e.target.value))
          }
        />
        <div className="value-display">0=Clear, 1=Frosted</div>
      </div>

      <div className="control-group">
        <label>
          Scattering Strength: {params.scatteringStrength.toFixed(1)}
        </label>
        <input
          type="range"
          min="0"
          max="5"
          step="0.1"
          value={params.scatteringStrength}
          onChange={e =>
            onParamChange('scatteringStrength', parseFloat(e.target.value))
          }
        />
      </div>

      <h3>Visualization</h3>

      <div className="control-group">
        <label>View Angle: {Math.round(params.viewAngle)}°</label>
        <input
          type="range"
          min="0"
          max="90"
          value={params.viewAngle}
          onChange={e => onParamChange('viewAngle', parseInt(e.target.value))}
        />
      </div>

      <div className="control-group">
        <label>Exposure: {params.exposure.toFixed(2)}</label>
        <input
          type="range"
          min="0.1"
          max="3"
          step="0.1"
          value={params.exposure}
          onChange={e =>
            onParamChange('exposure', parseFloat(e.target.value))
          }
        />
      </div>

      <div className="control-group">
        <label>Bloom Strength: {(params.bloomStrength * 100).toFixed(0)}%</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={params.bloomStrength}
          onChange={e =>
            onParamChange('bloomStrength', parseFloat(e.target.value))
          }
        />
      </div>
    </div>
  )
}

export default Controls
