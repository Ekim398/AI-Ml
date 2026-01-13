import React from 'react'
//for the undo clear export thingyies 
export default function Toolbar({
  tool,
  setTool,
  color,
  setColor,
  size,
  setSize,
  eraserSize,
  setEraserSize,
  // match App prop names
  stabilizerEnabled,
  setStabilizerEnabled,
  stabilizerLevel,
  setStabilizerLevel,
  // actions
  undo,
  clearAll,
  onExport,
  onSendOcr,
  onToggleHints,
  hintsVisible
}) {
  return (
    <div className="toolbar">
      <div className="tools">
        <button onClick={() => setTool('pen')} className={tool === 'pen' ? 'active' : ''}>Pen</button>
        <button onClick={() => setTool('eraser')} className={tool === 'eraser' ? 'active' : ''}>Eraser</button>
      </div>

      <div className="controls">
        <label>Color <input type="color" value={color} onChange={(e) => setColor(e.target.value)} /></label>
        <label>Size <input type="range" min="1" max="40" value={size} onChange={(e) => setSize(Number(e.target.value))} /></label>
        <label>Eraser <input type="range" min="4" max="60" value={eraserSize} onChange={(e) => setEraserSize(Number(e.target.value))} /></label>
      </div>

      <div className="stabilizer">
        <label>
          <input type="checkbox" checked={stabilizerEnabled} onChange={(e) => setStabilizerEnabled(e.target.checked)} /> Stabilizer
        </label>
        <label>
          Level <input type="range" min="1" max="12" value={stabilizerLevel} onChange={(e) => setStabilizerLevel(Number(e.target.value))} />
        </label>
      </div>

      <div className="actions">
        <button onClick={undo}>Undo</button>
        <button onClick={clearAll}>Clear</button>
        <button onClick={onExport}>Export PDF</button>
        <button className="ocr-btn" onClick={() => onSendOcr && onSendOcr()}>Sam!</button>
        <button className="hints-btn" onClick={() => onToggleHints && onToggleHints()}>
          {hintsVisible ? "Hide Hints" : "Show Hints"}
        </button>
      </div>
    </div>
  )
}
