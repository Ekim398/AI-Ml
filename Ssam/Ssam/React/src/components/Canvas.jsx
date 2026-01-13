//Pencil Stroke Functions and Background Canvas 
import React, { useRef, useEffect, useImperativeHandle } from 'react'

function drawStroke(ctx, stroke) {
  if (!ctx || !stroke) return
  ctx.save()
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.lineWidth = stroke.size
  if (stroke.tool === 'eraser') {
    ctx.globalCompositeOperation = 'destination-out'
  } else {
    ctx.globalCompositeOperation = 'source-over'
    ctx.strokeStyle = stroke.color
  }
  ctx.beginPath()
  const pts = stroke.points
  if (!pts || pts.length === 0) { ctx.restore(); return }
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i].x, pts[i].y)
  }
  ctx.stroke()
  ctx.restore()
}

function Canvas({
  id = 'canvas',
  strokes = [],
  // new API: addStroke(pageIndex, stroke)
  addStroke,
  pageIndex = 0,
  tool,
  color,
  size,
  eraserSize = 16,
  stabilizerEnabled = true,
  stabilizerLevel = 4,
  onStrokesChange,
  className
}, forwardedRef) {
  const canvasRef = useRef(null)
  const drawing = useRef(false)
  const current = useRef(null)
  const ctxRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) { console.warn('Canvas not mounted'); return }
    const ctx = canvas.getContext('2d')
    ctxRef.current = ctx

    const redraw = () => {
      try {
        const rect = canvas.getBoundingClientRect()
        if (rect.width === 0 || rect.height === 0) { console.warn('Canvas rect zero', rect); return }
        ctx.clearRect(0, 0, rect.width, rect.height)
        for (const s of strokes) drawStroke(ctx, s)
      } catch (err) {
        console.error('redraw error', err)
      }
    }

    const resize = () => {
      try {
        const rect = canvas.getBoundingClientRect()
        if (rect.width === 0 || rect.height === 0) { console.warn('resize: zero rect', rect); return }
        const dpr = window.devicePixelRatio || 1
        canvas.width = Math.max(1, Math.floor(rect.width * dpr))
        canvas.height = Math.max(1, Math.floor(rect.height * dpr))
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        redraw()
      } catch (err) { console.error('resize error', err) }
    }

    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // redraw strokes when strokes change (no backing-store resize)
  useEffect(() => {
    try {
      const canvas = canvasRef.current
      const ctx = ctxRef.current
      // debug: log strokes shape
      // eslint-disable-next-line no-console
      console.log('Canvas redraw props: strokes count=', strokes ? strokes.length : 0)
      if (strokes && strokes.length > 0) {
        // log first stroke info
        // eslint-disable-next-line no-console
        console.log(' first stroke points=', strokes[0].points ? strokes[0].points.length : 0)
      }
      if (!canvas || !ctx) { console.warn('redraw: missing canvas or ctx'); return }
      const cssW = canvas.clientWidth
      const cssH = canvas.clientHeight
      if (cssW === 0 || cssH === 0) { console.warn('redraw: client size zero', cssW, cssH); return }
      ctx.clearRect(0, 0, cssW, cssH)
      for (const s of strokes) drawStroke(ctx, s)
      console.log('redraw: strokes', strokes.length)
    } catch (err) { console.error('strokes redraw error', err) }
  }, [strokes])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = ctxRef.current

    const getPoint = (e) => {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      return {
        x,
        y,
        p: e.pressure ?? 0.5,
        t: Date.now(),
        pointerType: e.pointerType || 'unknown'
      }
    }

    const onPointerDown = (e) => {
      if (e.pointerId != null && canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId)
      drawing.current = true
      const pt = getPoint(e)
      const strokeSize = tool === 'eraser' ? eraserSize : size
      current.current = { tool, color, size: strokeSize, points: [pt], _buffer: [pt] }
      console.log('pointerdown', pt)
    }

    const onPointerMove = (e) => {
      try {
        if (!drawing.current || !current.current) return
        const pt = getPoint(e)
        const buf = current.current._buffer || []
        buf.push(pt)
        const maxBuf = Math.max(1, Math.min(50, Math.floor(stabilizerLevel)))
        while (buf.length > maxBuf) buf.shift()
        current.current._buffer = buf

        // compute smoothed x/y but keep pressure and pointerType from current event
        let outPt = pt
        if (stabilizerEnabled) {
          let sx = 0, sy = 0
          for (let i = 0; i < buf.length; i++) { sx += buf[i].x; sy += buf[i].y }
          outPt = { x: sx / buf.length, y: sy / buf.length, p: pt.p, t: Date.now(), pointerType: pt.pointerType }
        }

        current.current.points.push(outPt)

        const cssW = canvas.clientWidth
        const cssH = canvas.clientHeight
        if (cssW === 0 || cssH === 0) { console.warn('pointerMove: client size zero'); return }
        ctx.clearRect(0, 0, cssW, cssH)
        for (const s of strokes) drawStroke(ctx, s)
        drawStroke(ctx, current.current)
      } catch (err) { console.error('pointermove error', err) }
    }

    const onPointerUp = (e) => {
      try {
        if (!drawing.current) return
        drawing.current = false
        if (!current.current) return
        // debug log
        // eslint-disable-next-line no-console
        console.log('pointerup: adding stroke, points:', current.current.points ? current.current.points.length : 0)
        // only add non-empty strokes
        if (current.current.points && current.current.points.length > 0) {
          if (typeof addStroke === 'function') {
            addStroke(pageIndex, current.current)
            // eslint-disable-next-line no-console
            console.log('Canvas addStroke called for page', pageIndex)
            // notify parent of new strokes list if requested
            try {
              if (typeof onStrokesChange === 'function') onStrokesChange([...(strokes || []), current.current])
            } catch (err) { /* ignore */ }
          } else {
            console.error('Canvas: addStroke prop is required but not provided')
          }
        } else {
          // eslint-disable-next-line no-console
          console.log('pointerup: discarded empty stroke')
        }
        current.current = null
        if (e.pointerId != null && canvas.releasePointerCapture) canvas.releasePointerCapture(e.pointerId)
      } catch (err) { console.error('pointerup error', err) }
    }

    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)

    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    }
  }, [tool, color, size, eraserSize, addStroke, pageIndex, stabilizerEnabled, stabilizerLevel, strokes])

  // expose captureFrame via forwarded ref
  useImperativeHandle(forwardedRef, () => ({
    captureFrame: () => {
      return new Promise((resolve, reject) => {
        try {
          const canvas = canvasRef.current
          if (!canvas) return reject(new Error('Canvas not mounted'))
          canvas.toBlob((blob) => {
            if (blob) resolve(blob)
            else reject(new Error('toBlob returned null'))
          }, 'image/png')
        } catch (err) { reject(err) }
      })
    }
    ,
    getCanvas: () => canvasRef.current,
    getSize: () => {
      const c = canvasRef.current
      if (!c) return { width: 0, height: 0 }
      return { width: c.clientWidth || c.width || 0, height: c.clientHeight || c.height || 0 }
    }
  }))

  return (
    <canvas
      id={id}
      ref={canvasRef}
      className={className}
      style={{ width: '100%', height: '100%', touchAction: 'none' }}
    />
  )
}

export default React.forwardRef(Canvas)
