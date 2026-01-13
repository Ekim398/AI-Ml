//
import React, { useState, useRef } from 'react'
import Canvas from './components/Canvas'
import Toolbar from './components/Toolbar'
import PagesSidebar from './components/PagesSidebar'
import { ocrFrame, requestHintsSimple } from "./api";

export default function App() {
  // pages: array of strokes arrays -> pages[0] = []
  const [pages, setPages] = useState([[]])
  const [currentPage, setCurrentPage] = useState(0)
  const [tool, setTool] = useState('pen')
  const [color, setColor] = useState('#000000')
  const [size, setSize] = useState(3)
  const [eraserSize, setEraserSize] = useState(16)
  const [stabilizerEnabled, setStabilizerEnabled] = useState(true)
  const [stabilizerLevel, setStabilizerLevel] = useState(4)
  const canvasRef = useRef(null)
  const [ocrResult, setOcrResult] = useState(null)
  const [hintsResult, setHintsResult] = useState(null)
  const [hintsVisible, setHintsVisible] = useState(false)
  const OCR_SCALE = 2

  async function handleExportPdf() {
    try {
      if (!canvasRef.current || typeof canvasRef.current.captureFrame !== 'function') {
        console.warn('Canvas captureFrame not available')
        return
      }

      const pngBlob = await canvasRef.current.captureFrame()
      const bitmap = await createImageBitmap(pngBlob)
      const w = bitmap.width
      const h = bitmap.height
      const c = document.createElement('canvas')
      c.width = w
      c.height = h
      const ctx = c.getContext('2d')
      ctx.drawImage(bitmap, 0, 0)

      const jpegBlob = await new Promise((resolve) =>
        c.toBlob((b) => resolve(b), 'image/jpeg', 0.92)
      )
      if (!jpegBlob) throw new Error('Failed to encode PDF image')

      const pdfBlob = await buildPdfFromJpeg(jpegBlob, w, h)
      const url = URL.createObjectURL(pdfBlob)
      const a = document.createElement('a')
      a.href = url
      a.download = `notebook-page-${currentPage + 1}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (e) {
      console.error('export pdf error', e)
    }
  }

  async function buildPdfFromJpeg(jpegBlob, width, height) {
    const jpgBytes = new Uint8Array(await jpegBlob.arrayBuffer())
    const encoder = new TextEncoder()
    const parts = []
    const offsets = [0]
    let offset = 0

    const pushBytes = (bytes) => {
      parts.push(bytes)
      offset += bytes.length
    }

    const pushStr = (str) => pushBytes(encoder.encode(str))

    const concat = (...arrays) => {
      const total = arrays.reduce((sum, a) => sum + a.length, 0)
      const out = new Uint8Array(total)
      let at = 0
      arrays.forEach((a) => {
        out.set(a, at)
        at += a.length
      })
      return out
    }

    const addObject = (id, content) => {
      offsets[id] = offset
      pushStr(`${id} 0 obj\n`)
      if (typeof content === 'string') pushStr(content)
      else pushBytes(content)
      pushStr('\nendobj\n')
    }

    pushStr('%PDF-1.3\n')

    addObject(1, '<< /Type /Catalog /Pages 2 0 R >>')
    addObject(2, '<< /Type /Pages /Kids [3 0 R] /Count 1 >>')
    addObject(
      3,
      `<< /Type /Page /Parent 2 0 R /Resources << /XObject << /Im0 4 0 R >> /ProcSet [/PDF /ImageC] >> /MediaBox [0 0 ${width} ${height}] /Contents 5 0 R >>`
    )

    const imgHeader = encoder.encode(
      `<< /Type /XObject /Subtype /Image /Width ${width} /Height ${height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpgBytes.length} >>\nstream\n`
    )
    const imgFooter = encoder.encode('\nendstream')
    addObject(4, concat(imgHeader, jpgBytes, imgFooter))

    const contentStream = `q ${width} 0 0 ${height} 0 0 cm /Im0 Do Q`
    addObject(
      5,
      `<< /Length ${contentStream.length} >>\nstream\n${contentStream}\nendstream`
    )

    const xrefOffset = offset
    const count = offsets.length
    pushStr(`xref\n0 ${count}\n`)
    pushStr('0000000000 65535 f \n')
    for (let i = 1; i < count; i++) {
      const off = String(offsets[i]).padStart(10, '0')
      pushStr(`${off} 00000 n \n`)
    }
    pushStr(`trailer\n<< /Size ${count} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`)

    return new Blob(parts, { type: 'application/pdf' })
  }

  function canvasToOcrBlob(canvas, scale = 3) {
    return new Promise((resolve, reject) => {
      if (!canvas) return reject(new Error("canvas is null"));

      const out = document.createElement("canvas");
      out.width = Math.floor(canvas.width * scale);
      out.height = Math.floor(canvas.height * scale);

      const ctx = out.getContext("2d");
      if (!ctx) return reject(new Error("2d context not available"));

      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, out.width, out.height);

      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(canvas, 0, 0);

      out.toBlob((blob) => {
        if (!blob) return reject(new Error("toBlob failed"));
        resolve(blob);
      }, "image/png");
    });
  }


  async function handleSendToOCR() {
  try {
    if (!canvasRef.current || typeof canvasRef.current.getCanvas !== "function") {
      console.warn("Canvas getCanvas not available");
      return;
    }

    const drawingCanvas = canvasRef.current.getCanvas();
    if (!drawingCanvas) {
      console.warn("Canvas element not available");
      return;
    }

    const raw = await canvasToOcrBlob(drawingCanvas, OCR_SCALE);
    console.log("blob type:", raw.type, "bytes:", raw.size);
    console.log("raw frame bytes:", raw.size);

    const resp = await ocrFrame(raw);
    setOcrResult(resp);

    const extractText = (ocr) => {
      if (!ocr) return "";
      if (typeof ocr.full_text === "string") return ocr.full_text;
      if (typeof ocr.text === "string") return ocr.text;
      if (Array.isArray(ocr.text_annotations) && ocr.text_annotations[0]?.description) {
        return ocr.text_annotations[0].description;
      }
      if (Array.isArray(ocr.lines)) {
        return ocr.lines.map((l) => l.text || l.label || "").filter(Boolean).join("\n");
      }
      if (Array.isArray(ocr.words)) {
        return ocr.words.map((w) => w.text || w.label || "").filter(Boolean).join(" ");
      }
      return "";
    };

    const fullText = extractText(resp);
    const hints = await requestHintsSimple({ fullText, studentGoal: null, angleUnits: null });
    setHintsResult(hints);
    setHintsVisible(true);
  } catch (e) {
    console.error("send to OCR error", e);
    setOcrResult({ error: String(e) });
    setHintsResult(null);
  }
}

  const ocrBubbles = (() => {
    if (!ocrResult) return []

    const toArray = (value) => (Array.isArray(value) ? value : [])

    const candidates = []
    if (Array.isArray(ocrResult)) candidates.push(...ocrResult)
    if (ocrResult.annotations) candidates.push(...toArray(ocrResult.annotations))
    if (ocrResult.words) candidates.push(...toArray(ocrResult.words))
    if (ocrResult.lines) candidates.push(...toArray(ocrResult.lines))
    if (ocrResult.results) candidates.push(...toArray(ocrResult.results))
    if (ocrResult.text_annotations) candidates.push(...toArray(ocrResult.text_annotations))

    const parseBox = (item) => {
      if (!item) return null
      if (item.box && typeof item.box === 'object') {
        const { x, y, w, h } = item.box
        if ([x, y, w, h].every((v) => typeof v === 'number')) return { x, y, w, h }
      }
      if (item.bbox && Array.isArray(item.bbox) && item.bbox.length >= 4) {
        const [a, b, c, d] = item.bbox
        if ([a, b, c, d].every((v) => typeof v === 'number')) {
          let x = a
          let y = b
          let w = c
          let h = d
          if (c > a && d > b) {
            const w2 = c - a
            const h2 = d - b
            if (w2 > 1 && h2 > 1) {
              w = w2
              h = h2
            }
          }
          return { x, y, w, h }
        }
      }
      if (item.boundingBox && item.boundingBox.vertices) {
        const verts = item.boundingBox.vertices
        if (Array.isArray(verts) && verts.length) {
          const xs = verts.map((v) => v.x).filter((v) => typeof v === 'number')
          const ys = verts.map((v) => v.y).filter((v) => typeof v === 'number')
          if (xs.length && ys.length) {
            const x = Math.min(...xs)
            const y = Math.min(...ys)
            const w = Math.max(...xs) - x
            const h = Math.max(...ys) - y
            return { x, y, w, h }
          }
        }
      }
      return null
    }

    return candidates
      .map((item) => {
        const text = item.text || item.label || item.description || item.value || ''
        const box = parseBox(item)
        if (!text || !box) return null
        return { text, ...box }
      })
      .filter(Boolean)
  })()


  const addStrokeToPage = (pageIndex, stroke) => {
    console.log('addStrokeToPage', pageIndex, stroke)
    setPages((prev) => {
      const next = prev.map((p, i) => (i === pageIndex ? [...(p || []), stroke] : p))
      return next
    })
  }

  const undo = () => {
    setPages((prev) =>
      prev.map((p, i) => {
        if (i !== currentPage) return p
        if (!p || p.length === 0) return p
        const next = p.slice()
        while (next.length > 0 && (!next[next.length - 1]?.points || next[next.length - 1].points.length === 0)) {
          next.pop()
        }
        if (next.length > 0) next.pop()
        return next
      })
    )
  }

  const clearCurrentPage = () => {
    setPages(prev =>
      prev.map((p, i) => (i === currentPage ? [] : p))
    )
  }

  const addPage = () => {
    setPages(prev => [...prev, []])
    setCurrentPage(pages.length)
  }

  const deletePage = (index) => {
    if (pages.length === 1) return
    setPages(prev => {
      const next = prev.slice(0, index).concat(prev.slice(index + 1))
      if (currentPage >= next.length) setCurrentPage(Math.max(0, next.length - 1))
      return next
    })
  }

  return (
    <div className="app">
      <div className="app-shell">
        <PagesSidebar
          pages={pages}
          currentPage={currentPage}
          setCurrentPage={setCurrentPage}
          addPage={addPage}
          deletePage={deletePage}
        />
        <div className="workspace">
          <div className="toolbar-wrap">
            <Toolbar
              tool={tool}
              setTool={setTool}
              color={color}
              setColor={setColor}
              size={size}
              setSize={setSize}
              eraserSize={eraserSize}
              setEraserSize={setEraserSize}
              undo={undo}
              clearAll={clearCurrentPage}
              stabilizerEnabled={stabilizerEnabled}
              setStabilizerEnabled={setStabilizerEnabled}
              stabilizerLevel={stabilizerLevel}
              setStabilizerLevel={setStabilizerLevel}
              onExport={handleExportPdf}
              onSendOcr={handleSendToOCR}
              onToggleHints={() => setHintsVisible((prev) => !prev)}
              hintsVisible={hintsVisible}
            />
          </div>
          {ocrResult && ocrResult.error && (
            <div className="api-panel">
              <div className="api-response error">OCR error: {ocrResult.error}</div>
            </div>
          )}
          {hintsVisible && Array.isArray(hintsResult?.hints) && hintsResult.hints.length > 0 && (
            <div className="api-panel hints-panel">
              <div className="ocr-title">Hints</div>
              <div className="hints-list">
                {hintsResult.hints.map((hint, i) => (
                  <div key={`hint-list-${i}`} className="hint-list-item">
                    <div className="hint-title">{hint.title || `Hint ${i + 1}`}</div>
                    <div className="hint-text">{hint.hint || hint.text || ''}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="canvas-stage">
            {hintsVisible && Array.isArray(hintsResult?.hints) && hintsResult.hints.map((hint, i) => {
              const box = hint.focus_box || hint.bbox || hint.box
              let x = null
              let y = null
              let w = null
              let h = null
              if (Array.isArray(box) && box.length >= 4) {
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
              } else if (box && typeof box === 'object') {
                x = box.x
                y = box.y
                w = box.w
                h = box.h
              }
              if (![x, y, w, h].every((v) => typeof v === 'number')) return null
              const canvasEl = canvasRef.current && typeof canvasRef.current.getCanvas === 'function'
                ? canvasRef.current.getCanvas()
                : null
              const cssW = canvasEl ? (canvasEl.clientWidth || canvasEl.width) : 1
              const cssH = canvasEl ? (canvasEl.clientHeight || canvasEl.height) : 1
              const ocrW = canvasEl ? (canvasEl.width * OCR_SCALE) : 1
              const ocrH = canvasEl ? (canvasEl.height * OCR_SCALE) : 1
              const sx = ocrW ? cssW / ocrW : 1
              const sy = ocrH ? cssH / ocrH : 1
              const mapCoord = (v, s) => (v > 0 && v <= 1 ? `${v * 100}%` : `${v * s}px`)
              const style = {
                position: 'absolute',
                left: mapCoord(x, sx),
                top: mapCoord(y, sy),
                width: mapCoord(w, sx),
                height: mapCoord(h, sy),
                pointerEvents: 'none'
              }
              return (
                <div key={`hint-${i}`} className="hint-anchor" style={style}>
                  <div className="hint-bubble">
                    <div className="hint-title">{hint.title || `Hint ${i + 1}`}</div>
                    <div className="hint-text">{hint.hint || hint.text || ''}</div>
                  </div>
                </div>
              )
            })}
            {ocrBubbles.map((b, i) => {
              const toCss = (v) => (typeof v === 'number' ? (v > 0 && v <= 1 ? `${v * 100}%` : `${v}px`) : '0px')
              const style = {
                position: 'absolute',
                left: toCss(b.x),
                top: toCss(b.y),
                width: toCss(b.w),
                height: toCss(b.h),
                pointerEvents: 'none'
              }
              return (
                <div key={`ocr-${i}`} className="ocr-anchor" style={style}>
                  <div className="ocr-bubble">
                    <span>{b.text}</span>
                  </div>
                </div>
              )
            })}
            <Canvas
              ref={canvasRef}
              className="notebook-canvas"
              strokes={pages[currentPage] || []}
              addStroke={addStrokeToPage}
              pageIndex={currentPage}
              tool={tool}
              color={color}
              size={size}
              eraserSize={eraserSize}
              stabilizerEnabled={stabilizerEnabled}
              stabilizerLevel={stabilizerLevel}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
