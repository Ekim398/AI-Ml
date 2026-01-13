//pages 
import React from 'react'

export default function PagesSidebar({ pages, currentPage, setCurrentPage, addPage, deletePage }) {
  return (
    <aside className="pages">
      <div className="pages-header">
        <strong>Pages</strong>
        <button onClick={addPage}>+</button>
      </div>
      <ul>
        {pages.map((p, i) => {
          const count = Array.isArray(p) ? p.length : (p && p.strokes ? p.strokes.length : 0)
          return (
            <li key={(p && p.id) || i} className={i === currentPage ? 'active' : ''} onClick={() => setCurrentPage(i)}>
              <div className="thumb">Page {i + 1} ({count})</div>
              <button className="del" onClick={(e) => { e.stopPropagation(); deletePage(i) }}>Delete</button>
            </li>
          )
        })}
      </ul>
    </aside>
  )
}
