# React Notebook App (Vite)
Files:
src/
├── App.jsx          ← root component (stays here)
├── main.jsx         ← entry point
├── api.js           ← API calls
└── components/
    ├── Canvas.jsx
    ├── Toolbar.jsx
    └── PagesSidebar.jsx

Features:
- Draw with pen and eraser
- Multiple pages
- Undo / Clear
- Export current page as PNG

Handwriting stabilizer:
- Toggle `Stabilizer` in the toolbar to enable smoothing of pointer input.
- Adjust `Level` to change smoothing strength (higher = smoother, more latency).

Files of interest:
- [React/src/components/Canvas.jsx](React/src/components/Canvas.jsx)
- [React/src/components/Toolbar.jsx](React/src/components/Toolbar.jsx)
- [React/src/components/PagesSidebar.jsx](React/src/components/PagesSidebar.jsx)
- [React/src/App.jsx](React/src/App.jsx)

Run:

```bash
cd React
npm install
npm run dev
```
