import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './index.css'

// React 18 entry point. createRoot replaces the legacy ReactDOM.render.
//
// BrowserRouter lives here (not in App.jsx) so the router owns URL state for
// both pages while App.jsx stays a pure layout component.
//
// StrictMode intentionally double-invokes effects in development. Our canvas
// draw effect is written to be idempotent, so this only helps catch mistakes.
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
