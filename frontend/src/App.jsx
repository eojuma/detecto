import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import DetectionPage from './pages/DetectionPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'

// App is a pure layout component: it owns the chrome (brand + nav) and the
// route table. Routing itself is provided by BrowserRouter in main.jsx.
//
// NavLink exposes an isActive flag, so the active tab is styled without any
// manual location tracking.
function navClass({ isActive }) {
  return isActive ? 'active' : ''
}

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-nav">
        <div className="brand">
          de<span>tecto</span>
        </div>
        <nav>
          <NavLink to="/detect" className={navClass}>
            Detection
          </NavLink>
          <NavLink to="/history" className={navClass}>
            History
          </NavLink>
        </nav>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/detect" replace />} />
          <Route path="/detect" element={<DetectionPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<Navigate to="/detect" replace />} />
        </Routes>
      </main>
    </div>
  )
}
