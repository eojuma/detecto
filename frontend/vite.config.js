import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Why port 5173 explicitly?
//   The backend's CORS_ORIGINS allow-list contains http://localhost:5173.
//   If Vite chose a different port, every API call would be blocked by CORS.
//
// Why no dev proxy?
//   Proxying would bypass CORS and hide a misconfigured backend until
//   production. Calling the API directly (via VITE_API_BASE_URL) means the
//   browser exercises the real CORS path during development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
})
