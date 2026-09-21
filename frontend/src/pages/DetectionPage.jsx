import { useCallback, useEffect, useState } from 'react'
import { detect as detectApi } from '../api.js'
import ImageUpload from '../components/ImageUpload.jsx'
import BoxOverlay from '../components/BoxOverlay.jsx'
import DetectionPanel from '../components/DetectionPanel.jsx'

// DetectionPage owns the shared state and calls the API. The three children
// stay presentational, which keeps this file the single place where the
// request lifecycle lives.
export default function DetectionPage() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Create an object URL for the selected file, and revoke it on change or
  // unmount. Without the cleanup, every selection would leak a blob URL.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const handleSelect = useCallback((selected) => {
    setFile(selected)
    // A new image invalidates any previous result/error.
    setResult(null)
    setError(null)
  }, [])

  const handleDetect = useCallback(async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // annotated=false: we draw boxes client-side, so the server need not
      // spend CPU rendering them too.
      const data = await detectApi(file, { annotated: false })
      setResult(data)
    } catch (err) {
      // err is an ApiError from src/api.js; rendered by DetectionPanel.
      setError(err)
    } finally {
      setLoading(false)
    }
  }, [file])

  return (
    <div>
      <h1 className="page-title">Detection View</h1>
      <p className="page-subtitle">
        Upload an image to detect and count people.
      </p>

      <div className="detect-layout">
        <div className="detect-left">
          <ImageUpload
            onSelect={handleSelect}
            disabled={loading}
            selectedName={file?.name || null}
          />

          <div className="actions">
            <button type="button" onClick={handleDetect} disabled={!file || loading}>
              {loading ? 'Detecting…' : 'Detect people'}
            </button>
          </div>

          {/* Preview shows immediately after selection; boxes appear once a
              result exists. Before that, dimensions fall back to the image's
              natural size inside BoxOverlay. */}
          {previewUrl && (
            <BoxOverlay
              imageUrl={previewUrl}
              boxes={result?.boxes || []}
              imageWidth={result?.image_width || 0}
              imageHeight={result?.image_height || 0}
              alt={file?.name || 'Uploaded image'}
            />
          )}
        </div>

        <div className="detect-right">
          <DetectionPanel result={result} loading={loading} error={error} />
        </div>
      </div>
    </div>
  )
}
