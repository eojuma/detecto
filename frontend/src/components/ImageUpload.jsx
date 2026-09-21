import { useCallback, useRef, useState } from 'react'

// Client-side limits. These MIRROR the server rules for fast feedback, but the
// server remains the source of truth (it enforces its own MAX_UPLOAD_MB and
// magic-byte checks). Keep these in sync via VITE_MAX_UPLOAD_MB if you change
// the backend cap.
const ALLOWED_TYPES = ['image/jpeg', 'image/png']
const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB || 10)
const MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

/**
 * Drag-and-drop + file-picker input.
 *
 * Emits a validated File via onSelect. It deliberately does NOT render the
 * preview: the parent owns the displayed <img> and the canvas overlay so both
 * share identical geometry. Validation errors are rendered here as UI.
 *
 * @param {{ onSelect: (file: File) => void, disabled?: boolean, selectedName?: string }} props
 */
export default function ImageUpload({ onSelect, disabled = false, selectedName = null }) {
  const inputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)

  const validateAndEmit = useCallback(
    (file) => {
      if (!file) return
      if (!ALLOWED_TYPES.includes(file.type)) {
        setError(`Unsupported file type "${file.type || 'unknown'}". Use a JPEG or PNG.`)
        return
      }
      if (file.size === 0) {
        setError('That file is empty.')
        return
      }
      if (file.size > MAX_BYTES) {
        setError(
          `File is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is ${MAX_UPLOAD_MB} MB.`
        )
        return
      }
      setError(null)
      onSelect(file)
    },
    [onSelect]
  )

  function handleDrop(event) {
    event.preventDefault()
    setDragActive(false)
    if (disabled) return
    validateAndEmit(event.dataTransfer.files?.[0])
  }

  function handleChange(event) {
    validateAndEmit(event.target.files?.[0])
    // Reset the input value so selecting the same file again re-triggers change.
    event.target.value = ''
  }

  return (
    <div>
      <label
        className={`dropzone${dragActive ? ' drag' : ''}${disabled ? ' disabled' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_TYPES.join(',')}
          onChange={handleChange}
          disabled={disabled}
          hidden
        />
        <div className="dropzone-title">
          {selectedName ? (
            <>
              Selected: <strong>{selectedName}</strong>
            </>
          ) : (
            <>
              Drag &amp; drop an image here, or <span className="link">browse</span>
            </>
          )}
        </div>
        <div className="dropzone-hint">
          JPEG or PNG · up to {MAX_UPLOAD_MB} MB
          {selectedName ? ' · click or drop to replace' : ''}
        </div>
      </label>

      {/* Validation feedback is rendered as UI, never logged to the console. */}
      {error && (
        <div className="field-error" role="alert">
          {error}
        </div>
      )}
    </div>
  )
}
