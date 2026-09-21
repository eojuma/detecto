import { useCallback, useState } from 'react'

// BoxOverlay renders the image and its bounding boxes in ONE shared container.
//
// Why inline SVG (not canvas / positioned divs)?
//   The backend returns boxes in ORIGINAL image pixel coordinates. Setting the
//   SVG viewBox to `0 0 origW origH` makes the SVG's coordinate system identical
//   to the image's pixel space, so the browser scales boxes to any display size
//   for free -- no scale factors, no devicePixelRatio handling, always sharp.
//
// Why refs/state here?
//   We need the image's natural size to build the viewBox. The parent passes
//   the dimensions from the detection response when available; otherwise we
//   read them from the <img> on load.

const BOX_COLOR = '#3fb950'

/**
 * Lay out confidence labels so they do not overlap each other.
 *
 * Boxes are processed top-to-bottom. If a label would overlap one already
 * placed (and they share horizontal space), it is pushed above that label.
 * O(n^2) but n is small (<= ~50).
 */
function layoutLabels(boxes, fontSize, imageWidth, imageHeight) {
  const labelHeight = fontSize * 1.25
  const pad = fontSize * 0.2

  const ordered = boxes
    .map((box, index) => ({ box, index }))
    .sort((a, b) => a.box.y1 - b.box.y1)

  const placed = []
  const labels = new Array(boxes.length)

  for (const { box, index } of ordered) {
    const text = box.confidence.toFixed(2)
    const textWidth = text.length * fontSize * 0.62 + pad * 2

    let x = box.x1
    if (x + textWidth > imageWidth) x = Math.max(0, imageWidth - textWidth)

    // Prefer just above the box; fall back to inside its top edge.
    let y = box.y1 - labelHeight
    if (y < 0) y = box.y1

    // Resolve collisions against horizontally-overlapping labels.
    let moved = true
    let guard = 0
    while (moved && guard < boxes.length + 2) {
      moved = false
      for (const rect of placed) {
        const hOverlap = x < rect.x + rect.w && x + textWidth > rect.x
        const vOverlap = y < rect.y + rect.h && y + labelHeight > rect.y
        if (hOverlap && vOverlap) {
          y = rect.y - labelHeight - pad
          moved = true
        }
      }
      guard += 1
    }
    if (y < 0) y = 0

    const rect = { x, y, w: textWidth, h: labelHeight }
    placed.push(rect)
    labels[index] = { text, rect, textX: x + pad, textY: y + labelHeight - pad }
  }

  return { labels, labelHeight }
}

/**
 * @param {{
 *   imageUrl: string,
 *   boxes?: Array<{x1:number,y1:number,x2:number,y2:number,confidence:number}>,
 *   imageWidth?: number,
 *   imageHeight?: number,
 *   alt?: string,
 * }} props
 */
export default function BoxOverlay({
  imageUrl,
  boxes = [],
  imageWidth = 0,
  imageHeight = 0,
  alt = 'Uploaded image',
}) {
  const [natural, setNatural] = useState(null)

  const handleLoad = useCallback((event) => {
    setNatural({
      width: event.target.naturalWidth,
      height: event.target.naturalHeight,
    })
  }, [])

  // Prefer the dimensions the API reported (they match the detection frame);
  // fall back to the element's natural size for the pre-detection preview.
  const width = imageWidth || natural?.width || 0
  const height = imageHeight || natural?.height || 0
  const hasGeometry = width > 0 && height > 0

  const fontSize = hasGeometry ? Math.max(12, Math.min(width, height) * 0.028) : 16
  const { labels, labelHeight } = hasGeometry
    ? layoutLabels(boxes, fontSize, width, height)
    : { labels: [], labelHeight: 0 }

  return (
    <div className="overlay-wrap">
      <img
        className="overlay-img"
        src={imageUrl}
        alt={alt}
        onLoad={handleLoad}
        draggable={false}
      />

      {hasGeometry && boxes.length > 0 && (
        <svg
          className="overlay-svg"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`${boxes.length} person${boxes.length === 1 ? '' : 's'} detected`}
        >
          {boxes.map((box, index) => (
            <rect
              key={`box-${index}`}
              className="box-rect"
              x={box.x1}
              y={box.y1}
              width={Math.max(0, box.x2 - box.x1)}
              height={Math.max(0, box.y2 - box.y1)}
              stroke={BOX_COLOR}
              // non-scaling-stroke keeps the outline a crisp constant width
              // regardless of how large the image is displayed.
              vectorEffect="non-scaling-stroke"
            >
              <title>{`Person ${index + 1} — confidence ${box.confidence.toFixed(2)}`}</title>
            </rect>
          ))}

          {labels.map((label, index) => (
            <g key={`label-${index}`}>
              <rect
                className="box-label-plate"
                x={label.rect.x}
                y={label.rect.y}
                width={label.rect.w}
                height={label.rect.h}
                rx={labelHeight * 0.15}
                fill={BOX_COLOR}
              />
              <text
                className="box-label-text"
                x={label.textX}
                y={label.textY}
                fontSize={fontSize}
              >
                {label.text}
              </text>
            </g>
          ))}
        </svg>
      )}
    </div>
  )
}
