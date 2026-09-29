import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

const CANVAS_SIZE = 280;
const STROKE_WIDTH = 18;

/**
 * A freehand drawing surface for sketching a single digit.
 * Exposes imperative methods (via ref) to clear the canvas and export
 * its contents as a base64 PNG for the backend API.
 */
const DrawingCanvas = forwardRef(function DrawingCanvas(
  { onStrokeStart, disabled },
  ref
) {
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const lastPoint = useRef(null);
  const [hasDrawing, setHasDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#111318";
    ctx.lineWidth = STROKE_WIDTH;
  }, []);

  useImperativeHandle(ref, () => ({
    clear() {
      const ctx = canvasRef.current.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      setHasDrawing(false);
    },
    isEmpty() {
      return !hasDrawing;
    },
    exportBase64() {
      return canvasRef.current.toDataURL("image/png");
    },
  }));

  const getPoint = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    const scaleX = CANVAS_SIZE / rect.width;
    const scaleY = CANVAS_SIZE / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const startDraw = (e) => {
    if (disabled) return;
    e.preventDefault();
    isDrawing.current = true;
    lastPoint.current = getPoint(e);
    setHasDrawing(true);
    onStrokeStart?.();
  };

  const draw = (e) => {
    if (!isDrawing.current || disabled) return;
    e.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const point = getPoint(e);
    ctx.beginPath();
    ctx.moveTo(lastPoint.current.x, lastPoint.current.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint.current = point;
  };

  const endDraw = () => {
    isDrawing.current = false;
    lastPoint.current = null;
  };

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_SIZE}
      height={CANVAS_SIZE}
      className={`draw-canvas${disabled ? " draw-canvas--disabled" : ""}`}
      onMouseDown={startDraw}
      onMouseMove={draw}
      onMouseUp={endDraw}
      onMouseLeave={endDraw}
      onTouchStart={startDraw}
      onTouchMove={draw}
      onTouchEnd={endDraw}
    />
  );
});

export default DrawingCanvas;
