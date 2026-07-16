import { useEffect, useRef, useCallback, useState } from "react";
import { useStore } from "../store";
import { doc } from "../lib/document";
import { imageDataToCanvas } from "../lib/imaging";

type Handle = "nw" | "ne" | "sw" | "se" | null;

const SNAP = 64;

export default function Canvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const raf = useRef(0);

  const view = useStore((s) => s.view);
  const frame = useStore((s) => s.frame);
  const docRev = useStore((s) => s.docRev);
  const busy = useStore((s) => s.busy);
  const progress = useStore((s) => s.progress);
  const stamp = useStore((s) => s.stamp);
  const commitStamp = useStore((s) => s.commitStamp);
  const cancelStamp = useStore((s) => s.cancelStamp);
  const flipStamp = useStore((s) => s.flipStamp);
  const fitStampToFrame = useStore((s) => s.fitStampToFrame);
  const stampFromFile = useStore((s) => s.stampFromFile);

  const [dragOver, setDragOver] = useState(false);

  // Cached HTMLCanvas of the stamp's source pixels (rebuilt only when the
  // underlying image changes, not on every move/scale).
  const stampCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stampImgRef = useRef<ImageData | null>(null);
  if (stamp && stampImgRef.current !== stamp.img) {
    stampImgRef.current = stamp.img;
    stampCanvasRef.current = imageDataToCanvas(stamp.img);
  } else if (!stamp && stampImgRef.current) {
    stampImgRef.current = null;
    stampCanvasRef.current = null;
  }

  // stable getters
  const store = useStore;

  const draw = useCallback(() => {
    const cv = ref.current;
    const wrap = wrapRef.current;
    if (!cv || !wrap) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = wrap.clientWidth;
    const H = wrap.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) {
      cv.width = W * dpr;
      cv.height = H * dpr;
      cv.style.width = W + "px";
      cv.style.height = H + "px";
    }
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const { x: vx, y: vy, scale } = store.getState().view;
    const w2s = (wx: number, wy: number): [number, number] => [
      (wx - vx) * scale + W / 2,
      (wy - vy) * scale + H / 2,
    ];

    // --- backdrop: deep aurora + dotted grid ---
    const g = ctx.createRadialGradient(W / 2, H * 0.4, 40, W / 2, H * 0.5, Math.max(W, H) * 0.85);
    g.addColorStop(0, "#15120f");
    g.addColorStop(1, "#0a0908");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    const step = 48 * scale;
    if (step > 14) {
      const ox = ((-vx * scale + W / 2) % step + step) % step;
      const oy = ((-vy * scale + H / 2) % step + step) % step;
      ctx.fillStyle = "rgba(233,196,106,0.05)";
      for (let x = ox; x < W; x += step)
        for (let y = oy; y < H; y += step) ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
    }

    // --- world content ---
    const b = doc.bounds;
    const [bx, by] = w2s(b.x, b.y);
    ctx.imageSmoothingEnabled = scale < 3;
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 40;
    ctx.drawImage(doc.world, bx, by, b.w * scale, b.h * scale);
    ctx.restore();

    // --- outpaint frame ---
    const f = store.getState().frame;
    const [fx, fy] = w2s(f.x, f.y);
    const fw = f.w * scale;
    const fh = f.h * scale;

    // dim outside frame slightly for focus
    ctx.save();
    ctx.fillStyle = "rgba(10,9,8,0.28)";
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.rect(fx, fy, fw, fh);
    ctx.fill("evenodd");
    ctx.restore();

    // frame border — gilded, animated dash while busy
    const st = store.getState();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = st.busy ? "#f4d98b" : "#e9c46a";
    if (st.busy) {
      ctx.setLineDash([9, 7]);
      ctx.lineDashOffset = -(performance.now() / 40) % 16;
    } else {
      ctx.setLineDash([]);
    }
    ctx.strokeRect(fx + 0.5, fy + 0.5, fw, fh);
    ctx.setLineDash([]);

    // corner handles
    const hs = 7;
    ctx.fillStyle = "#14110f";
    ctx.strokeStyle = "#e9c46a";
    ctx.lineWidth = 1.5;
    for (const [hx, hy] of [
      [fx, fy], [fx + fw, fy], [fx, fy + fh], [fx + fw, fy + fh],
    ]) {
      ctx.beginPath();
      ctx.rect(hx - hs / 2, hy - hs / 2, hs, hs);
      ctx.fill();
      ctx.stroke();
    }

    // frame label
    ctx.font = "500 11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(216,205,182,0.8)";
    ctx.textBaseline = "bottom";
    ctx.fillText(`${f.w}×${f.h}`, fx + 2, fy - 6);

    // busy overlay glow inside frame
    if (st.busy) {
      const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 400);
      ctx.save();
      ctx.globalAlpha = 0.12 + pulse * 0.1;
      const fg = ctx.createLinearGradient(fx, fy, fx, fy + fh);
      fg.addColorStop(0, "rgba(233,196,106,0.5)");
      fg.addColorStop(1, "rgba(233,196,106,0)");
      ctx.fillStyle = fg;
      ctx.fillRect(fx, fy, fw, fh);
      ctx.restore();
    }

    // --- floating stamp (interactive photo placement) ---
    const sp = st.stamp;
    const spCanvas = stampCanvasRef.current;
    if (sp && spCanvas) {
      // dim the rest of the scene to focus on the stamp being placed
      ctx.save();
      ctx.fillStyle = "rgba(10,9,8,0.5)";
      ctx.fillRect(0, 0, W, H);
      ctx.restore();

      const [sx, sy] = w2s(sp.x, sp.y);
      const sw = sp.w * scale;
      const sh = sp.h * scale;

      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.6)";
      ctx.shadowBlur = 30;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      if (sp.flipH) {
        ctx.translate(sx + sw, sy);
        ctx.scale(-1, 1);
        ctx.drawImage(spCanvas, 0, 0, sw, sh);
      } else {
        ctx.drawImage(spCanvas, sx, sy, sw, sh);
      }
      ctx.restore();

      // gilded selection border
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#f4d98b";
      ctx.setLineDash([]);
      ctx.strokeRect(sx + 0.5, sy + 0.5, sw, sh);

      // corner handles (round — distinct from the frame's square handles)
      const hr = 5;
      ctx.fillStyle = "#14110f";
      ctx.strokeStyle = "#f4d98b";
      ctx.lineWidth = 1.5;
      for (const [hx, hy] of [
        [sx, sy], [sx + sw, sy], [sx, sy + sh], [sx + sw, sy + sh],
      ]) {
        ctx.beginPath();
        ctx.arc(hx, hy, hr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }

      // size label
      ctx.font = "500 11px 'JetBrains Mono', monospace";
      ctx.fillStyle = "rgba(216,205,182,0.9)";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${Math.round(sp.w)}×${Math.round(sp.h)}`, sx + 2, sy - 6);
    }
  }, [store]);

  // animation loop only while busy (for marching ants); else draw on demand
  useEffect(() => {
    draw();
  }, [draw, view, frame, docRev, stamp]);

  // drag-and-drop a photo onto the canvas → start a stamp
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const onOver = (e: DragEvent) => {
      if (e.dataTransfer && Array.from(e.dataTransfer.types).includes("Files")) {
        e.preventDefault();
        setDragOver(true);
      }
    };
    const onLeave = (e: DragEvent) => {
      if (!wrap.contains(e.relatedTarget as Node)) setDragOver(false);
    };
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = Array.from(e.dataTransfer?.files || []).find((f) => f.type.startsWith("image/"));
      if (file) stampFromFile(file);
    };
    wrap.addEventListener("dragover", onOver);
    wrap.addEventListener("dragleave", onLeave);
    wrap.addEventListener("drop", onDrop);
    return () => {
      wrap.removeEventListener("dragover", onOver);
      wrap.removeEventListener("dragleave", onLeave);
      wrap.removeEventListener("drop", onDrop);
    };
  }, [stampFromFile]);

  // paste an image from the clipboard → start a stamp
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const it of items) {
        if (it.type.startsWith("image/")) {
          const blob = it.getAsFile();
          if (blob) {
            e.preventDefault();
            stampFromFile(blob);
          }
          break;
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [stampFromFile]);

  useEffect(() => {
    if (!busy) return;
    const loop = () => {
      draw();
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [busy, draw]);

  useEffect(() => {
    const ro = new ResizeObserver(() => draw());
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, [draw]);

  // ---- pointer interaction ----
  useEffect(() => {
    const cv = ref.current;
    const wrap = wrapRef.current;
    if (!cv || !wrap) return;

    let mode: "none" | "pan" | "move" | "resize" | "stampMove" | "stampResize" = "none";
    let handle: Handle = null;
    let last = { x: 0, y: 0 };
    let startFrame = { x: 0, y: 0, w: 0, h: 0 };
    let startStamp = { x: 0, y: 0, w: 0, h: 0 };

    const rectOf = () => wrap.getBoundingClientRect();
    const s2w = (sx: number, sy: number): [number, number] => {
      const st = store.getState();
      const r = rectOf();
      return [
        st.view.x + (sx - r.width / 2) / st.view.scale,
        st.view.y + (sy - r.height / 2) / st.view.scale,
      ];
    };
    const w2s = (wx: number, wy: number): [number, number] => {
      const st = store.getState();
      const r = rectOf();
      return [(wx - st.view.x) * st.view.scale + r.width / 2, (wy - st.view.y) * st.view.scale + r.height / 2];
    };

    const hitHandle = (sx: number, sy: number): Handle => {
      const f = store.getState().frame;
      const corners: [Handle, number, number][] = [
        ["nw", f.x, f.y], ["ne", f.x + f.w, f.y],
        ["sw", f.x, f.y + f.h], ["se", f.x + f.w, f.y + f.h],
      ];
      for (const [h, wx, wy] of corners) {
        const [cx, cy] = w2s(wx, wy);
        if (Math.abs(sx - cx) < 11 && Math.abs(sy - cy) < 11) return h;
      }
      return null;
    };
    const insideFrame = (sx: number, sy: number) => {
      const f = store.getState().frame;
      const [ax, ay] = w2s(f.x, f.y);
      const [bx, by] = w2s(f.x + f.w, f.y + f.h);
      return sx >= ax && sx <= bx && sy >= ay && sy <= by;
    };

    const stampHandleAt = (sx: number, sy: number): Handle => {
      const sp = store.getState().stamp;
      if (!sp) return null;
      const corners: [Handle, number, number][] = [
        ["nw", sp.x, sp.y], ["ne", sp.x + sp.w, sp.y],
        ["sw", sp.x, sp.y + sp.h], ["se", sp.x + sp.w, sp.y + sp.h],
      ];
      for (const [h, wx, wy] of corners) {
        const [cx, cy] = w2s(wx, wy);
        if (Math.abs(sx - cx) < 13 && Math.abs(sy - cy) < 13) return h;
      }
      return null;
    };
    const insideStamp = (sx: number, sy: number) => {
      const sp = store.getState().stamp;
      if (!sp) return false;
      const [ax, ay] = w2s(sp.x, sp.y);
      const [bx, by] = w2s(sp.x + sp.w, sp.y + sp.h);
      return sx >= ax && sx <= bx && sy >= ay && sy <= by;
    };

    const onDown = (e: PointerEvent) => {
      const r = rectOf();
      const sx = e.clientX - r.left;
      const sy = e.clientY - r.top;
      last = { x: e.clientX, y: e.clientY };
      const st = store.getState();
      const wantPan = e.button === 1 || e.button === 2 || st.tool === "pan" || e.shiftKey;

      // A pending stamp is modal: pointer manipulates the stamp, not the frame.
      if (st.stamp && !wantPan) {
        const sh = stampHandleAt(sx, sy);
        if (sh) {
          mode = "stampResize";
          handle = sh;
          startStamp = { x: st.stamp.x, y: st.stamp.y, w: st.stamp.w, h: st.stamp.h };
          cv.setPointerCapture(e.pointerId);
          return;
        }
        if (insideStamp(sx, sy)) {
          mode = "stampMove";
          cv.setPointerCapture(e.pointerId);
          return;
        }
        // clicked outside the stamp — pan the canvas instead
        mode = "pan";
        cv.setPointerCapture(e.pointerId);
        return;
      }

      if (!wantPan) {
        const h = hitHandle(sx, sy);
        if (h) {
          mode = "resize";
          handle = h;
          startFrame = { ...st.frame };
          cv.setPointerCapture(e.pointerId);
          return;
        }
        if (insideFrame(sx, sy) || st.tool === "frame") {
          if (!insideFrame(sx, sy) && st.tool === "frame") {
            // reposition frame centered on click
            const [wx, wy] = s2w(sx, sy);
            st.setFrame({ x: Math.round(wx - st.frame.w / 2), y: Math.round(wy - st.frame.h / 2) });
          }
          mode = "move";
          startFrame = { ...store.getState().frame };
          cv.setPointerCapture(e.pointerId);
          return;
        }
      }
      mode = "pan";
      cv.setPointerCapture(e.pointerId);
    };

    const onMove = (e: PointerEvent) => {
      const st = store.getState();
      if (mode === "none") {
        // cursor feedback
        const r = rectOf();
        const sx = e.clientX - r.left, sy = e.clientY - r.top;
        if (st.stamp) {
          const sh = stampHandleAt(sx, sy);
          cv.style.cursor = sh
            ? sh === "nw" || sh === "se" ? "nwse-resize" : "nesw-resize"
            : insideStamp(sx, sy) ? "move" : "grab";
          return;
        }
        const h = hitHandle(sx, sy);
        cv.style.cursor = h
          ? h === "nw" || h === "se" ? "nwse-resize" : "nesw-resize"
          : insideFrame(sx, sy)
            ? "move"
            : st.tool === "pan" ? "grab" : "crosshair";
        return;
      }
      const dx = e.clientX - last.x;
      const dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };

      if (mode === "stampMove") {
        const sp = store.getState().stamp;
        if (sp) {
          st.updateStamp({
            x: Math.round(sp.x + dx / st.view.scale),
            y: Math.round(sp.y + dy / st.view.scale),
          });
        }
      } else if (mode === "stampResize" && handle) {
        const sp = store.getState().stamp;
        if (sp) {
          const aspect = startStamp.w / startStamp.h; // aspect-locked to the photo
          // opposite corner stays anchored
          const ax = handle.includes("w") ? startStamp.x + startStamp.w : startStamp.x;
          const ay = handle.includes("n") ? startStamp.y + startStamp.h : startStamp.y;
          const r = rectOf();
          const [wx, wy] = s2w(e.clientX - r.left, e.clientY - r.top);
          const MIN = 24;
          const newW = Math.max(MIN, Math.max(Math.abs(wx - ax), Math.abs(wy - ay) * aspect));
          const newH = newW / aspect;
          st.updateStamp({
            w: Math.round(newW),
            h: Math.round(newH),
            x: Math.round(handle.includes("w") ? ax - newW : ax),
            y: Math.round(handle.includes("n") ? ay - newH : ay),
          });
        }
      } else if (mode === "pan") {
        st.panBy(dx, dy);
      } else if (mode === "move") {
        const f = store.getState().frame;
        const nx = Math.round((f.x + dx / st.view.scale) / 8) * 8;
        const ny = Math.round((f.y + dy / st.view.scale) / 8) * 8;
        st.setFrame({ x: nx, y: ny });
      } else if (mode === "resize" && handle) {
        const r = rectOf();
        const [wx, wy] = s2w(e.clientX - r.left, e.clientY - r.top);
        let { x, y, w, h } = startFrame;
        const snap = (v: number) => Math.round(v / SNAP) * SNAP;
        if (handle.includes("e")) w = clampSize(snap(wx - x));
        if (handle.includes("s")) h = clampSize(snap(wy - y));
        if (handle.includes("w")) {
          const nx = snap(wx);
          w = clampSize(startFrame.x + startFrame.w - nx);
          x = startFrame.x + startFrame.w - w;
        }
        if (handle.includes("n")) {
          const ny = snap(wy);
          h = clampSize(startFrame.y + startFrame.h - ny);
          y = startFrame.y + startFrame.h - h;
        }
        st.setFrame({ x, y, w, h });
        st.setTile(w, h);
      }
    };

    const onUp = (e: PointerEvent) => {
      mode = "none";
      handle = null;
      try { cv.releasePointerCapture(e.pointerId); } catch { /* noop */ }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = rectOf();
      const factor = Math.exp(-e.deltaY * 0.0016);
      store.getState().zoomAt(factor, e.clientX - r.left, e.clientY - r.top, r.width, r.height);
    };
    const onContext = (e: Event) => e.preventDefault();

    cv.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    cv.addEventListener("wheel", onWheel, { passive: false });
    cv.addEventListener("contextmenu", onContext);
    return () => {
      cv.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      cv.removeEventListener("wheel", onWheel);
      cv.removeEventListener("contextmenu", onContext);
    };
  }, [store]);

  return (
    <div className="canvas-wrap" ref={wrapRef}>
      <canvas ref={ref} className="canvas-el" />
      {busy && (
        <div className="canvas-phase num">
          {phaseLabel(progress.phase)}
          {progress.detail ? ` · ${progress.detail}` : ""}
        </div>
      )}

      {stamp && (
        <div className="stamp-bar">
          <span className="stamp-hint">Placing photo</span>
          <button className="stamp-act" onClick={() => fitStampToFrame()}>Fit to frame</button>
          <button className="stamp-act" onClick={() => flipStamp()}>Flip</button>
          <button className="stamp-act ghost" onClick={() => cancelStamp()}>
            Cancel <span className="k num">esc</span>
          </button>
          <button className="stamp-act primary" onClick={() => commitStamp()}>
            Place <span className="k num">⏎</span>
          </button>
        </div>
      )}

      {dragOver && (
        <div className="drop-zone">
          <div className="drop-inner">
            <span className="drop-glyph">❖</span>
            Drop to stamp your photo
          </div>
        </div>
      )}
    </div>
  );
}

function clampSize(v: number) {
  return Math.min(1024, Math.max(256, v));
}

function phaseLabel(p: string) {
  switch (p) {
    case "downloading": return "downloading weights";
    case "compiling": return "compiling shaders";
    case "encoding": return "encoding";
    case "sampling": return "sampling";
    case "decoding": return "decoding";
    case "compositing": return "compositing";
    default: return "working";
  }
}
