import { useRef } from "react";
import { useStore } from "../store";
import { doc } from "../lib/document";
import { loadImageData } from "../lib/imaging";

export default function TopBar() {
  const fileRef = useRef<HTMLInputElement>(null);
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const clearDoc = useStore((s) => s.clearDoc);
  const exportPng = useStore((s) => s.exportPng);
  const bumpDoc = useStore((s) => s.bumpDoc);
  const setView = useStore((s) => s.setView);
  const setFrame = useStore((s) => s.setFrame);
  const toast = useStore((s) => s.toast);

  const onImport = async (file: File) => {
    try {
      const img = await loadImageData(file);
      const rect = doc.place(img, 0, 0);
      bumpDoc();
      // fit frame to image and view to content
      setFrame({ x: rect.x, y: rect.y, w: Math.min(1024, rect.w), h: Math.min(1024, rect.h) });
      fitToContent();
      toast("success", "Image placed — drag the frame past its edge to outpaint");
    } catch {
      toast("error", "Could not read that image");
    }
  };

  const fitToContent = () => {
    const cb = doc.contentBounds();
    if (!cb) {
      setView({ x: 0, y: 0, scale: 1 });
      return;
    }
    const pad = 1.35;
    const vw = window.innerWidth - 400;
    const vh = window.innerHeight - 140;
    const scale = Math.min(2, Math.max(0.1, Math.min(vw / (cb.w * pad), vh / (cb.h * pad))));
    setView({ x: cb.x + cb.w / 2, y: cb.y + cb.h / 2, scale });
  };

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">V</span>
        <div className="brand-txt">
          <span className="brand-name display">Vellum</span>
          <span className="brand-sub eyebrow">outpainting studio</span>
        </div>
      </div>

      <div className="topbar-actions">
        <button className="tb-btn" onClick={() => fileRef.current?.click()}>
          Import
        </button>
        <button className="tb-btn" onClick={exportPng}>Export</button>
        <span className="tb-sep" />
        <button className="tb-icon" title="Undo (⌘Z)" onClick={() => undo()}>↶</button>
        <button className="tb-icon" title="Redo (⌘⇧Z)" onClick={() => redo()}>↷</button>
        <span className="tb-sep" />
        <button className="tb-icon" title="Fit to content" onClick={fitToContent}>⤢</button>
        <button className="tb-icon danger" title="Clear canvas" onClick={() => clearDoc()}>⌫</button>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onImport(f);
          e.target.value = "";
        }}
      />
    </header>
  );
}
