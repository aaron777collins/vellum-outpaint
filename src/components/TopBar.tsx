import { useStore } from "../store";
import { doc } from "../lib/document";
import { pickImageFile } from "../lib/imaging";

export default function TopBar() {
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const clearDoc = useStore((s) => s.clearDoc);
  const exportPng = useStore((s) => s.exportPng);
  const setView = useStore((s) => s.setView);
  const stampFromFile = useStore((s) => s.stampFromFile);

  const onImport = async () => {
    const f = await pickImageFile();
    if (f) stampFromFile(f);
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
        <button className="tb-btn" onClick={onImport} title="Stamp a photo onto the canvas">
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
    </header>
  );
}
