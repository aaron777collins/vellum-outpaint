import { useStore, type ProviderId } from "../store";
import type { Tool } from "../store";

const TOOLS: { id: Tool; glyph: string; label: string }[] = [
  { id: "frame", glyph: "▣", label: "Frame — position the outpaint window" },
  { id: "pan", glyph: "✥", label: "Pan — drag the canvas (or hold Shift)" },
];

export default function Toolbar() {
  const tool = useStore((s) => s.tool);
  const setTool = useStore((s) => s.setTool);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const engineId = useStore((s) => s.engineId) as ProviderId;

  return (
    <nav className="toolbar">
      {TOOLS.map((t) => (
        <button
          key={t.id}
          className={`tool ${tool === t.id ? "on" : ""}`}
          title={t.label}
          onClick={() => setTool(t.id)}
        >
          {t.glyph}
        </button>
      ))}
      <span className="tool-sep" />
      <button className="tool" title="Zoom in" onClick={() => setView({ scale: Math.min(8, view.scale * 1.2) })}>+</button>
      <button className="tool" title="Zoom out" onClick={() => setView({ scale: Math.max(0.05, view.scale / 1.2) })}>−</button>
      <button className="tool small num" title="Reset zoom" onClick={() => setView({ scale: 1 })}>
        {Math.round(view.scale * 100)}
      </button>
      <span className="tool-flex" />
      <span className={`tool-engine ${engineId}`} title={`Engine: ${engineId}`}>
        {engineId === "webgpu" ? "GPU" : engineId === "remote" ? "WEB" : "DEMO"}
      </span>
    </nav>
  );
}
