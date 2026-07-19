import { useEffect } from "react";
import Canvas from "./components/Canvas";
import TopBar from "./components/TopBar";
import Toolbar from "./components/Toolbar";
import Rail from "./components/Rail";
import Toasts from "./components/Toasts";
import Onboarding from "./components/Onboarding";
import StatusBar from "./components/StatusBar";
import { useStore } from "./store";
import "./app.css";

export default function App() {
  const undo = useStore((s) => s.undo);
  const redo = useStore((s) => s.redo);
  const generate = useStore((s) => s.generate);
  const cancel = useStore((s) => s.cancel);
  const busy = useStore((s) => s.busy);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          generate();
        }
        return;
      }
      // A pending stamp claims Enter (place) and Escape (cancel).
      const stamp = useStore.getState().stamp;
      if (stamp) {
        if (e.key === "Enter") {
          e.preventDefault();
          useStore.getState().commitStamp();
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          useStore.getState().cancelStamp();
          return;
        }
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (busy) cancel();
        else generate();
      } else if (e.key === "Escape" && busy) {
        cancel();
      } else if (!e.metaKey && !e.ctrlKey && !e.altKey && (e.key === "[" || e.key === "]")) {
        // resize the erase brush
        const st = useStore.getState();
        if (st.tool === "erase") {
          e.preventDefault();
          st.setBrushSize(st.brushSize + (e.key === "]" ? 12 : -12));
        }
      } else if (!e.metaKey && !e.ctrlKey && !e.altKey && e.key.toLowerCase() === "e") {
        useStore.getState().setTool("erase");
      } else if (!e.metaKey && !e.ctrlKey && !e.altKey && (e.key.toLowerCase() === "b" || e.key.toLowerCase() === "v")) {
        useStore.getState().setTool("frame");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, generate, cancel, busy]);

  return (
    <div className="app">
      <TopBar />
      <div className="app-body">
        <Toolbar />
        <Canvas />
        <Rail />
      </div>
      <StatusBar />
      <Toasts />
      <Onboarding />
    </div>
  );
}
