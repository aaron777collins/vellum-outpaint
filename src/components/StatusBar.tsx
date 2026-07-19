import { useEffect, useState } from "react";
import { useStore } from "../store";
import { AMBIENT_VERSES, GENERATING_VERSES, type Verse } from "../lib/verses";

export default function StatusBar() {
  const busy = useStore((s) => s.busy);
  const view = useStore((s) => s.view);
  const frame = useStore((s) => s.frame);
  const [cursor, setCursor] = useState(0);

  // Rotate illuminations on a slow, steady cadence — the same tempo whether
  // idle or generating, so verses never flicker with per-step progress updates.
  useEffect(() => {
    const t = setInterval(() => setCursor((c) => c + 1), 16000);
    return () => clearInterval(t);
  }, []);

  const pool = busy ? GENERATING_VERSES : AMBIENT_VERSES;
  const shown: Verse = pool[cursor % pool.length];

  return (
    <footer className="statusbar">
      <div className="status-verse">
        <span className="verse-glyph">✧</span>
        <span className="verse-text display">“{shown.text}”</span>
        <span className="verse-ref eyebrow">{shown.ref}</span>
      </div>
      <div className="status-meta num">
        <span>x {Math.round(frame.x)}</span>
        <span>y {Math.round(frame.y)}</span>
        <span>{Math.round(view.scale * 100)}%</span>
      </div>
    </footer>
  );
}
