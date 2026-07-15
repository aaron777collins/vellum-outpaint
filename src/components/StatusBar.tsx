import { useEffect, useState } from "react";
import { useStore } from "../store";
import { nextAmbientVerse, type Verse } from "../lib/verses";
import { GENERATING_VERSES, pickVerse } from "../lib/verses";

export default function StatusBar() {
  const busy = useStore((s) => s.busy);
  const progress = useStore((s) => s.progress);
  const view = useStore((s) => s.view);
  const frame = useStore((s) => s.frame);
  const [verse, setVerse] = useState<Verse>(() => nextAmbientVerse());

  // rotate the ambient illumination while idle
  useEffect(() => {
    if (busy) return;
    const t = setInterval(() => setVerse(nextAmbientVerse()), 16000);
    return () => clearInterval(t);
  }, [busy]);

  const shown: Verse = busy
    ? pickVerse(GENERATING_VERSES, (progress.step ?? 0) + (progress.phase?.length ?? 0))
    : verse;

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
