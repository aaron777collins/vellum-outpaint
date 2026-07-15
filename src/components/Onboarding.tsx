import { useState } from "react";
import { useStore, ALL_PROVIDERS, getProvider, type ProviderId } from "../store";
import { LOADING_VERSES } from "../lib/verses";

const KEY = "vellum.seen.v1";

export default function Onboarding() {
  const [open, setOpen] = useState(() => localStorage.getItem(KEY) !== "1");
  const setEngine = useStore((s) => s.setEngine);
  const engineId = useStore((s) => s.engineId);

  if (!open) return null;

  const close = (id?: ProviderId) => {
    if (id) setEngine(id);
    localStorage.setItem(KEY, "1");
    setOpen(false);
  };

  const verse = LOADING_VERSES[1]; // "Let there be light"

  return (
    <div className="onboard-scrim" onClick={() => close()}>
      <div className="onboard" onClick={(e) => e.stopPropagation()}>
        <div className="onboard-mark">V</div>
        <h1 className="onboard-title display">Vellum</h1>
        <p className="onboard-verse display">“{verse.text}”</p>
        <p className="onboard-ref eyebrow">{verse.ref}</p>
        <p className="onboard-lede">
          An outpainting studio. Bring an image — or begin from nothing — and
          extend the frame outward. Stable Diffusion runs on <em>your</em> hardware;
          nothing is uploaded.
        </p>

        <div className="onboard-engines">
          {ALL_PROVIDERS.map((id) => {
            const p = getProvider(id);
            return (
              <button
                key={id}
                className={`onboard-engine ${id === engineId ? "on" : ""}`}
                onClick={() => close(id)}
              >
                <span className="oe-label">{p.caps.label}</span>
                <span className="oe-blurb">{p.caps.blurb}</span>
              </button>
            );
          })}
        </div>
        <button className="onboard-skip" onClick={() => close("demo")}>
          Just show me — start with the demo →
        </button>
      </div>
    </div>
  );
}
