import { useState } from "react";
import { useStore, ALL_PROVIDERS, getProvider } from "../store";

const TILE_PRESETS: [number, number, string][] = [
  [512, 512, "square"],
  [512, 768, "portrait"],
  [768, 512, "landscape"],
  [640, 640, "large"],
];

function Slider({
  label, value, min, max, step, onChange, fmt,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <label className="slider">
      <div className="slider-head">
        <span className="eyebrow">{label}</span>
        <span className="num slider-val">{fmt ? fmt(value) : value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

export default function Rail() {
  const params = useStore((s) => s.params);
  const setParams = useStore((s) => s.setParams);
  const tileW = useStore((s) => s.tileW);
  const tileH = useStore((s) => s.tileH);
  const setTile = useStore((s) => s.setTile);
  const setFrame = useStore((s) => s.setFrame);
  const frame = useStore((s) => s.frame);
  const generate = useStore((s) => s.generate);
  const cancel = useStore((s) => s.cancel);
  const busy = useStore((s) => s.busy);
  const engineId = useStore((s) => s.engineId);
  const setEngine = useStore((s) => s.setEngine);
  const engineStatus = useStore((s) => s.engineStatus);
  const engineError = useStore((s) => s.engineError);
  const loadEngine = useStore((s) => s.loadEngine);
  const remoteUrl = useStore((s) => s.remoteUrl);
  const setRemoteUrl = useStore((s) => s.setRemoteUrl);
  const progress = useStore((s) => s.progress);
  const loadPct = useStore((s) => s.loadPct);

  const [showNeg, setShowNeg] = useState(false);
  const [showAdv, setShowAdv] = useState(false);

  const applyTile = (w: number, h: number) => {
    setTile(w, h);
    // keep frame centered on its current center
    const cx = frame.x + frame.w / 2;
    const cy = frame.y + frame.h / 2;
    setFrame({ w, h, x: Math.round(cx - w / 2), y: Math.round(cy - h / 2) });
  };

  const provider = getProvider(engineId);
  const ready = engineStatus === "ready";
  const multiStep = !!provider.caps.multiStep;
  const showSampler = showAdv || multiStep;

  return (
    <aside className="rail">
      <div className="rail-scroll">
        {/* ---- prompt ---- */}
        <section className="panel">
          <div className="panel-title">
            <span className="eyebrow">The vision</span>
          </div>
          <textarea
            className="prompt"
            placeholder="a windswept coastline at golden hour, distant lighthouse, painterly…"
            value={params.prompt}
            onChange={(e) => setParams({ prompt: e.target.value })}
            rows={4}
            spellCheck={false}
          />
          <button className="mini-toggle" onClick={() => setShowNeg((v) => !v)}>
            {showNeg ? "− " : "+ "} exclude
          </button>
          {showNeg && (
            <textarea
              className="prompt prompt-neg"
              placeholder="what to avoid…"
              value={params.negativePrompt}
              onChange={(e) => setParams({ negativePrompt: e.target.value })}
              rows={2}
              spellCheck={false}
            />
          )}
        </section>

        {/* ---- canvas frame ---- */}
        <section className="panel">
          <div className="panel-title">
            <span className="eyebrow">Frame</span>
            <span className="num panel-aside">{tileW}×{tileH}</span>
          </div>
          <div className="tile-grid">
            {TILE_PRESETS.map(([w, h, name]) => (
              <button
                key={name}
                className={`tile-chip ${w === tileW && h === tileH ? "on" : ""}`}
                onClick={() => applyTile(w, h)}
              >
                <span className="tile-shape" style={{ aspectRatio: `${w}/${h}` }} />
                {name}
              </button>
            ))}
          </div>
        </section>

        {/* ---- guidance sliders ---- */}
        <section className="panel">
          <Slider
            label="Fidelity to source" value={params.strength} min={0.15} max={1} step={0.01}
            onChange={(v) => setParams({ strength: v })}
            fmt={(v) => `${Math.round((1 - v) * 100)}% kept`}
          />
          <div className="hint">Lower ⇒ blends with existing pixels · Higher ⇒ reinvents the frame</div>
          {showSampler && (
            <>
              {multiStep ? (
                <>
                  <Slider
                    label="Steps" value={params.steps} min={4} max={40} step={1}
                    onChange={(v) => setParams({ steps: v })}
                  />
                  <Slider
                    label="Guidance" value={params.guidance} min={1} max={12} step={0.1}
                    onChange={(v) => setParams({ guidance: v })}
                    fmt={(v) => v.toFixed(1)}
                  />
                  <div className="hint">More steps ⇒ finer detail (slower) · Guidance ⇒ how strictly it follows the prompt</div>
                </>
              ) : (
                <div className="hint">SD-Turbo paints in a single step — steps &amp; guidance are fixed. Switch to SD 1.5 for multi-step control.</div>
              )}
              <label className="slider">
                <div className="slider-head">
                  <span className="eyebrow">Seed</span>
                  <button
                    className="mini-toggle inline"
                    onClick={() => setParams({ seed: -1 })}
                  >
                    randomize
                  </button>
                </div>
                <input
                  className="seed-input num"
                  value={params.seed < 0 ? "" : params.seed}
                  placeholder="random"
                  onChange={(e) => {
                    const n = parseInt(e.target.value, 10);
                    setParams({ seed: isNaN(n) ? -1 : n });
                  }}
                />
              </label>
            </>
          )}
          {!multiStep && (
            <button className="mini-toggle" onClick={() => setShowAdv((v) => !v)}>
              {showAdv ? "− fewer controls" : "+ steps · guidance · seed"}
            </button>
          )}
        </section>
      </div>

      {/* ---- engine + generate (pinned) ---- */}
      <div className="rail-foot">
        <section className="panel engine-panel">
          <div className="panel-title">
            <span className="eyebrow">Engine</span>
            <span className={`dot ${engineStatus}`} />
          </div>
          <div className="engine-pick">
            {ALL_PROVIDERS.map((id) => {
              const p = getProvider(id);
              return (
                <button
                  key={id}
                  className={`engine-opt ${id === engineId ? "on" : ""}`}
                  onClick={() => setEngine(id)}
                  title={p.caps.blurb}
                >
                  {p.caps.label}
                  {p.caps.local && <span className="badge">private</span>}
                </button>
              );
            })}
          </div>
          <p className="engine-blurb">{provider.caps.blurb}</p>

          {engineId === "remote" && (
            <input
              className="url-input num"
              placeholder="http://localhost:7860"
              value={remoteUrl}
              onChange={(e) => setRemoteUrl(e.target.value)}
            />
          )}

          {engineStatus === "loading" && (
            <div className="load-bar">
              <div className="load-fill" style={{ width: `${loadPct}%` }} />
              <span className="load-txt num">
                {progress.detail || progress.phase} {loadPct ? `${loadPct}%` : ""}
              </span>
            </div>
          )}
          {engineStatus === "error" && <p className="engine-err">{engineError}</p>}
          {!ready && engineStatus !== "loading" && provider.caps.requiresLoad && (
            <button className="load-btn" onClick={() => loadEngine()}>
              {provider.caps.local ? "Download the model" : "Connect"}
            </button>
          )}
        </section>

        <button
          className={`generate ${busy ? "busy" : ""}`}
          onClick={() => (busy ? cancel() : generate())}
        >
          <span className="generate-label">
            {busy ? "Stop" : "Outpaint"}
          </span>
          <span className="generate-key num">{busy ? "esc" : "↵"}</span>
        </button>
      </div>
    </aside>
  );
}
