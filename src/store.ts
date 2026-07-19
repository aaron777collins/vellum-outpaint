import { create } from "zustand";
import { doc, type Rect } from "./lib/document";
import { ctxOf, imageDataToCanvas, loadImageData, makeCanvas } from "./lib/imaging";
import { captionImage } from "./lib/caption";
import {
  ALL_PROVIDERS,
  getProvider,
  remoteProvider,
  type EngineProgress,
  type ProviderId,
  type SamplingParams,
} from "./engine";
import { GenerationAbortError } from "./engine/types";

export type Tool = "pan" | "frame" | "move" | "stamp" | "erase";

export type ExpandDir = "left" | "right" | "up" | "down";

/**
 * A photo being interactively placed onto the canvas. It floats above the world
 * — movable and scalable — until the user commits it (Place) or cancels. This is
 * the "stamp" tool: import a photo, position/scale it, drop it in, then outpaint
 * around it. `x,y,w,h` are WORLD coordinates (same space as the frame). The
 * source pixels are kept at natural resolution so scaling stays crisp.
 */
export interface Stamp {
  img: ImageData;
  natW: number;
  natH: number;
  x: number;
  y: number;
  w: number;
  h: number;
  flipH: boolean;
}

export interface ViewState {
  x: number; // world coord at screen center
  y: number;
  scale: number;
}

const LS_KEY = "vellum.settings.v1";

interface Persisted {
  params: SamplingParams;
  tileW: number;
  tileH: number;
  engineId: ProviderId;
  remoteUrl: string;
  brushSize: number;
}

function loadPersisted(): Partial<Persisted> {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "{}");
  } catch {
    return {};
  }
}

const persisted = loadPersisted();

export interface Toast {
  id: number;
  kind: "info" | "error" | "success";
  msg: string;
}

interface AppState {
  // view
  view: ViewState;
  tool: Tool;
  setTool: (t: Tool) => void;
  setView: (v: Partial<ViewState>) => void;
  panBy: (dx: number, dy: number) => void;
  zoomAt: (factor: number, sx: number, sy: number, screenW: number, screenH: number) => void;

  // selection frame (world coords)
  frame: Rect;
  setFrame: (r: Partial<Rect>) => void;
  tileW: number;
  tileH: number;
  setTile: (w: number, h: number) => void;

  // erase brush (world-unit diameter)
  brushSize: number;
  setBrushSize: (d: number) => void;
  beginErase: () => void;
  eraseAt: (wx: number, wy: number) => void;

  // params
  params: SamplingParams;
  setParams: (p: Partial<SamplingParams>) => void;

  // engine
  engineId: ProviderId;
  setEngine: (id: ProviderId) => void;
  engineStatus: "idle" | "loading" | "ready" | "error";
  engineError: string | null;
  progress: EngineProgress;
  loadEngine: () => Promise<void>;
  remoteUrl: string;
  setRemoteUrl: (u: string) => void;

  // generation
  busy: boolean;
  loadPct: number;
  generate: () => Promise<void>;
  expand: (dir: ExpandDir) => Promise<void>;
  cancel: () => void;

  // auto-prompt (image captioning)
  captioning: boolean;
  suggestPrompt: () => Promise<void>;

  // stamp (interactive photo placement)
  stamp: Stamp | null;
  stampFromFile: (file: File | Blob) => Promise<void>;
  beginStamp: (img: ImageData) => void;
  updateStamp: (p: Partial<Stamp>) => void;
  flipStamp: () => void;
  fitStampToFrame: () => void;
  commitStamp: () => void;
  cancelStamp: () => void;

  // doc
  docRev: number;
  bumpDoc: () => void;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  clearDoc: () => void;
  exportPng: () => Promise<void>;

  // toasts
  toasts: Toast[];
  toast: (kind: Toast["kind"], msg: string) => void;
  dismiss: (id: number) => void;
}

let abortCtrl: AbortController | null = null;
let toastId = 1;

function savePersisted(s: AppState) {
  const p: Persisted = {
    params: s.params,
    tileW: s.tileW,
    tileH: s.tileH,
    engineId: s.engineId,
    remoteUrl: s.remoteUrl,
    brushSize: s.brushSize,
  };
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(p));
  } catch {
    /* ignore quota */
  }
}

export const useStore = create<AppState>((set, get) => ({
  view: { x: 0, y: 0, scale: 1 },
  tool: "frame",
  setTool: (t) => set({ tool: t }),
  setView: (v) => set((s) => ({ view: { ...s.view, ...v } })),
  panBy: (dx, dy) =>
    set((s) => ({ view: { ...s.view, x: s.view.x - dx / s.view.scale, y: s.view.y - dy / s.view.scale } })),
  zoomAt: (factor, sx, sy, screenW, screenH) =>
    set((s) => {
      const scale = Math.min(8, Math.max(0.05, s.view.scale * factor));
      // keep the world point under the cursor fixed
      const wx = s.view.x + (sx - screenW / 2) / s.view.scale;
      const wy = s.view.y + (sy - screenH / 2) / s.view.scale;
      const nx = wx - (sx - screenW / 2) / scale;
      const ny = wy - (sy - screenH / 2) / scale;
      return { view: { x: nx, y: ny, scale } };
    }),

  frame: { x: -256, y: -256, w: 512, h: 512 },
  setFrame: (r) => set((s) => ({ frame: { ...s.frame, ...r } })),
  tileW: persisted.tileW ?? 512,
  tileH: persisted.tileH ?? 512,
  setTile: (w, h) =>
    set((s) => {
      const nf = { ...s.frame, w, h };
      const next = { tileW: w, tileH: h, frame: nf };
      savePersisted({ ...s, ...next } as AppState);
      return next;
    }),

  brushSize: persisted.brushSize ?? 96,
  setBrushSize: (d) =>
    set((s) => {
      const brushSize = Math.min(512, Math.max(16, Math.round(d)));
      savePersisted({ ...s, brushSize } as AppState);
      return { brushSize };
    }),
  beginErase: () => doc.beginStroke(),
  eraseAt: (wx, wy) => {
    doc.eraseCircle(wx, wy, get().brushSize / 2);
    set({ docRev: doc.revision });
  },

  params: {
    prompt: "",
    negativePrompt: "blurry, low quality, distorted, watermark",
    steps: 1,
    guidance: 1,
    strength: 0.62,
    seed: -1,
    ...persisted.params,
  },
  setParams: (p) =>
    set((s) => {
      const params = { ...s.params, ...p };
      savePersisted({ ...s, params } as AppState);
      return { params };
    }),

  engineId: persisted.engineId ?? "webgpu",
  setEngine: (id) =>
    set((s) => {
      // Apply engine-appropriate sampler defaults so users don't have to tune
      // steps/guidance by hand when switching between turbo and full SD.
      const caps = getProvider(id).caps;
      const params = caps.multiStep
        ? { ...s.params, steps: caps.suggestedSteps ?? s.params.steps, guidance: caps.suggestedGuidance ?? s.params.guidance }
        : { ...s.params, steps: 1, guidance: 1 };
      const next = { engineId: id, engineStatus: "idle" as const, engineError: null, params };
      savePersisted({ ...s, ...next } as AppState);
      return next;
    }),
  engineStatus: "idle",
  engineError: null,
  progress: { phase: "idle" },
  loadPct: 0,
  remoteUrl: persisted.remoteUrl ?? "",
  setRemoteUrl: (u) =>
    set((s) => {
      remoteProvider().configure({ baseUrl: u });
      savePersisted({ ...s, remoteUrl: u } as AppState);
      return { remoteUrl: u };
    }),

  loadEngine: async () => {
    const { engineId, remoteUrl } = get();
    const provider = getProvider(engineId);
    if (engineId === "remote") remoteProvider().configure({ baseUrl: remoteUrl });
    if (provider.isLoaded()) {
      set({ engineStatus: "ready" });
      return;
    }
    set({ engineStatus: "loading", engineError: null, progress: { phase: "downloading" } });
    try {
      const ok = await provider.isAvailable();
      if (!ok && engineId.startsWith("webgpu")) {
        throw new Error(
          "This browser can't run WebGPU Stable Diffusion (needs Chrome/Edge 113+ with a shader-f16 GPU). Try the Atelier demo or a Remote WebUI.",
        );
      }
      await provider.load((p) =>
        set({ progress: p, loadPct: p.fraction ? Math.round(p.fraction * 100) : get().loadPct }),
      );
      set({ engineStatus: "ready", progress: { phase: "done" } });
      get().toast("success", `${provider.caps.label} ready`);
      // Local engines cache ~2.5 GB of weights. If the browser refused persistent
      // storage, that cache is best-effort and can be evicted — which is exactly
      // what makes the models re-download on a later visit. Tell the user plainly
      // (once) how to make it permanent instead of leaving it a mystery.
      if (engineId.startsWith("webgpu")) {
        try {
          const persisted = await navigator.storage?.persisted?.();
          if (persisted === false) {
            get().toast(
              "info",
              "Your browser hasn't made the model cache permanent, so it may re-download later. Install Vellum (address-bar ⊕ / ⋮ → Install) or bookmark it to keep the weights forever.",
            );
          }
        } catch {
          /* storage API unavailable */
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      set({ engineStatus: "error", engineError: msg, progress: { phase: "error" } });
      get().toast("error", msg);
    }
  },

  busy: false,
  generate: async () => {
    const s = get();
    if (s.busy) return;
    const provider = getProvider(s.engineId);
    if (!provider.isLoaded()) {
      await s.loadEngine();
      if (get().engineStatus !== "ready") return;
    }

    const rect: Rect = { ...s.frame };
    const known = doc.extract(rect);
    // is there anything to outpaint/continue from?
    let hasKnown = false;
    for (let i = 3; i < known.data.length; i += 4)
      if (known.data[i] > 8) { hasKnown = true; break; }

    // A prompt is only required when there's nothing to continue from. With
    // existing pixels in the frame, an empty prompt is a valid "just continue
    // the scene" request (SD outpaints from the surrounding context).
    if (!s.params.prompt.trim() && !hasKnown) {
      s.toast("error", "Describe what should appear — or place/erase something to continue from.");
      return;
    }
    abortCtrl = new AbortController();
    set({ busy: true, progress: { phase: "encoding" } });

    try {
      const res = await provider.generate(
        {
          ...s.params,
          width: rect.w,
          height: rect.h,
          initImage: hasKnown ? known : null,
          mask: null,
        },
        (p) => set({ progress: p, loadPct: p.fraction ? Math.round(p.fraction * 100) : get().loadPct }),
        abortCtrl.signal,
      );
      doc.commit(rect, res.image);
      set({ docRev: doc.revision, busy: false, progress: { phase: "done" } });
      get().toast("success", `Painted in ${(res.ms / 1000).toFixed(1)}s · seed ${res.seed}`);
    } catch (e: unknown) {
      set({ busy: false, progress: { phase: "idle" } });
      if (e instanceof GenerationAbortError) get().toast("info", "Generation cancelled");
      else get().toast("error", e instanceof Error ? e.message : String(e));
    } finally {
      abortCtrl = null;
    }
  },
  cancel: () => {
    abortCtrl?.abort();
  },

  // Move the frame so it straddles the current content's edge in `dir` — most of
  // it over EMPTY space (to be painted) with a band still covering existing
  // pixels (so the model continues the scene instead of inventing a disconnected
  // one). Then generate. This is the "press a button → the picture grows outward"
  // action that makes outpainting actually feel like outpainting.
  expand: async (dir) => {
    const s = get();
    if (s.busy) return;
    const cb = doc.contentBounds();
    // Nothing on the canvas yet → there's no edge to grow from; just paint here.
    if (!cb) {
      await s.generate();
      return;
    }
    const f = s.frame;
    const OVERLAP = 0.42; // fraction of the frame kept over existing content
    const snap = (v: number) => Math.round(v / 8) * 8;
    const cxC = cb.x + cb.w / 2;
    const cyC = cb.y + cb.h / 2;
    let x = f.x;
    let y = f.y;
    if (dir === "right") { x = snap(cb.x + cb.w - f.w * OVERLAP); y = snap(cyC - f.h / 2); }
    else if (dir === "left") { x = snap(cb.x - f.w * (1 - OVERLAP)); y = snap(cyC - f.h / 2); }
    else if (dir === "down") { y = snap(cb.y + cb.h - f.h * OVERLAP); x = snap(cxC - f.w / 2); }
    else if (dir === "up") { y = snap(cb.y - f.h * (1 - OVERLAP)); x = snap(cxC - f.w / 2); }
    set({ frame: { ...f, x, y } });
    await get().generate();
  },

  captioning: false,
  suggestPrompt: async () => {
    const s = get();
    if (s.captioning) return;
    // Prefer what's inside the frame (that's what the user is continuing); fall
    // back to the whole painted scene if the frame is empty.
    const framed = doc.extract(s.frame);
    let hasKnown = false;
    for (let i = 3; i < framed.data.length; i += 4)
      if (framed.data[i] > 8) { hasKnown = true; break; }
    const src = hasKnown ? framed : doc.extractContent();
    if (!src) {
      s.toast("info", "Paint or place something first, then I can suggest a prompt to continue it.");
      return;
    }
    set({ captioning: true });
    try {
      const text = await captionImage(src, (p) =>
        set({ loadPct: p.fraction != null ? Math.round(p.fraction * 100) : get().loadPct }),
      );
      if (text) {
        get().setParams({ prompt: text });
        get().toast("success", "Suggested a prompt from the scene — tweak it or just Outpaint.");
      } else {
        get().toast("info", "Couldn't read a clear subject — try describing it yourself.");
      }
    } catch (e: unknown) {
      get().toast("error", "Prompt suggestion failed: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      set({ captioning: false });
    }
  },

  // ---- stamp: interactive photo placement -------------------------------
  stamp: null,
  stampFromFile: async (file) => {
    try {
      const img = await loadImageData(file);
      get().beginStamp(img);
    } catch {
      get().toast("error", "Could not read that image");
    }
  },
  beginStamp: (img) =>
    set((s) => {
      const natW = img.width;
      const natH = img.height;
      // Fit the longest side to ~512 world units by default (matches the tile
      // scale) so big photos arrive at a sane, editable size. Small images keep
      // their native size. Centered on the current viewport center.
      const longest = Math.max(natW, natH);
      const scale = longest > 512 ? 512 / longest : 1;
      const w = Math.max(1, Math.round(natW * scale));
      const h = Math.max(1, Math.round(natH * scale));
      return {
        tool: "stamp" as const,
        stamp: {
          img, natW, natH, flipH: false,
          x: Math.round(s.view.x - w / 2),
          y: Math.round(s.view.y - h / 2),
          w, h,
        },
      };
    }),
  updateStamp: (p) => set((s) => (s.stamp ? { stamp: { ...s.stamp, ...p } } : {})),
  flipStamp: () => set((s) => (s.stamp ? { stamp: { ...s.stamp, flipH: !s.stamp.flipH } } : {})),
  fitStampToFrame: () =>
    set((s) => {
      if (!s.stamp) return {};
      const f = s.frame;
      const aspect = s.stamp.natW / s.stamp.natH;
      // contain within the frame while preserving aspect
      let w = f.w;
      let h = w / aspect;
      if (h > f.h) { h = f.h; w = h * aspect; }
      w = Math.round(w);
      h = Math.round(h);
      return {
        stamp: {
          ...s.stamp, w, h,
          x: Math.round(f.x + (f.w - w) / 2),
          y: Math.round(f.y + (f.h - h) / 2),
        },
      };
    }),
  commitStamp: () =>
    set((s) => {
      const st = s.stamp;
      if (!st) return {};
      const w = Math.max(1, Math.round(st.w));
      const h = Math.max(1, Math.round(st.h));
      // Render the (optionally flipped) photo to its world-pixel size, then paint
      // it into the document at the stamp's world rect.
      const c = makeCanvas(w, h);
      const ctx = ctxOf(c);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      if (st.flipH) { ctx.translate(w, 0); ctx.scale(-1, 1); }
      ctx.drawImage(imageDataToCanvas(st.img), 0, 0, w, h);
      const img = ctx.getImageData(0, 0, w, h);
      doc.commit({ x: st.x, y: st.y, w, h }, img);
      setTimeout(() => get().toast("success", "Photo placed — move the frame past its edge to outpaint"), 0);
      return { stamp: null, tool: "frame" as const, docRev: doc.revision };
    }),
  cancelStamp: () => set({ stamp: null, tool: "frame" }),

  docRev: 0,
  bumpDoc: () => set({ docRev: doc.revision }),
  undo: async () => {
    await doc.undo();
    set({ docRev: doc.revision });
  },
  redo: async () => {
    await doc.redo();
    set({ docRev: doc.revision });
  },
  clearDoc: () => {
    doc.clear();
    set({ docRev: doc.revision });
    get().toast("info", "Canvas cleared");
  },
  exportPng: async () => {
    const blob = await doc.exportPng();
    if (!blob) {
      get().toast("error", "Nothing to export yet.");
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vellum-${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
    get().toast("success", "Exported PNG");
  },

  toasts: [],
  toast: (kind, msg) =>
    set((s) => {
      const id = toastId++;
      setTimeout(() => get().dismiss(id), kind === "error" ? 7000 : 4000);
      return { toasts: [...s.toasts, { id, kind, msg }] };
    }),
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

export { ALL_PROVIDERS, getProvider };
export type { ProviderId };

// Dev/testing handle so automation can drive canvas state deterministically.
if (typeof window !== "undefined") {
  (window as unknown as { vellumStore?: typeof useStore }).vellumStore = useStore;
  (window as unknown as { vellumDoc?: typeof doc }).vellumDoc = doc;
}
