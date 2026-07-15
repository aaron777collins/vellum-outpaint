import { create } from "zustand";
import { doc, type Rect } from "./lib/document";
import {
  ALL_PROVIDERS,
  getProvider,
  remoteProvider,
  type EngineProgress,
  type ProviderId,
  type SamplingParams,
} from "./engine";
import { GenerationAbortError } from "./engine/types";

export type Tool = "pan" | "frame" | "move";

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
  cancel: () => void;

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
      const next = { engineId: id, engineStatus: "idle" as const, engineError: null };
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
      if (!ok && engineId === "webgpu") {
        throw new Error(
          "This browser can't run WebGPU Stable Diffusion (needs Chrome/Edge 113+ with a shader-f16 GPU). Try the Atelier demo or a Remote WebUI.",
        );
      }
      await provider.load((p) =>
        set({ progress: p, loadPct: p.fraction ? Math.round(p.fraction * 100) : get().loadPct }),
      );
      set({ engineStatus: "ready", progress: { phase: "done" } });
      get().toast("success", `${provider.caps.label} ready`);
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
    if (!s.params.prompt.trim()) {
      s.toast("error", "Describe what should appear in the frame first.");
      return;
    }
    abortCtrl = new AbortController();
    set({ busy: true, progress: { phase: "encoding" } });

    const rect: Rect = { ...s.frame };
    const known = doc.extract(rect);
    // is there anything to outpaint from?
    let hasKnown = false;
    for (let i = 3; i < known.data.length; i += 4)
      if (known.data[i] > 8) { hasKnown = true; break; }

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
}
