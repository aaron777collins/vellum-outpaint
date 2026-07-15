// ============================================================================
// Demo engine — a GPU-free procedural "diffuser".
// It does NOT call Stable Diffusion; it synthesises a plausible, prompt-tinted
// continuation of the edge-extended init so the entire canvas + outpaint +
// compositing pipeline is fully functional and demonstrable on any machine
// (including this GPU-less server, where the real WebGPU path can't be tested).
// ============================================================================

import {
  buildOutpaintInit,
  compositeFeathered,
  mulberry32,
  clamp8,
} from "../lib/imaging";
import type {
  DiffusionProvider,
  EngineProgress,
  GenerateRequest,
  GenerateResult,
  ProviderCapabilities,
} from "./types";
import { GenerationAbortError } from "./types";

function hashPrompt(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Warm palettes keyed off prompt words so the demo output visibly reacts to text.
function paletteFor(prompt: string): [number, number, number][] {
  const p = prompt.toLowerCase();
  const has = (...w: string[]) => w.some((x) => p.includes(x));
  if (has("sunset", "dusk", "amber", "gold", "fire", "warm"))
    return [[42, 22, 18], [180, 96, 44], [236, 176, 92], [250, 226, 170]];
  if (has("ocean", "sea", "water", "sky", "blue", "night", "cold"))
    return [[10, 18, 30], [34, 68, 104], [92, 140, 176], [186, 214, 232]];
  if (has("forest", "green", "moss", "jungle", "leaf", "grass"))
    return [[14, 26, 16], [40, 74, 44], [96, 132, 76], [186, 208, 150]];
  if (has("desert", "sand", "canyon", "clay", "earth"))
    return [[40, 26, 18], [128, 84, 52], [190, 140, 92], [232, 204, 158]];
  // default: gilded parchment
  return [[24, 20, 16], [86, 66, 40], [176, 140, 84], [232, 214, 172]];
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((res, rej) => {
    const t = setTimeout(res, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      rej(new GenerationAbortError());
    });
  });
}

export class DemoProvider implements DiffusionProvider {
  readonly id = "demo" as const;
  readonly caps: ProviderCapabilities = {
    local: true,
    requiresLoad: false,
    label: "Atelier (demo)",
    blurb:
      "A GPU-free procedural preview. Exercises the full outpaint pipeline instantly — swap to WebGPU for true Stable Diffusion.",
  };

  async isAvailable() {
    return true;
  }
  async load() {
    /* nothing to load */
  }
  isLoaded() {
    return true;
  }
  async dispose() {}

  async generate(
    req: GenerateRequest,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<GenerateResult> {
    const t0 = performance.now();
    const seed = req.seed >= 0 ? req.seed : (hashPrompt(req.prompt) ^ Date.now()) >>> 0;
    const { width: w, height: h } = req;

    const known =
      req.initImage ?? new ImageData(w, h); // all-transparent => full synth

    onProgress({ phase: "encoding", detail: "reading tile", fraction: 0.2 });
    await delay(120, signal);

    const { init } = buildOutpaintInit(known, seed);

    // Fake a sampling loop so the UI shows step progress like the real engine.
    const steps = Math.max(1, req.steps);
    const gen = new ImageData(new Uint8ClampedArray(init.data), w, h);
    const pal = paletteFor(req.prompt);
    const rnd = mulberry32(seed);

    for (let s = 0; s < steps; s++) {
      if (signal?.aborted) throw new GenerationAbortError();
      onProgress({
        phase: "sampling",
        step: s + 1,
        totalSteps: steps,
        fraction: (s + 1) / steps,
        detail: `step ${s + 1}/${steps}`,
      });
      paintPass(gen, known, pal, rnd, req.strength, (s + 1) / steps, seed + s);
      await delay(90, signal);
    }

    onProgress({ phase: "compositing", fraction: 0.95 });
    const feather = Math.round(Math.min(w, h) * 0.12) + 4;
    const composed = req.initImage
      ? compositeFeathered(known, gen, feather)
      : gen;

    onProgress({ phase: "done", fraction: 1 });
    return { image: composed, seed, ms: performance.now() - t0 };
  }
}

/** One procedural "denoise" pass: pulls fresh pixels toward a flowing palette. */
function paintPass(
  img: ImageData,
  known: ImageData,
  pal: [number, number, number][],
  rnd: () => number,
  strength: number,
  t: number,
  seed: number,
) {
  const { width: w, height: h } = img;
  const d = img.data;
  const kd = known.data;
  const freq = 0.008 + (seed % 7) * 0.0016;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      if (kd[i + 3] > 8) continue; // protect real pixels
      // smooth value-noise-ish field for banding/landscape feel
      const n =
        0.5 +
        0.5 *
          Math.sin(x * freq + Math.cos(y * freq * 1.7 + seed) * 2.2 + t * 3.14) *
          Math.cos(y * freq * 0.9 - seed * 0.01);
      const band = n * (pal.length - 1);
      const lo = Math.floor(band);
      const hi = Math.min(pal.length - 1, lo + 1);
      const f = band - lo;
      const c0 = pal[lo], c1 = pal[hi];
      const grain = (rnd() - 0.5) * 22 * (1 - t * 0.4);
      const tr = c0[0] + (c1[0] - c0[0]) * f + grain;
      const tg = c0[1] + (c1[1] - c0[1]) * f + grain;
      const tb = c0[2] + (c1[2] - c0[2]) * f + grain;
      // ease existing toward target — more per step, scaled by strength
      const k = 0.35 * strength + 0.15;
      d[i] = clamp8(d[i] + (tr - d[i]) * k);
      d[i + 1] = clamp8(d[i + 1] + (tg - d[i + 1]) * k);
      d[i + 2] = clamp8(d[i + 2] + (tb - d[i + 2]) * k);
      d[i + 3] = 255;
    }
  }
}
