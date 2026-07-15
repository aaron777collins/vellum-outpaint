// ============================================================================
// Remote engine — talk to a user-supplied Stable Diffusion server.
// Supports the AUTOMATIC1111 / Forge `/sdapi/v1/img2img` inpainting endpoint
// (the same API openOutpaint targets). This guarantees real SD output for
// anyone who already runs a WebUI, regardless of their browser's WebGPU
// support. The endpoint is configured in the UI and persisted.
// ============================================================================

import {
  buildOutpaintInit,
  compositeFeathered,
  imageDataToCanvas,
  loadImageData,
  makeCanvas,
  ctxOf,
} from "../lib/imaging";
import type {
  DiffusionProvider,
  EngineProgress,
  GenerateRequest,
  GenerateResult,
  ProviderCapabilities,
} from "./types";

export interface RemoteConfig {
  baseUrl: string; // e.g. http://localhost:7860
  sampler: string;
}

const DEFAULT: RemoteConfig = { baseUrl: "", sampler: "Euler a" };

function imageDataToPngDataUrl(img: ImageData): string {
  return imageDataToCanvas(img).toDataURL("image/png");
}

/** White where mask alpha==0 (regenerate the fresh region), black elsewhere. */
function maskFromKnown(known: ImageData): string {
  const c = makeCanvas(known.width, known.height);
  const ctx = ctxOf(c);
  const m = ctx.createImageData(known.width, known.height);
  for (let i = 0; i < known.data.length; i += 4) {
    const v = known.data[i + 3] > 8 ? 0 : 255;
    m.data[i] = m.data[i + 1] = m.data[i + 2] = v;
    m.data[i + 3] = 255;
  }
  ctx.putImageData(m, 0, 0);
  return c.toDataURL("image/png");
}

export class RemoteProvider implements DiffusionProvider {
  readonly id = "remote" as const;
  readonly caps: ProviderCapabilities = {
    local: false,
    requiresLoad: false,
    label: "Remote WebUI",
    blurb:
      "Point Vellum at your own AUTOMATIC1111 / Forge server for full-fat Stable Diffusion. Images are sent to that server.",
  };

  private cfg: RemoteConfig = { ...DEFAULT };

  configure(cfg: Partial<RemoteConfig>) {
    this.cfg = { ...this.cfg, ...cfg };
  }

  async isAvailable() {
    if (!this.cfg.baseUrl) return false;
    try {
      const r = await fetch(`${this.trim()}/sdapi/v1/sd-models`, {
        method: "GET",
      });
      return r.ok;
    } catch {
      return false;
    }
  }

  async load() {}
  isLoaded() {
    return !!this.cfg.baseUrl;
  }
  async dispose() {}

  private trim() {
    return this.cfg.baseUrl.replace(/\/+$/, "");
  }

  async generate(
    req: GenerateRequest,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<GenerateResult> {
    const t0 = performance.now();
    if (!this.cfg.baseUrl) throw new Error("No remote WebUI URL configured.");
    const { width: w, height: h } = req;
    const known = req.initImage ?? new ImageData(w, h);

    onProgress({ phase: "encoding", detail: "preparing tile", fraction: 0.15 });
    const seedVal = req.seed >= 0 ? req.seed : Math.floor(Math.random() * 2 ** 31);

    // Seed the fresh region so the server has structure to work with.
    const { init } = buildOutpaintInit(known, seedVal);

    const isOutpaint = !!req.initImage;
    const body: Record<string, unknown> = {
      prompt: req.prompt,
      negative_prompt: req.negativePrompt,
      steps: req.steps,
      cfg_scale: req.guidance,
      width: w,
      height: h,
      sampler_name: this.cfg.sampler,
      seed: seedVal,
      denoising_strength: req.strength,
    };
    if (isOutpaint) {
      body.init_images = [imageDataToPngDataUrl(init)];
      body.mask = maskFromKnown(known);
      body.inpainting_fill = 1; // original
      body.inpaint_full_res = false;
      body.mask_blur = Math.round(Math.min(w, h) * 0.06) + 4;
      body.initial_noise_multiplier = 1.0;
    }

    onProgress({ phase: "sampling", detail: "server sampling", fraction: 0.5 });
    const endpoint = isOutpaint ? "img2img" : "txt2img";
    const resp = await fetch(`${this.trim()}/sdapi/v1/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!resp.ok) {
      throw new Error(
        `Remote WebUI ${endpoint} failed (${resp.status}). Check the URL & CORS.`,
      );
    }
    const json = (await resp.json()) as { images?: string[] };
    if (!json.images?.length) throw new Error("Remote WebUI returned no image.");

    onProgress({ phase: "decoding", fraction: 0.85 });
    const out = await loadImageData(
      `data:image/png;base64,${json.images[0].replace(/^data:image\/\w+;base64,/, "")}`,
    );

    onProgress({ phase: "compositing", fraction: 0.95 });
    const feather = Math.round(Math.min(w, h) * 0.1) + 4;
    const composed = isOutpaint ? compositeFeathered(known, out, feather) : out;

    onProgress({ phase: "done", fraction: 1 });
    return { image: composed, seed: seedVal, ms: performance.now() - t0 };
  }
}
