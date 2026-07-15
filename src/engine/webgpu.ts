// ============================================================================
// WebGPU engine — Stable Diffusion running ENTIRELY in the browser on the
// client's own GPU via onnxruntime-web. Nothing leaves the machine.
//
// Two configurations share this one class:
//   • SD-Turbo  — 1-step distilled model, no CFG. Fast (~seconds).
//     Weights: schmuell/sd-turbo-ort-web (single-file fp16 ONNX).
//   • SD 1.5    — the original, non-distilled model. Real multi-step Euler
//     sampler with classifier-free guidance. Slower, more controllable.
//     Weights: nmkd/stable-diffusion-1.5-onnx-fp16 (unet uses external data).
//
// Single-step turbo constants: timestep 999, sigma 14.6146, vae scale 0.18215.
// Multi-step SD1.5 builds a scaled-linear sigma schedule (betas 0.00085→0.012)
// and runs k-diffusion style Euler: denoised = x - sigma*eps, x += eps*dσ.
//
// The code reads tensor names from each session at load time so it adapts to
// small naming differences between the two weight repos.
// ============================================================================

import * as ort from "onnxruntime-web/webgpu";
import {
  buildOutpaintInit,
  compositeFeathered,
} from "../lib/imaging";
import type {
  DiffusionProvider,
  EngineProgress,
  GenerateRequest,
  GenerateResult,
  ProviderCapabilities,
  ProviderId,
} from "./types";
import { GenerationAbortError } from "./types";

const ORT_VERSION = "1.27.0";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1;

// --- model manifest (runtime-overridable so URLs can be corrected in-app) ----
export interface ModelManifest {
  base: string;
  textEncoder: string;
  unet: string;
  /** External-data (.pb) tensor file for the unet, if the export uses one. */
  unetWeights?: string;
  vaeDecoder: string;
  vaeEncoder: string;
  tokenizer: string; // HF id for the CLIP tokenizer
  latent: number; // latent spatial dim (64 => 512px)
  /** 1-step distilled (turbo) vs. full multi-step diffusion. */
  turbo: boolean;
}

export const TURBO_MANIFEST: ModelManifest = {
  base: "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main",
  textEncoder: "text_encoder/model.onnx",
  unet: "unet/model.onnx",
  vaeDecoder: "vae_decoder/model.onnx",
  vaeEncoder: "vae_encoder/model.onnx",
  tokenizer: "Xenova/clip-vit-base-patch16",
  latent: 64,
  turbo: true,
};

// Full SD 1.5 (non-turbo). unet ships as a 1 MB graph + a 1.7 GB weights.pb
// external-data file; onnxruntime-web loads them together via `externalData`.
export const SD15_MANIFEST: ModelManifest = {
  base: "https://huggingface.co/nmkd/stable-diffusion-1.5-onnx-fp16/resolve/main",
  textEncoder: "text_encoder/model.onnx",
  unet: "unet/model.onnx",
  unetWeights: "unet/weights.pb",
  vaeDecoder: "vae_decoder/model.onnx",
  vaeEncoder: "vae_encoder/model.onnx",
  tokenizer: "Xenova/clip-vit-base-patch16",
  latent: 64,
  turbo: false,
};

// Backwards-compatible alias.
export const DEFAULT_MANIFEST = TURBO_MANIFEST;

const SIGMA_MAX = 14.6146;
const VAE_SCALE = 0.18215;
const TURBO_TIMESTEP = 999n;
const NUM_TRAIN = 1000;
const BETA_START = 0.00085;
const BETA_END = 0.012;

// ---------- fp16 helpers ------------------------------------------------------
function packF16(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = float32ToFloat16(src[i]);
  return out;
}
function unpackF16(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = float16ToFloat32(src[i]);
  return out;
}
function float32ToFloat16(val: number): number {
  floatView[0] = val;
  const x = int32View[0];
  const sign = (x >> 16) & 0x8000;
  let exp = ((x >> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    const round = 1 << (shift - 1);
    const h = (mant + round) >> shift;
    return sign | h;
  } else if (exp >= 0x1f) {
    return sign | 0x7c00;
  }
  const round = 0x00001000;
  if (mant & round) {
    mant += round;
    if (mant & 0x800000) { mant = 0; exp += 1; if (exp >= 0x1f) return sign | 0x7c00; }
  }
  return sign | (exp << 10) | (mant >> 13);
}
function float16ToFloat32(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  let val: number;
  if (exp === 0) val = frac / 1024 * Math.pow(2, -14);
  else if (exp === 0x1f) val = frac ? NaN : Infinity;
  else val = (1 + frac / 1024) * Math.pow(2, exp - 15);
  return sign ? -val : val;
}
const floatBuf = new ArrayBuffer(4);
const floatView = new Float32Array(floatBuf);
const int32View = new Int32Array(floatBuf);

// seeded gaussian noise (Box-Muller) over mulberry32
function seededGaussian(n: number, seed: number): Float32Array {
  let a = seed >>> 0;
  const rnd = () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(1e-7, rnd());
    const u2 = rnd();
    const mag = Math.sqrt(-2 * Math.log(u1));
    out[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

// SD training noise schedule (scaled-linear betas) -> per-timestep sigmas.
let TRAIN_SIGMAS: Float32Array | null = null;
function trainSigmas(): Float32Array {
  if (TRAIN_SIGMAS) return TRAIN_SIGMAS;
  const s = new Float32Array(NUM_TRAIN);
  const sqrtStart = Math.sqrt(BETA_START);
  const sqrtEnd = Math.sqrt(BETA_END);
  let acp = 1;
  for (let i = 0; i < NUM_TRAIN; i++) {
    const beta = Math.pow(sqrtStart + (sqrtEnd - sqrtStart) * (i / (NUM_TRAIN - 1)), 2);
    acp *= 1 - beta;
    s[i] = Math.sqrt((1 - acp) / acp);
  }
  TRAIN_SIGMAS = s;
  return s;
}

// Pick `steps` discrete timesteps (high noise -> low) plus a trailing sigma=0.
function buildSchedule(steps: number): { sigmas: number[]; ts: number[] } {
  const table = trainSigmas();
  const sigmas: number[] = [];
  const ts: number[] = [];
  for (let i = 0; i < steps; i++) {
    const frac = steps === 1 ? 0 : i / (steps - 1);
    const idx = Math.round((1 - frac) * (NUM_TRAIN - 1));
    ts.push(idx);
    sigmas.push(table[idx]);
  }
  sigmas.push(0);
  return { sigmas, ts };
}

export class WebGpuProvider implements DiffusionProvider {
  readonly id: ProviderId;
  readonly caps: ProviderCapabilities;

  private manifest: ModelManifest;
  private sessions: {
    text?: ort.InferenceSession;
    unet?: ort.InferenceSession;
    vaeDec?: ort.InferenceSession;
    vaeEnc?: ort.InferenceSession;
  } = {};
  private tokenizer: any = null;
  private loaded = false;
  private names: Record<string, { in: readonly string[]; out: readonly string[] }> = {};

  constructor(
    id: ProviderId = "webgpu",
    manifest: ModelManifest = TURBO_MANIFEST,
    caps?: ProviderCapabilities,
  ) {
    this.id = id;
    this.manifest = { ...manifest };
    this.caps = caps ?? {
      local: true,
      requiresLoad: true,
      label: "WebGPU · your GPU",
      blurb:
        "Stable Diffusion Turbo runs on your own graphics card. ~2.5 GB one-time download, then fully offline & private.",
    };
  }

  configure(m: Partial<ModelManifest>) {
    this.manifest = { ...this.manifest, ...m };
  }

  async isAvailable(): Promise<boolean> {
    const gpu = (navigator as any).gpu;
    if (!gpu) return false;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) return false;
      // fp16 weights need the shader-f16 GPU feature.
      return adapter.features?.has?.("shader-f16") ?? false;
    } catch {
      return false;
    }
  }

  isLoaded() {
    return this.loaded;
  }

  // Stream a URL into the Cache API (low JS-heap) and return its ArrayBuffer.
  // Teeing the body lets us report byte progress while the browser writes the
  // other branch straight to disk — avoiding the multi-GB triple-buffering that
  // previously OOM-crashed the tab ("this page can't be reached").
  private async fetchCached(
    url: string,
    onProgress: (p: EngineProgress) => void,
    label: string,
  ): Promise<ArrayBuffer> {
    const cache = await caches.open(this.cacheName());
    let hit = await cache.match(url);
    if (!hit) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Download failed: ${label} (${resp.status})`);
      const total = Number(resp.headers.get("content-length")) || 0;
      if (resp.body) {
        const [toCache, toCount] = resp.body.tee();
        const put = cache.put(
          url,
          new Response(toCache, total ? { headers: { "content-length": String(total) } } : undefined),
        );
        const reader = toCount.getReader();
        let received = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          received += value.length;
          onProgress({
            phase: "downloading",
            fraction: total ? received / total : undefined,
            detail: `${label} · ${(received / 1e6).toFixed(0)} MB${total ? ` / ${(total / 1e6).toFixed(0)} MB` : ""}`,
          });
        }
        await put;
      } else {
        const buf = await resp.arrayBuffer();
        await cache.put(url, new Response(buf));
      }
      hit = await cache.match(url);
      if (!hit) throw new Error(`Could not cache ${label}.`);
    }
    return hit.arrayBuffer();
  }

  private cacheName() {
    return `vellum-${this.id}`;
  }

  // Ask the browser to mark our storage bucket persistent. The multi-GB model
  // cache lives in the Cache API, which by default is "best-effort" storage and
  // gets EVICTED under disk pressure — that made the weights re-download on
  // every visit. Persistent storage is exempt from automatic eviction, so the
  // one-time download actually stays one-time. No-op if already granted or if
  // the API is unavailable (older browsers / insecure context).
  private async requestPersistence(onProgress: (p: EngineProgress) => void): Promise<void> {
    try {
      const storage = navigator.storage;
      if (!storage?.persist) return;
      if (await storage.persisted?.()) return; // already persistent — nothing to do
      const granted = await storage.persist();
      if (!granted) {
        // Chrome auto-decides based on site engagement; a denial just means the
        // cache remains evictable. Surface it so a re-download isn't a mystery.
        onProgress({
          phase: "downloading",
          detail: "note: browser did not grant persistent storage — cached models may be evicted later",
        });
      }
    } catch {
      /* storage API unavailable — proceed without persistence */
    }
  }

  private async makeSession(
    url: string,
    label: string,
    opts: ort.InferenceSession.SessionOptions,
    onProgress: (p: EngineProgress) => void,
  ): Promise<ort.InferenceSession> {
    const buf = await this.fetchCached(url, onProgress, label);
    onProgress({ phase: "compiling", detail: label });
    const s = await ort.InferenceSession.create(buf, opts);
    return s; // `buf` is now unreferenced and can be GC'd before the next model
  }

  async load(onProgress: (p: EngineProgress) => void): Promise<void> {
    if (this.loaded) return;
    if (!(await this.isAvailable())) {
      throw new Error(
        "WebGPU with shader-f16 is not available in this browser. Try Chrome/Edge 113+ on a discrete GPU, or use another engine.",
      );
    }
    // Mark storage persistent BEFORE downloading so the ~2.5 GB we're about to
    // cache survives across sessions instead of being evicted (root cause of the
    // "re-downloads every time" bug).
    await this.requestPersistence(onProgress);

    const m = this.manifest;
    const url = (p: string) => `${m.base}/${p}`;
    const L = m.latent;

    const common: ort.InferenceSession.SessionOptions = {
      executionProviders: ["webgpu"],
      enableMemPattern: false,
      enableCpuMemArena: false,
      extra: { session: { disable_prepacking: "1" } },
    };

    // Tokenizer (small; pulled from HF by transformers.js).
    onProgress({ phase: "downloading", detail: "tokenizer", fraction: 0.01 });
    const { AutoTokenizer } = await import("@huggingface/transformers");
    this.tokenizer = await AutoTokenizer.from_pretrained(m.tokenizer);

    // Text encoder
    this.sessions.text = await this.makeSession(url(m.textEncoder), "text encoder", {
      ...common,
      freeDimensionOverrides: { batch_size: 1 },
    }, onProgress);

    // UNet (largest). SD1.5 uses an external-data weights file.
    if (m.unetWeights) {
      const graphBuf = await this.fetchCached(url(m.unet), onProgress, "unet graph");
      const weightsBuf = await this.fetchCached(url(m.unetWeights), onProgress, "unet weights");
      onProgress({ phase: "compiling", detail: "unet" });
      this.sessions.unet = await ort.InferenceSession.create(graphBuf, {
        ...common,
        freeDimensionOverrides: {
          batch_size: 1, num_channels: 4, height: L, width: L, sequence_length: 77,
        },
        externalData: [
          // The graph references the .pb by its basename.
          { path: m.unetWeights.split("/").pop()!, data: new Uint8Array(weightsBuf) },
        ],
      } as ort.InferenceSession.SessionOptions);
    } else {
      this.sessions.unet = await this.makeSession(url(m.unet), "unet", {
        ...common,
        freeDimensionOverrides: {
          batch_size: 1, num_channels: 4, height: L, width: L, sequence_length: 77,
        },
      }, onProgress);
    }

    // VAE decoder
    this.sessions.vaeDec = await this.makeSession(url(m.vaeDecoder), "vae decoder", {
      ...common,
      freeDimensionOverrides: {
        batch_size: 1, num_channels_latent: 4, height_latent: L, width_latent: L,
      },
    }, onProgress);

    // VAE encoder (for img2img / outpaint)
    this.sessions.vaeEnc = await this.makeSession(url(m.vaeEncoder), "vae encoder", {
      ...common,
      freeDimensionOverrides: {
        batch_size: 1, num_channels: 3, height: L * 8, width: L * 8,
      },
    }, onProgress);

    this.names = {
      text: { in: this.sessions.text.inputNames, out: this.sessions.text.outputNames },
      unet: { in: this.sessions.unet.inputNames, out: this.sessions.unet.outputNames },
      vaeDec: { in: this.sessions.vaeDec.inputNames, out: this.sessions.vaeDec.outputNames },
      vaeEnc: { in: this.sessions.vaeEnc.inputNames, out: this.sessions.vaeEnc.outputNames },
    };

    this.loaded = true;
    onProgress({ phase: "done", fraction: 1 });
  }

  async dispose(): Promise<void> {
    for (const s of Object.values(this.sessions)) await s?.release?.();
    this.sessions = {};
    this.loaded = false;
  }

  // ------------------------------------------------------------------ encode
  private async encodeText(prompt: string): Promise<ort.Tensor> {
    const enc = await this.tokenizer(prompt, {
      padding: "max_length",
      max_length: 77,
      truncation: true,
      return_tensor: false,
    });
    const ids = (enc.input_ids as number[]).slice(0, 77);
    while (ids.length < 77) ids.push(0);
    const input = new ort.Tensor("int32", Int32Array.from(ids), [1, 77]);
    const inName = this.names.text.in[0];
    const res = await this.sessions.text!.run({ [inName]: input });
    return res[this.names.text.out[0]] as ort.Tensor;
  }

  private async vaeEncode(img: ImageData, L: number): Promise<Float32Array> {
    // pixels -> [1,3,H,W] in [-1,1], fp16
    const size = L * 8;
    const scaled = resizeImageData(img, size, size);
    const chw = new Float32Array(3 * size * size);
    const d = scaled.data;
    for (let p = 0, i = 0; p < d.length; p += 4, i++) {
      chw[i] = d[p] / 127.5 - 1;
      chw[size * size + i] = d[p + 1] / 127.5 - 1;
      chw[2 * size * size + i] = d[p + 2] / 127.5 - 1;
    }
    const t = new ort.Tensor("float16", packF16(chw), [1, 3, size, size]);
    const inName = this.names.vaeEnc.in[0];
    const res = await this.sessions.vaeEnc!.run({ [inName]: t });
    const out = res[this.names.vaeEnc.out[0]] as ort.Tensor;
    const latent = unpackF16(out.data as Uint16Array);
    for (let i = 0; i < latent.length; i++) latent[i] *= VAE_SCALE;
    return latent; // [1,4,L,L] flattened
  }

  // Run the unet once and return the predicted noise (eps) for a pre-scaled
  // latent. `scaled` is already x / sqrt(sigma^2+1).
  private async runUnet(
    scaled: Float32Array,
    timestep: bigint,
    hidden: ort.Tensor,
    L: number,
  ): Promise<Float32Array> {
    const sample = new ort.Tensor("float16", packF16(scaled), [1, 4, L, L]);
    const ts = new ort.Tensor("int64", BigInt64Array.from([timestep]), [1]);
    const feeds: Record<string, ort.Tensor> = {};
    const inNames = this.names.unet.in;
    for (const n of inNames) {
      const ln = n.toLowerCase();
      if (ln.includes("sample")) feeds[n] = sample;
      else if (ln.includes("timestep") || ln === "t") feeds[n] = ts;
      else if (ln.includes("hidden") || ln.includes("encoder")) feeds[n] = hidden;
    }
    if (!Object.keys(feeds).length && inNames.length >= 3) {
      feeds[inNames[0]] = sample; feeds[inNames[1]] = ts; feeds[inNames[2]] = hidden;
    }
    const res = await this.sessions.unet!.run(feeds);
    return unpackF16(res[this.names.unet.out[0]].data as Uint16Array);
  }

  private async vaeDecode(latent: Float32Array, L: number): Promise<ImageData> {
    const inp = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) inp[i] = latent[i] / VAE_SCALE;
    const t = new ort.Tensor("float16", packF16(inp), [1, 4, L, L]);
    const inName = this.names.vaeDec.in[0];
    const res = await this.sessions.vaeDec!.run({ [inName]: t });
    const out = res[this.names.vaeDec.out[0]] as ort.Tensor;
    const size = L * 8;
    const chw = unpackF16(out.data as Uint16Array);
    const img = new ImageData(size, size);
    const pl = size * size;
    for (let i = 0, p = 0; i < pl; i++, p += 4) {
      img.data[p] = clamp255((chw[i] / 2 + 0.5) * 255);
      img.data[p + 1] = clamp255((chw[pl + i] / 2 + 0.5) * 255);
      img.data[p + 2] = clamp255((chw[2 * pl + i] / 2 + 0.5) * 255);
      img.data[p + 3] = 255;
    }
    return img;
  }

  // ---------------------------------------------------------------- generate
  async generate(
    req: GenerateRequest,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<GenerateResult> {
    if (!this.loaded) throw new Error("WebGPU engine not loaded.");
    const t0 = performance.now();
    const L = this.manifest.latent;
    const size = L * 8;
    const seed = req.seed >= 0 ? req.seed : Math.floor(Math.random() * 2 ** 31);
    const check = () => { if (signal?.aborted) throw new GenerationAbortError(); };
    check();

    onProgress({ phase: "encoding", detail: "text", fraction: 0.08 });
    const hidden = await this.encodeText(req.prompt);
    // Unconditional embedding for classifier-free guidance (non-turbo only).
    const useCfg = !this.manifest.turbo && req.guidance > 1;
    const uncond = useCfg ? await this.encodeText(req.negativePrompt || "") : null;

    const knownAtSize = req.initImage ? resizeImageData(req.initImage, size, size) : null;
    const isImg = !!(req.initImage && knownAtSize);

    let latent: Float32Array;
    let finalLatent: Float32Array;

    if (this.manifest.turbo) {
      // ---- single-step turbo path ----
      let sigma: number;
      if (isImg && knownAtSize) {
        const { init } = buildOutpaintInit(knownAtSize, seed);
        onProgress({ phase: "encoding", detail: "vae", fraction: 0.25 });
        const z = await this.vaeEncode(init, L);
        sigma = Math.max(0.05, req.strength) * SIGMA_MAX;
        const noise = seededGaussian(z.length, seed);
        latent = new Float32Array(z.length);
        for (let i = 0; i < z.length; i++) latent[i] = z[i] + sigma * noise[i];
      } else {
        sigma = SIGMA_MAX;
        const noise = seededGaussian(4 * L * L, seed);
        latent = new Float32Array(noise.length);
        for (let i = 0; i < noise.length; i++) latent[i] = noise[i] * sigma;
      }
      check();
      onProgress({ phase: "sampling", step: 1, totalSteps: 1, fraction: 0.6, detail: "1 step (turbo)" });
      const cin = 1 / Math.sqrt(sigma * sigma + 1);
      const scaled = new Float32Array(latent.length);
      for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] * cin;
      const eps = await this.runUnet(scaled, TURBO_TIMESTEP, hidden, L);
      finalLatent = new Float32Array(latent.length);
      for (let i = 0; i < latent.length; i++) finalLatent[i] = latent[i] - sigma * eps[i];
    } else {
      // ---- multi-step Euler path (SD 1.5) ----
      const steps = Math.max(2, Math.min(50, Math.round(req.steps) || 20));
      const { sigmas, ts } = buildSchedule(steps);
      let start = 0;
      const n = 4 * L * L;
      const noise = seededGaussian(n, seed);
      if (isImg && knownAtSize) {
        const { init } = buildOutpaintInit(knownAtSize, seed);
        onProgress({ phase: "encoding", detail: "vae", fraction: 0.2 });
        const z = await this.vaeEncode(init, L);
        // strength => how many of the later steps to run (denoise from noised z0)
        start = Math.min(steps - 1, Math.max(0, Math.round((1 - req.strength) * steps)));
        latent = new Float32Array(n);
        for (let i = 0; i < n; i++) latent[i] = z[i] + sigmas[start] * noise[i];
      } else {
        latent = new Float32Array(n);
        for (let i = 0; i < n; i++) latent[i] = noise[i] * sigmas[start];
      }

      const total = steps - start;
      for (let i = start; i < steps; i++) {
        check();
        const sigma = sigmas[i];
        const cin = 1 / Math.sqrt(sigma * sigma + 1);
        const scaled = new Float32Array(latent.length);
        for (let k = 0; k < latent.length; k++) scaled[k] = latent[k] * cin;
        const tstep = BigInt(ts[i]);
        const epsCond = await this.runUnet(scaled, tstep, hidden, L);
        let eps = epsCond;
        if (useCfg && uncond) {
          const epsUncond = await this.runUnet(scaled, tstep, uncond, L);
          eps = new Float32Array(epsCond.length);
          for (let k = 0; k < eps.length; k++)
            eps[k] = epsUncond[k] + req.guidance * (epsCond[k] - epsUncond[k]);
        }
        // Euler: d = eps; x += d * (sigma_next - sigma)
        const dsig = sigmas[i + 1] - sigma;
        for (let k = 0; k < latent.length; k++) latent[k] += eps[k] * dsig;
        onProgress({
          phase: "sampling",
          step: i - start + 1,
          totalSteps: total,
          fraction: 0.15 + 0.7 * ((i - start + 1) / total),
          detail: `step ${i - start + 1}/${total}`,
        });
      }
      finalLatent = latent;
    }

    (hidden as any).dispose?.();
    (uncond as any)?.dispose?.();

    check();
    onProgress({ phase: "decoding", fraction: 0.9 });
    const decoded = await this.vaeDecode(finalLatent, L);

    const outAtReq = resizeImageData(decoded, req.width, req.height);
    let image = outAtReq;
    if (req.initImage) {
      onProgress({ phase: "compositing", fraction: 0.96 });
      const feather = Math.round(Math.min(req.width, req.height) * 0.1) + 4;
      image = compositeFeathered(req.initImage, outAtReq, feather);
    }
    onProgress({ phase: "done", fraction: 1 });
    return { image, seed, ms: performance.now() - t0 };
  }
}

// -------- small canvas utils (kept local to avoid import cycles) -------------
function resizeImageData(img: ImageData, w: number, h: number): ImageData {
  if (img.width === w && img.height === h) return img;
  const src = document.createElement("canvas");
  src.width = img.width; src.height = img.height;
  src.getContext("2d")!.putImageData(img, 0, 0);
  const dst = document.createElement("canvas");
  dst.width = w; dst.height = h;
  const dctx = dst.getContext("2d", { willReadFrequently: true })!;
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, w, h);
  return dctx.getImageData(0, 0, w, h);
}
function clamp255(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}
