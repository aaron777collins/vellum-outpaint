// ============================================================================
// WebGPU engine — Stable Diffusion (SD-Turbo) running ENTIRELY in the browser
// on the client's own GPU via onnxruntime-web. Nothing leaves the machine.
//
// Recipe verified against Microsoft's onnxruntime-inference-examples sd-turbo
// demo + the schmuell/sd-turbo-ort-web weights (text_encoder, unet, vae_decoder,
// vae_encoder — all fp16). Single-step Euler (timestep 999, sigma 14.6146,
// vae scale 0.18215). img2img/outpaint uses the VAE encoder + pixel-space
// feather composite so it shares the same seam-hiding path as the other engines.
//
// The code is defensive about exact tensor names (it reads them from each
// session at load time) because those couldn't be verified without a GPU here.
// Any failure surfaces a clear message and the app falls back to demo/remote.
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
  vaeDecoder: string;
  vaeEncoder: string;
  tokenizer: string; // HF id for the CLIP tokenizer
  latent: number; // latent spatial dim (64 => 512px)
}

export const DEFAULT_MANIFEST: ModelManifest = {
  base: "https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main",
  textEncoder: "text_encoder/model.onnx",
  unet: "unet/model.onnx",
  vaeDecoder: "vae_decoder/model.onnx",
  vaeEncoder: "vae_encoder/model.onnx",
  tokenizer: "Xenova/clip-vit-base-patch16",
  latent: 64,
};

const SIGMA_MAX = 14.6146;
const VAE_SCALE = 0.18215;
const TIMESTEP = 999n;
const CACHE = "vellum-sd-turbo";

// ---------- fp16 helpers ------------------------------------------------------
function f32to16(v: number): number {
  f16buf[0] = v;
  return f16view[0];
}
const f16buf = new Float32Array(1);
const f16view = new Uint16Array(f16buf.buffer); // NOTE: not a real f16 cast

// Proper float32 -> float16 bit packing.
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
function float32ToFloat16(val: float): number {
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
    let h = (mant + round) >> shift;
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
type float = number;
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

export class WebGpuProvider implements DiffusionProvider {
  readonly id = "webgpu" as const;
  readonly caps: ProviderCapabilities = {
    local: true,
    requiresLoad: true,
    label: "WebGPU · your GPU",
    blurb:
      "Stable Diffusion Turbo runs on your own graphics card. ~2.5 GB one-time download, then fully offline & private.",
  };

  private manifest: ModelManifest = { ...DEFAULT_MANIFEST };
  private sessions: {
    text?: ort.InferenceSession;
    unet?: ort.InferenceSession;
    vaeDec?: ort.InferenceSession;
    vaeEnc?: ort.InferenceSession;
  } = {};
  private tokenizer: any = null;
  private loaded = false;
  private names: Record<string, { in: readonly string[]; out: readonly string[] }> = {};

  configure(m: Partial<ModelManifest>) {
    this.manifest = { ...this.manifest, ...m };
  }

  async isAvailable(): Promise<boolean> {
    const gpu = (navigator as any).gpu;
    if (!gpu) return false;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) return false;
      // SD-Turbo fp16 weights need the shader-f16 GPU feature.
      return adapter.features?.has?.("shader-f16") ?? false;
    } catch {
      return false;
    }
  }

  isLoaded() {
    return this.loaded;
  }

  private async fetchCached(
    url: string,
    onProgress: (p: EngineProgress) => void,
    label: string,
  ): Promise<ArrayBuffer> {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) return hit.arrayBuffer();
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Download failed: ${label} (${resp.status})`);
    const total = Number(resp.headers.get("content-length")) || 0;
    const reader = resp.body?.getReader();
    if (!reader) {
      const buf = await resp.arrayBuffer();
      await cache.put(url, new Response(buf.slice(0)));
      return buf;
    }
    const chunks: Uint8Array[] = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress({
        phase: "downloading",
        fraction: total ? received / total : undefined,
        detail: `${label} · ${(received / 1e6).toFixed(0)} MB${total ? ` / ${(total / 1e6).toFixed(0)} MB` : ""}`,
      });
    }
    const merged = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    await cache.put(url, new Response(merged.slice(0).buffer));
    return merged.buffer;
  }

  async load(onProgress: (p: EngineProgress) => void): Promise<void> {
    if (this.loaded) return;
    if (!(await this.isAvailable())) {
      throw new Error(
        "WebGPU with shader-f16 is not available in this browser. Try Chrome/Edge 113+ on a discrete GPU, or use another engine.",
      );
    }
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
    const teBuf = await this.fetchCached(url(m.textEncoder), onProgress, "text encoder");
    onProgress({ phase: "compiling", detail: "text encoder" });
    this.sessions.text = await ort.InferenceSession.create(teBuf, {
      ...common,
      freeDimensionOverrides: { batch_size: 1 },
    });

    // UNet (largest)
    const unetBuf = await this.fetchCached(url(m.unet), onProgress, "unet");
    onProgress({ phase: "compiling", detail: "unet" });
    this.sessions.unet = await ort.InferenceSession.create(unetBuf, {
      ...common,
      freeDimensionOverrides: {
        batch_size: 1, num_channels: 4, height: L, width: L, sequence_length: 77,
      },
    });

    // VAE decoder
    const vdBuf = await this.fetchCached(url(m.vaeDecoder), onProgress, "vae decoder");
    onProgress({ phase: "compiling", detail: "vae decoder" });
    this.sessions.vaeDec = await ort.InferenceSession.create(vdBuf, {
      ...common,
      freeDimensionOverrides: {
        batch_size: 1, num_channels_latent: 4, height_latent: L, width_latent: L,
      },
    });

    // VAE encoder (for img2img / outpaint)
    const veBuf = await this.fetchCached(url(m.vaeEncoder), onProgress, "vae encoder");
    onProgress({ phase: "compiling", detail: "vae encoder" });
    this.sessions.vaeEnc = await ort.InferenceSession.create(veBuf, {
      ...common,
      freeDimensionOverrides: {
        batch_size: 1, num_channels: 3, height: L * 8, width: L * 8,
      },
    });

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

  private async unetStep(
    latent: Float32Array,
    hidden: ort.Tensor,
    sigma: number,
    L: number,
  ): Promise<Float32Array> {
    // scale_model_input
    const s = 1 / Math.sqrt(sigma * sigma + 1);
    const scaled = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) scaled[i] = latent[i] * s;
    const sample = new ort.Tensor("float16", packF16(scaled), [1, 4, L, L]);
    const ts = new ort.Tensor("int64", BigInt64Array.from([TIMESTEP]), [1]);
    const feeds: Record<string, ort.Tensor> = {};
    const inNames = this.names.unet.in;
    // map by heuristics on names
    for (const n of inNames) {
      const ln = n.toLowerCase();
      if (ln.includes("sample")) feeds[n] = sample;
      else if (ln.includes("timestep") || ln === "t") feeds[n] = ts;
      else if (ln.includes("hidden") || ln.includes("encoder")) feeds[n] = hidden;
    }
    // fallback positional if heuristics missed
    if (!Object.keys(feeds).length && inNames.length >= 3) {
      feeds[inNames[0]] = sample; feeds[inNames[1]] = ts; feeds[inNames[2]] = hidden;
    }
    const res = await this.sessions.unet!.run(feeds);
    const noise = unpackF16(res[this.names.unet.out[0]].data as Uint16Array);
    // Euler step to sigma_next = 0: denoised = latent - sigma * noise
    const denoised = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) denoised[i] = latent[i] - sigma * noise[i];
    return denoised;
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
    if (signal?.aborted) throw new GenerationAbortError();

    onProgress({ phase: "encoding", detail: "text", fraction: 0.1 });
    const hidden = await this.encodeText(req.prompt);

    const isImg = !!req.initImage;
    // known tile at working resolution (for compositing back)
    const knownAtSize = req.initImage ? resizeImageData(req.initImage, size, size) : null;

    let latent: Float32Array;
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

    if (signal?.aborted) throw new GenerationAbortError();
    onProgress({ phase: "sampling", step: 1, totalSteps: 1, fraction: 0.6, detail: "1 step (turbo)" });
    const denoised = await this.unetStep(latent, hidden, sigma, L);
    (hidden as any).dispose?.();

    onProgress({ phase: "decoding", fraction: 0.85 });
    const decoded = await this.vaeDecode(denoised, L);

    // resize decoded back to requested tile size, then composite
    const outAtReq = resizeImageData(decoded, req.width, req.height);
    let image = outAtReq;
    if (req.initImage) {
      onProgress({ phase: "compositing", fraction: 0.95 });
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

// silence unused (kept for potential gpu-buffer path)
void f32to16;
