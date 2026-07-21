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
  buildLatentMask,
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
// Ask ORT (and thus the WebGPU adapter request) for the HIGH-PERFORMANCE GPU.
// On hybrid-GPU laptops WebGPU otherwise defaults to the integrated GPU, whose
// fp16 support is often flaky / memory-starved — a prime suspect for the "works
// the first time, then the frame is black" report (the algorithm itself is
// verified correct on CPU end-to-end). This selects the discrete GPU instead.
try {
  (ort.env as unknown as { webgpu?: { powerPreference?: string } }).webgpu ??= {};
  (ort.env as unknown as { webgpu: { powerPreference?: string } }).webgpu.powerPreference =
    "high-performance";
} catch {
  /* older ort without env.webgpu — harmless */
}

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
const TURBO_TIMESTEP = 999;
const NUM_TRAIN = 1000;
const BETA_START = 0.00085;
const BETA_END = 0.012;

// --- robustness knobs --------------------------------------------------------
// A single ONNX op finishes in well under a second on any working GPU; if one
// takes longer than this the WebGPU device is almost certainly hung/lost (a lost
// device leaves run() pending FOREVER, which is the "never finishes / infinite
// loop" report). The watchdog converts that hang into a catchable error.
const OP_TIMEOUT_MS = 90_000;
const RELOAD_TIMEOUT_MS = 180_000;
// How many times to rebuild the sessions + retry a generation that came back
// degenerate before giving up. Bounded so a deterministically-black GPU can't
// loop endlessly.
const MAX_ATTEMPTS = 2;

// Thrown internally when a produced latent/frame is degenerate (all-black /
// NaN-poisoned). Caught by the attempt loop to trigger a bounded reload+retry.
class DegenerateFrameError extends Error {
  constructor(msg = "degenerate frame") {
    super(msg);
    this.name = "DegenerateFrameError";
  }
}

// Race a promise against a deadline AND an abort signal. A hung WebGPU device
// (device-lost commonly leaves run()/create() pending indefinitely) or a user
// cancel becomes a rejected promise instead of an infinite wait. Legitimate ops
// resolve far inside the deadline, so this never fires in the happy path.
function withDeadline<T>(p: Promise<T>, ms: number, label: string, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const done = (fn: (v: unknown) => void, v: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      fn(v);
    };
    const timer = setTimeout(
      () =>
        done(
          reject as (v: unknown) => void,
          new Error(
            `${label} timed out after ${Math.round(ms / 1000)}s — the GPU is unresponsive. ` +
              "Reload the page, or switch to the Remote engine.",
          ),
        ),
      ms,
    );
    const onAbort = () => done(reject as (v: unknown) => void, new GenerationAbortError());
    if (signal?.aborted) {
      done(reject as (v: unknown) => void, new GenerationAbortError());
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });
    p.then(
      (v) => done(resolve as (v: unknown) => void, v),
      (e) => done(reject as (v: unknown) => void, e),
    );
  });
}

// Replace non-finite values with 0 and clamp to a sane magnitude, in place. One
// fp16 overflow in a single op would otherwise turn the whole latent NaN and
// decode to a black frame; sanitizing keeps a stray overflow from poisoning the
// rest. Returns the fraction of values that were non-finite — a large fraction
// means the tensor is genuinely garbage (a real GPU failure), not a stray spike.
function sanitizeLatent(a: Float32Array, clampMag = 1e4): number {
  let bad = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!Number.isFinite(v)) {
      a[i] = 0;
      bad++;
    } else if (v > clampMag) a[i] = clampMag;
    else if (v < -clampMag) a[i] = -clampMag;
  }
  return a.length ? bad / a.length : 1;
}

// Health of a VAE-encoded latent. A well-formed SD latent (after the 0.18215
// scale) is small and varied — values roughly N(0, ~1), so RMS is order-1 and
// variance is clearly non-zero. The fp16 VAE encoder overflowing on a real photo
// instead yields either a NaN/Inf-riddled tensor (high `bad` fraction), a
// blown-up one (huge RMS), or — once sanitized/clamped — a near-constant one
// (dead). Any of those decodes to black. Sampled so it stays cheap on 16k-elem
// latents. Used to decide whether to retry the encode softer or bail to noise.
function latentHealth(a: Float32Array): { rms: number; dead: boolean } {
  let sum = 0;
  let sum2 = 0;
  let n = 0;
  const stride = Math.max(1, Math.floor(a.length / 4096));
  for (let i = 0; i < a.length; i += stride) {
    const v = a[i];
    sum += v;
    sum2 += v * v;
    n++;
  }
  if (n === 0) return { rms: 0, dead: true };
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  return { rms: Math.sqrt(sum2 / n), dead: !Number.isFinite(variance) || variance < 1e-8 };
}

// A latent that is essentially constant (near-zero variance) decodes to a flat
// frame — the pre-decode signature of a failed GPU run. Sampled so it stays cheap.
function latentIsDead(a: Float32Array): boolean {
  let sum = 0;
  let sum2 = 0;
  let n = 0;
  const stride = Math.max(1, Math.floor(a.length / 4096));
  for (let i = 0; i < a.length; i += stride) {
    const v = a[i];
    sum += v;
    sum2 += v * v;
    n++;
  }
  if (n === 0) return true;
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  return !Number.isFinite(variance) || variance < 1e-8;
}

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

// Per-session tensor metadata: input/output NAMES plus their declared DTYPES.
// The two weight repos DISAGREE on I/O precision, so we cannot assume fp16:
//   • SD-Turbo (schmuell): unet/vae_decoder take fp32 I/O; timestep is int64;
//     vae_encoder is genuinely fp16.
//   • SD 1.5 (nmkd):       unet is fully fp16 — including an fp16 timestep.
// We read each model's real dtypes at load and build/read tensors to match,
// converting from a canonical Float32Array. Feeding fp16 into an fp32 input
// (the previous hardcoded assumption) errored out every single generation.
interface SessMeta {
  inNames: readonly string[];
  outNames: readonly string[];
  in: Record<string, string>; // input  name -> Tensor.Type ("float16" | "float32" | "int64" | ...)
  out: Record<string, string>; // output name -> Tensor.Type
}
function readSessMeta(s: ort.InferenceSession): SessMeta {
  const inT: Record<string, string> = {};
  const outT: Record<string, string> = {};
  for (const m of s.inputMetadata) if ((m as any).isTensor) inT[m.name] = String((m as any).type);
  for (const m of s.outputMetadata) if ((m as any).isTensor) outT[m.name] = String((m as any).type);
  return { inNames: s.inputNames, outNames: s.outputNames, in: inT, out: outT };
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
  private meta: Record<string, SessMeta> = {};
  // Serialize generations at the provider level. ONNX sessions are NOT reentrant:
  // two overlapping run()s on the same session corrupt each other's GPU buffers
  // (a route to black/garbage frames). The store already guards, but a provider
  // that can be driven from anywhere must protect itself too.
  private inFlight = false;

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
      // Prefer the discrete GPU (see powerPreference note at module top).
      const adapter =
        (await gpu.requestAdapter({ powerPreference: "high-performance" })) ??
        (await gpu.requestAdapter());
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
  //
  // "RE-DOWNLOADS EVERY REFRESH" ROOT CAUSE (diagnosed 2026-07-16, with a real
  // headless-Chromium reproduction): it is NOT a cache-key/Vary problem — a
  // bare-URL `cache.match` (even for the streamed tee() put above) is verified to
  // survive a full browser-context close+reopen. The real cause is EVICTION:
  // `navigator.storage.persist()` returns false on a fresh, low-engagement origin
  // (confirmed in-browser), which leaves the ~2.5 GB weight cache as *best-effort*
  // storage — and the browser reclaims best-effort storage under disk pressure,
  // so on the next visit the entry is simply gone. The durable fix lives in
  // `requestPersistence()` + the PWA manifest (installable / bookmarked / engaged
  // origins get persistence GRANTED, which exempts the cache from eviction); when
  // the browser still declines, we surface that plainly so a re-download is an
  // understood state, not a mystery. `{ ignoreVary: true }` below is kept purely
  // as cheap defensiveness against any future header-bearing cache key.
  private async fetchCached(
    url: string,
    onProgress: (p: EngineProgress) => void,
    label: string,
  ): Promise<ArrayBuffer> {
    const cache = await caches.open(this.cacheName());
    const MATCH = { ignoreVary: true, ignoreSearch: true } as const;
    let hit = await cache.match(url, MATCH);
    if (!hit) {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Download failed: ${label} (${resp.status})`);
      const total = Number(resp.headers.get("content-length")) || 0;
      // Store under a clean synthetic response: no Vary, no cross-origin headers,
      // just the length so progress works on a later cached read.
      const cacheHeaders = total ? { "content-length": String(total) } : undefined;
      if (resp.body) {
        const [toCache, toCount] = resp.body.tee();
        const put = cache.put(url, new Response(toCache, cacheHeaders ? { headers: cacheHeaders } : undefined));
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
        await cache.put(url, new Response(buf, cacheHeaders ? { headers: cacheHeaders } : undefined));
      }
      hit = await cache.match(url, MATCH);
      // Extremely defensive: if the store silently didn't land (private-mode
      // quota, etc.), fall back to a direct fetch so the load still succeeds
      // this session rather than hard-failing.
      if (!hit) {
        onProgress({ phase: "downloading", detail: `${label} · caching unavailable, using direct download` });
        const direct = await fetch(url);
        if (!direct.ok) throw new Error(`Could not cache or fetch ${label} (${direct.status}).`);
        return direct.arrayBuffer();
      }
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

    // UNet free-dim overrides. The two repos name their symbolic dims
    // differently (turbo: batch_size/num_channels/sequence_length; SD 1.5:
    // batch/channels/sequence), so we supply BOTH — unknown keys are ignored,
    // and this keeps SD 1.5's shapes static instead of dynamic.
    const unetDims = {
      batch_size: 1, num_channels: 4, height: L, width: L, sequence_length: 77,
      batch: 1, channels: 4, sequence: 77,
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
        freeDimensionOverrides: unetDims,
        externalData: [
          // The graph references the .pb by its basename.
          { path: m.unetWeights.split("/").pop()!, data: new Uint8Array(weightsBuf) },
        ],
      } as ort.InferenceSession.SessionOptions);
    } else {
      this.sessions.unet = await this.makeSession(url(m.unet), "unet", {
        ...common,
        freeDimensionOverrides: unetDims,
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

    this.meta = {
      text: readSessMeta(this.sessions.text),
      unet: readSessMeta(this.sessions.unet),
      vaeDec: readSessMeta(this.sessions.vaeDec),
      vaeEnc: readSessMeta(this.sessions.vaeEnc),
    };

    this.loaded = true;
    onProgress({ phase: "done", fraction: 1 });
  }

  async dispose(): Promise<void> {
    for (const s of Object.values(this.sessions)) await s?.release?.();
    this.sessions = {};
    this.loaded = false;
  }

  // Tear down and rebuild every session. Weights come from the Cache API, so no
  // re-download happens — this just re-creates the ONNX sessions (and the WebGPU
  // device), used to recover from a GPU that started returning empty frames.
  private async reloadSessions(onProgress: (p: EngineProgress) => void): Promise<void> {
    await this.dispose();
    await this.load(onProgress);
  }

  // Run one ONNX session under the hang/abort watchdog. Every GPU inference in
  // this file goes through here so none of them can wedge the pipeline forever.
  private runSess(
    sess: ort.InferenceSession,
    feeds: Record<string, ort.Tensor>,
    label: string,
    signal?: AbortSignal,
  ): Promise<ort.InferenceSession.OnnxValueMapType> {
    return withDeadline(sess.run(feeds), OP_TIMEOUT_MS, `${label}`, signal);
  }

  // ------------------------------------------------------ dtype-aware tensors
  // Build an ORT tensor for input `name` of session `key`, converting canonical
  // Float32 data into whatever precision the model actually declares.
  private makeFloatInput(key: string, name: string, f32: Float32Array, dims: readonly number[]): ort.Tensor {
    const t = this.meta[key].in[name];
    const d = dims as number[];
    if (t === "float16") return new ort.Tensor("float16", packF16(f32), d);
    if (t === "float32") return new ort.Tensor("float32", f32, d);
    throw new Error(`Unexpected float input dtype "${t}" for ${key}.${name}`);
  }

  // Build the timestep/sigma scalar in the unet's declared type — int64 for
  // SD-Turbo, fp16 for SD 1.5, etc.
  private makeStep(key: string, name: string, value: number): ort.Tensor {
    const t = this.meta[key].in[name];
    if (t === "int64") return new ort.Tensor("int64", BigInt64Array.from([BigInt(Math.round(value))]), [1]);
    if (t === "int32") return new ort.Tensor("int32", Int32Array.from([Math.round(value)]), [1]);
    if (t === "float16") return new ort.Tensor("float16", packF16(Float32Array.from([value])), [1]);
    if (t === "float32") return new ort.Tensor("float32", Float32Array.from([value]), [1]);
    throw new Error(`Unexpected timestep dtype "${t}" for ${key}.${name}`);
  }

  // Read any float output tensor back to a canonical Float32Array, then release
  // the source tensor's GPU/CPU buffer. We COPY on the float32 path (`.slice()`)
  // so the returned array stays valid after dispose — never hand back a view into
  // a buffer we're about to free.
  private readFloatOutput(t: ort.Tensor): Float32Array {
    let out: Float32Array;
    if (t.type === "float16") out = unpackF16(t.data as Uint16Array);
    else if (t.type === "float32") out = (t.data as Float32Array).slice();
    else throw new Error(`Unexpected output dtype "${t.type}"`);
    (t as unknown as { dispose?: () => void }).dispose?.();
    return out;
  }

  // ------------------------------------------------------------------ encode
  private async encodeText(prompt: string, signal?: AbortSignal): Promise<{ data: Float32Array; dims: readonly number[] }> {
    const enc = await this.tokenizer(prompt, {
      padding: "max_length",
      max_length: 77,
      truncation: true,
      return_tensor: false,
    });
    const ids = (enc.input_ids as number[]).slice(0, 77);
    while (ids.length < 77) ids.push(0);
    const inName = this.meta.text.inNames[0];
    const input = this.meta.text.in[inName] === "int64"
      ? new ort.Tensor("int64", BigInt64Array.from(ids.map((v) => BigInt(v))), [1, 77])
      : new ort.Tensor("int32", Int32Array.from(ids), [1, 77]);
    const res = await this.runSess(this.sessions.text!, { [inName]: input }, "text encoder", signal);
    const out = res[this.meta.text.outNames[0]] as ort.Tensor;
    // Capture dims BEFORE readFloatOutput (which disposes the tensor). Return
    // canonical float32; the unet builds its own hidden tensor in the precision
    // IT wants (fp32 for turbo, fp16 for SD 1.5).
    const dims = out.dims;
    return { data: this.readFloatOutput(out), dims };
  }

  // Run the VAE encoder once. `contrast` (0..1) lerps the input toward mid-grey
  // BEFORE encoding: the encoder is the sole fp16 stage in the turbo pipeline and
  // fp16 VAE encoders famously overflow their internal activations on
  // high-contrast, out-of-distribution inputs (real photographs) — emitting Inf
  // that unpacks to a non-finite latent, i.e. a guaranteed black decode. Shrinking
  // the input's dynamic range shrinks those activations and often dodges the
  // overflow while preserving structure/colour. Does NOT throw: returns the
  // sanitized latent plus the fraction of values that were non-finite, so the
  // caller can retry softer or fall back to noise instead of committing a black
  // tile. (Sanitize still runs so Inf/NaN never propagate downstream.)
  private async vaeEncode(
    img: ImageData,
    L: number,
    signal?: AbortSignal,
    contrast = 1,
  ): Promise<{ latent: Float32Array; bad: number }> {
    // pixels -> [1,3,H,W] in [-1,1]; makeFloatInput casts to the model's dtype
    const size = L * 8;
    const scaled = resizeImageData(img, size, size);
    const chw = new Float32Array(3 * size * size);
    const d = scaled.data;
    for (let p = 0, i = 0; p < d.length; p += 4, i++) {
      chw[i] = (d[p] / 127.5 - 1) * contrast;
      chw[size * size + i] = (d[p + 1] / 127.5 - 1) * contrast;
      chw[2 * size * size + i] = (d[p + 2] / 127.5 - 1) * contrast;
    }
    const inName = this.meta.vaeEnc.inNames[0];
    const t = this.makeFloatInput("vaeEnc", inName, chw, [1, 3, size, size]);
    const res = await this.runSess(this.sessions.vaeEnc!, { [inName]: t }, "vae encoder", signal);
    const out = res[this.meta.vaeEnc.outNames[0]] as ort.Tensor;
    const latent = this.readFloatOutput(out);
    for (let i = 0; i < latent.length; i++) latent[i] *= VAE_SCALE;
    const bad = sanitizeLatent(latent);
    return { latent, bad }; // [1,4,L,L] flattened
  }

  // Robustly turn `init` pixels into an img2img/outpaint init latent. The fp16
  // VAE encoder can overflow on real photos (see vaeEncode) — the actual root
  // cause of "stamp a photo → black outpaint". Rather than reload the GPU (the
  // overflow is DETERMINISTIC, so a reload + the same photo overflows identically)
  // we degrade gracefully: retry the encode at progressively lower contrast, and
  // if the encoder still can't produce a healthy latent, return null so the
  // caller seeds from pure noise (txt2img-style). The known pixels are preserved
  // regardless by the feathered composite downstream, so the worst outcome is a
  // less-coherent outpaint — never a black one.
  private async encodeInitLatent(
    init: ImageData,
    L: number,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<Float32Array | null> {
    const contrasts = [1, 0.75, 0.5];
    for (let i = 0; i < contrasts.length; i++) {
      if (signal?.aborted) throw new GenerationAbortError();
      const { latent, bad } = await this.vaeEncode(init, L, signal, contrasts[i]);
      const h = latentHealth(latent);
      // Accept only a genuinely well-formed latent: essentially no non-finite
      // values, varied (not collapsed), and not blown up. Healthy SD latents sit
      // around RMS ~1, so 50 is loose headroom that still rejects overflow.
      if (bad <= 0.01 && !h.dead && h.rms < 50) return latent;
      const softer = i + 1 < contrasts.length;
      // eslint-disable-next-line no-console
      console.warn(
        `[vellum] VAE encode unstable (bad=${(bad * 100).toFixed(1)}%, rms=${h.rms.toFixed(1)}, dead=${h.dead}) at contrast ${contrasts[i]} — ${softer ? "retrying softer" : "falling back to noise init"}.`,
      );
      onProgress({
        phase: "encoding",
        detail: softer
          ? "photo encode overflowed fp16 — retrying softer…"
          : "photo encode unstable — seeding fresh area from noise…",
      });
    }
    return null;
  }

  // Run the unet once and return the predicted noise (eps) for a pre-scaled
  // latent. `scaled` is already x / sqrt(sigma^2+1). `hidden` is canonical fp32.
  private async runUnet(
    scaled: Float32Array,
    timestep: number,
    hidden: { data: Float32Array; dims: readonly number[] },
    L: number,
    signal?: AbortSignal,
  ): Promise<Float32Array> {
    const feeds: Record<string, ort.Tensor> = {};
    const inNames = this.meta.unet.inNames;
    for (const n of inNames) {
      const ln = n.toLowerCase();
      if (ln.includes("sample")) feeds[n] = this.makeFloatInput("unet", n, scaled, [1, 4, L, L]);
      else if (ln.includes("timestep") || ln === "t") feeds[n] = this.makeStep("unet", n, timestep);
      else if (ln.includes("hidden") || ln.includes("encoder")) feeds[n] = this.makeFloatInput("unet", n, hidden.data, hidden.dims);
    }
    const res = await this.runSess(this.sessions.unet!, feeds, "unet", signal);
    const eps = this.readFloatOutput(res[this.meta.unet.outNames[0]] as ort.Tensor);
    // Keep a single overflow from poisoning every subsequent step. A wholly
    // non-finite eps means the GPU run failed — bail to a reload+retry.
    const bad = sanitizeLatent(eps);
    if (bad > 0.25) throw new DegenerateFrameError(`unet produced ${Math.round(bad * 100)}% non-finite values`);
    return eps;
  }

  private async vaeDecode(latent: Float32Array, L: number, signal?: AbortSignal): Promise<ImageData> {
    const inp = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i++) inp[i] = latent[i] / VAE_SCALE;
    const inName = this.meta.vaeDec.inNames[0];
    const t = this.makeFloatInput("vaeDec", inName, inp, [1, 4, L, L]);
    const res = await this.runSess(this.sessions.vaeDec!, { [inName]: t }, "vae decoder", signal);
    const out = res[this.meta.vaeDec.outNames[0]] as ort.Tensor;
    const size = L * 8;
    const chw = this.readFloatOutput(out);
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
  // Public entry: serialize, then run the pipeline under a bounded reload+retry
  // loop. Every failure mode here is finite — a degenerate/black frame triggers
  // at most MAX_ATTEMPTS-1 reloads, a hung GPU trips the per-op watchdog, and an
  // abort short-circuits immediately. There is no path that can spin forever.
  async generate(
    req: GenerateRequest,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<GenerateResult> {
    if (!this.loaded) throw new Error("WebGPU engine not loaded.");
    if (this.inFlight) {
      throw new Error("A generation is already running — wait for it to finish or cancel it.");
    }
    this.inFlight = true;
    try {
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        if (signal?.aborted) throw new GenerationAbortError();
        try {
          return await this.runPipeline(req, onProgress, signal);
        } catch (e) {
          // Abort and genuine hangs are not retryable — surface them at once.
          if (e instanceof GenerationAbortError) throw e;
          const degenerate = e instanceof DegenerateFrameError;
          if (degenerate && attempt < MAX_ATTEMPTS) {
            // Rebuild the sessions + WebGPU device (weights are cached, so this is
            // fast) and try once more. Guarded by its own deadline so a device
            // that's wedged during teardown can't hang the reload forever.
            onProgress({ phase: "compiling", detail: "GPU returned an empty frame — reinitializing…" });
            try {
              await withDeadline(this.reloadSessions(onProgress), RELOAD_TIMEOUT_MS, "GPU reinitialization", signal);
            } catch (re) {
              if (re instanceof GenerationAbortError) throw re;
              throw new Error(
                "Couldn't recover the GPU after an empty frame. Reload the page, or switch to the " +
                  "SD 1.5 or Remote engine. (" + (re instanceof Error ? re.message : String(re)) + ")",
              );
            }
            continue;
          }
          if (degenerate) {
            throw new Error(
              "The GPU returned an empty (black) frame twice, even after reinitializing. This usually " +
                "means the browser is on a low-power/integrated GPU or is out of video memory — not your " +
                "prompt. Try: reload the page, close other GPU-heavy tabs, switch to the Remote engine, " +
                "or use a machine with a discrete GPU.",
            );
          }
          throw e;
        }
      }
      // Unreachable: the loop either returns, retries, or throws.
      throw new Error("Generation failed after all attempts.");
    } finally {
      this.inFlight = false;
    }
  }

  // One full text→(vae)→unet→vae pass. Throws DegenerateFrameError when a latent
  // or the decoded frame is empty/NaN-poisoned, GenerationAbortError on cancel.
  private async runPipeline(
    req: GenerateRequest,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<GenerateResult> {
    const t0 = performance.now();
    const L = this.manifest.latent;
    const size = L * 8;
    const seed = req.seed >= 0 ? req.seed : Math.floor(Math.random() * 2 ** 31);
    const check = () => { if (signal?.aborted) throw new GenerationAbortError(); };
    check();

    onProgress({ phase: "encoding", detail: "text", fraction: 0.08 });
    const hidden = await this.encodeText(req.prompt, signal);
    // Unconditional embedding for classifier-free guidance (non-turbo only).
    const useCfg = !this.manifest.turbo && req.guidance > 1;
    const uncond = useCfg ? await this.encodeText(req.negativePrompt || "", signal) : null;

    const knownAtSize = req.initImage ? resizeImageData(req.initImage, size, size) : null;
    const isImg = !!(req.initImage && knownAtSize);

    // ---- init latent + latent-space outpaint mask ------------------------
    // Encode the edge-extended known tile to a latent we can LOCK in place, and
    // build a mask marking the fresh (to-generate) region. Re-locking the known
    // latent every sampling step is what makes the fresh region continue the
    // scene (verified: without it the model denoises a whole new, unrelated tile
    // — the "photos overlap, no blend" report). z0 is null only when the fp16
    // VAE encoder overflowed; we then fall back to noise (txt2img) and still
    // pixel-composite the real photo back at the seam.
    const n = 4 * L * L;
    let z0: Float32Array | null = null;
    let mask: Float32Array | null = null; // [4·L·L], 1 = generate fresh, 0 = keep known
    let anyUnknown = false;
    let anyKnown = false;
    if (isImg && knownAtSize) {
      const { init } = buildOutpaintInit(knownAtSize, seed);
      onProgress({ phase: "encoding", detail: "vae", fraction: 0.22 });
      z0 = await this.encodeInitLatent(init, L, onProgress, signal);
      if (z0) {
        mask = buildLatentMask(knownAtSize, L);
        for (let i = 0; i < L * L; i++) {
          if (mask[i] > 0.02) anyUnknown = true;
          else anyKnown = true;
          if (anyUnknown && anyKnown) break;
        }
      }
    }
    check();

    const turbo = this.manifest.turbo;
    const steps = turbo ? 1 : Math.max(2, Math.min(50, Math.round(req.steps) || 20));
    const { sigmas, ts } = buildSchedule(steps);
    const noise = seededGaussian(n, seed);

    // An outpaint/inpaint (known pixels AND a fresh region) is driven by the
    // mask, so it generates the fresh region fully (start at max sigma) while the
    // lock preserves the surroundings. A plain img2img over a FULLY-known tile
    // (no fresh region) has nothing to outpaint, so it honours `strength` the
    // classic way. z0-less runs (txt2img / encode overflow) also start at max.
    const outpaint = !!(z0 && mask && anyUnknown);
    const plainImg2img = !!(z0 && !anyUnknown && anyKnown);
    let start = 0;
    if (plainImg2img) {
      // Fraction of the sigma ladder to traverse. For turbo (1 step) we can't
      // start partway, so approximate strength by scaling the single sigma below.
      start = turbo ? 0 : Math.min(steps - 1, Math.max(0, Math.round((1 - req.strength) * steps)));
    }
    const startSigma =
      plainImg2img && turbo ? Math.max(0.05, req.strength) * SIGMA_MAX : sigmas[start];

    // Seed the latent. Outpaint & txt2img noise fully; plain img2img noises z0 by
    // its start sigma. The known region (mask 0) is set to the re-noised true
    // latent so step 1 already sees correct surroundings.
    const latent = new Float32Array(n);
    if (z0) for (let i = 0; i < n; i++) latent[i] = z0[i] + startSigma * noise[i];
    else for (let i = 0; i < n; i++) latent[i] = noise[i] * startSigma;

    const lockKnown = (sigma: number, toClean: boolean) => {
      if (!outpaint || !z0 || !mask) return;
      for (let k = 0; k < n; k++) {
        const known = toClean ? z0[k] : z0[k] + sigma * noise[k];
        latent[k] = mask[k] * latent[k] + (1 - mask[k]) * known;
      }
    };

    const total = steps - start;
    for (let i = start; i < steps; i++) {
      check();
      const sigma = turbo && plainImg2img ? startSigma : sigmas[i];
      // Re-pin the known region to the true latent at this noise level before
      // predicting, so the fresh region always denoises against real context.
      lockKnown(sigma, false);
      const cin = 1 / Math.sqrt(sigma * sigma + 1);
      const scaled = new Float32Array(n);
      for (let k = 0; k < n; k++) scaled[k] = latent[k] * cin;
      const tstep = turbo ? TURBO_TIMESTEP : ts[i];
      const epsCond = await this.runUnet(scaled, tstep, hidden, L, signal);
      let eps = epsCond;
      if (useCfg && uncond) {
        const epsUncond = await this.runUnet(scaled, tstep, uncond, L, signal);
        eps = new Float32Array(epsCond.length);
        for (let k = 0; k < eps.length; k++)
          eps[k] = epsUncond[k] + req.guidance * (epsCond[k] - epsUncond[k]);
      }
      // Euler step. For turbo (1 step) sigma_next is 0, so x = x - sigma·eps.
      const dsig = (turbo && plainImg2img ? 0 : sigmas[i + 1]) - sigma;
      for (let k = 0; k < n; k++) latent[k] += eps[k] * dsig;
      onProgress({
        phase: "sampling",
        step: i - start + 1,
        totalSteps: total,
        fraction: 0.15 + 0.7 * ((i - start + 1) / total),
        detail: turbo ? "1 step (turbo)" : `step ${i - start + 1}/${total}`,
      });
    }
    // Final lock: pin the known region to the CLEAN encoded latent so the
    // surroundings decode as the real scene (the pixel composite then restores
    // them exactly and feathers only the seam).
    lockKnown(0, true);
    const finalLatent = latent;

    check();
    // Pre-decode guard: a NaN-poisoned or flat latent will only ever decode to a
    // black frame, so catch it here (cheap) and let the attempt loop reload+retry
    // instead of paying for a decode that we already know is dead.
    sanitizeLatent(finalLatent);
    if (latentIsDead(finalLatent)) {
      throw new DegenerateFrameError("latent collapsed to a constant before decode");
    }

    onProgress({ phase: "decoding", fraction: 0.9 });
    const decoded = await this.vaeDecode(finalLatent, L, signal);

    // Post-decode safety net: an all-black / flat frame is a GPU-execution
    // failure (device lost, VRAM exhaustion, low-power GPU). Signal the attempt
    // loop to reload+retry rather than committing a black tile to the canvas.
    if (isDegenerateFrame(decoded)) {
      throw new DegenerateFrameError("decoder returned a flat frame");
    }

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

// True when a decoded frame is degenerate (uniform / all-black) — the signature
// of a failed WebGPU decode. A real image, even a dark one, has large luminance
// variance (hundreds+); a failed decode is essentially flat, so a near-zero
// variance is an unambiguous "the GPU gave us nothing" signal. Sampled (~4k px)
// so it stays cheap on large frames.
function isDegenerateFrame(img: ImageData): boolean {
  const d = img.data;
  let sum = 0;
  let sum2 = 0;
  let n = 0;
  const stride = 4 * Math.max(1, Math.floor(d.length / 4 / 4096));
  for (let p = 0; p + 2 < d.length; p += stride) {
    const l = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
    sum += l;
    sum2 += l * l;
    n++;
  }
  if (n === 0) return false;
  const mean = sum / n;
  const variance = sum2 / n - mean * mean;
  return variance < 2;
}
