// ============================================================================
// Engine contract — every generation backend implements DiffusionProvider.
// The rest of the app only ever talks to this interface, so the in-browser
// WebGPU engine, a remote A1111/ComfyUI server, and the local demo engine are
// fully interchangeable.
// ============================================================================

export type ProviderId = "webgpu" | "remote" | "demo";

export interface SamplingParams {
  prompt: string;
  negativePrompt: string;
  steps: number;
  guidance: number;
  /** 0..1 — how much of the init image to preserve (img2img/outpaint). */
  strength: number;
  seed: number; // -1 => random
}

/** A single outpaint/inpaint job. All images are ImageData in the target frame. */
export interface GenerateRequest extends SamplingParams {
  width: number;
  height: number;
  /** Existing pixels for the tile being (out)painted. Null => pure txt2img. */
  initImage: ImageData | null;
  /**
   * White = regenerate, black = keep. Feathered greyscale allowed.
   * Null => regenerate everything (txt2img).
   */
  mask: ImageData | null;
}

export type ProgressPhase =
  | "idle"
  | "downloading"
  | "compiling"
  | "encoding"
  | "sampling"
  | "decoding"
  | "compositing"
  | "done"
  | "error";

export interface EngineProgress {
  phase: ProgressPhase;
  /** 0..1 for the current phase, or undefined when indeterminate. */
  fraction?: number;
  /** e.g. step 1/1, or MB downloaded. */
  detail?: string;
  /** current step for sampling phase */
  step?: number;
  totalSteps?: number;
}

export interface GenerateResult {
  image: ImageData;
  seed: number;
  ms: number;
}

export interface ProviderCapabilities {
  /** Runs entirely on the client GPU with nothing leaving the machine. */
  local: boolean;
  /** Needs weights fetched/compiled before first use. */
  requiresLoad: boolean;
  /** Human label + one-line description for the engine picker. */
  label: string;
  blurb: string;
}

export interface DiffusionProvider {
  readonly id: ProviderId;
  readonly caps: ProviderCapabilities;

  /** True if this provider can run in the current environment right now. */
  isAvailable(): Promise<boolean>;

  /** Download/compile weights. Idempotent. */
  load(onProgress: (p: EngineProgress) => void): Promise<void>;

  isLoaded(): boolean;

  generate(
    req: GenerateRequest,
    onProgress: (p: EngineProgress) => void,
    signal?: AbortSignal,
  ): Promise<GenerateResult>;

  /** Free GPU/session memory. */
  dispose(): Promise<void>;
}

export class GenerationAbortError extends Error {
  constructor() {
    super("generation aborted");
    this.name = "GenerationAbortError";
  }
}
