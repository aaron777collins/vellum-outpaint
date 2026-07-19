// ============================================================================
// Auto-prompt — "detect the best prompt to continue the scene".
// Runs a small image-captioning model (vit-gpt2) fully in the browser via
// @huggingface/transformers on CPU/wasm (no WebGPU needed), so suggesting a
// prompt stays on-device like everything else. The model (~200 MB) is fetched
// lazily on first use and cached by the browser, so it never delays startup.
// ============================================================================

import { ctxOf, makeCanvas } from "./imaging";

// The caption model. vit-gpt2 is small, reliable on wasm, and gives short,
// prompt-shaped descriptions ("a painting of a mountain range at sunset").
const CAPTION_MODEL = "Xenova/vit-gpt2-image-captioning";

export interface CaptionProgress {
  /** 0..1 while the model downloads, undefined once it's running. */
  fraction?: number;
  detail: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let captioner: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loading: Promise<any> | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function getCaptioner(onProgress: (p: CaptionProgress) => void): Promise<any> {
  if (captioner) return captioner;
  if (!loading) {
    loading = (async () => {
      const { pipeline } = await import("@huggingface/transformers");
      const pipe = await pipeline("image-to-text", CAPTION_MODEL, {
        // Full-precision weights. The repo's DEFAULT int8 build uses MatMulNBits
        // ops that onnxruntime-web (1.27) can't build a session for — it throws
        // "Missing required scale … DequantizeLinear". fp32 is a bit larger but
        // creates a session reliably and captions on CPU/wasm in a second or two.
        dtype: "fp32",
        device: "wasm",
        // Per-file download progress so the UI can show a real bar the first time.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        progress_callback: (p: any) => {
          if (p?.status === "progress" && typeof p.progress === "number") {
            onProgress({
              fraction: p.progress / 100,
              detail: `downloading caption model · ${Math.round(p.progress)}%`,
            });
          }
        },
      });
      captioner = pipe;
      return pipe;
    })();
  }
  try {
    return await loading;
  } catch (e) {
    loading = null; // let a later attempt retry the download
    throw e;
  }
}

/**
 * Suggest a continuation prompt for `img`. Transparency is flattened onto a
 * neutral mid-grey first (the captioner would otherwise read transparent pixels
 * as black and describe a dark void). Returns a trimmed caption string, or ""
 * if the model produced nothing.
 */
export async function captionImage(
  img: ImageData,
  onProgress: (p: CaptionProgress) => void,
): Promise<string> {
  const pipe = await getCaptioner(onProgress);
  onProgress({ detail: "reading the scene…" });

  // Flatten onto neutral grey so partly-transparent frames caption sensibly.
  const c = makeCanvas(img.width, img.height);
  const ctx = ctxOf(c);
  ctx.fillStyle = "#7c7469";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.putImageData(compositeOverGrey(img), 0, 0);
  const dataUrl = c.toDataURL("image/png");

  const out = await pipe(dataUrl);
  // Pipeline returns [{ generated_text }] (or occasionally a bare object).
  const text = Array.isArray(out)
    ? (out[0]?.generated_text ?? "")
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    : ((out as any)?.generated_text ?? "");
  return String(text).trim();
}

/** Alpha-composite `img` over opaque mid-grey so no transparent pixels remain. */
function compositeOverGrey(img: ImageData): ImageData {
  const out = new ImageData(img.width, img.height);
  const s = img.data;
  const d = out.data;
  const G = 124; // matches the grey fill above
  for (let i = 0; i < s.length; i += 4) {
    const a = s[i + 3] / 255;
    d[i] = Math.round(s[i] * a + G * (1 - a));
    d[i + 1] = Math.round(s[i + 1] * a + G * (1 - a));
    d[i + 2] = Math.round(s[i + 2] * a + G * (1 - a));
    d[i + 3] = 255;
  }
  return out;
}
