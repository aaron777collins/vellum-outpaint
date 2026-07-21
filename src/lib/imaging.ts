// ============================================================================
// Imaging — the canvas math behind outpainting.
// Feathered masks, edge/mirror fill for the fresh region, seeded noise, and
// the feathered composite that blends a freshly generated tile back into the
// scene without a visible seam. Pure functions, no DOM globals beyond canvas.
// ============================================================================

/** Deterministic PRNG so a given seed always paints the same noise. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

export function ctxOf(c: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas context unavailable");
  return ctx;
}

export function blankImageData(w: number, h: number): ImageData {
  return new ImageData(Math.max(1, w | 0), Math.max(1, h | 0));
}

export function imageDataToCanvas(img: ImageData): HTMLCanvasElement {
  const c = makeCanvas(img.width, img.height);
  ctxOf(c).putImageData(img, 0, 0);
  return c;
}

export function canvasToImageData(c: HTMLCanvasElement): ImageData {
  return ctxOf(c).getImageData(0, 0, c.width, c.height);
}

/**
 * Build the init tile for an outpaint: existing pixels are copied in, and any
 * area with zero alpha (the fresh expanse) is filled by mirror-extending the
 * nearest known edge, then lightly blurred + seeded-noised. This gives the
 * diffusion model coherent structure to denoise from instead of hard black,
 * which is what makes outpaint seams disappear.
 */
export function buildOutpaintInit(
  known: ImageData,
  seed: number,
): { init: ImageData; hasKnown: boolean } {
  const { width: w, height: h } = known;
  const src = known.data;
  const out = new ImageData(w, h);
  const dst = out.data;

  // Column/row extents of known (alpha>8) pixels, for mirror sampling.
  let anyKnown = false;
  for (let i = 3; i < src.length; i += 4) {
    if (src[i] > 8) { anyKnown = true; break; }
  }
  if (!anyKnown) {
    // Nothing to extend from — fill with mid-grey noise so txt2img has a base.
    const rnd = mulberry32(seed || 1);
    for (let i = 0; i < dst.length; i += 4) {
      const n = 110 + Math.floor(rnd() * 40);
      dst[i] = dst[i + 1] = dst[i + 2] = n;
      dst[i + 3] = 255;
    }
    return { init: out, hasKnown: false };
  }

  const idx = (x: number, y: number) => (y * w + x) * 4;
  const isKnown = (x: number, y: number) => src[idx(x, y) + 3] > 8;

  // For every fresh pixel, walk outward to the nearest known pixel in the 4
  // cardinal directions and average them — cheap directional inpaint prior.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = idx(x, y);
      if (src[o + 3] > 8) {
        dst[o] = src[o];
        dst[o + 1] = src[o + 1];
        dst[o + 2] = src[o + 2];
        dst[o + 3] = 255;
        continue;
      }
      let r = 0, g = 0, b = 0, n = 0;
      // scan four directions
      const dirs = [
        [1, 0], [-1, 0], [0, 1], [0, -1],
      ];
      for (const [dx, dy] of dirs) {
        let cx = x, cy = y, steps = 0;
        while (cx >= 0 && cx < w && cy >= 0 && cy < h && steps < Math.max(w, h)) {
          if (isKnown(cx, cy)) {
            const k = idx(cx, cy);
            r += src[k]; g += src[k + 1]; b += src[k + 2]; n++;
            break;
          }
          cx += dx; cy += dy; steps++;
        }
      }
      if (n === 0) { r = g = b = 118; n = 1; }
      dst[o] = Math.round(r / n);
      dst[o + 1] = Math.round(g / n);
      dst[o + 2] = Math.round(b / n);
      dst[o + 3] = 255;
    }
  }

  // Soften the extended region + sprinkle seeded noise so it isn't flat.
  const blurred = boxBlurMasked(out, known, 6);
  const rnd = mulberry32((seed || 1) ^ 0x9e3779b9);
  const bd = blurred.data;
  for (let i = 0; i < bd.length; i += 4) {
    if (known.data[i + 3] > 8) continue; // don't disturb real pixels
    const j = (rnd() - 0.5) * 26;
    bd[i] = clamp8(bd[i] + j);
    bd[i + 1] = clamp8(bd[i + 1] + j);
    bd[i + 2] = clamp8(bd[i + 2] + j);
  }
  return { init: blurred, hasKnown: true };
}

/** Box blur that only writes to pixels NOT known in `keep` (protects real image). */
function boxBlurMasked(img: ImageData, keep: ImageData, radius: number): ImageData {
  const { width: w, height: h } = img;
  const src = img.data;
  const out = new ImageData(new Uint8ClampedArray(src), w, h);
  const dst = out.data;
  const r = Math.max(1, radius | 0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (y * w + x) * 4;
      if (keep.data[o + 3] > 8) continue;
      let sr = 0, sg = 0, sb = 0, c = 0;
      for (let dy = -r; dy <= r; dy += 2) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -r; dx <= r; dx += 2) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const k = (yy * w + xx) * 4;
          sr += src[k]; sg += src[k + 1]; sb += src[k + 2]; c++;
        }
      }
      if (c) {
        dst[o] = sr / c; dst[o + 1] = sg / c; dst[o + 2] = sb / c; dst[o + 3] = 255;
      }
    }
  }
  return out;
}

/**
 * Blend weight for compositing a freshly-generated tile back onto the scene.
 * Returns per-pixel alpha in [0,1] where 1 = use the generated pixel, 0 = keep
 * the original known pixel:
 *
 *   • Unknown (transparent) pixels → 1: the fresh expanse is fully generated.
 *   • Known pixels → a ramp from ~1 at the seam down to 0 once you're `feather`
 *     px inside the existing image.
 *
 * The ramp is the whole point of outpainting: instead of butting the generated
 * region hard against the original (a visible seam / "the photos just overlap"),
 * the new pixels CROSS-FADE into the existing photo over a soft band, so the
 * extension looks continuous. The weight is driven by each known pixel's
 * distance to the nearest unknown pixel — i.e. how deep inside the real image it
 * sits — computed with a two-pass chamfer distance transform.
 */
export function featherMaskFromKnown(known: ImageData, feather: number): Float32Array {
  const { width: w, height: h } = known;
  const f = Math.max(1, feather | 0);
  const n = w * h;
  const INF = 1e9;

  // dist = distance from each pixel to the nearest UNKNOWN (transparent) pixel.
  // Seed 0 in the unknown region; it grows as we march into the known region.
  const dist = new Float32Array(n);
  for (let i = 0, p = 3; i < n; i++, p += 4) dist[i] = known.data[p] > 8 ? INF : 0;

  const relax = (i: number, j: number, cost: number) => {
    if (dist[j] + cost < dist[i]) dist[i] = dist[j] + cost;
  };
  // forward pass (top-left → bottom-right)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (dist[i] === 0) continue;
      if (x > 0) relax(i, i - 1, 1);
      if (y > 0) relax(i, i - w, 1);
      if (x > 0 && y > 0) relax(i, i - w - 1, 1.4142);
      if (x < w - 1 && y > 0) relax(i, i - w + 1, 1.4142);
    }
  // backward pass (bottom-right → top-left)
  for (let y = h - 1; y >= 0; y--)
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (dist[i] === 0) continue;
      if (x < w - 1) relax(i, i + 1, 1);
      if (y < h - 1) relax(i, i + w, 1);
      if (x < w - 1 && y < h - 1) relax(i, i + w + 1, 1.4142);
      if (x > 0 && y < h - 1) relax(i, i + w - 1, 1.4142);
    }

  const alpha = new Float32Array(n);
  for (let i = 0, p = 3; i < n; i++, p += 4) {
    // Unknown → fully generated. Known → fade generated out over `feather` px.
    alpha[i] = known.data[p] > 8 ? Math.max(0, 1 - dist[i] / f) : 1;
  }
  return alpha;
}

/**
 * Composite a generated tile over the known tile using the feathered weight from
 * `featherMaskFromKnown`, so the fresh region is fully painted and cross-fades
 * smoothly into the existing image across the seam (no hard overlap boundary).
 * `known` and `generated` must share dimensions.
 */
export function compositeFeathered(
  known: ImageData,
  generated: ImageData,
  feather: number,
): ImageData {
  const { width: w, height: h } = known;
  const alpha = featherMaskFromKnown(known, feather);
  const out = new ImageData(w, h);
  const k = known.data, g = generated.data, o = out.data;
  for (let i = 0, p = 0; i < alpha.length; i++, p += 4) {
    const a = alpha[i]; // 1 => generated, 0 => known
    const ia = 1 - a;
    // Where known is transparent, alpha is 1 (see featherMaskFromKnown), so the
    // transparent RGB never leaks in — a plain lerp is correct everywhere.
    o[p] = Math.round(g[p] * a + k[p] * ia);
    o[p + 1] = Math.round(g[p + 1] * a + k[p + 1] * ia);
    o[p + 2] = Math.round(g[p + 2] * a + k[p + 2] * ia);
    o[p + 3] = 255;
  }
  return out;
}

export function clamp8(v: number): number {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** Round n up to the nearest multiple of `m` (diffusion needs /8 or /64). */
export function roundTo(n: number, m: number): number {
  return Math.max(m, Math.round(n / m) * m);
}

/** Open the OS file picker and resolve the chosen image File (or null). */
export function pickImageFile(): Promise<File | null> {
  return new Promise((res) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "image/*";
    inp.style.display = "none";
    let settled = false;
    const done = (f: File | null) => {
      if (settled) return;
      settled = true;
      inp.remove();
      res(f);
    };
    inp.onchange = () => done(inp.files?.[0] ?? null);
    // `cancel` fires in modern browsers when the dialog is dismissed.
    inp.addEventListener("cancel", () => done(null));
    document.body.appendChild(inp);
    inp.click();
  });
}

/** Load a File/Blob/URL into ImageData. */
export async function loadImageData(src: Blob | string): Promise<ImageData> {
  const url = typeof src === "string" ? src : URL.createObjectURL(src);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const im = new Image();
      im.crossOrigin = "anonymous";
      im.onload = () => res(im);
      im.onerror = () => rej(new Error("image decode failed"));
      im.src = url;
    });
    const c = makeCanvas(img.naturalWidth, img.naturalHeight);
    ctxOf(c).drawImage(img, 0, 0);
    return canvasToImageData(c);
  } finally {
    if (typeof src !== "string") URL.revokeObjectURL(url);
  }
}
