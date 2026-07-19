// ============================================================================
// Document — the growable infinite-canvas "world".
// A single offscreen canvas holds every committed pixel; it expands as you
// outpaint outward. Kept outside React state (mutable, ref-counted by a
// revision integer) so painting large tiles never triggers a re-render storm.
// ============================================================================

import { makeCanvas, ctxOf } from "./imaging";

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Snapshot {
  originX: number;
  originY: number;
  data: string; // dataURL
}

const PAD = 4096; // headroom when first placing content

export class VellumDocument {
  world: HTMLCanvasElement;
  /** World-space coordinate of the canvas's (0,0) pixel. */
  originX = 0;
  originY = 0;
  revision = 0;

  private undoStack: Snapshot[] = [];
  private redoStack: Snapshot[] = [];

  constructor() {
    this.world = makeCanvas(PAD, PAD);
    this.originX = -PAD / 2;
    this.originY = -PAD / 2;
  }

  get bounds(): Rect {
    return { x: this.originX, y: this.originY, w: this.world.width, h: this.world.height };
  }

  /** True content bounds (non-transparent pixels), in world coords, or null. */
  contentBounds(): Rect | null {
    const ctx = ctxOf(this.world);
    const { width: w, height: h } = this.world;
    const d = ctx.getImageData(0, 0, w, h).data;
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (d[(y * w + x) * 4 + 3] > 8) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null;
    return {
      x: this.originX + minX,
      y: this.originY + minY,
      w: maxX - minX + 1,
      h: maxY - minY + 1,
    };
  }

  /** Ensure the world canvas covers `rect` (world coords); grow if needed. */
  ensure(rect: Rect) {
    const b = this.bounds;
    const nx = Math.min(b.x, rect.x - 64);
    const ny = Math.min(b.y, rect.y - 64);
    const nx2 = Math.max(b.x + b.w, rect.x + rect.w + 64);
    const ny2 = Math.max(b.y + b.h, rect.y + rect.h + 64);
    if (nx === b.x && ny === b.y && nx2 === b.x + b.w && ny2 === b.y + b.h) return;
    const newW = Math.ceil(nx2 - nx);
    const newH = Math.ceil(ny2 - ny);
    const next = makeCanvas(newW, newH);
    ctxOf(next).drawImage(this.world, b.x - nx, b.y - ny);
    this.world = next;
    this.originX = nx;
    this.originY = ny;
    this.revision++;
  }

  worldToCanvas(x: number, y: number): [number, number] {
    return [x - this.originX, y - this.originY];
  }

  /** Extract the known pixels within a world rect (transparent where empty). */
  extract(rect: Rect): ImageData {
    const c = makeCanvas(rect.w, rect.h);
    const [cx, cy] = this.worldToCanvas(rect.x, rect.y);
    ctxOf(c).drawImage(
      this.world,
      cx, cy, rect.w, rect.h,
      0, 0, rect.w, rect.h,
    );
    return ctxOf(c).getImageData(0, 0, rect.w, rect.h);
  }

  /** Paint an image tile into the world at a rect (world coords). */
  commit(rect: Rect, img: ImageData) {
    this.pushUndo();
    this.ensure(rect);
    const [cx, cy] = this.worldToCanvas(rect.x, rect.y);
    const tile = makeCanvas(img.width, img.height);
    ctxOf(tile).putImageData(img, 0, 0);
    const ctx = ctxOf(this.world);
    ctx.drawImage(tile, 0, 0, img.width, img.height, cx, cy, rect.w, rect.h);
    this.revision++;
  }

  /**
   * Begin a user edit stroke: snapshot current state for undo ONCE, so a whole
   * drag (many erase dabs) collapses to a single undo step. Call at pointer-down.
   */
  beginStroke() {
    this.pushUndo();
  }

  /**
   * Erase a soft-edged circle at a WORLD point — makes those pixels transparent
   * (alpha 0) so the next generate treats them as "unknown" and repaints them
   * (inpaint). The radial falloff feathers the erased edge so the regenerated
   * fill blends instead of leaving a hard disc. No undo push here: the caller
   * runs beginStroke() once per drag so an entire stroke is one undo step.
   */
  eraseCircle(wx: number, wy: number, radius: number) {
    const r = Math.max(1, radius);
    const [cx, cy] = this.worldToCanvas(wx, wy);
    const ctx = ctxOf(this.world);
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    const g = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r);
    g.addColorStop(0, "rgba(0,0,0,1)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    this.revision++;
  }

  /** Tight content-bounds pixels (opaque region), or null if empty. Used to
   *  give the auto-prompt captioner the actual painted scene, not the padding. */
  extractContent(): ImageData | null {
    const cb = this.contentBounds();
    if (!cb) return null;
    return this.extract(cb);
  }

  /** Place an imported image, centered at a world point. */
  place(img: ImageData, centerX: number, centerY: number) {
    const rect: Rect = {
      x: Math.round(centerX - img.width / 2),
      y: Math.round(centerY - img.height / 2),
      w: img.width,
      h: img.height,
    };
    this.commit(rect, img);
    return rect;
  }

  clear() {
    this.pushUndo();
    ctxOf(this.world).clearRect(0, 0, this.world.width, this.world.height);
    this.revision++;
  }

  // ---- history ----
  private snap(): Snapshot {
    return { originX: this.originX, originY: this.originY, data: this.world.toDataURL() };
  }
  private pushUndo() {
    this.undoStack.push(this.snap());
    if (this.undoStack.length > 30) this.undoStack.shift();
    this.redoStack = [];
  }
  canUndo() { return this.undoStack.length > 0; }
  canRedo() { return this.redoStack.length > 0; }

  async undo() {
    const s = this.undoStack.pop();
    if (!s) return;
    this.redoStack.push(this.snap());
    await this.restore(s);
  }
  async redo() {
    const s = this.redoStack.pop();
    if (!s) return;
    this.undoStack.push(this.snap());
    await this.restore(s);
  }
  private async restore(s: Snapshot) {
    const img = await loadDataUrl(s.data);
    const c = makeCanvas(img.width, img.height);
    ctxOf(c).drawImage(img, 0, 0);
    this.world = c;
    this.originX = s.originX;
    this.originY = s.originY;
    this.revision++;
  }

  /** Export the tight content region as a PNG blob. */
  async exportPng(): Promise<Blob | null> {
    const cb = this.contentBounds();
    if (!cb) return null;
    const c = makeCanvas(cb.w, cb.h);
    const [cx, cy] = this.worldToCanvas(cb.x, cb.y);
    ctxOf(c).drawImage(this.world, cx, cy, cb.w, cb.h, 0, 0, cb.w, cb.h);
    return new Promise((res) => c.toBlob((b) => res(b), "image/png"));
  }
}

function loadDataUrl(url: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = url;
  });
}

// Single shared document instance for the app.
export const doc = new VellumDocument();
