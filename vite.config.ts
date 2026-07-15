import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// COOP/COEP make the page cross-origin-isolated (crossOriginIsolated === true),
// which onnxruntime-web wants for its threaded/WebGPU backend. `credentialless`
// (rather than require-corp) still lets us fetch the multi-GB model weights
// cross-origin from huggingface.co and the ORT wasm from jsdelivr. The
// production nginx config sets the same pair.
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: { middlewares: { use: (fn: (req: unknown, res: { setHeader: (k: string, v: string) => void }, next: () => void) => void) => void } }) {
    server.middlewares.use((_req, res, next) => {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), crossOriginIsolation],
  // onnxruntime-web ships prebuilt wasm; don't let esbuild try to pre-bundle it.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 4000,
  },
  worker: { format: "es" },
});
