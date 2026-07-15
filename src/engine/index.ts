// Engine registry — single place the app resolves a provider by id.
import type { DiffusionProvider, ProviderId } from "./types";
import { DemoProvider } from "./demo";
import { RemoteProvider } from "./remote";
import { WebGpuProvider, TURBO_MANIFEST, SD15_MANIFEST } from "./webgpu";

export * from "./types";
export { DEFAULT_MANIFEST } from "./webgpu";
export type { ModelManifest } from "./webgpu";
export type { RemoteConfig } from "./remote";

const singletons: Partial<Record<ProviderId, DiffusionProvider>> = {};

function makeProvider(id: ProviderId): DiffusionProvider {
  switch (id) {
    case "webgpu":
      return new WebGpuProvider("webgpu", TURBO_MANIFEST, {
        local: true,
        requiresLoad: true,
        label: "SD-Turbo · your GPU",
        blurb:
          "Distilled 1-step Stable Diffusion on your own graphics card. ~2.5 GB one-time download, then instant, offline & private.",
      });
    case "webgpu-sd15":
      return new WebGpuProvider("webgpu-sd15", SD15_MANIFEST, {
        local: true,
        requiresLoad: true,
        label: "SD 1.5 · your GPU",
        blurb:
          "The original, non-distilled Stable Diffusion 1.5 with full multi-step sampling & guidance. ~2.3 GB download — slower but more controllable.",
        multiStep: true,
        suggestedSteps: 20,
        suggestedGuidance: 7.5,
      });
    case "remote":
      return new RemoteProvider();
    default:
      return new DemoProvider();
  }
}

export function getProvider(id: ProviderId): DiffusionProvider {
  if (!singletons[id]) singletons[id] = makeProvider(id);
  return singletons[id]!;
}

export function webgpuProvider(): WebGpuProvider {
  return getProvider("webgpu") as WebGpuProvider;
}
export function remoteProvider(): RemoteProvider {
  return getProvider("remote") as RemoteProvider;
}

export const ALL_PROVIDERS: ProviderId[] = ["webgpu", "webgpu-sd15", "remote", "demo"];
