// Engine registry — single place the app resolves a provider by id.
import type { DiffusionProvider, ProviderId } from "./types";
import { DemoProvider } from "./demo";
import { RemoteProvider } from "./remote";
import { WebGpuProvider } from "./webgpu";

export * from "./types";
export { DEFAULT_MANIFEST } from "./webgpu";
export type { ModelManifest } from "./webgpu";
export type { RemoteConfig } from "./remote";

const singletons: Partial<Record<ProviderId, DiffusionProvider>> = {};

export function getProvider(id: ProviderId): DiffusionProvider {
  if (!singletons[id]) {
    singletons[id] =
      id === "webgpu"
        ? new WebGpuProvider()
        : id === "remote"
          ? new RemoteProvider()
          : new DemoProvider();
  }
  return singletons[id]!;
}

export function webgpuProvider(): WebGpuProvider {
  return getProvider("webgpu") as WebGpuProvider;
}
export function remoteProvider(): RemoteProvider {
  return getProvider("remote") as RemoteProvider;
}

export const ALL_PROVIDERS: ProviderId[] = ["webgpu", "remote", "demo"];
