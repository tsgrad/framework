import { loadNativeAddon } from '../native.js';
import { CpuBackend } from './cpu_backend.js';
import type { BackendKind, DeviceBackend } from './device_backend.js';
import { NativeBackend } from './native_backend.js';

export type BackendPreference = 'auto' | 'cpu' | 'native' | 'cuda' | 'webgpu';
export interface BackendInitOptions {
    backend?: BackendPreference;
}

let activeBackend: DeviceBackend | null = null;

export function getBackend(): DeviceBackend {
    if (!activeBackend) {
        throw new Error('No tensor backend is initialized. Call await initBackend() before creating tensors.');
    }
    return activeBackend;
}

export function getBackendKind(): BackendKind | null {
    return activeBackend?.kind ?? null;
}

async function createNative(requiredKind?: 'cuda'): Promise<DeviceBackend | null> {
    const addon = await loadNativeAddon();
    if (!addon) return null;
    const backend = new NativeBackend(addon);
    return requiredKind && backend.kind !== requiredKind ? null : backend;
}

async function createWebGpu(): Promise<DeviceBackend | null> {
    const { tryCreateWebGpuBackend } = await import('./webgpu/webgpu_backend.js');
    return tryCreateWebGpuBackend();
}

export async function initBackend(options: BackendInitOptions = {}): Promise<DeviceBackend> {
    const envPreference = typeof process !== 'undefined'
        ? process.env?.TSGRAD_BACKEND as BackendPreference | undefined
        : undefined;
    const preference = options.backend ?? envPreference ?? 'auto';
    if (activeBackend) {
        const matches = preference === 'auto'
            || preference === activeBackend.kind
            || (preference === 'native' && (activeBackend.kind === 'cuda' || activeBackend.kind === 'native-cpu'));
        if (!matches) {
            throw new Error(`Backend is already initialized as "${activeBackend.kind}", not "${preference}"`);
        }
        return activeBackend;
    }

    if (preference === 'cpu') activeBackend = new CpuBackend();
    else if (preference === 'native') activeBackend = await createNative();
    else if (preference === 'cuda') activeBackend = await createNative('cuda');
    else if (preference === 'webgpu') activeBackend = await createWebGpu();
    else {
        activeBackend = await createNative();
        activeBackend ??= await createWebGpu();
        activeBackend ??= new CpuBackend();
    }

    if (!activeBackend) throw new Error(`Requested backend "${preference}" is unavailable`);
    return activeBackend;
}

export async function synchronizeBackend(): Promise<void> {
    await getBackend().synchronize();
}

/** Intended for isolated tests that have released all tensors. */
export function setBackendForTesting(backend: DeviceBackend | null): void {
    activeBackend = backend;
}

export type { BackendKind, DeviceBackend } from './device_backend.js';
