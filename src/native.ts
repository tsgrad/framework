import type { Shape, Stride } from './tensor_data.js';

export interface NativeStorage {
    readonly __nativeStorage: unique symbol;
}

export type MapNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
) => void;

export type ZipNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    a: NativeStorage, aShape: Shape, aStride: Stride,
    b: NativeStorage, bShape: Shape, bStride: Stride,
) => void;

export type ReduceNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    dim: number,
) => void;

export type ConvNative = (
    out: NativeStorage, outShape: Shape, outStride: Stride,
    input: NativeStorage, inShape: Shape, inStride: Stride,
    weight: NativeStorage, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
) => void;

// Names match the strings addon.cc puts on `exports`.
export type NativeAddon = {
    mapNeg: MapNative;
    mapInv: MapNative;
    mapSigmoid: MapNative;
    mapRelu: MapNative;
    mapLog: MapNative;
    mapExp: MapNative;
    mapId: MapNative;
    zipAdd: ZipNative;
    zipSub: ZipNative;
    zipMul: ZipNative;
    zipDiv: ZipNative;
    zipLt: ZipNative;
    zipEq: ZipNative;
    zipIsClose: ZipNative;
    reduceMax: ReduceNative;
    reduceSum: ReduceNative;
    reduceProd: ReduceNative;
    matmul: ZipNative;
    conv1d: ConvNative;
    conv2d: ConvNative;
    alloc: (numel: number, fill?: number) => NativeStorage;
    empty: (numel: number) => NativeStorage;
    retain: (storage: NativeStorage) => void;
    release: (storage: NativeStorage) => void;
    fill: (storage: NativeStorage, value: number) => void;
    copyToHost: (storage: NativeStorage) => Float32Array;
    copyFromHost: (storage: NativeStorage, values: Float32Array) => void;
    synchronize: () => void;
    backendInfo?: () => { kind: 'cuda' | 'cpu' };
    sgd: (
        param: NativeStorage, shape: Shape, stride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        lr: number,
    ) => void;
    sgdMomentum: (
        param: NativeStorage, shape: Shape, stride: Stride,
        grad: NativeStorage, gradShape: Shape, gradStride: Stride,
        velocity: NativeStorage, velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number,
    ) => void;
};

// Set by initBackend(). Keeping this named export preserves the low-level
// escape hatch used by native-specific tests without importing Node built-ins
// into browser bundles at module evaluation time.
export let native: NativeAddon | null = null;

export async function loadNativeAddon(): Promise<NativeAddon | null> {
    if (native) return native;
    if (typeof process === 'undefined' || !process.versions?.node) return null;

    const moduleName = 'node:module';
    const fsName = 'node:fs';
    const urlName = 'node:url';
    const [{ createRequire }, { existsSync }, { fileURLToPath }] = await Promise.all([
        import(/* @vite-ignore */ moduleName) as Promise<typeof import('node:module')>,
        import(/* @vite-ignore */ fsName) as Promise<typeof import('node:fs')>,
        import(/* @vite-ignore */ urlName) as Promise<typeof import('node:url')>,
    ]);
    const addonRelativePath = ['..', 'backend', 'build', 'Release', 'tsgrad_native.node'].join('/');
    const addonPath = fileURLToPath(new URL(addonRelativePath, import.meta.url));
    if (!existsSync(addonPath)) return null;
    try {
        native = createRequire(import.meta.url)(addonPath) as NativeAddon;
    } catch (error) {
        if (process.env?.TSGRAD_DEBUG_BACKEND) console.warn('Unable to load native tensor backend', error);
        return null;
    }
    return native;
}
