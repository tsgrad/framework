import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Shape, Stride } from './tensor_data.js';

export interface NativeStorage {
    readonly __nativeStorage: unique symbol;
}

const require = createRequire(import.meta.url);

// CMake will write this path
const addonPath = fileURLToPath(new URL('../backend/build/Release/tsgrad_native.node', import.meta.url));

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
    zipMul: ZipNative;
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
    copyToHost: (storage: NativeStorage) => Float32Array;
    copyFromHost: (storage: NativeStorage, values: Float32Array) => void;
};

// Missing .node means every caller stays on the TypeScript loops.
export const native: NativeAddon | null = existsSync(addonPath)
    ? require(addonPath) as NativeAddon
    : null;
