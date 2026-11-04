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
    softmax: (out: NativeStorage, outShape: Shape, outStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, dim: number) => void;
    softmaxBackward: (out: NativeStorage, outShape: Shape, outStride: Stride, softmax: NativeStorage, softmaxShape: Shape, softmaxStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, dim: number) => void;
    logSoftmax: (out: NativeStorage, outShape: Shape, outStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, dim: number) => void;
    logSoftmaxBackward: (out: NativeStorage, outShape: Shape, outStride: Stride, logSoftmax: NativeStorage, logSoftmaxShape: Shape, logSoftmaxStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, dim: number) => void;
    softmaxCrossEntropy: (loss: NativeStorage, lossShape: Shape, lossStride: Stride, softmax: NativeStorage, softmaxShape: Shape, softmaxStride: Stride, logits: NativeStorage, logitsShape: Shape, logitsStride: Stride, target: NativeStorage, targetShape: Shape, targetStride: Stride, dim: number) => void;
    softmaxCrossEntropyBackward: (gradLogits: NativeStorage, gradLogitsShape: Shape, gradLogitsStride: Stride, softmax: NativeStorage, softmaxShape: Shape, softmaxStride: Stride, target: NativeStorage, targetShape: Shape, targetStride: Stride, dim: number) => void;
    reluBackward: (out: NativeStorage, outShape: Shape, outStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride) => void;
    sigmoidBackward: (out: NativeStorage, outShape: Shape, outStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, output: NativeStorage, outputShape: Shape, outputStride: Stride) => void;
    expand: (out: NativeStorage, outShape: Shape, outStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride) => void;
    biasAct: (out: NativeStorage, outShape: Shape, outStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, bias: NativeStorage, biasShape: Shape, biasStride: Stride, act: number) => void;
    biasActBackward: (gradInput: NativeStorage, gradInputShape: Shape, gradInputStride: Stride, gradBias: NativeStorage, gradBiasShape: Shape, gradBiasStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, saved: NativeStorage, savedShape: Shape, savedStride: Stride, bias: NativeStorage, biasShape: Shape, biasStride: Stride, act: number) => void;
    layerNorm: (out: NativeStorage, outShape: Shape, outStride: Stride, mean: NativeStorage, meanShape: Shape, meanStride: Stride, rstd: NativeStorage, rstdShape: Shape, rstdStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, weight: NativeStorage, weightShape: Shape, weightStride: Stride, bias: NativeStorage, biasShape: Shape, biasStride: Stride, eps: number) => void;
    layerNormBackward: (gradInput: NativeStorage, gradInputShape: Shape, gradInputStride: Stride, gradWeight: NativeStorage, gradWeightShape: Shape, gradWeightStride: Stride, gradBias: NativeStorage, gradBiasShape: Shape, gradBiasStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, mean: NativeStorage, meanShape: Shape, meanStride: Stride, rstd: NativeStorage, rstdShape: Shape, rstdStride: Stride, weight: NativeStorage, weightShape: Shape, weightStride: Stride) => void;
    maxPool2d: (out: NativeStorage, outShape: Shape, outStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, kh: number, kw: number) => void;
    maxPool2dBackward: (gradInput: NativeStorage, gradInputShape: Shape, gradInputStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, output: NativeStorage, outputShape: Shape, outputStride: Stride, kh: number, kw: number) => void;
    avgPool2d: (out: NativeStorage, outShape: Shape, outStride: Stride, input: NativeStorage, inShape: Shape, inStride: Stride, kh: number, kw: number) => void;
    avgPool2dBackward: (gradInput: NativeStorage, gradInputShape: Shape, gradInputStride: Stride, grad: NativeStorage, gradShape: Shape, gradStride: Stride, kh: number, kw: number) => void;
    uniform: (storage: NativeStorage, low: number, high: number) => void;
};

// Missing .node means every caller stays on the TypeScript loops.
export const native: NativeAddon | null = existsSync(addonPath)
    ? require(addonPath) as NativeAddon
    : null;
