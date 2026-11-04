import * as operators from './operators.js';
import { tensorMap as slowMap, tensorReduce as slowReduce, tensorZip as slowZip } from './tensor_ops.js';
import { fastMatMul, fastConv1d, fastConv2d } from './fast_ops.js';
import { native, type NativeStorage } from './native.js';
import type { Shape, Storage, Stride, TensorBuffer } from './tensor_data.js';

type MapCall = (
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
) => void;

type ZipCall = (
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    a: TensorBuffer, aShape: Shape, aStride: Stride,
    b: TensorBuffer, bShape: Shape, bStride: Stride,
) => void;

type ReduceCall = (
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    dim: number,
) => void;


// Same signatures as tensor_ops.ts / fast_ops.ts, so tensor_functions.ts can
// import these names instead. An unknown operator, or a missing .node file,
// keeps the existing TypeScript implementation.

export function tensorMap(func: (x: number) => number): MapCall {
    if (native) {
        if (func === operators.neg) return native.mapNeg as MapCall;
        if (func === operators.inv) return native.mapInv as MapCall;
        if (func === operators.sigmoid) return native.mapSigmoid as MapCall;
        if (func === operators.relu) return native.mapRelu as MapCall;
        if (func === operators.log) return native.mapLog as MapCall;
        if (func === operators.exp) return native.mapExp as MapCall;
        if (func === operators.id) return native.mapId as MapCall;
    }
    return slowMap(func) as MapCall;
}

export function tensorZip(func: (x: number, y: number) => number): ZipCall {
    if (native) {
        if (func === operators.add) return native.zipAdd as ZipCall;
        if (func === operators.sub) return native.zipSub as ZipCall;
        if (func === operators.mul) return native.zipMul as ZipCall;
        if (func === operators.div) return native.zipDiv as ZipCall;
        if (func === operators.lt) return native.zipLt as ZipCall;
        if (func === operators.eq) return native.zipEq as ZipCall;
        if (func === operators.isClose) return native.zipIsClose as ZipCall;
    }
    return slowZip(func) as ZipCall;
}

export function tensorReduce(func: (x: number, y: number) => number, base: number): ReduceCall {
    if (native) {
        if (func === operators.max && base === -Infinity) return native.reduceMax as ReduceCall;
        if (func === operators.add && base === 0) return native.reduceSum as ReduceCall;
        if (func === operators.mul && base === 1) return native.reduceProd as ReduceCall;
    }
    return slowReduce(func, base) as ReduceCall;
}

export function matMul(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    aStorage: TensorBuffer, aShape: Shape, aStride: Stride,
    bStorage: TensorBuffer, bShape: Shape, bStride: Stride,
): void {
    if (!native) {
        fastMatMul(out as Storage, outShape, outStride, aStorage as Storage, aShape, aStride, bStorage as Storage, bShape, bStride);
        return;
    }
    native.matmul(out as NativeStorage, outShape, outStride, aStorage as NativeStorage, aShape, aStride, bStorage as NativeStorage, bShape, bStride);
}

export function tensorConv1d(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    weight: TensorBuffer, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
): void {
    if (!native) {
        fastConv1d(out as Storage, outShape, outStride, input as Storage, inShape, inStride, weight as Storage, weightShape, weightStride, reverse);
        return;
    }
    native.conv1d(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, weight as NativeStorage, weightShape, weightStride, reverse);
}

export function tensorConv2d(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    weight: TensorBuffer, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
): void {
    if (!native) {
        fastConv2d(out as Storage, outShape, outStride, input as Storage, inShape, inStride, weight as Storage, weightShape, weightStride, reverse);
        return;
    }
    native.conv2d(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, weight as NativeStorage, weightShape, weightStride, reverse);
}

function needNative(): NonNullable<typeof native> {
    if (!native) throw "native storage requires the addon";
    return native;
}

export function nativeSoftmax(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    dim: number,
): void {
    needNative().softmax(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, dim);
}

export function nativeSoftmaxBackward(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    softmax: TensorBuffer, softmaxShape: Shape, softmaxStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    dim: number,
): void {
    needNative().softmaxBackward(out as NativeStorage, outShape, outStride, softmax as NativeStorage, softmaxShape, softmaxStride, grad as NativeStorage, gradShape, gradStride, dim);
}

export function nativeLogSoftmax(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    dim: number,
): void {
    needNative().logSoftmax(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, dim);
}

export function nativeLogSoftmaxBackward(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    logSoftmax: TensorBuffer, logSoftmaxShape: Shape, logSoftmaxStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    dim: number,
): void {
    needNative().logSoftmaxBackward(out as NativeStorage, outShape, outStride, logSoftmax as NativeStorage, logSoftmaxShape, logSoftmaxStride, grad as NativeStorage, gradShape, gradStride, dim);
}

export function nativeSoftmaxCrossEntropy(
    loss: TensorBuffer, lossShape: Shape, lossStride: Stride,
    softmax: TensorBuffer, softmaxShape: Shape, softmaxStride: Stride,
    logits: TensorBuffer, logitsShape: Shape, logitsStride: Stride,
    target: TensorBuffer, targetShape: Shape, targetStride: Stride,
    dim: number,
): void {
    needNative().softmaxCrossEntropy(loss as NativeStorage, lossShape, lossStride, softmax as NativeStorage, softmaxShape, softmaxStride, logits as NativeStorage, logitsShape, logitsStride, target as NativeStorage, targetShape, targetStride, dim);
}

export function nativeSoftmaxCrossEntropyBackward(
    gradLogits: TensorBuffer, gradLogitsShape: Shape, gradLogitsStride: Stride,
    softmax: TensorBuffer, softmaxShape: Shape, softmaxStride: Stride,
    target: TensorBuffer, targetShape: Shape, targetStride: Stride,
    dim: number,
): void {
    needNative().softmaxCrossEntropyBackward(gradLogits as NativeStorage, gradLogitsShape, gradLogitsStride, softmax as NativeStorage, softmaxShape, softmaxStride, target as NativeStorage, targetShape, targetStride, dim);
}

export function nativeReluBackward(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
): void {
    needNative().reluBackward(out as NativeStorage, outShape, outStride, grad as NativeStorage, gradShape, gradStride, input as NativeStorage, inShape, inStride);
}

export function nativeSigmoidBackward(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    output: TensorBuffer, outputShape: Shape, outputStride: Stride,
): void {
    needNative().sigmoidBackward(out as NativeStorage, outShape, outStride, grad as NativeStorage, gradShape, gradStride, output as NativeStorage, outputShape, outputStride);
}

export function nativeExpand(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
): void {
    needNative().expand(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride);
}

export function nativeBiasAct(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    bias: TensorBuffer, biasShape: Shape, biasStride: Stride,
    act: number,
): void {
    needNative().biasAct(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, bias as NativeStorage, biasShape, biasStride, act);
}

export function nativeBiasActBackward(
    gradInput: TensorBuffer, gradInputShape: Shape, gradInputStride: Stride,
    gradBias: TensorBuffer, gradBiasShape: Shape, gradBiasStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    saved: TensorBuffer, savedShape: Shape, savedStride: Stride,
    bias: TensorBuffer, biasShape: Shape, biasStride: Stride,
    act: number,
): void {
    needNative().biasActBackward(gradInput as NativeStorage, gradInputShape, gradInputStride, gradBias as NativeStorage, gradBiasShape, gradBiasStride, grad as NativeStorage, gradShape, gradStride, saved as NativeStorage, savedShape, savedStride, bias as NativeStorage, biasShape, biasStride, act);
}

export function nativeLayerNorm(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    mean: TensorBuffer, meanShape: Shape, meanStride: Stride,
    rstd: TensorBuffer, rstdShape: Shape, rstdStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    weight: TensorBuffer, weightShape: Shape, weightStride: Stride,
    bias: TensorBuffer, biasShape: Shape, biasStride: Stride,
    eps: number,
): void {
    needNative().layerNorm(out as NativeStorage, outShape, outStride, mean as NativeStorage, meanShape, meanStride, rstd as NativeStorage, rstdShape, rstdStride, input as NativeStorage, inShape, inStride, weight as NativeStorage, weightShape, weightStride, bias as NativeStorage, biasShape, biasStride, eps);
}

export function nativeLayerNormBackward(
    gradInput: TensorBuffer, gradInputShape: Shape, gradInputStride: Stride,
    gradWeight: TensorBuffer, gradWeightShape: Shape, gradWeightStride: Stride,
    gradBias: TensorBuffer, gradBiasShape: Shape, gradBiasStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    mean: TensorBuffer, meanShape: Shape, meanStride: Stride,
    rstd: TensorBuffer, rstdShape: Shape, rstdStride: Stride,
    weight: TensorBuffer, weightShape: Shape, weightStride: Stride,
): void {
    needNative().layerNormBackward(gradInput as NativeStorage, gradInputShape, gradInputStride, gradWeight as NativeStorage, gradWeightShape, gradWeightStride, gradBias as NativeStorage, gradBiasShape, gradBiasStride, grad as NativeStorage, gradShape, gradStride, input as NativeStorage, inShape, inStride, mean as NativeStorage, meanShape, meanStride, rstd as NativeStorage, rstdShape, rstdStride, weight as NativeStorage, weightShape, weightStride);
}

export function nativeMaxPool2d(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    kh: number, kw: number,
): void {
    needNative().maxPool2d(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, kh, kw);
}

export function nativeMaxPool2dBackward(
    gradInput: TensorBuffer, gradInputShape: Shape, gradInputStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    output: TensorBuffer, outputShape: Shape, outputStride: Stride,
    kh: number, kw: number,
): void {
    needNative().maxPool2dBackward(gradInput as NativeStorage, gradInputShape, gradInputStride, grad as NativeStorage, gradShape, gradStride, input as NativeStorage, inShape, inStride, output as NativeStorage, outputShape, outputStride, kh, kw);
}

export function nativeAvgPool2d(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    kh: number, kw: number,
): void {
    needNative().avgPool2d(out as NativeStorage, outShape, outStride, input as NativeStorage, inShape, inStride, kh, kw);
}

export function nativeAvgPool2dBackward(
    gradInput: TensorBuffer, gradInputShape: Shape, gradInputStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    kh: number, kw: number,
): void {
    needNative().avgPool2dBackward(gradInput as NativeStorage, gradInputShape, gradInputStride, grad as NativeStorage, gradShape, gradStride, kh, kw);
}
