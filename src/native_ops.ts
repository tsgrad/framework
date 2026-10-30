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
        if (func === operators.mul) return native.zipMul as ZipCall;
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
