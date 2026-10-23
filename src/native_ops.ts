import * as operators from './operators.js';
import { tensorMap as slowMap, tensorReduce as slowReduce, tensorZip as slowZip } from './tensor_ops.js';
import { fastMatMul, fastConv1d, fastConv2d } from './fast_ops.js';
import { native } from './native.js';
import type { Shape, Storage, Stride } from './tensor_data.js';

// Same signatures as tensor_ops.ts / fast_ops.ts, so tensor_functions.ts can
// import these names instead. An unknown operator, or a missing .node file,
// keeps the existing TypeScript implementation.

export function tensorMap(func: (x: number) => number) {
    if (native) {
        if (func === operators.neg) return native.mapNeg;
        if (func === operators.inv) return native.mapInv;
        if (func === operators.sigmoid) return native.mapSigmoid;
        if (func === operators.relu) return native.mapRelu;
        if (func === operators.log) return native.mapLog;
        if (func === operators.exp) return native.mapExp;
        if (func === operators.id) return native.mapId;
    }
    return slowMap(func);
}

export function tensorZip(func: (x: number, y: number) => number) {
    if (native) {
        if (func === operators.add) return native.zipAdd;
        if (func === operators.mul) return native.zipMul;
        if (func === operators.lt) return native.zipLt;
        if (func === operators.eq) return native.zipEq;
        if (func === operators.isClose) return native.zipIsClose;
    }
    return slowZip(func);
}

export function tensorReduce(func: (x: number, y: number) => number, base: number) {
    if (native) {
        if (func === operators.max && base === -Infinity) return native.reduceMax;
        if (func === operators.add && base === 0) return native.reduceSum;
        if (func === operators.mul && base === 1) return native.reduceProd;
    }
    return slowReduce(func, base);
}

export function matMul(
    out: Storage, outShape: Shape, outStride: Stride,
    aStorage: Storage, aShape: Shape, aStride: Stride,
    bStorage: Storage, bShape: Shape, bStride: Stride,
): void {
    if (!native) {
        fastMatMul(out, outShape, outStride, aStorage, aShape, aStride, bStorage, bShape, bStride);
        return;
    }
    native.matmul(out, outShape, outStride, aStorage, aShape, aStride, bStorage, bShape, bStride);
}

export function tensorConv1d(
    out: Storage, outShape: Shape, outStride: Stride,
    input: Storage, inShape: Shape, inStride: Stride,
    weight: Storage, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
): void {
    if (!native) {
        fastConv1d(out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse);
        return;
    }
    native.conv1d(out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse);
}

export function tensorConv2d(
    out: Storage, outShape: Shape, outStride: Stride,
    input: Storage, inShape: Shape, inStride: Stride,
    weight: Storage, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
): void {
    if (!native) {
        fastConv2d(out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse);
        return;
    }
    native.conv2d(out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse);
}
