import * as operators from './operators.js';
import { getBackend } from './backend/index.js';
import type { BinaryOp, ReduceOp, UnaryOp } from './backend/device_backend.js';
import type { Shape, Stride, TensorBuffer } from './tensor_data.js';

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


function unaryName(func: (x: number) => number): UnaryOp {
    if (func === operators.neg) return 'neg';
    if (func === operators.inv) return 'inv';
    if (func === operators.sigmoid) return 'sigmoid';
    if (func === operators.relu) return 'relu';
    if (func === operators.log) return 'log';
    if (func === operators.exp) return 'exp';
    if (func === operators.id) return 'id';
    throw new Error(`Unsupported unary tensor operator: ${func.name}`);
}

export function tensorMap(func: (x: number) => number): MapCall {
    const op = unaryName(func);
    return (out, outShape, outStride, input, inShape, inStride) =>
        getBackend().map(op, out, outShape, outStride, input, inShape, inStride);
}

function binaryName(func: (x: number, y: number) => number): BinaryOp {
    if (func === operators.add) return 'add';
    if (func === operators.sub) return 'sub';
    if (func === operators.mul) return 'mul';
    if (func === operators.div) return 'div';
    if (func === operators.lt) return 'lt';
    if (func === operators.eq) return 'eq';
    if (func === operators.isClose) return 'isClose';
    throw new Error(`Unsupported binary tensor operator: ${func.name}`);
}

export function tensorZip(func: (x: number, y: number) => number): ZipCall {
    const op = binaryName(func);
    return (out, outShape, outStride, a, aShape, aStride, b, bShape, bStride) =>
        getBackend().zip(op, out, outShape, outStride, a, aShape, aStride, b, bShape, bStride);
}

export function tensorReduce(func: (x: number, y: number) => number, base: number): ReduceCall {
    let op: ReduceOp;
    if (func === operators.max && base === -Infinity) op = 'max';
    else if (func === operators.add && base === 0) op = 'sum';
    else if (func === operators.mul && base === 1) op = 'prod';
    else throw new Error(`Unsupported tensor reduction: ${func.name} with base ${base}`);
    return (out, outShape, outStride, input, inShape, inStride, dim) =>
        getBackend().reduce(op, out, outShape, outStride, input, inShape, inStride, dim);
}

export function matMul(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    aStorage: TensorBuffer, aShape: Shape, aStride: Stride,
    bStorage: TensorBuffer, bShape: Shape, bStride: Stride,
): void {
    getBackend().matmul(out, outShape, outStride, aStorage, aShape, aStride, bStorage, bShape, bStride);
}

export function tensorConv1d(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    weight: TensorBuffer, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
): void {
    getBackend().conv1d(out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse);
}

export function tensorConv2d(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    input: TensorBuffer, inShape: Shape, inStride: Stride,
    weight: TensorBuffer, weightShape: Shape, weightStride: Stride,
    reverse: boolean,
): void {
    getBackend().conv2d(out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse);
}

export function tensorScatter(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    indicies: TensorBuffer, indiciesShape: Shape, indiciesStride: Stride,
    grad: TensorBuffer, gradShape: Shape, gradStride: Stride,
    dim: number,
): void {
    getBackend().scatter(out, outShape, outStride, indicies, indiciesShape, indiciesStride, grad, gradShape, gradStride, dim);
}

export function tensorGather(
    out: TensorBuffer, outShape: Shape, outStride: Stride,
    indicies: TensorBuffer, indiciesShape: Shape, indiciesStride: Stride,
    values: TensorBuffer, valuesShape: Shape, valuesStride: Stride,
    dim: number
): void {
    getBackend().gather(out, outShape, outStride, indicies, indiciesShape, indiciesStride, values, valuesShape, valuesStride, dim);
}