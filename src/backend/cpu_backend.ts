import * as operators from '../operators.js';
import { tensorMap, tensorReduce, tensorZip } from '../tensor_ops.js';
import type {
    BackendStorage, BinaryOp, DeviceBackend, ReduceOp, Shape, Stride, UnaryOp,
} from './device_backend.js';

const unary = {
    neg: operators.neg, inv: operators.inv, sigmoid: operators.sigmoid,
    relu: operators.relu, log: operators.log, exp: operators.exp, id: operators.id,
} satisfies Record<UnaryOp, (x: number) => number>;

const binary = {
    add: operators.add, sub: operators.sub, mul: operators.mul, div: operators.div,
    lt: operators.lt, eq: operators.eq, isClose: operators.isClose,
} satisfies Record<BinaryOp, (a: number, b: number) => number>;

const reductions = {
    max: [operators.max, -Infinity],
    sum: [operators.add, 0],
    prod: [operators.mul, 1],
} satisfies Record<ReduceOp, [(a: number, b: number) => number, number]>;

function host(storage: BackendStorage): Float32Array {
    if (!(storage instanceof Float32Array)) throw new Error('CPU backend requires Float32Array storage');
    return storage;
}

function positionToIndex(ordinal: number, shape: Shape, out: number[]): void {
    for (let i = shape.length - 1; i >= 0; i--) {
        out[i] = ordinal % shape[i]!;
        ordinal = Math.trunc(ordinal / shape[i]!);
    }
}

function indexToPosition(index: number[], stride: Stride): number {
    let position = 0;
    for (let i = 0; i < stride.length; i++) position += index[i]! * stride[i]!;
    return position;
}

export class CpuBackend implements DeviceBackend {
    readonly kind = 'cpu' as const;
    readonly supportsSynchronousRead = true;

    isStorage(value: unknown): boolean { return value instanceof Float32Array; }
    alloc(numel: number, fill = 0): Float32Array {
        const out = new Float32Array(numel);
        if (fill !== 0) out.fill(fill);
        return out;
    }
    empty(numel: number): Float32Array { return new Float32Array(numel); }
    retain(_storage: BackendStorage): void {}
    release(_storage: BackendStorage): void {}
    fill(storage: BackendStorage, value: number): void { host(storage).fill(value); }
    copyFromHost(storage: BackendStorage, values: Float32Array): void {
        const target = host(storage);
        if (target.length !== values.length) throw new Error('copyFromHost length does not match storage');
        target.set(values);
    }
    copyToHost(storage: BackendStorage): Float32Array { return host(storage); }
    async copyToHostAsync(storage: BackendStorage): Promise<Float32Array> { return host(storage); }
    async synchronize(): Promise<void> {}

    map(op: UnaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride): void {
        tensorMap(unary[op])(host(out), outShape, outStride, host(input), inShape, inStride);
    }

    zip(op: BinaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        tensorZip(binary[op])(host(out), outShape, outStride, host(a), aShape, aStride, host(b), bShape, bStride);
    }

    reduce(op: ReduceOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        const [fn, base] = reductions[op];
        tensorReduce(fn, base)(host(out), outShape, outStride, host(input), inShape, inStride, dim);
    }

    matmul(out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        if (aShape.length !== 2 || bShape.length !== 2) throw new Error('CPU matmul requires rank-2 tensors');
        const output = host(out), av = host(a), bv = host(b);
        const m = aShape[0]!, k = aShape[1]!, n = bShape[1]!;
        for (let row = 0; row < m; row++) {
            for (let col = 0; col < n; col++) {
                let total = 0;
                for (let inner = 0; inner < k; inner++)
                    total += av[row * aStride[0]! + inner * aStride[1]!]!
                        * bv[inner * bStride[0]! + col * bStride[1]!]!;
                output[row * outStride[0]! + col * outStride[1]!] = total;
            }
        }
    }

    conv1d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const output = host(out), x = host(input), w = host(weight);
        for (let batch = 0; batch < outShape[0]!; batch++) {
            for (let oc = 0; oc < outShape[1]!; oc++) {
                for (let ow = 0; ow < outShape[2]!; ow++) {
                    let total = 0;
                    for (let ic = 0; ic < inShape[1]!; ic++) {
                        for (let kernel = 0; kernel < weightShape[2]!; kernel++) {
                            const at = reverse ? ow - kernel : ow + kernel;
                            if (at >= 0 && at < inShape[2]!)
                                total += x[batch*inStride[0]! + ic*inStride[1]! + at*inStride[2]!]!
                                    * w[oc*weightStride[0]! + ic*weightStride[1]! + kernel*weightStride[2]!]!;
                        }
                    }
                    output[batch*outStride[0]! + oc*outStride[1]! + ow*outStride[2]!] = total;
                }
            }
        }
    }

    conv2d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const output = host(out), x = host(input), w = host(weight);
        for (let batch = 0; batch < outShape[0]!; batch++) {
            for (let oc = 0; oc < outShape[1]!; oc++) {
                for (let oh = 0; oh < outShape[2]!; oh++) {
                    for (let ow = 0; ow < outShape[3]!; ow++) {
                        let total = 0;
                        for (let ic = 0; ic < inShape[1]!; ic++) {
                            for (let kh = 0; kh < weightShape[2]!; kh++) {
                                const ih = reverse ? oh - kh : oh + kh;
                                if (ih < 0 || ih >= inShape[2]!) continue;
                                for (let kw = 0; kw < weightShape[3]!; kw++) {
                                    const iw = reverse ? ow - kw : ow + kw;
                                    if (iw >= 0 && iw < inShape[3]!)
                                        total += x[batch*inStride[0]! + ic*inStride[1]! + ih*inStride[2]! + iw*inStride[3]!]!
                                            * w[oc*weightStride[0]! + ic*weightStride[1]! + kh*weightStride[2]! + kw*weightStride[3]!]!;
                                }
                            }
                        }
                        output[batch*outStride[0]! + oc*outStride[1]! + oh*outStride[2]! + ow*outStride[3]!] = total;
                    }
                }
            }
        }
    }

    sgd(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, _gradShape: Shape, gradStride: Stride, lr: number): void {
        const p = host(param), g = host(grad), index = new Array(shape.length).fill(0);
        const size = shape.reduce((a, b) => a * b, 1);
        for (let i = 0; i < size; i++) {
            positionToIndex(i, shape, index);
            p[indexToPosition(index, stride)]! -= lr * g[indexToPosition(index, gradStride)]!;
        }
    }

    sgdMomentum(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, _gradShape: Shape, gradStride: Stride,
        velocity: BackendStorage, _velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number): void {
        const p = host(param), g = host(grad), v = host(velocity);
        const index = new Array(shape.length).fill(0);
        const size = shape.reduce((a, b) => a * b, 1);
        for (let i = 0; i < size; i++) {
            positionToIndex(i, shape, index);
            const pp = indexToPosition(index, stride);
            const gp = indexToPosition(index, gradStride);
            const vp = indexToPosition(index, velocityStride);
            const next = beta * v[vp]! + (1 - beta) * g[gp]!;
            v[vp] = next;
            p[pp]! -= lr * next;
        }
    }
}
