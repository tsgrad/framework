import type { NativeAddon, NativeStorage } from '../native.js';
import type {
    BackendStorage, BinaryOp, DeviceBackend, ReduceOp, Shape, Stride, UnaryOp,
} from './device_backend.js';

function nativeStorage(storage: BackendStorage): NativeStorage {
    if (storage instanceof Float32Array) throw new Error('Native backend requires NativeStorage');
    return storage as NativeStorage;
}

export class NativeBackend implements DeviceBackend {
    readonly kind: 'cuda' | 'native-cpu';
    readonly supportsSynchronousRead = true;

    constructor(readonly addon: NativeAddon) {
        this.kind = addon.backendInfo?.().kind === 'cuda' ? 'cuda' : 'native-cpu';
    }

    isStorage(value: unknown): boolean {
        return typeof value === 'object' && value !== null && !(value instanceof Float32Array);
    }
    alloc(numel: number, fill = 0): NativeStorage { return this.addon.alloc(numel, fill); }
    empty(numel: number): NativeStorage { return this.addon.empty(numel); }
    retain(storage: BackendStorage): void { this.addon.retain(nativeStorage(storage)); }
    release(storage: BackendStorage): void { this.addon.release(nativeStorage(storage)); }
    fill(storage: BackendStorage, value: number): void { this.addon.fill(nativeStorage(storage), value); }
    copyFromHost(storage: BackendStorage, values: Float32Array): void { this.addon.copyFromHost(nativeStorage(storage), values); }
    copyToHost(storage: BackendStorage): Float32Array { return this.addon.copyToHost(nativeStorage(storage)); }
    async copyToHostAsync(storage: BackendStorage): Promise<Float32Array> { return this.copyToHost(storage); }
    async synchronize(): Promise<void> { this.addon.synchronize(); }

    map(op: UnaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride): void {
        const fn = {
            neg: this.addon.mapNeg, inv: this.addon.mapInv, sigmoid: this.addon.mapSigmoid,
            relu: this.addon.mapRelu, log: this.addon.mapLog, exp: this.addon.mapExp, id: this.addon.mapId,
        }[op];
        fn(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride);
    }

    zip(op: BinaryOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        const fn = {
            add: this.addon.zipAdd, sub: this.addon.zipSub, mul: this.addon.zipMul,
            div: this.addon.zipDiv, lt: this.addon.zipLt, eq: this.addon.zipEq,
            isClose: this.addon.zipIsClose,
        }[op];
        fn(nativeStorage(out), outShape, outStride, nativeStorage(a), aShape, aStride, nativeStorage(b), bShape, bStride);
    }

    reduce(op: ReduceOp, out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        const fn = { max: this.addon.reduceMax, sum: this.addon.reduceSum, prod: this.addon.reduceProd }[op];
        fn(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride, dim);
    }

    matmul(out: BackendStorage, outShape: Shape, outStride: Stride,
        a: BackendStorage, aShape: Shape, aStride: Stride,
        b: BackendStorage, bShape: Shape, bStride: Stride): void {
        this.addon.matmul(nativeStorage(out), outShape, outStride, nativeStorage(a), aShape, aStride, nativeStorage(b), bShape, bStride);
    }
    conv1d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        this.addon.conv1d(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride, nativeStorage(weight), weightShape, weightStride, reverse);
    }
    conv2d(out: BackendStorage, outShape: Shape, outStride: Stride,
        input: BackendStorage, inShape: Shape, inStride: Stride,
        weight: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        this.addon.conv2d(nativeStorage(out), outShape, outStride, nativeStorage(input), inShape, inStride, nativeStorage(weight), weightShape, weightStride, reverse);
    }
    sgd(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride, lr: number): void {
        this.addon.sgd(nativeStorage(param), shape, stride, nativeStorage(grad), gradShape, gradStride, lr);
    }
    sgdMomentum(param: BackendStorage, shape: Shape, stride: Stride,
        grad: BackendStorage, gradShape: Shape, gradStride: Stride,
        velocity: BackendStorage, velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number): void {
        this.addon.sgdMomentum(
            nativeStorage(param), shape, stride,
            nativeStorage(grad), gradShape, gradStride,
            nativeStorage(velocity), velocityShape, velocityStride,
            lr, beta,
        );
    }
}
