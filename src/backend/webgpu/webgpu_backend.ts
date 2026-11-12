import type {
    BackendStorage, BinaryOp, DeviceBackend, ReduceOp, Shape, Stride, UnaryOp,
} from '../device_backend.js';
import {
    alignedMapShader, alignedZipShader, CONV1D_SHADER, CONV2D_SHADER, FILL_SHADER,
    MATMUL_SHADER, MAX_DIMS, reduceShader, SGD_MOMENTUM_SHADER, SGD_SHADER,
    stridedMapShader, stridedZipShader, WORKGROUP_SIZE,
} from './webgpu_kernels.js';
import { WebGPUStorage } from './webgpu_storage.js';

// WebGPU flag values are fixed by the specification. TypeScript's current DOM
// declarations expose the interfaces but not these runtime constant objects.
const BUFFER_USAGE = {
    MAP_READ: 0x0001,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
} as const;
const MAP_MODE_READ = 0x0001;

function gpuStorage(storage: BackendStorage): WebGPUStorage {
    if (!(storage instanceof WebGPUStorage) || storage.released)
        throw new Error('WebGPU backend requires live WebGPUStorage');
    return storage;
}

function product(shape: Shape): number {
    return shape.reduce((a, b) => a * b, 1);
}

function positionToIndex(ordinal: number, shape: Shape, out: number[]): void {
    for (let i = shape.length - 1; i >= 0; i--) {
        out[i] = ordinal % shape[i]!;
        ordinal = Math.trunc(ordinal / shape[i]!);
    }
}

function leadingOffset(index: number[], stride: Stride, count: number): number {
    let position = 0;
    for (let i = 0; i < count; i++) position += index[i]! * stride[i]!;
    return position;
}

function same(a: Shape, b: Shape): boolean {
    return a.length === b.length && a.every((v, i) => v === b[i]);
}

function contiguous(shape: Shape, stride: Stride): boolean {
    let expected = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (shape[i]! > 1 && stride[i] !== expected) return false;
        expected *= shape[i]!;
    }
    return true;
}

function floatBits(value: number): number {
    const buffer = new ArrayBuffer(4);
    new Float32Array(buffer)[0] = value;
    return new Uint32Array(buffer)[0]!;
}

export class WebGpuBackend implements DeviceBackend {
    readonly kind = 'webgpu' as const;
    readonly supportsSynchronousRead = false;
    private readonly pipelines = new Map<string, GPUComputePipeline>();
    private readonly free = new Map<number, GPUBuffer[]>();
    private pendingDestroy: GPUBuffer[] = [];

    constructor(readonly device: GPUDevice) {}

    isStorage(value: unknown): boolean { return value instanceof WebGPUStorage && !value.released; }

    private takeBuffer(byteSize: number): GPUBuffer {
        const list = this.free.get(byteSize);
        const reused = list?.pop();
        if (reused) return reused;
        return this.device.createBuffer({
            size: byteSize,
            usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST,
        });
    }

    alloc(numel: number, fill = 0): WebGPUStorage {
        const byteSize = Math.max(4, numel * Float32Array.BYTES_PER_ELEMENT);
        const storage = new WebGPUStorage(this.takeBuffer(byteSize), numel, byteSize);
        this.fill(storage, fill);
        return storage;
    }

    empty(numel: number): WebGPUStorage {
        const byteSize = Math.max(4, numel * Float32Array.BYTES_PER_ELEMENT);
        return new WebGPUStorage(this.takeBuffer(byteSize), numel, byteSize);
    }

    retain(storage: BackendStorage): void { gpuStorage(storage).owners++; }

    release(storage: BackendStorage): void {
        const value = gpuStorage(storage);
        value.owners--;
        if (value.owners < 0) throw new Error('WebGPU storage released too many times');
        if (value.owners === 0) {
            value.released = true;
            // Every command uses this device's single ordered queue. Reusing
            // the same GPUBuffer is safe: any later write is submitted after
            // all earlier reads that referenced it. Never destroy pooled
            // data buffers until backend shutdown.
            this.free.set(value.byteSize, [...(this.free.get(value.byteSize) ?? []), value.buffer]);
        }
    }

    private pipeline(key: string, shader: string): GPUComputePipeline {
        let pipeline = this.pipelines.get(key);
        if (!pipeline) {
            pipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: this.device.createShaderModule({ code: shader }), entryPoint: 'main' },
            });
            this.pipelines.set(key, pipeline);
        }
        return pipeline;
    }

    private params(values: number[], uniform = false): GPUBuffer {
        const data = Uint32Array.from(values);
        const byteSize = Math.max(uniform ? 16 : 4, Math.ceil(data.byteLength / 4) * 4);
        const buffer = this.device.createBuffer({
            size: byteSize,
            usage: (uniform ? BUFFER_USAGE.UNIFORM : BUFFER_USAGE.STORAGE) | BUFFER_USAGE.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(buffer.getMappedRange()).set(data);
        buffer.unmap();
        return buffer;
    }

    private pad(values: number[]): number[] {
        if (values.length > MAX_DIMS) throw new Error(`WebGPU tensors support at most ${MAX_DIMS} dimensions`);
        if (values.some(v => v < 0 || !Number.isSafeInteger(v))) throw new Error('WebGPU shapes and strides must be non-negative integers');
        return [...values, ...new Array(MAX_DIMS - values.length).fill(0)];
    }

    private submit(
        key: string,
        shader: string,
        entries: GPUBindGroupEntry[],
        workgroups: [number, number?, number?],
        temporary: GPUBuffer[],
    ): void {
        const pipeline = this.pipeline(key, shader);
        const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroups[0], workgroups[1] ?? 1, workgroups[2] ?? 1);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        this.deferDestroy(...temporary);
    }

    private deferDestroy(...buffers: GPUBuffer[]): void {
        this.pendingDestroy.push(...buffers);
        if (this.pendingDestroy.length >= 128) {
            void this.flushPendingDestroy();
        }
    }

    private async flushPendingDestroy(): Promise<void> {
        // Capture only buffers used by work submitted before this fence.
        // Later submissions append to a fresh array and are cleaned by a
        // later fence, so concurrent readback cannot destroy their params.
        const batch = this.pendingDestroy.splice(0);
        await this.device.queue.onSubmittedWorkDone();
        batch.forEach(buffer => buffer.destroy());
    }

    fill(storage: BackendStorage, value: number): void {
        const target = gpuStorage(storage);
        if (target.numel === 0) return;
        if (value === 0) {
            const encoder = this.device.createCommandEncoder();
            encoder.clearBuffer(target.buffer);
            this.device.queue.submit([encoder.finish()]);
            return;
        }
        const params = this.params([target.numel, floatBits(value)], true);
        this.submit('fill', FILL_SHADER, [
            { binding: 0, resource: { buffer: target.buffer } },
            { binding: 1, resource: { buffer: params } },
        ], [Math.ceil(target.numel / WORKGROUP_SIZE)], [params]);
    }

    copyFromHost(storage: BackendStorage, values: Float32Array): void {
        const target = gpuStorage(storage);
        if (values.length !== target.numel) throw new Error('copyFromHost length does not match storage');
        if (values.length > 0) this.device.queue.writeBuffer(target.buffer, 0, values);
    }

    copyToHost(_storage: BackendStorage): Float32Array {
        throw new Error('WebGPU readback is asynchronous; use readStorageAsync() or Tensor.toFloat32()');
    }

    async copyToHostAsync(storage: BackendStorage): Promise<Float32Array> {
        const source = gpuStorage(storage);
        if (source.numel === 0) return new Float32Array();
        const staging = this.device.createBuffer({
            size: source.byteSize,
            usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
        });
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(source.buffer, 0, staging, 0, source.byteSize);
        this.device.queue.submit([encoder.finish()]);
        const cleanup = this.flushPendingDestroy();
        await staging.mapAsync(MAP_MODE_READ);
        const result = new Float32Array(staging.getMappedRange()).slice(0, source.numel);
        staging.unmap();
        staging.destroy();
        await cleanup;
        return result;
    }

    async synchronize(): Promise<void> {
        await this.flushPendingDestroy();
    }

    private copyStorage(source: WebGPUStorage, target: WebGPUStorage): void {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(source.buffer, 0, target.buffer, 0, Math.min(source.byteSize, target.byteSize));
        this.device.queue.submit([encoder.finish()]);
    }

    map(op: UnaryOp, outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage);
        if (out.buffer === input.buffer) {
            const temp = this.empty(out.numel);
            this.map(op, temp, outShape, outStride, input, inShape, inStride);
            this.copyStorage(temp, out);
            this.release(temp);
            return;
        }
        const size = product(outShape);
        if (size === 0) return;
        const aligned = same(outShape, inShape) && contiguous(outShape, outStride) && contiguous(inShape, inStride);
        const params = aligned
            ? this.params([size], true)
            : this.params([size, outShape.length, inShape.length, 0,
                ...this.pad(outShape), ...this.pad(outStride), ...this.pad(inShape), ...this.pad(inStride)]);
        this.submit(`map:${op}:${aligned}`, aligned ? alignedMapShader(op) : stridedMapShader(op), [
            { binding: 0, resource: { buffer: input.buffer } },
            { binding: 1, resource: { buffer: out.buffer } },
            { binding: 2, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    zip(op: BinaryOp, outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        aStorage: BackendStorage, aShape: Shape, aStride: Stride,
        bStorage: BackendStorage, bShape: Shape, bStride: Stride): void {
        const out = gpuStorage(outStorage), a = gpuStorage(aStorage), b = gpuStorage(bStorage);
        if (out.buffer === a.buffer || out.buffer === b.buffer) {
            const temp = this.empty(out.numel);
            this.zip(op, temp, outShape, outStride, a, aShape, aStride, b, bShape, bStride);
            this.copyStorage(temp, out);
            this.release(temp);
            return;
        }
        const size = product(outShape);
        if (size === 0) return;
        const aligned = same(outShape, aShape) && same(outShape, bShape)
            && contiguous(outShape, outStride) && contiguous(aShape, aStride) && contiguous(bShape, bStride);
        const params = aligned
            ? this.params([size], true)
            : this.params([size, outShape.length, aShape.length, bShape.length,
                ...this.pad(outShape), ...this.pad(outStride),
                ...this.pad(aShape), ...this.pad(aStride),
                ...this.pad(bShape), ...this.pad(bStride)]);
        this.submit(`zip:${op}:${aligned}`, aligned ? alignedZipShader(op) : stridedZipShader(op), [
            { binding: 0, resource: { buffer: a.buffer } },
            { binding: 1, resource: { buffer: b.buffer } },
            { binding: 2, resource: { buffer: out.buffer } },
            { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    reduce(op: ReduceOp, outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride, dim: number): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage);
        const outSize = product(outShape);
        if (outSize === 0) return;
        const params = this.params([outSize, inShape.length, dim, inShape[dim]!,
            ...this.pad(outShape), ...this.pad(outStride), ...this.pad(inShape), ...this.pad(inStride)]);
        this.submit(`reduce:${op}`, reduceShader(op), [
            { binding: 0, resource: { buffer: input.buffer } },
            { binding: 1, resource: { buffer: out.buffer } },
            { binding: 2, resource: { buffer: params } },
        ], [outSize], [params]);
    }

    matmul(outStorage: BackendStorage, _outShape: Shape, outStride: Stride,
        aStorage: BackendStorage, aShape: Shape, aStride: Stride,
        bStorage: BackendStorage, bShape: Shape, bStride: Stride): void {
        if (aShape.length !== bShape.length || aShape.length < 2)
            throw new Error('WebGPU matmul requires matching ranks of at least 2');
        const rank = aShape.length;
        for (let i = 0; i < rank - 2; i++)
            if (aShape[i] !== bShape[i]) throw new Error('WebGPU matmul batch dimensions must match');
        const m = aShape[rank - 2]!, k = aShape[rank - 1]!, n = bShape[rank - 1]!;
        if (k !== bShape[rank - 2]) throw new Error('WebGPU matmul inner dimensions must match');

        const batchRank = rank - 2;
        const batchShape = aShape.slice(0, batchRank);
        let batch = 1;
        for (const dim of batchShape) batch *= dim;
        if (batch === 0 || m === 0 || n === 0) return;

        const out = gpuStorage(outStorage), a = gpuStorage(aStorage), b = gpuStorage(bStorage);
        const batchIndex = new Array(batchRank).fill(0);
        for (let ordinal = 0; ordinal < batch; ordinal++) {
            let aBase = 0, bBase = 0, outBase = 0;
            if (batchRank > 0) {
                positionToIndex(ordinal, batchShape, batchIndex);
                aBase = leadingOffset(batchIndex, aStride, batchRank);
                bBase = leadingOffset(batchIndex, bStride, batchRank);
                outBase = leadingOffset(batchIndex, outStride, batchRank);
            }
            const params = this.params([
                m, n, k,
                outStride[rank - 2]!, outStride[rank - 1]!,
                aStride[rank - 2]!, aStride[rank - 1]!,
                bStride[rank - 2]!, bStride[rank - 1]!,
                aBase, bBase, outBase,
            ]);
            this.submit('matmul:tiled16', MATMUL_SHADER, [
                { binding: 0, resource: { buffer: a.buffer } }, { binding: 1, resource: { buffer: b.buffer } },
                { binding: 2, resource: { buffer: out.buffer } }, { binding: 3, resource: { buffer: params } },
            ], [Math.ceil(n / 16), Math.ceil(m / 16)], [params]);
        }
    }

    conv1d(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride,
        weightStorage: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage), weight = gpuStorage(weightStorage);
        const values = [outShape[0]!, outShape[1]!, outShape[2]!, inShape[1]!, inShape[2]!, weightShape[2]!, Number(reverse),
            ...outStride, ...inStride, ...weightStride];
        const params = this.params(values);
        this.submit('conv1d', CONV1D_SHADER, [
            { binding: 0, resource: { buffer: input.buffer } }, { binding: 1, resource: { buffer: weight.buffer } },
            { binding: 2, resource: { buffer: out.buffer } }, { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(product(outShape) / WORKGROUP_SIZE)], [params]);
    }

    conv2d(outStorage: BackendStorage, outShape: Shape, outStride: Stride,
        inputStorage: BackendStorage, inShape: Shape, inStride: Stride,
        weightStorage: BackendStorage, weightShape: Shape, weightStride: Stride, reverse: boolean): void {
        const out = gpuStorage(outStorage), input = gpuStorage(inputStorage), weight = gpuStorage(weightStorage);
        const values = [outShape[0]!, outShape[1]!, outShape[2]!, outShape[3]!, inShape[1]!, inShape[2]!, inShape[3]!,
            weightShape[2]!, weightShape[3]!, Number(reverse), ...outStride, ...inStride, ...weightStride];
        const params = this.params(values);
        this.submit('conv2d', CONV2D_SHADER, [
            { binding: 0, resource: { buffer: input.buffer } }, { binding: 1, resource: { buffer: weight.buffer } },
            { binding: 2, resource: { buffer: out.buffer } }, { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(product(outShape) / WORKGROUP_SIZE)], [params]);
    }

    sgd(paramStorage: BackendStorage, shape: Shape, stride: Stride,
        gradStorage: BackendStorage, _gradShape: Shape, gradStride: Stride, lr: number): void {
        if (!contiguous(shape, stride) || !contiguous(shape, gradStride)) throw new Error('WebGPU SGD requires contiguous tensors');
        const param = gpuStorage(paramStorage), grad = gpuStorage(gradStorage), size = product(shape);
        const params = this.params([size, floatBits(lr)]);
        this.submit('sgd', SGD_SHADER, [
            { binding: 0, resource: { buffer: param.buffer } }, { binding: 1, resource: { buffer: grad.buffer } },
            { binding: 2, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }

    sgdMomentum(paramStorage: BackendStorage, shape: Shape, stride: Stride,
        gradStorage: BackendStorage, _gradShape: Shape, gradStride: Stride,
        velocityStorage: BackendStorage, _velocityShape: Shape, velocityStride: Stride,
        lr: number, beta: number): void {
        if (!contiguous(shape, stride) || !contiguous(shape, gradStride) || !contiguous(shape, velocityStride))
            throw new Error('WebGPU SGD momentum requires contiguous tensors');
        const param = gpuStorage(paramStorage), grad = gpuStorage(gradStorage), velocity = gpuStorage(velocityStorage);
        const size = product(shape), params = this.params([size, floatBits(lr), floatBits(beta)]);
        this.submit('sgdMomentum', SGD_MOMENTUM_SHADER, [
            { binding: 0, resource: { buffer: param.buffer } }, { binding: 1, resource: { buffer: grad.buffer } },
            { binding: 2, resource: { buffer: velocity.buffer } }, { binding: 3, resource: { buffer: params } },
        ], [Math.ceil(size / WORKGROUP_SIZE)], [params]);
    }
}

export async function tryCreateWebGpuBackend(): Promise<WebGpuBackend | null> {
    let gpu: { requestAdapter(): Promise<GPUAdapter | null> } | undefined;
    const browserGpu = (globalThis.navigator as Navigator & { gpu?: typeof gpu } | undefined)?.gpu;
    if (browserGpu) {
        gpu = browserGpu;
    } else if (typeof process !== 'undefined' && process.versions?.node) {
        let module: { globals: object; create(args: unknown[]): unknown };
        try {
            module = await import(/* @vite-ignore */ 'webgpu');
        } catch (error) {
            if (process.env?.TSGRAD_DEBUG_BACKEND) console.warn('Unable to load Node WebGPU provider', error);
            return null;
        }
        Object.assign(globalThis, module.globals);
        gpu = module.create([]) as unknown as typeof gpu;
    }
    if (!gpu) return null;
    const adapter = await gpu.requestAdapter();
    if (!adapter) return null;
    return new WebGpuBackend(await adapter.requestDevice());
}
