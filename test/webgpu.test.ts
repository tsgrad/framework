import { beforeAll, describe, expect, test } from 'vitest';
import { setBackendForTesting } from '../src/backend/index.js';
import { tryCreateWebGpuBackend, WebGpuBackend } from '../src/backend/webgpu/webgpu_backend.js';
import { Tensor } from '../src/tensor.js';
import { conv1d, gather, scatter } from '../src/tensor_functions.js';
import { readStorageAsync, TensorData } from '../src/tensor_data.js';

const backend = await tryCreateWebGpuBackend();

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) expect(actual[i]).toBeCloseTo(expected[i]!);
}

describe.skipIf(!backend)('WebGPU backend', () => {
    beforeAll(() => setBackendForTesting(backend as WebGpuBackend));

    test('keeps chained map, zip, and reduce operations on device', async () => {
        const input = new Tensor(new TensorData(new Float32Array([-2, 1, -4, 3]), [2, 2]));
        const result = input.relu().add(2).mul(input.relu()).sum(1);
        expectClose(await result.toFloat32(), [3, 15]);
    });

    test('supports broadcast and permuted views', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
        const b = new Tensor(new TensorData(new Float32Array([10, 20]), [2, 1]));
        expectClose(await a.add(b).toFloat32(), [11, 12, 13, 24, 25, 26]);
        expectClose(await a.permute(1, 0).tocontiguous().toFloat32(), [1, 4, 2, 5, 3, 6]);
    });

    test('covers unary, comparison, and reduction operators', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4]), [2, 2]));
        expectClose(await a.neg().exp().toFloat32(), [Math.exp(-1), Math.exp(-2), Math.exp(-3), Math.exp(-4)]);
        expectClose(await a.sigmoid().toFloat32(), [0.7310586, 0.880797, 0.952574, 0.982014]);
        expectClose(await a.sub(2).div(2).toFloat32(), [-0.5, 0, 0.5, 1]);
        expectClose(await a.lt(3).toFloat32(), [1, 1, 0, 0]);
        expectClose(await a.eq(3).toFloat32(), [0, 0, 1, 0]);
        expectClose(await a.max(1).toFloat32(), [2, 4]);
    });

    test('runs tiled matmul and convolution', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
        const b = new Tensor(new TensorData(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]));
        expectClose(await a.matmul(b).toFloat32(), [58, 64, 139, 154]);

        const input = new TensorData(new Float32Array([1, 2, 3, 4]), [1, 1, 4]);
        const weight = new TensorData(new Float32Array([1, 0, -1]), [1, 1, 3]);
        expectClose(await readStorageAsync(conv1d(input, weight)._storage), [-2, -2, 3, 4]);
    });

    test('runs matmul backward through strided views', async () => {
        const a = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
        const b = new Tensor(new TensorData(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]));
        a._requiresGrad(true);
        b._requiresGrad(true);
        a.matmul(b).sum(0).sum(1).backward();
        expectClose(await a.grad!.toFloat32(), [15, 19, 23, 15, 19, 23]);
        expectClose(await b.grad!.toFloat32(), [5, 5, 7, 7, 9, 9]);
    });

    test('updates parameters with SGD kernels', async () => {
        const param = backend!.alloc(3);
        const grad = backend!.alloc(3);
        backend!.copyFromHost(param, new Float32Array([1, 2, 3]));
        backend!.copyFromHost(grad, new Float32Array([0.5, -1, 2]));
        backend!.sgd(param, [3], [1], grad, [3], [1], 0.1);
        expectClose(await backend!.copyToHostAsync(param), [0.95, 2.1, 2.8]);
        backend!.release(param);
        backend!.release(grad);
    });

    test('gathers through strided index and value views', async () => {
        const values = new TensorData(new Float32Array([10, 11, 12, 20, 21, 22]), [2, 3]);
        const indices = new TensorData(new Float32Array([1, 0, 0, 1]), [2, 2]);
        expectClose(await readStorageAsync(gather(indices, values, 1)._storage), [11, 10, 20, 21]);

        const column = new TensorData(new Float32Array([0, 1, 2, 0]), [2, 2]);
        expectClose(await readStorageAsync(gather(column, values.permute(1, 0), 0)._storage), [10, 21, 12, 20]);

        const wide = new TensorData(new Float32Array([1, 0, 2, 0, 1, 1]), [2, 3]);
        const table = new TensorData(new Float32Array([10, 20, 11, 21, 12, 22, 13, 23]), [4, 2]);
        expectClose(await readStorageAsync(gather(wide.permute(1, 0), table, 0)._storage), [11, 20, 10, 21, 12, 21]);

        const out = backend!.alloc(4);
        const indexStorage = backend!.alloc(4);
        const valueStorage = backend!.alloc(6);
        backend!.copyFromHost(indexStorage, new Float32Array([1, 0, 0, 1]));
        backend!.copyFromHost(valueStorage, new Float32Array([10, 11, 12, 20, 21, 22]));
        backend!.gather(out, [2, 2], [1, 2], indexStorage, [2, 2], [2, 1], valueStorage, [2, 3], [3, 1], 1);
        expectClose(await backend!.copyToHostAsync(out), [11, 20, 10, 21]);
        backend!.release(out);
        backend!.release(indexStorage);
        backend!.release(valueStorage);
    });

    test('scatter-adds duplicate gradient entries', async () => {
        const indices = new TensorData(new Float32Array([1, 0, 0, 1]), [2, 2]);
        const grad = new TensorData(new Float32Array([1, 2, 3, 4]), [2, 2]);
        const out = TensorData.fill([2, 3], 0);
        scatter(out, indices, grad, 1);
        expectClose(await readStorageAsync(out._storage), [2, 1, 0, 3, 4, 0]);

        const flipped = TensorData.fill([2, 3], 0);
        scatter(flipped, indices, grad.permute(1, 0), 1);
        expectClose(await readStorageAsync(flipped._storage), [3, 1, 0, 2, 4, 0]);

        const repeated = TensorData.fill([1, 3], 100);
        scatter(repeated, new TensorData(new Float32Array([0, 0]), [1, 2]), new TensorData(new Float32Array([5, 7]), [1, 2]), 1);
        expectClose(await readStorageAsync(repeated._storage), [112, 100, 100]);
    });

    test('reuses pooled buffers without violating queue order', async () => {
        const first = backend!.alloc(4, 7);
        const firstBuffer = first.buffer;
        backend!.release(first);
        const second = backend!.alloc(4, 0);
        expect(second.buffer).toBe(firstBuffer);
        expectClose(await backend!.copyToHostAsync(second), [0, 0, 0, 0]);
        backend!.release(second);
    });

    test('readback cleanup does not destroy parameters for later submits', async () => {
        const input = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4]), [4]));
        const first = input.add(1);
        const readback = first.toFloat32();
        let later = input;
        for (let i = 0; i < 20; i++) later = later.mul(1.01).add(0.25);
        expectClose(await readback, [2, 3, 4, 5]);
        const values = await later.toFloat32();
        expect(values.every(Number.isFinite)).toBe(true);
    });
});
