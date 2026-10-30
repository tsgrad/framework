import { describe, test, expect } from "vitest"
import { Tensor } from "../src/tensor.js"
import { readStorage, TensorData } from "../src/tensor_data";
import { conv1d, conv2d } from "../src/tensor_functions.js";
import { fastConv1d, fastConv2d } from "../src/fast_ops.js";
import { native } from "../src/native.js";

test("Matmul test", () => {
    const A = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
    const B = new Tensor(new TensorData(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]));

    const C = A.matmul(B);
    const expected = new Float32Array([58, 64, 139, 154]);
    const actual = readStorage(C.data._storage);
    expect(actual.length).toEqual(expected.length);
    expected.forEach((val, i) => {expect(val).toBeCloseTo(actual[i])});
});

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>) {
    expect(actual.length).toEqual(expected.length);
    for (let i = 0; i < expected.length; i++)
        expect(actual[i]).toBeCloseTo(expected[i]!);
}

function rand(seed: number, n: number): Float32Array {
    const out = new Float32Array(n);
    let s = seed >>> 0;
    for (let i = 0; i < n; i++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        out[i] = s / 0x100000000 * 2 - 1;
    }
    return out;
}

test("Conv1d test", () => {
    expect(native).not.toBeNull();
    const input = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4]), [1, 1, 4]));
    const weight = new Tensor(new TensorData(new Float32Array([1, 0, -1]), [1, 1, 3]));
    expectClose(readStorage(input.conv1d(weight).data._storage), [-2, -2, 3, 4]);

    const channels = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [1, 2, 3]));
    const kernel = new Tensor(new TensorData(new Float32Array([1, 1, 1, -1]), [1, 2, 2]));
    expectClose(readStorage(channels.conv1d(kernel).data._storage), [2, 4, 9]);
});

test("Conv2d test", () => {
    const input = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4]), [1, 1, 2, 2]));
    const weight = new Tensor(new TensorData(new Float32Array([1, 0, 0, -1]), [1, 1, 2, 2]));
    expectClose(readStorage(input.conv2d(weight).data._storage), [-3, 2, 3, 4]);
});

test("Conv native matches the TypeScript kernels", () => {
    expect(native).not.toBeNull();
    let conv1Calls = 0;
    let conv2Calls = 0;
    const orig1 = native!.conv1d;
    const orig2 = native!.conv2d;
    native!.conv1d = (...args) => { conv1Calls++; return orig1(...args); };
    native!.conv2d = (...args) => { conv2Calls++; return orig2(...args); };

    try {
        const input = new TensorData(rand(1, 1 * 2 * 4), [1, 2, 4]);
        const weight = new TensorData(rand(2, 3 * 2 * 3), [3, 2, 3]);
        const forward = new Tensor(input).conv1d(new Tensor(weight));
        const expected = new Float32Array(forward.data.size);
        fastConv1d(expected, forward.data._shape, forward.data._stride, readStorage(input._storage), input._shape, input._stride, readStorage(weight._storage), weight._shape, weight._stride, false);
        expectClose(readStorage(forward.data._storage), expected);

        const reversed = conv1d(input, weight, true);
        const expectedReverse = new Float32Array(reversed.size);
        fastConv1d(expectedReverse, reversed._shape, reversed._stride, readStorage(input._storage), input._shape, input._stride, readStorage(weight._storage), weight._shape, weight._stride, true);
        expectClose(readStorage(reversed._storage), expectedReverse);

        const narrow = conv1d(input, weight, false, weight.shape[2]);
        const expectedNarrow = new Float32Array(narrow.size);
        fastConv1d(expectedNarrow, narrow._shape, narrow._stride, readStorage(input._storage), input._shape, input._stride, readStorage(weight._storage), weight._shape, weight._stride, false);
        expectClose(readStorage(narrow._storage), expectedNarrow);

        const strided = native!.alloc(4);
        const stridedInput = native!.alloc(4);
        const stridedWeight = native!.alloc(6);
        native!.copyFromHost(stridedInput, new Float32Array([1, 2, 3, 4]));
        native!.copyFromHost(stridedWeight, new Float32Array([1, 99, 0, 99, -1, 99]));
        native!.conv1d(
            strided, [1, 1, 4], [4, 4, 1],
            stridedInput, [1, 1, 4], [4, 4, 1],
            stridedWeight, [1, 1, 3], [6, 6, 2],
            false,
        );
        expectClose(native!.copyToHost(strided), [-2, -2, 3, 4]);

        const bigIn = new TensorData(rand(3, 2 * 4 * 32), [2, 4, 32]);
        const bigW = new TensorData(rand(4, 4 * 4 * 3), [4, 4, 3]);
        const big = conv1d(bigIn, bigW);
        const expectedBig = new Float32Array(big.size);
        fastConv1d(expectedBig, big._shape, big._stride, readStorage(bigIn._storage), bigIn._shape, bigIn._stride, readStorage(bigW._storage), bigW._shape, bigW._stride, false);
        expectClose(readStorage(big._storage), expectedBig);

        const image = new TensorData(rand(5, 2 * 2 * 8 * 8), [2, 2, 8, 8]);
        const filter = new TensorData(rand(6, 2 * 2 * 3 * 3), [2, 2, 3, 3]);
        const conv = new Tensor(image).conv2d(new Tensor(filter));
        const expectedConv = new Float32Array(conv.data.size);
        fastConv2d(expectedConv, conv.data._shape, conv.data._stride, readStorage(image._storage), image._shape, image._stride, readStorage(filter._storage), filter._shape, filter._stride, false);
        expectClose(readStorage(conv.data._storage), expectedConv);

        const flipped = image.permute(1, 0, 2, 3);
        const flippedWeight = filter.permute(1, 0, 2, 3);
        const backspatial = conv2d(flipped, flippedWeight, true);
        const expectedSpatial = new Float32Array(backspatial.size);
        fastConv2d(expectedSpatial, backspatial._shape, backspatial._stride, readStorage(flipped._storage), flipped._shape, flipped._stride, readStorage(flippedWeight._storage), flippedWeight._shape, flippedWeight._stride, true);
        expectClose(readStorage(backspatial._storage), expectedSpatial);

        expect(conv1Calls).toBeGreaterThan(0);
        expect(conv2Calls).toBeGreaterThan(0);
    } finally {
        native!.conv1d = orig1;
        native!.conv2d = orig2;
    }
});

test("Conv backward matches a finite difference", () => {
    const inputVals = rand(7, 1 * 2 * 5);
    const weightVals = rand(8, 2 * 2 * 3);
    const input = new Tensor(new TensorData(inputVals, [1, 2, 5]));
    const weight = new Tensor(new TensorData(weightVals, [2, 2, 3]));
    input._requiresGrad(true);
    weight._requiresGrad(true);
    input.conv1d(weight).backward();

    const loss = (i: Float32Array, w: Float32Array) => {
        const y = new Tensor(new TensorData(i, [1, 2, 5])).conv1d(new Tensor(new TensorData(w, [2, 2, 3])));
        let sum = 0;
        for (const v of readStorage(y.data._storage)) sum += v;
        return sum;
    };
    const eps = 1e-2;
    const inputGrad = readStorage(input.grad!.data._storage);
    for (let i = 0; i < inputVals.length; i++) {
        const up = Float32Array.from(inputVals);
        const down = Float32Array.from(inputVals);
        up[i] += eps;
        down[i] -= eps;
        expect(inputGrad[i]).toBeCloseTo((loss(up, weightVals) - loss(down, weightVals)) / (2 * eps));
    }
    const weightGrad = readStorage(weight.grad!.data._storage);
    for (let i = 0; i < weightVals.length; i++) {
        const up = Float32Array.from(weightVals);
        const down = Float32Array.from(weightVals);
        up[i] += eps;
        down[i] -= eps;
        expect(weightGrad[i]).toBeCloseTo((loss(inputVals, up) - loss(inputVals, down)) / (2 * eps));
    }
});