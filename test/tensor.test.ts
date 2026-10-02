import { describe, test, expect } from "vitest"
import { Tensor } from "../src/tensor.js"
import { TensorData } from "../src/tensor_data";

test("Matmul test", () => {
    const A = new Tensor(new TensorData(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
    const B = new Tensor(new TensorData(new Float32Array([7, 8, 9, 10, 11, 12]), [3, 2]));

    const C = A.matmul(B);
    const expected = new Float32Array([58, 64, 139, 154]);
    expect(C.data._storage.length).toEqual(expected.length);
    expected.forEach((val, i) => {expect(val).toBeCloseTo(C.data._storage[i])});
});