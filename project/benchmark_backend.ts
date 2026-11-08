import { initBackend, synchronizeBackend, type BackendPreference } from "../src/backend/index.js";
import { Tensor, tensorScope } from "../src/tensor.js";
import { TensorData } from "../src/tensor_data.js";

const requested = (process.argv[2] ?? "auto") as BackendPreference;
const backend = await initBackend({ backend: requested });
const size = Number(process.argv[3] ?? 1000);
const iterations = Number(process.argv[4] ?? 2000);

const valuesA = new Float32Array(size * size);
const valuesB = new Float32Array(size * size);
for (let i = 0; i < valuesA.length; i++) {
    valuesA[i] = (i % 31) / 31;
    valuesB[i] = (i % 17) / 17;
}

const a = new Tensor(new TensorData(valuesA, [size, size]));
const b = new Tensor(new TensorData(valuesB, [size, size]));

for (let i = 0; i < 5; i++) tensorScope(() => {a.matmul(b).relu();});
await synchronizeBackend();

const started = performance.now();
for (let i = 0; i < iterations; i++) tensorScope(() => {a.matmul(b).relu();});
await synchronizeBackend();
const elapsed = performance.now() - started;

console.log({
    backend: backend.kind,
    matrix: `${size}x${size}`,
    iterations,
    totalMs: elapsed,
    averageMs: elapsed / iterations,
});
