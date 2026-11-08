import { expect, test } from "vitest";
import { initBackend } from "../src/backend/index.js";
import { Conv2d, Flatten, Linear, LogSoftmax, MaxPool2d, ReLU } from "../src/layers.js";
import { Layer, Sequential } from "../src/model.js";
import { loadNativeAddon } from "../src/native.js";
import { Tensor } from "../src/tensor.js";
import { TensorData } from "../src/tensor_data.js";

const nativeAddon = await loadNativeAddon();
await initBackend({ backend: nativeAddon ? "native" : "cpu" });

class RecordingLayer extends Layer {
    constructor(
        private readonly id: string,
        private readonly calls: string[],
    ) {
        super();
    }

    forward(input: Tensor): Tensor {
        this.calls.push(this.id);
        return input;
    }
}

test("Sequential executes layers in order and registers added modules", () => {
    const calls: string[] = [];
    const first = new RecordingLayer("first", calls);
    const second = new RecordingLayer("second", calls);
    const third = new RecordingLayer("third", calls);
    const model = new Sequential(first, second);

    expect(model.add(third)).toBe(model);
    expect(model.layers).toEqual([first, second, third]);
    expect(model.modules()).toEqual([first, second, third]);

    const input = new Tensor(TensorData.fill([1], 1));
    expect(model.forward(input)).toBe(input);
    expect(calls).toEqual(["first", "second", "third"]);

    model.eval();
    expect(model.layers.every(layer => !layer.training)).toBe(true);
    model.train();
    expect(model.layers.every(layer => layer.training)).toBe(true);
});

test("concrete layers compose into a batch-first image model", async () => {
    const model = new Sequential(
        new Conv2d(1, 2, 3, 3),
        new ReLU(),
        new MaxPool2d([2, 2]),
        new Flatten(),
        new Linear(8, 3),
        new LogSoftmax(1),
    );

    expect(model.parameters()).toHaveLength(4);
    expect(model.namedParameters().map(([name]) => name)).toEqual([
        "0.weights",
        "0.bias",
        "4.weights",
        "4.bias",
    ]);

    const output = model.forward(new Tensor(TensorData.fill([1, 1, 4, 4], 1)));
    expect(output.shape()).toEqual([1, 3]);

    const values = await output.toFloat32();
    expect(values.every(Number.isFinite)).toBe(true);
    expect(Array.from(values, Math.exp).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
});

test("Flatten preserves the leading batch dimension", () => {
    const input = new Tensor(TensorData.fill([2, 3, 4], 1));
    expect(new Flatten().forward(input).shape()).toEqual([2, 12]);
});
