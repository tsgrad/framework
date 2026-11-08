import { randomFloat } from "./helperfunctions.js";
import { Parameter } from "./module.js";
import { Layer } from "./model.js";
import { Tensor } from "./tensor.js";
import { logsoftmax, maxpool2d } from "./tensor_functions.js";
import { Shape, TensorData, writeStorage } from "./tensor_data.js";

function heParameter(fanIn: number, shape: Shape): Tensor {
    const bound = Math.sqrt(6 / fanIn);
    const data = TensorData.fill(shape);
    const values = new Float32Array(data.size);
    for (let i = 0; i < values.length; i++)
        values[i] = randomFloat(-bound, bound);
    writeStorage(data._storage, values);

    const tensor = new Tensor(data);
    tensor._requiresGrad(true);
    return tensor;
}

function zeroParameter(shape: Shape): Tensor {
    const tensor = new Tensor(TensorData.fill(shape, 0));
    tensor._requiresGrad(true);
    return tensor;
}

export class Linear extends Layer {
    readonly inSize: number;
    readonly outSize: number;
    weights: Parameter;
    bias: Parameter;

    constructor(inSize: number, outSize: number) {
        super();
        this.inSize = inSize;
        this.outSize = outSize;
        this.weights = this.addParameter("weights", heParameter(inSize, [inSize, outSize]));
        this.bias = this.addParameter("bias", zeroParameter([outSize]));
    }

    forward(input: Tensor): Tensor {
        return input
            .matmul(this.weights.value as Tensor)
            .add(this.bias.value as Tensor);
    }
}

export class Conv2d extends Layer {
    readonly inChannels: number;
    readonly outChannels: number;
    readonly kernelHeight: number;
    readonly kernelWidth: number;
    weights: Parameter;
    bias: Parameter;

    constructor(inChannels: number, outChannels: number, kernelHeight: number, kernelWidth: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.weights = this.addParameter(
            "weights",
            heParameter(
                inChannels * kernelHeight * kernelWidth,
                [outChannels, inChannels, kernelHeight, kernelWidth],
            ),
        );
        this.bias = this.addParameter("bias", zeroParameter([outChannels, 1, 1]));
    }

    forward(input: Tensor): Tensor {
        return input
            .conv2d(this.weights.value as Tensor)
            .add(this.bias.value as Tensor);
    }
}

export class ReLU extends Layer {
    forward(input: Tensor): Tensor {
        return input.relu();
    }
}

export class MaxPool2d extends Layer {
    readonly kernel: [number, number];

    constructor(kernel: [number, number]) {
        super();
        this.kernel = [...kernel];
    }

    forward(input: Tensor): Tensor {
        return maxpool2d(input, this.kernel);
    }
}

export class Flatten extends Layer {
    forward(input: Tensor): Tensor {
        if (input.dims() < 2)
            throw new Error("Flatten expects a batch dimension");

        const batch = input.shape()[0];
        return input.view(batch, input.size() / batch);
    }
}

export class LogSoftmax extends Layer {
    readonly dim: number;

    constructor(dim: number) {
        super();
        this.dim = dim;
    }

    forward(input: Tensor): Tensor {
        return logsoftmax(input, this.dim);
    }
}
