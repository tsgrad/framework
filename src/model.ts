import { Module } from "./module.js";
import { Tensor } from "./tensor.js";

export abstract class Model extends Module {
    abstract forward(input: Tensor): Tensor;
}

export abstract class Layer extends Module {
    abstract forward(input: Tensor): Tensor;
}

export class Sequential extends Model {
    readonly layers: Layer[];

    constructor(...layers: Layer[]) {
        super();
        this.layers = [];
        for (const layer of layers)
            this.add(layer);
    }

    add(layer: Layer): this {
        this.addModule(`${this.layers.length}`, layer);
        this.layers.push(layer);
        return this;
    }

    forward(input: Tensor): Tensor {
        let output = input;
        for (const layer of this.layers)
            output = layer.forward(output);
        return output;
    }
}
