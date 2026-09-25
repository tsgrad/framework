import { randomFloat } from "../src/helperfunctions"
import { Module, Parameter } from "../src/module"
import { Tensor } from "../src/tensor";
import { Shape, TensorData } from "../src/tensor_data"
import * as operators from "../src/operators"

function RParam(...shape: Shape): Parameter{
    // Random tensor where each cell is -1 to 1 with shape as the shape
    return new Parameter(new Tensor(new TensorData(Array.from({ length: operators.prod(shape) }, () => randomFloat(-1, 1)), shape)));
}

type ActivationFunction = "relu" | "sigmoid";

export interface NetworkLayer{
    layer: Layer;
    activationFunction: ActivationFunction
}

export abstract class Network extends Module{
    layers: NetworkLayer[];

    constructor(){
        super();
        this.layers = [];
    }

    forward(x: Tensor): Tensor{
        let h = x;
        for (let i = 0; i < this.layers.length; i++){
            h = this.layers[i].layer.forward(h);
            h = h[this.layers[i].activationFunction]();
        }
        return h;
    }

    addLayer(type: new (inSize: number, outSize: number) => Layer, activationFunction: ActivationFunction, inSize: number, outSize: number){
        let layer = new type(inSize, outSize);
        this.addModule(`layer_${this.layers.length}`, layer);
        this.layers.push({layer: layer, activationFunction: activationFunction});
    }
}

export abstract class Layer extends Module{
    inSize: number;
    outSize: number;
    constructor(inSize: number, outSize: number){
        super();
        this.inSize = inSize;
        this.outSize = outSize;
    }

    abstract forward(x: Tensor): Tensor;
}

export class Linear extends Layer{
    weights: Parameter;
    bias: Parameter;
    constructor(inSize: number, outSize: number){
        super(inSize, outSize);
        this.weights = this.addParameter(`weights`, RParam(inSize, outSize));
        this.bias = this.addParameter(`bias`, RParam(outSize));
    }

    forward(inputs: Tensor): Tensor{
        return inputs.mul(this.weights.value as Tensor).sum(0).add(this.bias.value as Tensor);
    }
}

export class SimpleNetwork extends Network {
    constructor() {
        super();
        //this.addLayer(Linear, "relu", 2, 2);
        this.addLayer(Linear, "sigmoid", 2, 1);
    }
}

function defaultLogFn(epoch: number, totalLoss: number, correct: number, losses: number): void{
    console.log(`Epoch ${epoch} loss ${totalLoss} correct ${correct}`);
}
