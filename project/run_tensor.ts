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
        // we want output = [batchsize, outsize]
        return (this.weights.value as Tensor) // first need to reshape weights and inputs so they can be multiplied together
            .view(1, ...(this.weights.value as Tensor).shape()) // now [insize, outsize] is [1, insize, outsize]
            .mul(
                inputs.view(...inputs.shape(), 1) // now inputs is [batchsize, insize, 1]
            ) // so we are multiplying [1, insize, outsize] * [batchsize, insize, 1] = [batchsize, insize, outsize]
            .sum(1) // sum across insize so we have [batchsize, 1, outsize]
            .view(inputs.shape()[0], this.outSize) // actually reduce the dimensions by using view so now we have [batchsize, outsize]
            .add( // second we need to add bias [outsize]
                this.bias.value as Tensor
            ); // we can do [batchsize, outsize] + [outsize] since add will broadcast [outsize] to [batchsize, outsize]
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
