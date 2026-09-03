import { Module, Parameter } from "../src/module";
import { randomFloat } from "../src/helperfunctions";
import { Scalar } from "../src/scalar";
import { Graph, datasets } from "../src/datasets";
import { SGD } from "../src/optim";


type ActivationFunction = "relu" | "leakyrelu" | "sigmoid";

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

    forward(x: Scalar[]): Scalar[]{
        let h = x;
        for (let i = 0; i < this.layers.length; i++){
            h = this.layers[i].layer.forward(h);
            h = h.map(value => { return value[this.layers[i].activationFunction](); } );
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

    abstract forward(x: Scalar[]): Scalar[];
}

export class Linear extends Layer{
    weights: Parameter[][];
    bias: Parameter[];
    constructor(inSize: number, outSize: number){
        super(inSize, outSize);
        this.weights = [];
        this.bias = [];
        for (let i = 0; i < inSize; i++){
            this.weights.push([]);
            for (let j = 0; j < outSize; j++){
                this.weights[i].push(this.addParameter(`weight_${i}_${j}`, new Scalar(randomFloat(-1, 1))));
            }
        }
        for (let j = 0; j < outSize; j++)
            this.bias.push(this.addParameter(`bias_${j}`, new Scalar(randomFloat(-1, 1))));
    }

    forward(inputs: Scalar[]): Scalar[]{
        let y: Scalar[] = [];
        for (let i = 0; i < this.bias.length; i++)
            y.push(this.bias[i].value);

        for (let i = 0; i < inputs.length; i++){
            for (let j = 0; j < y.length; j++)
                y[j] = y[j].add(inputs[i].mul(this.weights[i][j].value));
        }
        return y;
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

export class ScalarTrain{
    learningRate: number = 0;
    maxEpochs: number = 0;
    model: Network;

    constructor(){
        this.model = new SimpleNetwork();
    }

    runOne(x: number[]): Scalar[]{
        return this.model.forward([new Scalar(x[0], undefined, "x_1"), new Scalar(x[1], undefined, "x_2")]);
    }

    train(data: Graph, learningRate: number, maxEpochs: number = 500, logFn: Function = defaultLogFn): void{
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        let optim = new SGD(this.model.parameters(), learningRate);
    
        let losses: number[] = [];
        for (let epoch = 1; epoch < maxEpochs + 1; epoch++){
            let totalLoss = 0;
            let correct = 0;
            optim.zeroGrad();

            let loss: Scalar;
            for (let i = 0; i < data.n; i++){
                let [x1, x2] = data.x[i];
                let y: number = data.y[i];

                let sX1 = new Scalar(x1), sX2 = new Scalar(x2);
                let out: Scalar = this.model.forward([sX1, sX2])[0];

                let prob: Scalar;
                if (y === 1){
                    prob = out;
                    correct += out.data > 0.5 ? 1 : 0;
                }
                else{
                    prob = out.neg().add(1);
                    correct += out.data < 0.5 ? 1 : 0;
                }

                loss = prob.log().neg();
                totalLoss += loss.data;

                loss.div(data.n).backward();
            }

            if (data.n != 0)
                losses.push(totalLoss);

            optim.step();

            if (epoch % 5 === 0 || epoch == maxEpochs)
                logFn(epoch, totalLoss, correct, losses);
        }
    }
}
