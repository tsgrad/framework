import { Module, Parameter } from "../src/module";
import { randomFloat } from "../test/testhelperfunctions";
import { Scalar } from "../src/scalar";

class Network extends Module{
    layer1: Linear;
    layer2: Linear;
    layer3: Linear;
    constructor(hiddenLayers: any){
        super();
        this.layer1 = new Linear(2, hiddenLayers);
        this.layer2 = new Linear(hiddenLayers, hiddenLayers);
        this.layer3 = new Linear(hiddenLayers, 1);
    }

    forward(x: Scalar[]): Scalar{
        let h: Scalar[] = this.layer1.forward(x);
        let middle: Scalar[] = [];
        for (const val of h)
            middle.push(val.relu());
        h = this.layer2.forward(middle);

        let end: Scalar[] = [];
        for (const val of h)
            end.push(val.relu());

        return this.layer3.forward(end)[0].sigmoid();
    }
}

class Linear extends Module{
    weights: Parameter[][];
    bias: Parameter[];
    constructor(inSize: number, outSize: number){
        super();
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

function defaultLogFn(epoch: number, totalLoss: number, correct: number, losses: number): void{
    console.log(`Epoch ${epoch} loss ${totalLoss} correct ${correct}`);
}

