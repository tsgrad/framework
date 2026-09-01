import { Module, Parameter } from "../src/module";
import { randomFloat } from "../src/helperfunctions";
import { Scalar } from "../src/scalar";
import { Graph, datasets } from "../src/datasets";
import { SGD } from "../src/optim";

class Network extends Module{
    layer1: Linear;
    layer2: Linear;
    layer3: Linear;
    constructor(hiddenLayers: number){
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

export class ScalarTrain{
    learningRate: number = 0;
    maxEpochs: number = 0;

    hiddenLayers: number;
    model: Network;


    constructor(hiddenLayers: number){
        this.hiddenLayers = hiddenLayers;
        this.model = new Network(hiddenLayers);
    }

    runOne(x: number[]): Scalar{
        return this.model.forward([new Scalar(x[0], undefined, "x_1"), new Scalar(x[1], undefined, "x_2")]);
    }

    train(data: Graph, learningRate: number, maxEpochs: number = 500, logFn: Function = defaultLogFn): void{
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        this.model = new Network(this.hiddenLayers);
        let optim = new SGD(this.model.parameters(), learningRate);

        let losses: Scalar[] = [];
        for (let epoch = 1; epoch < maxEpochs + 1; epoch++){
            let totalLoss = 0;
            let correct = 0;
            optim.zeroGrad();

            let loss: Scalar;
            for (let i = 0; i < data.n; i++){
                let [x1, x2] = data.x[i];
                let y: number = data.y[i];

                let sX1 = new Scalar(x1), sX2 = new Scalar(x2);
                let out: Scalar = this.model.forward([sX1, sX2]);

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
                losses.push(loss!);

            optim.step();

            if (epoch % 10 === 0 || epoch == maxEpochs)
                logFn(epoch, totalLoss, correct, losses);
        }
    }
}