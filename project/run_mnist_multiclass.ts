import { randomFloat } from "../src/helperfunctions.js"
import { Module, Parameter } from "../src/module.js"
import { Tensor } from "../src/tensor.js";
import { Shape, TensorData } from "../src/tensor_data.js"
import * as operators from "../src/operators.js"
import { Graph } from "../src/datasets.js";
import { Optimizer, SGD, SGDMomentum } from "../src/optim.js";
import { dropout, logsoftmax, maxpool2d } from "../src/tensor_functions.js";

const BATCH = 16;

// Number of classes (10 digits)
const C = 10;

//Size of images (height and width)
const H = 28, W = 28;

function sharedTensorData(values: ArrayLike<number>, shape: Shape): TensorData {
    const data = TensorData.fill(shape);
    data._storage.set(values);
    return data;
}

function RParam(...shape: Shape): Tensor{
    // Random tensor where each cell is -0.05 to 0.05 with shape as the shape
    const data = TensorData.fill(shape);
    for (let i = 0; i < data._storage.length; i++) {
        data._storage[i] = randomFloat(-0.05, 0.05);
    }
    const tensor = new Tensor(data);
    tensor._requiresGrad(true);
    return tensor;
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

    addLayer(type: new (inSize: number, outSize: number, ...args: number[]) => Layer, activationFunction: ActivationFunction, inSize: number, outSize: number, ...args: any[]){
        let layer = new type(inSize, outSize, ...args);
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
        return inputs.matmul(this.weights.value as Tensor).add(this.bias.value as Tensor);

        /* we have matmul now so this is no longer needed
        // we want output = [batchsize, outsize]
        return (this.weights.value as Tensor) // first need to reshape weights and inputs so they can be multiplied together
            .view(1, this.inSize, this.outSize) // now [insize, outsize] is [1, insize, outsize]
            .mul(
                inputs.view(...inputs.shape(), 1) // now inputs is [batchsize, insize, 1]
            ) // so we are multiplying [1, insize, outsize] * [batchsize, insize, 1] = [batchsize, insize, outsize]
            .sum(1) // sum across insize so we have [batchsize, 1, outsize]
            .view(inputs.shape()[0], this.outSize) // actually reduce the dimensions by using view so now we have [batchsize, outsize]
            .add( // second we need to add bias [outsize]
                this.bias.value as Tensor
            ); // we can do [batchsize, outsize] + [outsize] since add will broadcast [outsize] to [batchsize, outsize]
        */
    }
}

export class Conv2d extends Layer{
    weights: Parameter;
    bias: Parameter;
    constructor(inChannels: number, outChannels: number, kh: number, kw: number){
        super(inChannels, outChannels);
        this.weights = this.addParameter(`weights`, RParam(outChannels, inChannels, kh, kw));
        this.bias = this.addParameter(`bias`, RParam(outChannels, 1, 1));
    }

    forward(inputs: Tensor): Tensor{
        return inputs.conv2d(this.weights.value as Tensor).add(this.bias.value as Tensor);
    }
}

export class MNistNetwork extends Network {
    conv1: Conv2d;
    conv2: Conv2d;
    linear1: Linear;
    linear2: Linear;
    mid?: Tensor;
    out?: Tensor;
    constructor() {
        super();
        this.conv1 = new Conv2d(1, 4, 3, 3);
        this.conv2 = new Conv2d(4, 8, 3, 3);
        this.linear1 = new Linear(392, 64);
        this.linear2 = new Linear(64, C);
    }
    forward(x: Tensor): Tensor {
        this.mid = this.conv1.forward(x).relu();
        this.out = this.conv2.forward(this.mid).relu();
        const pooled = maxpool2d(this.out, [4, 4]);
        const batch = pooled.shape()[0];
        // 8 channels × 7 × 7 = 392
        let h = pooled.view(batch, 392);
        h = this.linear1.forward(h).relu();
        h = dropout(h, 0.25, !this.training);
        h = this.linear2.forward(h);
        return logsoftmax(h, 1);
    }
}

function defaultLogFn(epoch: number, totalLoss: number, correct: number, total: number): void{
    console.log(`Epoch ${epoch} loss ${totalLoss} correct ${correct}/${total}`);
}

export type Image = number[][]; //[28][28]
export type OneHotLabel = number[]; //[10]
export type ImageData = [Image[], OneHotLabel[]];

export type ImageLogFn = (
    epoch: number,
    totalLoss: number,
    correct: number,
    total: number,
    losses: number[],
    model: MNistNetwork,
) => void;

export class ImageTrain{
    learningRate: number = 0;
    maxEpochs: number = 0;
    model: Network;
    optim: Optimizer;

    constructor(){
        this.model = new MNistNetwork();
        this.optim = new SGDMomentum(this.model.parameters(), this.learningRate);
    }

    private imagesToTensor(images: Image[]): Tensor{
        return new Tensor(sharedTensorData(images.flat(2), [images.length, 1, H, W]));
    }

    private labelsToTensor(labels: OneHotLabel[]): Tensor {
        return new Tensor(sharedTensorData(labels.flat(), [labels.length, C]));
    }

    runOne(x: Image): Tensor{
        // evaluates one image without effecting model training
        const wasTraining = this.model.training;
        this.model.eval();

        const res: Tensor = this.model.forward(this.imagesToTensor([x])).detach();
        if (wasTraining) this.model.train();

        return res;
    }

    runMany(x: Image[]): Tensor{
        // evaluates one or more images without effecting model training
        const wasTraining = this.model.training;
        this.model.eval();

        const res: Tensor = this.model.forward(this.imagesToTensor(x)).detach();
        if (wasTraining) this.model.train();

        return res;
    }

    private evaluate(data: ImageData): number {
        const [images, labels] = data;
        let correct = 0;
        this.model.eval();
        for (let start = 0; start < images.length; start += BATCH) {
            const imageBatch = images.slice(start, start + BATCH);
            const labelBatch = labels.slice(start, start + BATCH);
            const batchSize = imageBatch.length;
            const output = this.model.forward(this.imagesToTensor(imageBatch)).detach();

            for (let i = 0; i < batchSize; i++) {
                let predicted = 0;
                for (let j = 1; j < C; j++) {
                    if (output.data._storage[i * C + j] > output.data._storage[i * C + predicted]){
                        predicted = j;
                    }
                }
                if (labelBatch[i][predicted] === 1) {
                    correct++;
                }
            }
        }
        return correct;
    }

    train(dataTrain: ImageData, dataVal: ImageData, learningRate: number, maxEpochs: number = 500, logFn: Function = defaultLogFn): void{
        const [trainImages, trainLabels] = dataTrain;
        if (trainImages.length !== trainLabels.length) throw "Training image and label counts differ";
        
        this.model = new MNistNetwork();
        this.optim = new SGDMomentum(this.model.parameters(), learningRate);
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        
        const losses: number[] = [];

        const starttime = performance.now();

        for (let epoch = 1; epoch < maxEpochs + 1; epoch++){
            let totalLoss = 0;
            this.model.train();
            for (let start = 0; start < trainImages.length; start += BATCH){
                const imageBatch = trainImages.slice(start, start + BATCH);
                const labelBatch = trainLabels.slice(start, start + BATCH);
                const batchSize = imageBatch.length;

                const x = this.imagesToTensor(imageBatch);
                const y = this.labelsToTensor(labelBatch);

                this.optim.zeroGrad();

                const out = this.model.forward(x);
                const loss = out.mul(y).sum(1).sum(0).div(batchSize).neg();

                loss.backward();
                this.optim.step();

                totalLoss += loss.item();
                losses.push(loss.item());
            }

            if (epoch % 10 === 0 || epoch == maxEpochs){
                const correct = this.evaluate(dataVal);
                logFn(epoch, totalLoss, correct, losses);
            }
        }

        const endtime = performance.now();
        console.log(`Time to train took ${endtime - starttime}`);
    }
}