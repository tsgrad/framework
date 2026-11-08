import { Tensor, tensorScope } from "../src/tensor.js";
import { readStorageAsync, Shape, TensorData, writeStorage } from "../src/tensor_data.js"
import { initBackend } from "../src/backend/index.js";
import { Optimizer, SGDMomentum } from "../src/optim.js";
import { Conv2d, Flatten, Linear, LogSoftmax, MaxPool2d, ReLU } from "../src/layers.js";
import { Sequential } from "../src/model.js";
import { destroyPool } from "../src/fast_ops.js";

const BATCH = 16;

// Number of classes (10 digits)
const C = 10;

//Size of images (height and width)
const H = 28, W = 28;

function sharedTensorData(values: ArrayLike<number>, shape: Shape): TensorData {
    const data = TensorData.fill(shape);
    writeStorage(data._storage, values);
    return data;
}

export class MNistNetwork extends Sequential {
    constructor() {
        super(
            new Conv2d(1, 4, 3, 3),
            new ReLU(),
            new Conv2d(4, 8, 3, 3),
            new ReLU(),
            new MaxPool2d([4, 4]),
            new Flatten(),
            new Linear(392, 64),
            new ReLU(),
            new Linear(64, C),
            new LogSoftmax(1),
        );
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
    model: MNistNetwork;
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

    private initializeParameterGradients(): void {
        for (const parameter of this.model.parameters()) {
            const value = parameter.value;
            if (value instanceof Tensor && value.grad === undefined)
                value.grad = Tensor.zeros(value.shape());
        }
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

    private async evaluate(data: ImageData): Promise<number> {
        const [images, labels] = data;
        let correct = 0;
        this.model.eval();
        for (let start = 0; start < images.length; start += BATCH) {
            const imageBatch = images.slice(start, start + BATCH);
            const labelBatch = labels.slice(start, start + BATCH);
            const batchSize = imageBatch.length;
            const output = this.model.forward(this.imagesToTensor(imageBatch)).detach();

            const scores = await readStorageAsync(output.data._storage);
            for (let i = 0; i < batchSize; i++) {
                let predicted = 0;
                for (let j = 1; j < C; j++) {
                    if (scores[i * C + j] > scores[i * C + predicted]){
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

    async train(dataTrain: ImageData, dataVal: ImageData, learningRate: number, maxEpochs: number = 500, logFn: Function = defaultLogFn): Promise<void>{
        const [trainImages, trainLabels] = dataTrain;
        if (trainImages.length !== trainLabels.length) throw "Training image and label counts differ";
        
        this.model = new MNistNetwork();
        this.optim = new SGDMomentum(this.model.parameters(), learningRate);
        // Gradients persist between batches, so create them outside the batch scope.
        this.initializeParameterGradients();
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

                const lossValue = await tensorScope(() => {
                    const x = this.imagesToTensor(imageBatch);
                    const y = this.labelsToTensor(labelBatch);

                    this.optim.zeroGrad();

                    const out = this.model.forward(x);
                    const loss = out.mul(y).sum(1).sum(0).div(batchSize).neg();

                    loss.backward();
                    this.optim.step();

                    return loss.itemAsync();
                });
                totalLoss += lossValue;
                losses.push(lossValue);
            }

            if (epoch % 1 === 0 || epoch == maxEpochs){
                const correct = await this.evaluate(dataVal);
                logFn(epoch, totalLoss, correct, dataVal[0].length);
            }
        }

        const endtime = performance.now();
        console.log(`Time to train took ${endtime - starttime}`);
    }
}

// Actual testing down here
import { fileURLToPath } from "node:url";
import { loadMnist } from "./mnist_data.js";
await initBackend({ backend: "auto" });
const dataDirectory = fileURLToPath(new URL("./data/mnist/", import.meta.url));
const dataTrain = await loadMnist(dataDirectory, "train", 0, 500);
const dataVal = await loadMnist(dataDirectory, "train", 10000, 500);

const trainer = new ImageTrain();
try{
    await trainer.train(dataTrain, dataVal, 0.01, 10);
}
finally{
    destroyPool();
}
