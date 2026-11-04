import { datasets } from "../src/datasets.js";
import { Linear, Network, TensorTrain } from "./run_tensor.js";
import { destroyPool } from "../src/fast_ops.js";
import { initBackend } from "../src/backend/index.js";


export class Network1 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "sigmoid", 2, 1);
    }
}

export class Network2 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "relu", 2, 3);
        this.addLayer(Linear, "relu", 3, 3);
        this.addLayer(Linear, "sigmoid", 3, 1);
    }
}

export class Network6 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "relu", 2, 10);
        this.addLayer(Linear, "relu", 10, 10);
        this.addLayer(Linear, "relu", 10, 10);
        this.addLayer(Linear, "relu", 10, 10);
        this.addLayer(Linear, "sigmoid", 10, 1);
    }
}

export class NetworkMultithreaded extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "relu", 2, 64);
        this.addLayer(Linear, "relu", 64, 64);
        this.addLayer(Linear, "sigmoid", 64, 1);
    }
}

// Test 1
export async function test1(): Promise<void>{
    console.log("Test 1");
    let PTS = 50;
    let RATE = 0.5;
    let data = datasets.Simple(PTS);

    let tensorTrain = new TensorTrain();
    tensorTrain.model = new NetworkMultithreaded();
    await tensorTrain.train(data, RATE, 100);
}

export async function test6(): Promise<void>{
    console.log("Test 6");
    let PTS = 1000;
    let RATE = 0.5;
    let data = datasets.Diag(PTS);

    let tensorTrain = new TensorTrain();
    tensorTrain.model = new Network6();
    await tensorTrain.train(data, RATE, 100);
}

await initBackend({ backend: "auto" });
try {
    await test1();
} finally {
    destroyPool();
}