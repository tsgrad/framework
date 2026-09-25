import { datasets } from "../src/datasets";
import { Linear, Network, TensorTrain } from "./run_tensor";


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

// Test 1
export function test1(): void{
    console.log("Test 1");
    let PTS = 50;
    let RATE = 0.5;
    let data = datasets.Simple(PTS);

    let tensorTrain = new TensorTrain();
    tensorTrain.model = new Network1();
    tensorTrain.train(data, RATE);
}

export function test6(): void{
    console.log("Test 6");
    let PTS = 1000;
    let RATE = 0.5;
    let data = datasets.Diag(PTS);

    let tensorTrain = new TensorTrain();
    tensorTrain.model = new Network6();
    tensorTrain.train(data, RATE, 10000);
}

test6();