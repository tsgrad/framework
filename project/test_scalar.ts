import { datasets } from "../src/datasets";
import { Sigmoid } from "../src/scalar_functions";
import { Network, Linear, NetworkLayer, ScalarTrain, SimpleNetwork } from "./run_scalar";

export class Network1 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "sigmoid", 2, 1);
    }
}

export class Network2 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "leakyrelu", 2, 3);
        this.addLayer(Linear, "leakyrelu", 3, 3);
        this.addLayer(Linear, "sigmoid", 3, 1);
    }
}

export class Network3 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "relu", 2, 3);
        this.addLayer(Linear, "sigmoid", 3, 1);
    }
}

export class Network6 extends Network{
    constructor(){
        super();
        this.addLayer(Linear, "leakyrelu", 2, 10);
        this.addLayer(Linear, "leakyrelu", 10, 10);
        this.addLayer(Linear, "leakyrelu", 10, 10);
        this.addLayer(Linear, "leakyrelu", 10, 10);
        this.addLayer(Linear, "sigmoid", 10, 1);
    }
}

// Test 1
export function test1(): void{
    console.log("Test 1");
    let PTS = 50;
    let RATE = 0.5;
    let data = datasets.Simple(PTS);

    let scalarTrain = new ScalarTrain();
    scalarTrain.model = new Network1();

    new ScalarTrain().train(data, RATE);
}

// Test 2
export function test2(): void{
    console.log("Test 2");
    let PTS = 50;
    let RATE = 10;
    let data = datasets.Diag(PTS);

    let scalarTrain = new ScalarTrain();
    scalarTrain.model = new Network2();
    new ScalarTrain().train(data, RATE, 500);
}

export function test3(): void{
    console.log("Test 3");
    let PTS = 50;
    let RATE = 0.5;
    let data = datasets.Split(PTS);

    let scalarTrain = new ScalarTrain();
    scalarTrain.model = new Network3();
    new ScalarTrain().train(data, RATE, 5000);
}

export function test6(): void{
    console.log("Test 6");
    let PTS = 1000;
    let RATE = 0.5;
    let data = datasets.Diag(PTS);

    let scalarTrain = new ScalarTrain();
    scalarTrain.model = new Network6();
    new ScalarTrain().train(data, RATE, 10000);
}

test6();