import { datasets } from "../src/datasets";
import { Network, Linear, NetworkLayer, ScalarTrain } from "./run_scalar";


// Test 1
let PTS = 50;
let HIDDEN = 2;
let RATE = 0.5;
let data = datasets.Simple(PTS);

new ScalarTrain().train(data, RATE);