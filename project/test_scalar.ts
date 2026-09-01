import { datasets } from "../src/datasets";
import { ScalarTrain } from "./run_scalar";

let PTS = 50;
let HIDDEN = 2;
let RATE = 0.5;
let data = datasets.Simple(PTS);

new ScalarTrain(HIDDEN).train(data, RATE);