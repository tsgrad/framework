import { Parameter } from "./module";
import { Scalar } from "./scalar";

export abstract class Optimizer{
    parameters: Parameter[];
    constructor(parameters: Parameter[]){
        this.parameters = parameters;
    }
}