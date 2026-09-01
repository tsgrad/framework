import { Parameter } from "./module";
import { Scalar } from "./scalar";

export abstract class Optimizer{
    parameters: Parameter[];
    constructor(parameters: Parameter[]){
        this.parameters = parameters;
    }
}

export class SGD extends Optimizer{
    lr: number; // learning rate
    constructor(parameters: Parameter[], lr: number = 1.0){
        super(parameters);
        this.lr = lr;
    }

    zeroGrad(): void{
        for (const p of this.parameters){
            if ("derivative" in p.value){
                if (p.value.derivative !== undefined)
                    p.value.derivative = undefined;
            }
        }
    }

    step(): void{
        for (const p of this.parameters){
            if (p.value.derivative !== undefined)
                p.update(new Scalar(p.value.data - this.lr * p.value.derivative));
        }
    }
}
