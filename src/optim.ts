import { Queue } from "./datastructures";
import { Parameter } from "./module";
import { Scalar } from "./scalar";

export abstract class Optimizer{
    parameters: Parameter[];
    constructor(parameters: Parameter[]){
        this.parameters = parameters;
    }

    abstract zeroGrad(): void;
    abstract step(): void;
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

export class SGDMomentum extends Optimizer{
    lr: number; // learning rate
    beta: number
    prev: Map<Parameter, number>;
    constructor(parameters: Parameter[], lr: number = 1.0, beta: number = 0.9){
        super(parameters);
        this.lr = lr;
        this.beta = beta;
        this.prev = new Map<Parameter, number>();
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
            if (p.value.derivative !== undefined){
                let oldVelocity = this.prev.getOrInsert(p, p.value.derivative);
                let velocity = this.beta * oldVelocity + (1 - this.beta) * p.value.derivative;
                p.update(new Scalar(p.value.data - this.lr * velocity));
                this.prev.set(p, velocity);
            }
        }
    }
}
