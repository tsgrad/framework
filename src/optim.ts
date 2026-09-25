import { Variable } from "./autodiff";
import { Queue } from "./datastructures";
import { Parameter } from "./module";
import { Scalar } from "./scalar";
import { Tensor } from "./tensor";

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
                if (p.value.derivative !== null && p.value.derivative !== undefined)
                    p.value.derivative = 0;
            }

            if ("grad" in p.value){
                if (p.value.grad !== null && p.value.grad !== undefined)
                    p.value.grad = undefined;
            }
        }
    }

    step() {
        for (let p of this.parameters) {
            if (p.value instanceof Scalar) {
                const grad = p.value.derivative ?? 0;
                p.value.data -= this.lr * grad;
            }
            else if (p.value instanceof Tensor) {
                if (p.value.grad !== undefined) {
                    for (let i = 0; i < p.value.data._storage.length; i++)
                        p.value.data._storage[i] -= this.lr * p.value.grad.data._storage[i];
                }
            }
        }
    }
}

export class SGDMomentum extends Optimizer{
    lr: number; // learning rate
    beta: number
    prev: Map<Parameter, Variable["data"]>;
    constructor(parameters: Parameter[], lr: number = 1.0, beta: number = 0.9){
        super(parameters);
        this.lr = lr;
        this.beta = beta;
        this.prev = new Map<Parameter, Variable["data"]>();
    }

    zeroGrad(): void{
        for (const p of this.parameters){
            if ("derivative" in p.value){
                if (p.value.derivative !== null && p.value.derivative !== undefined)
                    p.value.derivative = 0;
            }

            if ("grad" in p.value){
                if (p.value.grad !== null && p.value.grad !== undefined)
                    p.value.grad = undefined;
            }
        }
    }

    step(): void{
        for (const p of this.parameters){
            if (p.value instanceof Scalar){
                if (p.value.derivative !== undefined){
                    let oldVelocity = this.prev.get(p);

                    if (oldVelocity === undefined)
                        oldVelocity = 0;

                    let velocity = this.beta * oldVelocity + (1 - this.beta) * p.value.derivative;
                    p.value.data -= this.lr * velocity;

                    this.prev.set(p, velocity);
                }
            }
            else if (p.value instanceof Tensor) {
                if (p.value.grad !== undefined) {
                    let oldVelocity = this.prev.get(p);

                    if (oldVelocity === undefined) 
                        oldVelocity = Array(p.value.data._storage.length).fill(0);

                    for (let i = 0; i < p.value.data._storage.length; i++){
                        const velocity = this.beta * oldVelocity[i] + (1 - this.beta) * p.value.grad.data._storage[i];
                        p.value.data._storage[i] -= this.lr * velocity;
                        oldVelocity[i] = velocity;
                    }
                    
                    this.prev.set(p, oldVelocity);
                }
            }
        }
    }
}
