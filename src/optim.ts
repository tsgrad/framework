import { Variable } from "./autodiff.js";
import { Queue } from "./datastructures.js";
import { Parameter } from "./module.js";
import { Scalar } from "./scalar.js";
import { Tensor } from "./tensor.js";
import { readStorage, writeStorage } from "./tensor_data.js";

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
                    const values = readStorage(p.value.data._storage);
                    const grads = readStorage(p.value.grad.data._storage);
                    for (let i = 0; i < values.length; i++)
                        values[i] -= this.lr * grads[i];
                    writeStorage(p.value.data._storage, values);
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
                    const values = readStorage(p.value.data._storage);
                    const grads = readStorage(p.value.grad.data._storage);
                    let oldVelocity = this.prev.get(p);

                    if (oldVelocity === undefined)
                        oldVelocity = Array(values.length).fill(0);

                    for (let i = 0; i < values.length; i++){
                        const velocity = this.beta * oldVelocity[i] + (1 - this.beta) * grads[i];
                        values[i] -= this.lr * velocity;
                        oldVelocity[i] = velocity;
                    }

                    writeStorage(p.value.data._storage, values);
                    this.prev.set(p, oldVelocity);
                }
            }
        }
    }
}
