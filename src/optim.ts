import { Variable } from "./autodiff.js";
import { Queue } from "./datastructures.js";
import { Parameter } from "./module.js";
import { native, type NativeStorage } from "./native.js";
import { Scalar } from "./scalar.js";
import { Tensor } from "./tensor.js";
import { readStorage, TensorBuffer, writeStorage } from "./tensor_data.js";

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
                if (p.value.grad instanceof Tensor)
                    p.value.grad.fill_(0);
                else if (p.value.grad !== null && p.value.grad !== undefined)
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
                    if (native) {
                        native.sgd(
                            p.value.data._storage as NativeStorage, p.value.data._shape, p.value.data._stride,
                            p.value.grad.data._storage as NativeStorage, p.value.grad.data._shape, p.value.grad.data._stride,
                            this.lr,
                        );
                    } else {
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
}

export class SGDMomentum extends Optimizer{
    lr: number; // learning rate
    beta: number
    prev: Map<Parameter, number | TensorBuffer>;
    constructor(parameters: Parameter[], lr: number = 1.0, beta: number = 0.9){
        super(parameters);
        this.lr = lr;
        this.beta = beta;
        this.prev = new Map<Parameter, number | TensorBuffer>();
    }

    zeroGrad(): void{
        for (const p of this.parameters){
            if ("derivative" in p.value){
                if (p.value.derivative !== null && p.value.derivative !== undefined)
                    p.value.derivative = 0;
            }

            if ("grad" in p.value){
                if (p.value.grad instanceof Tensor)
                    p.value.grad.fill_(0);
                else if (p.value.grad !== null && p.value.grad !== undefined)
                    p.value.grad = undefined;
            }
        }
    }

    step(): void{
        for (const p of this.parameters){
            if (p.value instanceof Scalar){
                if (p.value.derivative !== undefined){
                    let oldVelocity = this.prev.get(p);

                    if (typeof oldVelocity !== "number")
                        oldVelocity = 0;

                    let velocity = this.beta * oldVelocity + (1 - this.beta) * p.value.derivative;
                    p.value.data -= this.lr * velocity;

                    this.prev.set(p, velocity);
                }
            }
            else if (p.value instanceof Tensor) {
                if (p.value.grad !== undefined) {
                    let velocity = this.prev.get(p);
                    if (velocity === undefined || typeof velocity === "number") {
                        velocity = native
                            ? native.alloc(p.value.data.size, 0)
                            : new Float32Array(p.value.data.size);
                        this.prev.set(p, velocity);
                    }

                    if (native) {
                        const buffer = velocity as NativeStorage;
                        const shape = p.value.data._shape;
                        const stride = p.value.data._stride;
                        native.sgdMomentum(
                            p.value.data._storage as NativeStorage, shape, stride,
                            p.value.grad.data._storage as NativeStorage, p.value.grad.data._shape, p.value.grad.data._stride,
                            buffer, shape, stride,
                            this.lr, this.beta,
                        );
                    } else {
                        const values = readStorage(p.value.data._storage);
                        const grads = readStorage(p.value.grad.data._storage);
                        const oldVelocity = velocity as Float32Array;
                        for (let i = 0; i < values.length; i++){
                            const next = this.beta * oldVelocity[i] + (1 - this.beta) * grads[i];
                            values[i] -= this.lr * next;
                            oldVelocity[i] = next;
                        }
                        writeStorage(p.value.data._storage, values);
                    }
                }
            }
        }
    }
}
