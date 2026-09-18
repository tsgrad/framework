import { Context } from "./autodiff";
import { TensorData, UserShape } from "./tensor_data";
import { TensorBackend } from "./tensor_ops";

export class History{
    lastFn : Function | undefined;
    ctx: Context | undefined;
    inputs: Tensor[];

    constructor(lstFn: Function | undefined = undefined, ct: Context | undefined = undefined, inpts: Tensor[] = []){
        this.lastFn = lstFn;
        this.ctx = ct;
        this.inputs = inpts;
    }
}

var _tensorCount = 0;
export class Tensor{
    // Tensor is a generalization of Scalar in that it is a Variable that
    // handles multidimensional arrays.

    backend: TensorBackend;
    history: History | undefined;
    grad: Tensor | undefined;
    _tensor: TensorData;
    uniqueId: number;
    name: string;
    f: TensorBackend;

    constructor(tensor: TensorData, history: History | undefined = undefined, 
        name: string | undefined = undefined, backend: TensorBackend | undefined = undefined){
        _tensorCount++;
        this.uniqueId = _tensorCount;
        if (backend === undefined)
            throw "Backend must be defined when initializing Tensor";

        this._tensor = tensor;
        this.history = history;
        this.backend = backend;
        this.grad = undefined;
        if (name !== undefined)
            this.name = name;
        else
            this.name = this.uniqueId.toString();

        this.f = backend;
    }

    _requiresGrad(x: boolean): void{ this.history = new History(); }
    requiresGrad(): boolean { return this.history !== undefined; }

    shape(): UserShape{ return this._tensor._shape; }
    size(): number{ return this._tensor.size; }
    dims(): number{ return this._tensor.dims; }
}