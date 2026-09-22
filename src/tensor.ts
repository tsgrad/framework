import { Context } from "./autodiff";
import { TensorData, UserShape } from "./tensor_data";
import { TensorFunction } from "./tensor_functions";
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

    detach(): Tensor{
        return new Tensor(this._tensor, undefined, undefined, this.backend);
    }

    item(): number{
        if (this._tensor.size !== 1)
            throw "item() called on tensor of size " + this._tensor.size;

        return this._tensor._storage[0];
    }

    static apply(f: typeof TensorFunction, vals: Tensor[]): Tensor{
        let rawVals: Tensor[] = [];
        let needGrad = false;

        for (const val of vals){
            if (val.requiresGrad())
                needGrad = true;
            rawVals.push(val.detach());
        }

        // Create context
        let ctx = new Context(!needGrad);

        // Call forward with the variables
        let c: Tensor = f.forward(ctx, ...rawVals);
        
        // Create new variable from the result with a new history
        let history = new History(f, ctx, vals);
        return new Tensor(c._tensor, history, undefined, c.backend);
    }
}