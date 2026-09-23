import { Context } from "./autodiff";
import { TensorData, UserShape, Shape } from "./tensor_data";
import { Add, TensorFunction, Neg, Mul, Inv, LT, EQ, IsClose, Sigmoid, ReLU, Log, Exp } from "./tensor_functions";

export class History{
    lastFn : typeof TensorFunction | undefined;
    ctx: Context | undefined;
    inputs: Tensor[];

    constructor(lstFn: typeof TensorFunction | undefined = undefined, ct: Context | undefined = undefined, inpts: Tensor[] = []){
        this.lastFn = lstFn;
        this.ctx = ct;
        this.inputs = inpts;
    }
}

var _tensorCount = 0;
export class Tensor{
    // Tensor is a generalization of Scalar in that it is a Variable that
    // handles multidimensional arrays.

    history: History | undefined;
    grad: Tensor | undefined;
    _tensor: TensorData;
    uniqueId: number;
    name: string;

    constructor(tensor: TensorData, history: History | undefined = undefined, name: string | undefined = undefined){
        _tensorCount++;
        this.uniqueId = _tensorCount;

        this._tensor = tensor;
        this.history = history;
        this.grad = undefined;
        if (name !== undefined)
            this.name = name;
        else
            this.name = this.uniqueId.toString();
    }

    _requiresGrad(x: boolean): void{ this.history = new History(); }
    requiresGrad(): boolean { return this.history !== undefined; }

    shape(): UserShape{ return this._tensor._shape; }
    size(): number{ return this._tensor.size; }
    dims(): number{ return this._tensor.dims; }

    detach(): Tensor{
        return new Tensor(this._tensor);
    }

    item(): number{
        if (this._tensor.size !== 1)
            throw "item() called on tensor of size " + this._tensor.size;

        return this._tensor._storage[0];
    }

    _ensureTensor(x: Tensor | number){
        if (x instanceof Tensor)
            return x;
        else
            return new Tensor(new TensorData([x], [1], [1]));
    }

    //functions

    add(b: Tensor | number): Tensor{
        return Tensor.apply(Add, this, this._ensureTensor(b));
    }

    sub(b: Tensor | number): Tensor{
        return Tensor.apply(Add, this, Tensor.apply(Neg, this._ensureTensor(b)));
    }

    mul(b: Tensor | number): Tensor{
        return Tensor.apply(Mul, this, this._ensureTensor(b));
    }

    div(b: Tensor | number): Tensor{
        return Tensor.apply(Mul, this, Tensor.apply(Inv, this._ensureTensor(b)));
    }

    /* Once we add matrix multiplication
    matmul(b: Tensor | number): Tensor{
        return Tensor.apply(MatMul, this, this._ensureTensor(b));
    }*/

    lt(b: Tensor | number): Tensor{
        return Tensor.apply(LT, this, this._ensureTensor(b));
    }

    gt(b: Tensor | number): Tensor{
        return Tensor.apply(LT, this._ensureTensor(b), this);
    }

    eq(b: Tensor | number): Tensor{
        return Tensor.apply(EQ, this, this._ensureTensor(b));
    }

    isClose(b: Tensor | number): Tensor{
        return Tensor.apply(IsClose, this, this._ensureTensor(b));
    }

    neg(): Tensor{
        return Tensor.apply(Neg, this);
    }

    sigmoid(): Tensor{
        return Tensor.apply(Sigmoid, this);
    }

    relu(): Tensor{
        return Tensor.apply(ReLU, this);
    }

    log(): Tensor{
        return Tensor.apply(Log, this);
    }

    exp(): Tensor{
        return Tensor.apply(Exp, this);
    }

    static zeros(shape: Shape): Tensor{
        return new Tensor(TensorData.fill(shape, 0));
    }

    static ones(shape: Shape): Tensor{
        return new Tensor(TensorData.fill(shape, 1));
    }

    static apply(f: typeof TensorFunction, ...vals: Tensor[]): Tensor{
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
        let history: History | undefined = undefined;
        if (needGrad)
            history = new History(f, ctx, vals);

        return new Tensor(c._tensor, history);
    }
}