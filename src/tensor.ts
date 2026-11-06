import { backpropagate, Context, inBackward, Variable } from "./autodiff.js";
import { readStorage, readStorageAsync, TensorData, UserShape, Shape } from "./tensor_data.js";
import { getBackend } from "./backend/index.js";
import { tensorMap, tensorZip } from "./native_ops.js";
import * as operators from "./operators.js";
import { Add, TensorFunction, Neg, Mul, Inv, LT, EQ, IsClose, Sigmoid, ReLU, Log, Exp, Sum, Permute, ToContiguous, View, MatMul, Conv1d, Conv2d, Max, Sub, Div } from "./tensor_functions.js";

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
export class Tensor implements Variable{
    // Tensor is a generalization of Scalar in that it is a Variable that
    // handles multidimensional arrays.

    history: History | undefined;
    grad: Tensor | undefined;
    data: TensorData;
    uniqueId: number;
    name: string;

    constructor(tensor: TensorData, history: History | undefined = undefined, name: string | undefined = undefined){
        _tensorCount++;
        this.uniqueId = _tensorCount;

        this.data = tensor;
        this.history = history;
        this.grad = undefined;
        if (name !== undefined)
            this.name = name;
        else
            this.name = this.uniqueId.toString();
    }

    free(): void { this.data.release(); }

    _requiresGrad(x: boolean): void{ this.history = new History(); }
    requiresGrad(): boolean { return this.history !== undefined; }

    shape(): UserShape{ return this.data._shape; }
    size(): number{ return this.data.size; }
    dims(): number{ return this.data.dims; }

    detach(): Tensor{
        return new Tensor(this.data);
    }

    isLeaf(): boolean{
        return this.history !== undefined && this.history.lastFn === undefined;
    }

    isConstant(): boolean{
        return this.history === undefined;
    }

    parents(): Tensor[]{
        return this.history?.inputs ?? [];
    }

    item(): number{
        if (this.data.size !== 1)
            throw "item() called on tensor of size " + this.data.size;

        return readStorage(this.data._storage)[0];
    }

    async itemAsync(): Promise<number> {
        if (this.data.size !== 1)
            throw "itemAsync() called on tensor of size " + this.data.size;
        return (await readStorageAsync(this.data._storage))[0]!;
    }

    toFloat32(): Promise<Float32Array> {
        return readStorageAsync(this.data._storage);
    }

    _ensureTensor(x: Tensor | number){
        if (x instanceof Tensor)
            return x;
        else
            return new Tensor(new TensorData(new Float32Array([x]), [1], [1]));
    }

    //functions

    add(b: Tensor | number): Tensor{
        return Tensor.apply(Add, this, this._ensureTensor(b));
    }

    sub(b: Tensor | number): Tensor{
        return Tensor.apply(Sub, this, this._ensureTensor(b));
    }

    mul(b: Tensor | number): Tensor{
        return Tensor.apply(Mul, this, this._ensureTensor(b));
    }

    div(b: Tensor | number): Tensor{
        return Tensor.apply(Div, this, this._ensureTensor(b));
    }

    private _sameShape(other: Tensor): boolean {
        const a = this.shape();
        const b = other.shape();
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }

    add_(b: Tensor | number): Tensor {
        if (this.requiresGrad()) throw "add_ on a tensor that requires grad";
        const other = this._ensureTensor(b);
        if (!this._sameShape(other)) throw "add_ requires the same shape";
        tensorZip(operators.add)(this.data._storage, this.data._shape, this.data._stride, this.data._storage, this.data._shape, this.data._stride, other.data._storage, other.data._shape, other.data._stride);
        return this;
    }

    mul_(b: Tensor | number): Tensor {
        if (this.requiresGrad()) throw "mul_ on a tensor that requires grad";
        const other = this._ensureTensor(b);
        if (!this._sameShape(other)) throw "mul_ requires the same shape";
        tensorZip(operators.mul)(this.data._storage, this.data._shape, this.data._stride, this.data._storage, this.data._shape, this.data._stride, other.data._storage, other.data._shape, other.data._stride);
        return this;
    }

    relu_(): Tensor {
        if (this.requiresGrad()) throw "relu_ on a tensor that requires grad";
        tensorMap(operators.relu)(this.data._storage, this.data._shape, this.data._stride, this.data._storage, this.data._shape, this.data._stride);
        return this;
    }

    fill_(value: number): Tensor {
        getBackend().fill(this.data._storage, value);
        return this;
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

    matmul(b: Tensor): Tensor{
        return Tensor.apply(MatMul, this, b);
    }

    conv1d(weight: Tensor): Tensor{
        return Tensor.apply(Conv1d, this, weight);
    }

    conv2d(weight: Tensor): Tensor{
        return Tensor.apply(Conv2d, this, weight);
    }

    max(dim: number): Tensor{
        return Tensor.apply(Max(dim), this);
    }

    sum(dim: number): Tensor{
        return Tensor.apply(Sum(dim), this);
    }

    permute(...order: number[]): Tensor{
        return Tensor.apply(Permute(order), this); 
    }

    view(...shape: Shape): Tensor{
        return Tensor.apply(View(shape), this); 
    }

    tocontiguous(): Tensor{
        return Tensor.apply(ToContiguous, this);
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
            if (!inBackward() && val.requiresGrad())
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

        return new Tensor(c.data, history);
    }

    accumulateDerivative(grad: Tensor): void{
        if (this.grad === undefined)
            this.grad = new Tensor(TensorData.fill(this.shape(), 0));
        this.grad.add_(grad);
    }

    chainRule(gradOutput: Tensor): [Tensor, Tensor][]{
        if (this.history === undefined || this.history.lastFn === undefined || this.history.ctx === undefined)
            throw "chainRule called with incomplete history";

        const grads: Tensor[] = this.history.lastFn.backward(this.history.ctx, gradOutput);

        if (grads.length != this.history.inputs.length)
            throw "backward gave grads with dimension size " + grads.length + " but expected " + this.history.inputs.length;

        const res: [Tensor, Tensor][] = [];
        for (let i = 0; i < this.history.inputs.length; i++)
            res.push([this.history.inputs[i], grads[i]]);

        return res;
    }

    backward(gradOutput: Tensor | undefined = undefined): void{
        if (gradOutput === undefined)
            gradOutput = Tensor.ones(this.shape());
        backpropagate(this, gradOutput);
    }
}