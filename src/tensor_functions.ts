import { Tensor } from "./tensor";
import { Shape, shapeBroadcast, TensorData } from "./tensor_data";
import { Context } from "./autodiff";
import * as operators from "./operators";
import { tensorMap, tensorZip, tensorReduce } from "./tensor_ops";

export function neg(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.neg)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function inv(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.inv)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function sigmoid(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.sigmoid)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function relu(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.relu)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function log(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.log)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function exp(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.exp)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function id(a: TensorData): TensorData{
    const out = TensorData.fill(a.shape, 0);
    tensorMap(operators.id)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function add(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.fill(outShape, 0);
    tensorZip(operators.add)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function mul(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.fill(outShape, 0);
    tensorZip(operators.mul)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}


export function lt(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.fill(outShape, 0);
    tensorZip(operators.lt)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function eq(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.fill(outShape, 0);
    tensorZip(operators.eq)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function isclose(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.fill(outShape, 0);
    tensorZip(operators.isClose)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

// for reducing
export function sum(a: TensorData, dim: number): TensorData{
    const outShape = [...a.shape];
    if (dim < a.shape.length && dim >= 0)
        outShape[dim] = 1;

    const out = TensorData.fill(outShape, 0);
    tensorReduce(operators.add, 0)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, dim);
    return out;
}

export function prod(a: TensorData, dim: number): TensorData{
    const outShape = [...a.shape];
    if (dim < a.shape.length && dim >= 0)
        outShape[dim] = 1;

    const out = TensorData.fill(outShape, 0);
    tensorReduce(operators.mul, 1)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, dim);
    return out;
}

export function view(a: TensorData, shape: Shape): TensorData{
    const stridesShouldBe = TensorData.calculateStrides(a._shape);
    if (stridesShouldBe.every((val, idx) => {val === a._shape[idx]}) === false)
        throw "view received non-contiguous Tensor, received: " + a;

    const shapeSize = operators.prod(shape);
    if (operators.prod(shape) !== operators.prod(a._shape))
        throw "view received different sizes, received: " + a + " and shape: " + shape;

    return new TensorData(a._storage, shape, stridesShouldBe);
}

export function tocontiguous(a: TensorData): TensorData{
    const stridesShouldBe = TensorData.calculateStrides(a._shape);
    if (stridesShouldBe.every((val, idx) => {val === a._shape[idx]}) === true)
        return a; // already contiguous
    return id(a);
}

export abstract class TensorFunction{
    static forward(ctx: Context, ...t: Tensor[]): Tensor{
        throw new Error("Subclass must create forward");
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        throw new Error("Subclass must create backward");
    }
}

export function unbroadcast(a: Tensor, original: Shape): Tensor{
    let res = a;
    while (res.dims() > original.length){ // first make shape length match (same num of dimensions)
        res = res.sum(0); // shrink from the left, e.g., (shapes) og = [2, 1] res = [3, 2, 1] should cut off to res = [1, 2, 1]
        res = res.view(...res.shape().slice(1)); // then res = [2, 1]
    }

    for (let i = 0; i < original.length; i++){ // second reduce across each dimension as needed
        if (original[i] === 1 && res.shape()[i] > 1)
            res = res.sum(i);
    }

    return res;
}

export class Neg extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        return new Tensor(neg(a._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput.neg()];
    }
}

// I don't like it, and it's kinda messy using Tensor operations inside the Operation classes but it works...
export class Inv extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(inv(a._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.neg().div(a.mul(a))];
    }
}

export class Add extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(add(a._tensor, b._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput, a.shape()), unbroadcast(gradOutput, b.shape())];
    }
}

export class Mul extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(mul(a._tensor, b._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput.mul(b), a.shape()), unbroadcast(gradOutput.mul(a), b.shape())];
    }
}

export class Sigmoid extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = new Tensor(sigmoid(a._tensor));
        ctx.saveForBackward(res);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [res] = ctx.savedValues;
        return [gradOutput.mul(res).mul(res.neg().add(1))];
    }
}

export class ReLU extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(relu(a._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        let relumask = new Tensor(lt(TensorData.fill(a._tensor.shape, 0), a._tensor)); // relumask[i] = 1 if a[i] > zeros[i] else 0
        return [gradOutput.mul(relumask)];
    }
}

export class Log extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(log(a._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.div(a)];
    }
}

export class Exp extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = new Tensor(exp(a._tensor));
        ctx.saveForBackward(res);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [res] = ctx.savedValues;
        return [gradOutput.mul(res)];
    }
}

export class LT extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(lt(a._tensor, b._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class EQ extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(eq(a._tensor, b._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class IsClose extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(isclose(a._tensor, b._tensor));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class ToContiguous extends TensorFunction {
    static forward(ctx: Context, a: Tensor): Tensor {
        return new Tensor(tocontiguous(a._tensor));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
        return [gradOutput];
    }
}

export function Sum(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(sum(a._tensor, dim));
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            let [a] = ctx.savedValues;
            return [gradOutput.mul(Tensor.ones(a))];
        }
    };
}

export function Permute(order: number[]): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            return new Tensor(a._tensor.permute(...order));
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const inverseOrder = new Array(order.length);

            for (let i = 0; i < order.length; i++)
                inverseOrder[order[i]!] = i;

            return [new Tensor(gradOutput._tensor.permute(...inverseOrder))];
        }
    };
}

export function View(newShape: Shape): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(view(a._tensor, newShape));
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            let [a] = ctx.savedValues;
            return [gradOutput.tocontiguous().view(...a.shape())];
        }
    };
}