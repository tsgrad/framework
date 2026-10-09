import { Tensor } from "./tensor.js";
import { Shape, shapeBroadcast, TensorData } from "./tensor_data.js";
import { Context } from "./autodiff.js";
import * as operators from "./operators.js";
// swap out if wanted, tensor_ops has non-multithreaded, the fastTensor stuff is multithreaded
//import { tensorMap, tensorZip, tensorReduce } from "./tensor_ops.js";
import {fastTensorMap as tensorMap, fastTensorZip as tensorZip, fastTensorReduce as tensorReduce, fastMatMul as matMul, fastConv1d as tensorConv1d} from "./fast_ops.js";

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

export function matmul(a: TensorData, b: TensorData): TensorData{
    if (a._shape.length != 2 || b._shape.length != 2)
        throw "both a and b's shapes must be of length 2 for now";
    let out = TensorData.fill([a._shape[0], b._shape[1]]);
    matMul(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function conv1d(input: TensorData, weight: TensorData, reverse: boolean = false, outWidth?: number): TensorData{
    /*
     """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output
    */
    
    const batch = input._shape[0], inChannels = input._shape[1], w = input._shape[2];
    const outChannel = weight._shape[0], inChannels2 = weight._shape[1], kw = weight._shape[2];

    if (inChannels != inChannels2) throw "in channels must be same between both input shape and weight shape";

    let out = TensorData.fill([batch, outChannel, outWidth ? outWidth : w]);
    tensorConv1d(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, weight._storage, weight._shape, weight._stride, reverse);
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
    if (stridesShouldBe.every((val, idx) => val === a._stride[idx]) === false)
        throw "view received non-contiguous Tensor, received: " + a;

    if (operators.prod(shape) !== operators.prod(a._shape))
        throw "view received different sizes, received: " + a + " and shape: " + shape;

    return new TensorData(a._storage, shape);
}

export function tocontiguous(a: TensorData): TensorData{
    const stridesShouldBe = TensorData.calculateStrides(a._shape);
    if (stridesShouldBe.every((val, idx) => val === a._stride[idx]) === true)
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
        return new Tensor(neg(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput.neg()];
    }
}

// I don't like it, and it's kinda messy using Tensor operations inside the Operation classes but it works...
export class Inv extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(inv(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.neg().div(a.mul(a))];
    }
}

export class Add extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(add(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput, a.shape()), unbroadcast(gradOutput, b.shape())];
    }
}

export class Mul extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(mul(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput.mul(b), a.shape()), unbroadcast(gradOutput.mul(a), b.shape())];
    }
}

export class Sigmoid extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = new Tensor(sigmoid(a.data));
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
        return new Tensor(relu(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        let relumask = new Tensor(lt(TensorData.fill(a.data.shape, 0), a.data)); // relumask[i] = 1 if a[i] > zeros[i] else 0
        return [gradOutput.mul(relumask)];
    }
}

export class Log extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(log(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.div(a)];
    }
}

export class Exp extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = new Tensor(exp(a.data));
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
        return new Tensor(lt(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class EQ extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(eq(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class IsClose extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(isclose(a.data, b.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [Tensor.zeros(a.shape()), Tensor.zeros(b.shape())];
    }
}

export class MatMul extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(matmul(a.data, b.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [gradOutput.matmul(b.permute(1, 0)), a.permute(1, 0).matmul(gradOutput)];
    }
}

export class Conv1d extends TensorFunction{
    static forward(ctx: Context, input: Tensor, weight: Tensor): Tensor{
        ctx.saveForBackward(input, weight);
        return new Tensor(conv1d(input.data, weight.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [input, weight] = ctx.savedValues;
        return [new Tensor(conv1d(gradOutput.data, weight.permute(1, 0, 2).data, true)), new Tensor(conv1d(input.permute(1, 0, 2).data, gradOutput.permute(1, 0, 2).data, false, weight.shape()[2])).permute(1, 0, 2)];
    }
}

export class ToContiguous extends TensorFunction {
    static forward(ctx: Context, a: Tensor): Tensor {
        return new Tensor(tocontiguous(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
        return [gradOutput];
    }
}

export function Sum(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(sum(a.data, dim));
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            let [a] = ctx.savedValues;
            return [gradOutput.mul(Tensor.ones(a.shape()))];
        }
    };
}

export function Permute(order: number[]): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            return new Tensor(a.data.permute(...order));
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const inverseOrder = new Array(order.length);

            for (let i = 0; i < order.length; i++)
                inverseOrder[order[i]!] = i;

            return [new Tensor(gradOutput.data.permute(...inverseOrder))];
        }
    };
}

export function View(newShape: Shape): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            ctx.saveForBackward(a);
            return new Tensor(view(a.data, newShape));
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            let [a] = ctx.savedValues;
            return [gradOutput.tocontiguous().view(...a.shape())];
        }
    };
}