import { Tensor } from "./tensor.js";
import { Shape, shapeBroadcast, TensorData } from "./tensor_data.js";
import { Context } from "./autodiff.js";
import * as operators from "./operators.js";
// swap out if wanted, tensor_ops has non-multithreaded, the fastTensor stuff is multithreaded
//import { tensorMap, tensorZip, tensorReduce } from "./tensor_ops.js";
//import {/*fastTensorMap as tensorMap, fastTensorZip as tensorZip, fastTensorReduce as tensorReduce, fastMatMul as matMul,*/ fastConv1d as tensorConv1d, fastConv2d as tensorConv2d} from "./fast_ops.js";
import {tensorMap, tensorZip, tensorReduce, matMul, tensorConv1d, tensorConv2d, nativeSoftmax, nativeSoftmaxBackward, nativeLogSoftmax, nativeLogSoftmaxBackward, nativeSoftmaxCrossEntropy, nativeSoftmaxCrossEntropyBackward, nativeReluBackward, nativeSigmoidBackward, nativeExpand, nativeBiasAct, nativeBiasActBackward, nativeLayerNorm, nativeLayerNormBackward, nativeMaxPool2d, nativeMaxPool2dBackward, nativeAvgPool2d, nativeAvgPool2dBackward } from "./native_ops.js"
import { native } from "./native.js";
import { randomFloat } from "./helperfunctions.js";

export function neg(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.neg)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function inv(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.inv)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function sigmoid(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.sigmoid)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function relu(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.relu)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function log(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.log)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function exp(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.exp)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function id(a: TensorData): TensorData{
    const out = TensorData.empty(a.shape);
    tensorMap(operators.id)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride);
    return out;
}

export function max(input: TensorData, dim: number): TensorData{
    const outShape = [...input._shape];
    outShape[dim] = 1;
    const out = TensorData.empty(outShape)
    tensorReduce(operators.max, -Infinity)(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, dim);
    return out;
}

export function add(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.add)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function mul(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.mul)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function lt(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.lt)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function eq(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.eq)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function sub(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.sub)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function div(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.div)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function isclose(a: TensorData, b: TensorData): TensorData{
    const outShape = shapeBroadcast(a._shape, b._shape);
    const out = TensorData.empty(outShape);
    tensorZip(operators.isClose)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, b._storage, b._shape, b._stride);
    return out;
}

export function matmul(a: TensorData, b: TensorData): TensorData{
    if (a._shape.length != 2 || b._shape.length != 2)
        throw "both a and b's shapes must be of length 2 for now";
    let out = TensorData.empty([a._shape[0], b._shape[1]]);
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

    let out = TensorData.empty([batch, outChannel, outWidth ? outWidth : w]);
    tensorConv1d(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, weight._storage, weight._shape, weight._stride, reverse);
    return out;
}

//                                                                                      new h then w
export function conv2d(input: TensorData, weight: TensorData, reverse: boolean = false, ...outDims : number[]): TensorData{
    const batch = input._shape[0], inChannels = input._shape[1], h = outDims[0] ?? input._shape[2], w = outDims[1] ?? input._shape[3];
    const outChannel = weight._shape[0], inChannels2 = weight._shape[1], kh = weight._shape[2], kw = weight._shape[3];

    if (inChannels != inChannels2) throw "in channels must be same between both input shape and weight shape";

    let out = TensorData.empty([batch, outChannel, h, w]);
    tensorConv2d(out._storage, out._shape, out._stride, input._storage, input._shape, input._stride, weight._storage, weight._shape, weight._stride, reverse);
    return out;
}

// for reducing
export function sum(a: TensorData, dim: number): TensorData{
    const outShape = [...a.shape];
    if (dim < a.shape.length && dim >= 0)
        outShape[dim] = 1;

    const out = TensorData.empty(outShape);
    tensorReduce(operators.add, 0)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, dim);
    return out;
}

export function prod(a: TensorData, dim: number): TensorData{
    const outShape = [...a.shape];
    if (dim < a.shape.length && dim >= 0)
        outShape[dim] = 1;

    const out = TensorData.empty(outShape);
    tensorReduce(operators.mul, 1)(out._storage, out._shape, out._stride, a._storage, a._shape, a._stride, dim);
    return out;
}

export function view(a: TensorData, shape: Shape): TensorData{
    const stridesShouldBe = TensorData.calculateStrides(a._shape);
    if (stridesShouldBe.every((val, idx) => val === a._stride[idx]) === false)
        throw "view received non-contiguous Tensor, received: " + a;

    if (operators.prod(shape) !== operators.prod(a._shape))
        throw "view received different sizes, received: " + a + " and shape: " + shape;

    return TensorData.share(a._storage, shape);
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

export function tile2d(input: Tensor, kernel : [number, number]): [Tensor, number, number]{
    // reshapes a Tensor of shape [batch, channel, height, width] to [batch, channel, newHeight, newWidth, kernelHeight * kernelWidth] returning the new tensor as well as the new height and width
    // assumes that the input size is divisible by the kernel size
    // also assumes stride to be equal to kernel size, so no overlapping stuff

    const batch = input.shape()[0], channel = input.shape()[1], height = input.shape()[2], width = input.shape()[3];
    const kh = kernel[0], kw = kernel[1];
    if (height % kh !== 0 || width % kw !== 0) throw "Expected input size in tile2d to be divisible by kernel size";

    const newHeight = height / kh;
    const newWidth = width / kw;

    return [input.view(batch, channel, newHeight, kh, newWidth, kw).permute(0, 1, 2, 4, 3, 5).tocontiguous().view(batch, channel, newHeight, newWidth, kh * kw), newHeight, newWidth];
}

function poolFallback(input: Tensor, kernel: [number, number], average: boolean): Tensor {
    const batch = input.shape()[0], channel = input.shape()[1];
    const kh = kernel[0], kw = kernel[1];
    const [tiled, newHeight, newWidth] = tile2d(input, kernel);
    if (average)
        return tiled.sum(4).div(kh * kw).view(batch, channel, newHeight, newWidth);
    return tiled.max(4).view(batch, channel, newHeight, newWidth);
}

export function AvgPool2d(kh: number, kw: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor): Tensor {
            const batch = input.shape()[0], channel = input.shape()[1], height = input.shape()[2], width = input.shape()[3];
            const out = TensorData.empty([batch, channel, Math.floor(height / kh), Math.floor(width / kw)]);
            nativeAvgPool2d(out._storage, out._shape, out._stride, input.data._storage, input.data._shape, input.data._stride, kh, kw);
            ctx.saveForBackward(input);
            return new Tensor(out);
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [input] = ctx.savedValues as Tensor[];
            const grad = TensorData.empty(input.shape());
            nativeAvgPool2dBackward(grad._storage, grad._shape, grad._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, kh, kw);
            return [new Tensor(grad)];
        }
    };
}

export function MaxPool2d(kh: number, kw: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor): Tensor {
            const batch = input.shape()[0], channel = input.shape()[1], height = input.shape()[2], width = input.shape()[3];
            const out = TensorData.empty([batch, channel, Math.floor(height / kh), Math.floor(width / kw)]);
            nativeMaxPool2d(out._storage, out._shape, out._stride, input.data._storage, input.data._shape, input.data._stride, kh, kw);
            const y = new Tensor(out);
            ctx.saveForBackward(input, y);
            return y;
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [input, y] = ctx.savedValues as Tensor[];
            const grad = TensorData.empty(input.shape());
            nativeMaxPool2dBackward(grad._storage, grad._shape, grad._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, input.data._storage, input.data._shape, input.data._stride, y.data._storage, y.data._shape, y.data._stride, kh, kw);
            return [new Tensor(grad)];
        }
    };
}

export function avgpool2d(input: Tensor, kernel: [number, number]): Tensor{
    if (!native) return poolFallback(input, kernel, true);
    return Tensor.apply(AvgPool2d(kernel[0], kernel[1]), input);
}

export function maxpool2d(input: Tensor, kernel: [number, number]): Tensor{
    if (!native) return poolFallback(input, kernel, false);
    return Tensor.apply(MaxPool2d(kernel[0], kernel[1]), input);
}

export function rparam(shape: Shape, lowerBound: number = -1, upperBound: number = 1): Tensor{
    // Uniform values in [lowerBound, upperBound). The addon writes them in parallel.
    const data = TensorData.empty(shape);
    if (native && !(data._storage instanceof Float32Array)) {
        native.uniform(data._storage, lowerBound, upperBound);
        return new Tensor(data);
    }

    const values = data._storage as Float32Array;
    for (let i = 0; i < values.length; i++)
        values[i] = randomFloat(lowerBound, upperBound);
    return new Tensor(data);
}

export function dropout(input: Tensor, rate: number, ignore: boolean = false): Tensor{
    if (rate < 0.0 || rate > 1.0) throw "rate must be in [0, 1)";

    if (ignore || rate == 0.0) return input;

    const threshold = TensorData.fill(input.shape(), 1 - rate);
    const random = rparam(input.shape(), 0, 1);

    const mask = new Tensor(lt(random.data, threshold));
    return input.mul(mask).div(1.0 - rate);
}

export function argmax(input: Tensor, dim: number): Tensor{
    return new Tensor(max(input.data, dim)).eq(input);
}

export function Softmax(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            const out = TensorData.empty(a.shape());
            nativeSoftmax(out._storage, out._shape, out._stride, a.data._storage, a.data._shape, a.data._stride, dim);
            const y = new Tensor(out);
            ctx.saveForBackward(y);
            return y;
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [y] = ctx.savedValues as Tensor[];
            const grad = TensorData.empty(y.shape());
            nativeSoftmaxBackward(grad._storage, grad._shape, grad._stride, y.data._storage, y.data._shape, y.data._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, dim);
            return [new Tensor(grad)];
        }
    };
}

export function LogSoftmax(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            const out = TensorData.empty(a.shape());
            nativeLogSoftmax(out._storage, out._shape, out._stride, a.data._storage, a.data._shape, a.data._stride, dim);
            const y = new Tensor(out);
            ctx.saveForBackward(y);
            return y;
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [y] = ctx.savedValues as Tensor[];
            const grad = TensorData.empty(y.shape());
            nativeLogSoftmaxBackward(grad._storage, grad._shape, grad._stride, y.data._storage, y.data._shape, y.data._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, dim);
            return [new Tensor(grad)];
        }
    };
}

export function softmax(input: Tensor, dim: number): Tensor{
    if (!native) {
        const exps = input.sub(new Tensor(max(input.data, dim))).exp();
        return exps.div(exps.sum(dim));
    }
    return Tensor.apply(Softmax(dim), input);
}

export function logsoftmax(input: Tensor, dim: number): Tensor{
    if (!native) {
        const shifted = input.sub(new Tensor(max(input.data, dim)));
        return shifted.sub(shifted.exp().sum(dim).log());
    }
    return Tensor.apply(LogSoftmax(dim), input);
}

function crossEntropyFallback(logits: Tensor, target: Tensor, dim: number): Tensor {
    if (dim < 0 || dim >= logits.dims()) throw "softmaxCrossEntropy received dim outside the boundaries of the shape";
    const rows = logits.size() / logits.shape()[dim];
    const logp = logsoftmax(logits, dim);
    let nll = target.mul(logp).sum(dim).neg();
    const dims = nll.dims();
    for (let i = 0; i < dims; i++)
        nll = nll.sum(i);
    return nll.div(rows);
}

export function SoftmaxCrossEntropy(dim: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, logits: Tensor, target: Tensor): Tensor {
            const probs = TensorData.empty(logits.shape());
            const loss = TensorData.empty([1]);
            nativeSoftmaxCrossEntropy(loss._storage, loss._shape, loss._stride, probs._storage, probs._shape, probs._stride, logits.data._storage, logits.data._shape, logits.data._stride, target.data._storage, target.data._shape, target.data._stride, dim);
            ctx.saveForBackward(new Tensor(probs), target);
            return new Tensor(loss);
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [probs, target] = ctx.savedValues as Tensor[];
            const grad = TensorData.empty(probs.shape());
            nativeSoftmaxCrossEntropyBackward(grad._storage, grad._shape, grad._stride, probs.data._storage, probs.data._shape, probs.data._stride, target.data._storage, target.data._shape, target.data._stride, dim);
            return [new Tensor(grad).mul(gradOutput), new Tensor(TensorData.fill(target.shape(), 0))];
        }
    };
}

export function softmaxCrossEntropy(logits: Tensor, target: Tensor, dim: number): Tensor {
    if (!native) return crossEntropyFallback(logits, target, dim);
    return Tensor.apply(SoftmaxCrossEntropy(dim), logits, target);
}

export type BiasActivation = "relu" | "sigmoid";

function actCode(activation: BiasActivation): number {
    return activation === "relu" ? 0 : 1;
}

function sameShape(a: Tensor, b: Tensor): boolean {
    const left = a.shape();
    const right = b.shape();
    if (left.length !== right.length) return false;
    for (let i = 0; i < left.length; i++) {
        if (left[i] !== right[i]) return false;
    }
    return true;
}

export function BiasAct(activation: BiasActivation): typeof TensorFunction {
    const act = actCode(activation);
    return class extends TensorFunction {
        static forward(ctx: Context, input: Tensor, bias: Tensor): Tensor {
            const out = TensorData.empty(shapeBroadcast(input.shape(), bias.shape()));
            nativeBiasAct(out._storage, out._shape, out._stride, input.data._storage, input.data._shape, input.data._stride, bias.data._storage, bias.data._shape, bias.data._stride, act);
            const y = new Tensor(out);
            ctx.saveForBackward(input, bias, y);
            return y;
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [input, bias, y] = ctx.savedValues as Tensor[];
            const saved = activation === "relu" ? input : y;
            const gradInput = TensorData.empty(gradOutput.shape());
            const gradBias = TensorData.empty(bias.shape());
            nativeBiasActBackward(gradInput._storage, gradInput._shape, gradInput._stride, gradBias._storage, gradBias._shape, gradBias._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, saved.data._storage, saved.data._shape, saved.data._stride, bias.data._storage, bias.data._shape, bias.data._stride, act);
            let gIn = new Tensor(gradInput);
            if (!sameShape(gIn, input)) gIn = unbroadcast(gIn, input.shape());
            return [gIn, new Tensor(gradBias)];
        }
    };
}

export function layerNorm(x: Tensor, weight: Tensor, bias: Tensor, eps: number = 1e-5): Tensor {
    if (x.dims() < 1) throw "layerNorm expected a last dimension";
    if (!x.shape()[x.dims() - 1]) throw "layerNorm received an empty last dimension";
    if (!native) {
        const dim = x.dims() - 1;
        const n = x.shape()[dim];
        const mean = x.sum(dim).div(n);
        const centered = x.sub(mean);
        const variance = centered.mul(centered).sum(dim).div(n);
        const rstd = variance.add(eps).log().mul(-0.5).exp();
        return centered.mul(rstd).mul(weight).add(bias);
    }
    return Tensor.apply(LayerNorm(eps), x, weight, bias);
}

export function LayerNorm(eps: number): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, x: Tensor, weight: Tensor, bias: Tensor): Tensor {
            const last = x.shape()[x.dims() - 1];
            const rows = x.size() / last;
            const out = TensorData.empty(x.shape());
            const mean = TensorData.empty([rows]);
            const rstd = TensorData.empty([rows]);
            nativeLayerNorm(out._storage, out._shape, out._stride, mean._storage, mean._shape, mean._stride, rstd._storage, rstd._shape, rstd._stride, x.data._storage, x.data._shape, x.data._stride, weight.data._storage, weight.data._shape, weight.data._stride, bias.data._storage, bias.data._shape, bias.data._stride, eps);
            ctx.saveForBackward(x, new Tensor(mean), new Tensor(rstd), weight, bias);
            return new Tensor(out);
        }
        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const [x, mean, rstd, weight, bias] = ctx.savedValues as Tensor[];
            const gradX = TensorData.empty(x.shape());
            const gradW = TensorData.empty(weight.shape());
            const gradB = TensorData.empty(bias.shape());
            nativeLayerNormBackward(gradX._storage, gradX._shape, gradX._stride, gradW._storage, gradW._shape, gradW._stride, gradB._storage, gradB._shape, gradB._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, x.data._storage, x.data._shape, x.data._stride, mean.data._storage, mean.data._shape, mean.data._stride, rstd.data._storage, rstd.data._shape, rstd.data._stride, weight.data._storage, weight.data._shape, weight.data._stride);
            return [new Tensor(gradX), new Tensor(gradW), new Tensor(gradB)];
        }
    };
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

export class Sub extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(sub(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [unbroadcast(gradOutput, a.shape()), unbroadcast(gradOutput.neg(), b.shape())];
    }
}

export class Div extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return new Tensor(div(a.data, b.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        const gradA = gradOutput.div(b);
        const gradB = gradOutput.neg().mul(a).div(b.mul(b));
        return [unbroadcast(gradA, a.shape()), unbroadcast(gradB, b.shape())];
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
        let [res] = ctx.savedValues as Tensor[];
        if (!native) return [gradOutput.mul(res).mul(res.neg().add(1))];
        const grad = TensorData.empty(res.shape());
        nativeSigmoidBackward(grad._storage, grad._shape, grad._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, res.data._storage, res.data._shape, res.data._stride);
        return [new Tensor(grad)];
    }
}

export class ReLU extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return new Tensor(relu(a.data));
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues as Tensor[];
        if (!native) {
            let relumask = new Tensor(lt(TensorData.fill(a.data.shape, 0), a.data)); // relumask[i] = 1 if a[i] > zeros[i] else 0
            return [gradOutput.mul(relumask)];
        }
        const grad = TensorData.empty(a.shape());
        nativeReluBackward(grad._storage, grad._shape, grad._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride, a.data._storage, a.data._shape, a.data._stride);
        return [new Tensor(grad)];
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

        // derivative of input, then weight
        return [new Tensor(conv1d(gradOutput.data, weight.permute(1, 0, 2).data, true)), new Tensor(conv1d(input.permute(1, 0, 2).data, gradOutput.permute(1, 0, 2).data, false, weight.shape()[2])).permute(1, 0, 2).tocontiguous()];
    }
}

export class Conv2d extends TensorFunction{
    static forward(ctx: Context, input: Tensor, weight: Tensor): Tensor{
        ctx.saveForBackward(input, weight);
        return new Tensor(conv2d(input.data, weight.data));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [input, weight] = ctx.savedValues;

        // derivative of input, then weight
        return [new Tensor(conv2d(gradOutput.data, weight.permute(1, 0, 2, 3).data, true)), new Tensor(conv2d(input.permute(1, 0, 2, 3).data, gradOutput.permute(1, 0, 2, 3).data, false, weight.shape()[2], weight.shape()[3])).permute(1, 0, 2, 3).tocontiguous()];
    }
}

export class Max extends TensorFunction{
    static forward(ctx: Context, input: Tensor, dim: Tensor): Tensor{
        const d = dim.item();
        ctx.saveForBackward(input, dim);
        return new Tensor(max(input.data, d));
    }

    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [input, dim] = ctx.savedValues;
        const d = dim.item();

        //                                          Dim has no derivative so just return a tensor with [0]
        return [argmax(input, d).mul(gradOutput), new Tensor(TensorData.fill([1], 0))];
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
            let [a] = ctx.savedValues as Tensor[];
            if (!native) return [gradOutput.mul(Tensor.ones(a.shape()))];
            const grad = TensorData.empty(a.shape());
            nativeExpand(grad._storage, grad._shape, grad._stride, gradOutput.data._storage, gradOutput.data._shape, gradOutput.data._stride);
            return [new Tensor(grad)];
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