import { Tensor } from "./tensor";
import { Context } from "./autodiff";
import { lt } from "./operators";

export abstract class TensorFunction{
    static forward(ctx: Context, ...t: Tensor[]): Tensor{
        throw new Error("Subclass must create forward");
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        throw new Error("Subclass must create backward");
    }
}

export class Neg extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        return a.f.negMap();
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput.f.negMap(gradOutput)];
    }
}

// I don't like it, and it's kinda messy using Tensor operations inside the Operation classes but it works...
export class Inv extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return a.f.invMap();
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.neg().div(a.mul(a))];
    }
}

export class Add extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        return a.f.addZip(a, b);
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput, gradOutput];
    }
}

export class Mul extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        ctx.saveForBackward(a, b);
        return a.f.mulZip(a, b);
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a, b] = ctx.savedValues;
        return [gradOutput.mul(b), gradOutput.mul(a)];
    }
}

export class Sigmoid extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = a.f.sigmoidMap(a);
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
        return a.f.reluMap(a);
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        //let [a] = ctx.savedValues;
        //let relumask = Tensor.zeros(gradOutput.backend, a._tensor.shape);
        //currently broken will fix later
        //return [gradOutput.mul(relumask)];
    }
}

export class Log extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        ctx.saveForBackward(a);
        return a.f.logMap(a);
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.div(a)];
    }
}

export class Exp extends TensorFunction{
    static forward(ctx: Context, a: Tensor): Tensor{
        let res = a.f.expMap(a);
        ctx.saveForBackward(res);
        return res;
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        let [a] = ctx.savedValues;
        return [gradOutput.mul(a)];
    }
}

export class Sum extends TensorFunction{
    static forward(ctx: Context, a: Tensor, dim: Tensor): Tensor{
        ctx.saveForBackward(a.shape, dim);
        return a.f.addReduce(a, dim.item());
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput, gradOutput._ensureTensor(0.0)];
    }
}

export class LT extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        return a.f.ltZip(a, b);
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput.mul(0)];
    }
}

export class EQ extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        return a.f.eqZip(a, b);
    }
    static backward(ctx: Context, gradOutput: Tensor): Tensor[]{
        return [gradOutput.mul(0)];
    }
}

export class IsClose extends TensorFunction{
    static forward(ctx: Context, a: Tensor, b: Tensor): Tensor{
        return a.f.isCloseZip(a, b);
    }
}

export function Permute(order: number[]): typeof TensorFunction {
    return class extends TensorFunction {
        static forward(ctx: Context, a: Tensor): Tensor {
            return new Tensor(a._tensor.permute(...order), undefined, undefined, a.backend);
        }

        static backward(ctx: Context, gradOutput: Tensor): Tensor[] {
            const inverseOrder = new Array(order.length);

            for (let i = 0; i < order.length; i++)
                inverseOrder[order[i]!] = i;

            return [new Tensor(gradOutput._tensor.permute(...inverseOrder), undefined, undefined, gradOutput.backend)];
        }
    };
}