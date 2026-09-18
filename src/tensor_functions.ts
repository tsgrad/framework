import { Tensor } from "./tensor";
import { Context } from "./autodiff";

export class TensorFunction{
    static forward(ctx: Context, ...inputs: Tensor[]): Tensor{
        throw new Error("Subclass must create forward");
    }
    static backward(ctx: Context, ...t: Tensor[]): Tensor[]{
        throw new Error("Subclass must create backward");
    }
}

export class Neg extends TensorFunction{
    static forward(ctx: Context, ...inputs: Tensor[]): Tensor{
        throw new Error("Subclass must create forward");
    }
    static backward(ctx: Context, ...t: Tensor[]): Tensor[]{
        return [t[0].f.negMap()]
    }
}