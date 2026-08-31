import { add, eq, exp, inv, log, lt, mul, neg, relu, sigmoid } from "../src/operators"
import { Context } from "../src/autodiff"

export class ScalarFunction{
    static backward(ctx: Context, dOut: number): number[]{
        throw new Error("Subclass must create backward");
    }
    static forward(ctx: Context, ...inputs: number[]): number{
        throw new Error("Subclass must create forward");
    }
}

export class Add extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return add(a, b);
    }

    static backward(ctx: Context, d: number): number[]{
        return [d, d];
    }
}

export class Log extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        ctx.saveForBackward(a);
        return log(a);
    }

    static backward(ctx: Context, d: number): number[]{
        let [x] = ctx.savedValues;
        let val = x != 0 ? x : 1e-10;
        return [d * inv(val)];
    }
}

export class Mul extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        ctx.saveForBackward(a, b);
        return mul(a, b);
    }

    static backward(ctx: Context, d: number): number[]{
        let [x, y] = ctx.savedValues;
        //let x_prime = y;
        //let y_prime = x;
        return [d * y, d * x]; 
    }
}

export class Inv extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        ctx.saveForBackward(a);
        return inv(a);
    }

    static backward(ctx: Context, d: number): number[]{
        let [x] = ctx.savedValues;
        return [-d * inv(x * x)];
    }
}

export class Neg extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return neg(a);
    }

    static backward(ctx: Context, d: number): number[]{
        return [-d]; 
    }
}

export class Sigmoid extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        ctx.saveForBackward(a);
        return sigmoid(a);
    }

    static backward(ctx: Context, d: number): number[]{
        let [x] = ctx.savedValues;
        return [d * exp(-x) * inv(((1 + exp(-x)) ** 2))]
    }
}

export class ReLU extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        ctx.saveForBackward(a);
        return relu(a);
    }

    static backward(ctx: Context, d: number): number[]{
        let [x] = ctx.savedValues;
        if (x > 0)
            return [d];
        return [0];
    }
}

export class Exp extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        ctx.saveForBackward(a);
        return exp(a);
    }

    static backward(ctx: Context, d: number): number[]{
        let [x] = ctx.savedValues;
        return [d * exp(x)];
    }
}

export class LT extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return lt(a, b);
    }

    static backward(ctx: Context, d: number): number[]{
        return [0, 0];
    }
}

export class EQ extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return eq(a, b);
    }

    static backward(ctx: Context, d: number): number[]{
        return [0, 0];
    }
}
