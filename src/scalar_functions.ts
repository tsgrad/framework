import { add, eq, exp, inv, log, lt, mul, neg, relu, sigmoid } from "../src/operators"
import { Context } from "../src/autodiff"

export class ScalarFunction{
    static backward(ctx: Context, dOut: number): number{
        throw new Error("Subclass must create backward");
    }
    static forward(ctx: Context, ...inputs: number[]): number{
        throw new Error("Subclass must create forward");
    }

    // Ill make apply later it seems like a dependency mess
}

export class Add extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return add(a, b);
    }
}

export class Log extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return log(a);
    }
}

export class Mul extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return mul(a, b);
    }
}

export class Inv extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return inv(a);
    }
}

export class Neg extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return neg(a);
    }
}

export class Sigmoid extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return sigmoid(a);
    }
}

export class ReLU extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return relu(a);
    }
}

export class Exp extends ScalarFunction{
    static forward(ctx: Context, a: number): number{
        return exp(a);
    }
}

export class LT extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return lt(a, b);
    }
}

export class EQ extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return eq(a, b);
    }
}