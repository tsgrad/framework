import { Context, Variable } from "./autodiff";
import { Add, EQ, Exp, Inv, Log, LT, Mul, Neg, ReLU, ScalarFunction, Sigmoid } from "./scalar_functions";

export class ScalarHistory{
    // Tracks the history of `Function` operations that were used
    // to construct the current variable

    lastFn: typeof ScalarFunction | undefined;
    ctx: Context | undefined;
    inputs: Scalar[] = [];

    constructor(lastFn: typeof ScalarFunction | undefined = undefined, ctx: Context | undefined = undefined, inputs: Scalar[] = []){
        this.lastFn = lastFn;
        this.ctx = ctx;
        this.inputs = inputs;
    }
};

type ScalarLike = (number | Scalar);

export class Scalar implements Variable{
    // Scalar values for autodifferentiation tracking
    // Can only be manipulated by `ScalarFunction`

    private static _varCount: number = 0;

    history: ScalarHistory | undefined;
    derivative: number | undefined;
    data: number;
    uniqueId: number;
    name: string;

    constructor(data: number, history: ScalarHistory = new ScalarHistory(), name: string | undefined = undefined){
        Scalar._varCount++;
        this.uniqueId = Scalar._varCount;
        this.data = data;
        this.history = history;

        if (name !== undefined)
            this.name = name;
        else
            this.name = this.uniqueId.toString();
    }

    static apply(f: typeof ScalarFunction, ...vals: ScalarLike[]){
        let rawVals: number[] = [];
        let scalars: Scalar[] = [];

        for (const val of vals){
            if (val instanceof Scalar){ 
                scalars.push(val);
                rawVals.push(val.data);
            }
            else{
                scalars.push(new Scalar(val));
                rawVals.push(val);
            }
        }

        // Create context
        let ctx = new Context();

        // Call forward with the variables
        let c: number = f.forward(ctx, ...rawVals);
        
        // Create new variable from the result with a new history
        let back = new ScalarHistory(f, ctx, scalars);
        return new Scalar(c, back);
    }

    add(b: ScalarLike): Scalar{
        return Scalar.apply(Add, this, b);
    }

    sub(b: ScalarLike): Scalar{
        return Scalar.apply(Add, this, Scalar.apply(Neg, b));
    }

    mul(b: ScalarLike): Scalar{
        return Scalar.apply(Mul, this, b);
    }

    div(b: ScalarLike): Scalar{
        return Scalar.apply(Mul, this, Scalar.apply(Inv, b));
    }

    // We might not need this since all our calls will look like a.div(b) but gonna add it for now
    rdiv(b: ScalarLike): Scalar{
        return Scalar.apply(Mul, Scalar.apply(Inv, this), b);
    }

    lt(b: ScalarLike): Scalar{
        return Scalar.apply(LT, this, b);
    }

    gt(b: ScalarLike): Scalar{
        return Scalar.apply(LT, b, this);
    }

    // Maybe needed? Kept for consistency
    bool(): boolean{
        return !!this.data; 
    }

    eq(b: ScalarLike): Scalar{
        return Scalar.apply(EQ, this, b);
    }

    neg(): Scalar{
        return Scalar.apply(Neg, this);
    }

    log(): Scalar{
        return Scalar.apply(Log, this);
    }

    exp(): Scalar{
        return Scalar.apply(Exp, this);
    }

    sigmoid(): Scalar{
        return Scalar.apply(Sigmoid, this);
    }

    relu(): Scalar{
        return Scalar.apply(ReLU, this);
    }

    accumulateDerivative(x: number): void{
        if (!this.isLeaf())
            throw new Error("Only leaf variables can have derivatives.");
        
        if (this.derivative === undefined)
            this.derivative = 0.0;

        this.derivative += x;
    }

    isLeaf(): boolean{
        // True if this variable was created by the user (no `lastFn`)
        return (this.history !== undefined && this.history.lastFn === undefined);
    }

    isConstant(): boolean{
        return (this.history === undefined);
    }

    parents(): Scalar[]{
        if (this.history === undefined)
            throw new Error("history is undefined");
        return this.history.inputs;
    }

    chainRule(gradient: number): [Variable, number][]{
        let history = this.history;
        if (history === undefined) throw new Error("History is undefined in chainrule");
        if (history.lastFn === undefined) throw new Error("History has no lastFn in chainrule");
        if (history.ctx === undefined) throw new Error("Context is undefined in chainrule");

        let parentGradients: number[] = history.lastFn.backward(history.ctx, gradient);
        let res: [Variable, number][] = [];
        for (let i = 0; i < parentGradients.length; i++){
            res.push([history.inputs[i], parentGradients[i]]);
        }
        return res;
    }
}

