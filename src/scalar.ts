import { Context } from "./autodiff";
import { ScalarFunction } from "./scalar_functions";

class ScalarHistory{
    // Tracks the history of `Function` operations that were used
    // to construct the current variable

    lastFn: ScalarFunction | undefined;
    ctx: Context | undefined;
    inputs: Scalar[] = [];

    constructor(lastFn: ScalarFunction | undefined = undefined, ctx: Context | undefined = undefined, inputs: Scalar[] = []){
        this.lastFn = lastFn;
        this.ctx = ctx;
        this.inputs = inputs;
    }
};

class Scalar{
    // Scalar values for autodifferentiation tracking
    // Can only be manipulated by `ScalarFunction`

    private static _varCount: number = 0;

    history: ScalarFunction | undefined;
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

    static apply(f: typeof ScalarFunction, ...vals: (number | Scalar)[]){
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
}

