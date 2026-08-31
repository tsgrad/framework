import { describe, test, expect } from "vitest"
import { centralDifference, Context } from "../src/autodiff"
import { id, add, mul, exp, isClose } from "../src/operators"
import { ScalarFunction } from "../src/scalar_functions"
import { Scalar, ScalarHistory } from "../src/scalar"

test("Test central difference", () => {
    expect(isClose(centralDifference(id, [5]), 1.0)).toEqual(1);
    expect(isClose(centralDifference(add, [5, 10]), 1.0)).toEqual(1);
    expect(isClose(centralDifference(mul, [5, 10]), 10.0)).toEqual(1);
    expect(isClose(centralDifference(mul, [5, 10], 1), 5.0)).toEqual(1);
    expect(isClose(centralDifference(exp, [2]), exp(2))).toEqual(1);
});

class Function1 extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        return a + b + 10;
    }
    static backward(ctx: Context, dOutput: number): number[]{
        return [dOutput, dOutput];
    }
};

class Function2 extends ScalarFunction{
    static forward(ctx: Context, a: number, b: number): number{
        ctx.saveForBackward(a, b);
        return a * b + a;
    }
    static backward(ctx: Context, dOutput: number): number[]{
        let [x, y] = ctx.savedValues;
        return [dOutput * (y + 1), dOutput * x];
    }
};

describe("Tests for the autodifferentiation machinery", () => {
    test("Test chainRule 1", () => {
        let x = new Scalar(0.0);
        let constant = new Scalar(0.0, new ScalarHistory(Function1, new Context(), [x, x]));
        let back = constant.chainRule(5);
        expect(back.length).toEqual(2);
        expect(back[0][1]).toEqual(5);
    });

    test("Test chainRule 2", () => {
        //Check that constants are ignored and variables get derivatives
        let constant = 10;
        let x = new Scalar(5);
        let y = Scalar.apply(Function2, constant, x);
        let back = y.chainRule(5);
        expect(back.length).toEqual(2);
        expect(back[1][1]).toEqual(5 * 10);
    });

    test("Test chainRule 3", () => {
        let var1 = new Scalar(5);
        let var2 = new Scalar(10);

        let y = Scalar.apply(Function2, var1, var2);
        let back = y.chainRule(5);
        expect(back.length).toEqual(2);
        expect(back[0][1]).toEqual(5 * (10 + 1));
        expect(back[1][1]).toEqual(5 * 5);
    });
});
