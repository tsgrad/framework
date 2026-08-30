import { describe, test, expect } from "vitest"
import { Scalar } from "../src/scalar"
import { randomInt } from "./testhelperfunctions";
import { isClose, log, relu } from "../src/operators";

// Toggle to see what the histories of the scalars look like
var verbose = true;

describe("Test Scalar Functions", () => {
    let a = randomInt(-10, 10), b = randomInt(-10, 10);
    test("Simple add", () => {
        let c = new Scalar(a).add(new Scalar(b));
        if (verbose) console.log(c.history);
        expect(isClose(c.data, a + b)).toEqual(1);
    });

    test("Simple mul", () => {
        let c = new Scalar(a).mul(new Scalar(b));
        if (verbose) console.log(c.history);
        expect(isClose(c.data, a * b)).toEqual(1);
    });
    
    test("Simple relu", () => {
        let c = new Scalar(a).relu().add(new Scalar(b).relu());
        if (verbose) console.log(c.history);
        expect(isClose(c.data, relu(a) + relu(b))).toEqual(1);
    });

    test("Simple log", () => {
        let x = Math.abs(a) || 1;
        let y = Math.abs(b) || 1;

        let c = new Scalar(x).log().add(new Scalar(y).log());
        if (verbose) console.log(c.history);
        expect(isClose(c.data, log(x) + log(y))).toEqual(1);
    });
    test("Simple equalities", () => {
        let x = new Scalar(a)
        let y = new Scalar(b);
        expect((!!(x.lt(y)).data) === (a < b)).toBe(true);
        expect((!!(x.gt(y)).data) === (a > b)).toBe(true);
        expect((!!(x.eq(y)).data) === (a === b)).toBe(true);
        
        if (verbose) {
            console.log(x.history);
            console.log(y.history);
        }
    });
});
