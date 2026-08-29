import { describe, test, expect } from "vitest"
import { centralDifference } from "../src/autodiff"
import { id, add, mul, exp, isClose } from "../src/operators"

test("Test central difference", () => {
    expect(isClose(centralDifference(id, [5]), 1.0)).toEqual(1);
    expect(isClose(centralDifference(add, [5, 10]), 1.0)).toEqual(1);
    expect(isClose(centralDifference(mul, [5, 10]), 10.0)).toEqual(1);
    expect(isClose(centralDifference(mul, [5, 10], 1), 5.0)).toEqual(1);
    expect(isClose(centralDifference(exp, [2]), exp(2))).toEqual(1);
});