import { describe, test, expect } from "vitest"
import { mul, id, add, neg, lt, eq, max, isClose, sigmoid, relu, log, exp, inv, logBack, invBack, reluBack, addLists, sum, zipWith, prod } from "../src/operators";
import { randomInt } from "./testhelperfunctions";

describe("sigmoid", () =>{
    test("Always between 0 and 1", () =>{
        // expecting a big positive number to be close to 1 but not greater than 1.
        expect(isClose(sigmoid(100), 1) && sigmoid(100) <= 1).toBe(true);
        
        // expecting a big negative number to be close to 0 but not less than 0.
        expect(isClose(sigmoid(-100), 0) && sigmoid(-100) >= 0).toBe(true);
    });

    test("One minus sigmoid should be the same as sigmoid of the negative", () =>{
        expect(isClose(1 - sigmoid(2), sigmoid(-2))).toBe(1);
    });

    test("Sigmoid should cross 0.5 at 0", () =>{
        expect(isClose(sigmoid(0), 0.5)).toBe(1);
    });

    test("Sigmoid is strictly increasing", () => {
        for (let i = 1; i < 20; i++){
            expect(sigmoid(-1 + (0.1 * i)) > sigmoid(-1 + (0.1 * (i - 1)))).toBe(true);
        }
    });
});

describe("Logical properties", () =>{
    const values = [-2, -1, 0, 1, 2, 3];

    test("Test the transitive property of lt (a < b and b < c implies a < c)", () =>{
        for (const a of values) {
            for (const b of values) {
                for (const c of values) {
                    if (lt(a, b) && lt(b, c)) {
                        expect(lt(a, c)).toBe(1);
                    }
                }
            }
        }
    });

    test("mul is symmetric", () =>{
        for (const a of values) {
            for (const b of values) {
                expect(mul(a, b)).toEqual(mul(b, a));
            }
        }
    });

    test("make sure mul, and add are distributive", () =>{
        for (const a of values) {
            for (const b of values) {
                for (const c of values) { // + 0 is because in javascript -0 and +0 are different, thank you javascript
                    expect(mul(c, add(a, b)) + 0).toEqual(add(mul(c, a), mul(c, b)) + 0);
                }
            }
        }
    });
});

describe("Higher order functions", () =>{
    test("Test addLists and zipWith", ()=>{
        let a = randomInt(-100, 100), b = randomInt(-100, 100), c = randomInt(-100, 100), d = randomInt(-100, 100);
        let combined = addLists([a, b], [c, d]);
        expect(combined[0] + 0).toEqual(add(a, c) + 0);
        expect(combined[1] + 0).toEqual(add(b, d) + 0);
    });

    test("Test distributive property of sum and addLists", () => {
        // Testing (a + b) + (c + d) = (a + c) + (b + d)
        let a = randomInt(-100, 100), b = randomInt(-100, 100), c = randomInt(-100, 100), d = randomInt(-100, 100);
        expect(add(sum([a, b]), sum([c, d])) + 0).toEqual(sum(addLists([a, b], [c, d])) + 0);
    });

    test("Test prod", () => {
        let a = randomInt(-100, 100), b = randomInt(-100, 100), c = randomInt(-100, 100);
        expect(prod([a, b, c]) + 0).toEqual(a * b * c + 0);
    });
})