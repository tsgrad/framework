import { describe, test, expect } from "vitest"
import { Module, Parameter } from "../src/module"
import { randomInt, randomFloat } from "./testhelperfunctions";

class ModuleA1 extends Module{
    p1: Parameter;
    nonParam: number;
    a: ModuleA2;
    b: ModuleA3;

    constructor(){
        super();
        this.p1 = new Parameter(5);
        this.nonParam = 10;
        this.a = new ModuleA2();
        this.b = new ModuleA3();
    }
}

class ModuleA2 extends Module{
    p2: Parameter;
    constructor(){
        super();
        this.p2 = new Parameter(10);
    }
}

class ModuleA3 extends Module{
    c: ModuleA4;
    constructor(){
        super();
        this.c = new ModuleA4();
    }
}

class ModuleA4 extends Module{
    p3: Parameter;
    constructor(){
        super();
        this.p3 = new Parameter(15);
    }
}

describe("Basic test of Module tree structure", () => {
    let mod = new ModuleA1();

    //console.log(mod);
    test("Constructor for ModuleA1 works as expected", () =>{
        expect(mod.p1.value).toEqual(5);
        expect(mod.nonParam).toEqual(10);
    });

    test ("namedParameters() is correct and maintains hierarchy", () => {
        let namedParams : [string, Parameter][] = mod.namedParameters(); 
        let np = new Map<string, Parameter>(namedParams);
            
        expect(np.get("p1")?.value).toEqual(5);
        expect(np.get("a.p2")?.value).toEqual(10);
        expect(np.get("b.c.p3")?.value).toEqual(15);
    });
});

// These tests generate a stack of modules of varying sizes to check properties
var VALA = 50.0, VALB = 100.0;

class Module1 extends Module{
    moduleA: Module2;
    moduleB: Module2;
    parameterA: Parameter;
    
    constructor(sizeA: number, sizeB: number, val: number){
        super();
        this.moduleA = new Module2(sizeA);
        this.moduleB = new Module2(sizeB);
        this.parameterA = new Parameter(val);
    }
}

class Module2 extends Module{
    parameterA: Parameter;
    parameterB: Parameter;
    nonParameter: number;
    moduleC: Module3;

    constructor(extra: number = 0){
        super();
        this.parameterA = new Parameter(VALA);
        this.parameterB = new Parameter(VALB);
        this.nonParameter = 10;
        this.moduleC = new Module3();
        for (let i = 0; i < extra; i++)
            this.addParameter(`extraParameter${i}`, 0);
    }
}
class Module3 extends Module{
    parameterA: Parameter;
    constructor(){
        super();
        this.parameterA = new Parameter(VALA);
    }
}

describe("Advanced - Check properties of single module (no extras)", () => {
    let mod = new Module2();
    test("eval() should switch training to false", () => { 
        mod.eval();
        expect(mod.training).toBe(false);
    });

    test("train() should switch training to true", () => { 
        mod.train();
        expect(mod.training).toBe(true);
    });

    test("Should be 3 parameters in the Module2 object by default", () => {
        expect(mod.parameters().length).toEqual(3);
    });

});

describe("Advanced - Check properties of single module (with extras)", () => {
    let sizeA = randomInt(1, 100), sizeB = randomInt(1, 100);

    test(`mod should have ${sizeA} + 3 parameters after running addParameters() ${sizeA} times`, () => {
        let mod = new Module2(sizeA);
        expect(mod.parameters().length).toEqual(sizeA + 3);
    });

    test("_parameters should be updated correctly", () => {
        let mod = new Module2(sizeB);
        let namedParams : [string, Parameter][] = mod.namedParameters(); 
        let np = new Map<string, Parameter>(namedParams);
        expect(np.get("parameterA")?.value).toEqual(VALA);
        expect(np.get("parameterB")?.value).toEqual(VALB);
        expect(np.get("extraParameter0")?.value).toEqual(0);
    });
});

describe("Advanced - Check properties of a stacked module", () => {
    let sizeA = randomInt(1, 100), sizeB = randomInt(1, 100);
    let val = randomFloat(-100, 100);
    let mod = new Module1(sizeA, sizeB, val);
    test("eval() should switch training to false", () => {
        mod.eval();
        expect(mod.training).toBe(false);
        expect(mod.moduleA.training).toBe(false);
        expect(mod.moduleB.training).toBe(false);
    });

    test("train() should switch training to true", () => { 
        mod.train();
        expect(mod.training).toBe(true);
        expect(mod.moduleA.training).toBe(true);
        expect(mod.moduleB.training).toBe(true);
    });

    test(`Should be ${sizeA + sizeB + 7} parameters in namedParameters()`, () => {
        expect(mod.parameters().length).toEqual(sizeA + sizeB + 7);
    });

    let namedParams : [string, Parameter][] = mod.namedParameters(); 
    let np = new Map<string, Parameter>(namedParams);

    test("_parameters should be updated correctly", () => {
        expect(np.get("parameterA")?.value).toEqual(val);
        expect(np.get("moduleA.parameterA")?.value).toEqual(VALA);
        expect(np.get("moduleA.parameterB")?.value).toEqual(VALB);
        expect(np.get("moduleB.parameterA")?.value).toEqual(VALA);
        expect(np.get("moduleB.parameterB")?.value).toEqual(VALB);
    });
});