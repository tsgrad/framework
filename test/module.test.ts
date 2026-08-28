import { describe, test, expect } from "vitest"
import { Module, Parameter } from "../src/module"

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

describe("Test Module tree structure", () => {
    let mod = new ModuleA1();
    let namedParams : [string, Parameter][] = mod.namedParameters(); 
    let np = new Map<string, Parameter>(namedParams);

    console.log(mod);
    test("Constructor for ModuleA1 works as expected", () =>{
        expect(mod.p1.value).toEqual(5);
        expect(mod.nonParam).toEqual(10);
    });

    test ("namedParameters() is correct and maintains hierarchy", () => {
        expect(np.get("p1")?.value).toEqual(5);
        expect(np.get("a.p2")?.value).toEqual(10);
        expect(np.get("b.c.p3")?.value).toEqual(15);
    });
});

// These tests generate a stack of modules of varying sizes to check properties
var VAL_A = 50.0, VAL_B = 100.0;
