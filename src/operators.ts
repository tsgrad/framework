function mul(a: number, b: number): number{
    return a * b;
}

function id(a: number): number{
    return a;
}

function add(a: number, b: number): number{
    return a + b;
}

// Just means flip the sign
function neg(a: number): number{
    return -a;
}

function lt(a: number, b: number): number{
    return a < b ? 1 : 0;
}

function eq(a: number, b: number): number{
    return a === b ? 1 : 0;
}

function max(a : number, b: number): number{
    return a > b ? a : b;
}