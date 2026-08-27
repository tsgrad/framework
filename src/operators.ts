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

function is_close(a: number, b: number): number{
    return Math.abs(a - b) < 1e-2 ? 1 : 0;
}

function sigmoid(a: number): number{
    if (a >= 0)
        return 1.0/(1.0 + Math.exp(-a));  // e^a
    else
        return Math.exp(a) / (1.0 + Math.exp(a));
}

function relu(a: number): number{
    return Math.max(0, a);
}

function log(a: number): number{
    return Math.log(a); // ln(a)
}

function exp(a: number): number{
    return Math.exp(a); // e^a
}

// reciprocal
function inv(a: number): number{
    return 1.0 / a;
}