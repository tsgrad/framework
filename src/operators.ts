export function mul(a: number, b: number): number{
    return a * b;
}

export function id(a: number): number{
    return a;
}

export function add(a: number, b: number): number{
    return a + b;
}

// Just means flip the sign
export function neg(a: number): number{
    return -a;
}

export function lt(a: number, b: number): number{
    return a < b ? 1 : 0;
}

export function eq(a: number, b: number): number{
    return a === b ? 1 : 0;
}

export function max(a : number, b: number): number{
    return a > b ? a : b;
}

export function is_close(a: number, b: number): number{
    return Math.abs(a - b) < 1e-2 ? 1 : 0;
}

export function sigmoid(a: number): number{
    if (a >= 0)
        return 1.0/(1.0 + Math.exp(-a));  // e^a
    else
        return Math.exp(a) / (1.0 + Math.exp(a));
}

export function relu(a: number): number{
    return Math.max(0, a);
}

export function log(a: number): number{
    return Math.log(a); // ln(a)
}

export function exp(a: number): number{
    return Math.exp(a); // e^a
}

// reciprocal
export function inv(a: number): number{
    return 1.0 / a;
}

export function log_back(a: number, b: number): number{
    return b / a;
}

export function inv_back(a: number, b: number): number{
    return -b / (a * a);
}

export function relu_back(a: number, b: number): number{
    return a > 0 ? b : 0;
}

// For this task 0.3 section for now I'm going to group everything under array
// This might need to be changed to a general class of iterable later
export function map(func: Function, it: Array<number>): Array<number>{
    for (let i = 0; i < it.length; i++)
        func(it[i]);
    return it;
}

export function zipWith(func: Function, it1: Array<number>, it2: Array<number>): Array<number>{
    if (it1.length < it2.length){
        for (let i = 0; i < it1.length; i++)
            it2[i] = func(it1[i], it2[i]);
        return it2;
    }
    else{
        for (let i = 0; i < it2.length; i++)
            it1[i] = func(it1[i], it2[i]);
        return it1;
    }
}

export function reduce(func: Function, it: Array<number>): number{
    let res : number = 0;
    for (let i = 0; i < it.length; i++)
        res += func(it[i]);
    return res;
}