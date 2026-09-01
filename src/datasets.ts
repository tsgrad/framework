import { randomFloat } from "./helperfunctions";

function makePts(n: number): [number, number][]{
    let res: [number, number][] = [];
    for (let i = 0; i < n; i++){
        let x = randomFloat(0, 1), y = randomFloat(0, 1);
        res.push([x, y]);
    }
    return res;
}

export class Graph{
    n: number;
    x: [number, number][];
    y: number[];

    constructor(n: number, x: [number, number][], y: number[]){
        this.n = n;
        this.x = x;
        this.y = y;
    }
}

function simple(n: number): Graph{
    let x = makePts(n);
    let y = [];
    for (const [x1, x2] of x){
        y.push(x1 < 0.5 ? 1 : 0);
    }
    return new Graph(n, x, y);
}

function diag(n: number): Graph{
    let x = makePts(n);
    let y = [];
    for (const [x1, x2] of x){
        y.push(x1 + x2 < 0.5 ? 1 : 0);
    }
    return new Graph(n, x, y);
}

function split(n: number): Graph{
    let x = makePts(n);
    let y = [];
    for (const [x1, x2] of x){
        y.push(x1 < 0.2 || x1 > 0.8 ? 1 : 0);
    }
    return new Graph(n, x, y);
}

function xor(n: number): Graph{
    let x = makePts(n);
    let y = [];
    for (const [x1, x2] of x){
        y.push((x1 < 0.5 && x2 > 0.5) || (x1 > 0.5 && x2 < 0.5) ? 1 : 0);
    }
    return new Graph(n, x, y);
}

function circle(n: number): Graph{
    let x = makePts(n);
    let y = [];
    for (const [x1, x2] of x){
        let a = x1 - 0.5, b = x2 - 0.5;
        y.push(a * a + b * b > 0.1 ? 1 : 0);
    }
    return new Graph(n, x, y);
}

function spiral(n: number): Graph{
    function x(t: number): number { return t * Math.cos(t) / 20.0};
    function y(t: number): number { return t * Math.sin(t) / 20.0};

    const X: [number, number][] = [];

    for (let i = 5; i < 5 + Math.floor(n / 2); i++) {
        const t = 10.0 * (i / Math.floor(n / 2));
        X.push([x(t) + 0.5, y(t) + 0.5]);
    }

    for (let i = 5; i < 5 + Math.floor(n / 2); i++) {
        const t = -10.0 * (i / Math.floor(n / 2));
        X.push([y(t) + 0.5, x(t) + 0.5]);
    }

    const y2: number[] = [
        ...Array(Math.floor(n / 2)).fill(0),
        ...Array(Math.floor(n / 2)).fill(1)
    ];

    return new Graph(n, X, y2);
}

export const datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}