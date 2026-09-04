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
        y.push(Math.abs(x1 + x2) < 1 ? 1 : 0);
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

function makeMoons(n: number, noise = 0.1): Graph {
    const samples: { point: [number, number]; label: number }[] = [];
    const outerCount = Math.floor(n / 2);
    const innerCount = n - outerCount;

    const gaussian = (): number => {
        const u = Math.max(Math.random(), Number.EPSILON);
        const v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };

    const add = (point: [number, number], label: number): void => {
        samples.push({
            point: [
                point[0] + noise * gaussian(),
                point[1] + noise * gaussian(),
            ],
            label
        });
    };

    for (let i = 0; i < outerCount; i++) {
        const angle = Math.PI * i / Math.max(outerCount - 1, 1);
        add([Math.cos(angle), Math.sin(angle)], 0);
    }

    for (let i = 0; i < innerCount; i++) {
        const angle = Math.PI * i / Math.max(innerCount - 1, 1);
        add([1 - Math.cos(angle), 0.5 - Math.sin(angle)], 1);
    }

    for (let i = samples.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [samples[i], samples[j]] = [samples[j], samples[i]];
    }

    return new Graph(samples.length, samples.map(sample => sample.point), samples.map(sample => sample.label));
}

export const datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
    "Moon": makeMoons
}