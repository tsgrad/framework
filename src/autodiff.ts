import { Queue } from "./datastructures.js";

export function centralDifference(f: Function, vals: number[], arg: number = 0, epsilon: number = 1e-6){
    let original = vals[arg];
    vals[arg] = original + epsilon;
    let v1 = f(...vals);
    vals[arg] = original - epsilon;
    let v2 = f(...vals);
    vals[arg] = original;

    return (v1 - v2) / (2 * epsilon);
}

export class Context{
    // Context class is used by `Function` to store information during forward pass
    constructor(noGrad: boolean = false, savedValues = []){
        this.noGrad = noGrad;
        this.savedValues = savedValues;
    };

    noGrad: boolean;
    savedValues: any[];

    saveForBackward(...values: any[]): void {
        if (this.noGrad)
            return;
        this.savedValues = values;
    }

    savedTensors(): any[]{
        return this.savedValues;
    }
}

export interface Variable{
    uniqueId: number;
    history: any;
    data: any;
    accumulateDerivative(x: any): void;
    isLeaf(): boolean;
    isConstant(): boolean;
    parents(): Variable[];
    add(b: Variable | number): Variable;
    chainRule(dOutput: any): [Variable, any][];
}

export function topologicalSort(variable: Variable): Variable[] {
    const seen = new Set<number>();
    const res: Variable[] = [];

    function dfs(node: Variable): void {
        if (seen.has(node.uniqueId))
            return;

        seen.add(node.uniqueId);
        for (const parent of node.parents())
            dfs(parent);

        res.push(node);
    }

    dfs(variable);
    return res.reverse();
}

let backwardDepth = 0;

export function inBackward(): boolean {
    return backwardDepth > 0;
}

function releaseGrad(grad: any, held: Map<number, any>): void {
    if (grad == null || typeof grad !== "object" || typeof grad.data?.release !== "function") return;
    for (const value of held.values()) {
        if (value === grad) return;
    }
    grad.data.release();
}

function detachTape(val: Variable): void {
    const history = val.history;
    if (!history) return;
    history.inputs = [];
    if (history.ctx) history.ctx.savedValues = [];
    history.lastFn = undefined;
}

export function backpropagate(start: Variable, dStart: any): void{
    backwardDepth++;
    try {
        let order: Variable[] = topologicalSort(start);
        let derivatives: Map<number, any> = new Map<number, any>();
        derivatives.set(start.uniqueId, dStart);

        for (const val of order){
            if (val.isLeaf()){
                const grad = derivatives.get(val.uniqueId);
                if (grad !== undefined) val.accumulateDerivative(grad);
                derivatives.delete(val.uniqueId);
                releaseGrad(grad, derivatives);
            }
            else if (val.isConstant()){
                continue;
            }
            else{
                const grad = derivatives.get(val.uniqueId);
                let valDerivatives = val.chainRule(grad);
                derivatives.delete(val.uniqueId);
                for (const [parent, d] of valDerivatives){
                    const cur = derivatives.get(parent.uniqueId);
                    if (cur === undefined){
                        derivatives.set(parent.uniqueId, d);
                    }
                    else if (typeof cur === "number"){
                        derivatives.set(parent.uniqueId, cur + d);
                        releaseGrad(d, derivatives);
                    }
                    else{
                        derivatives.set(parent.uniqueId, cur.add(d));
                        releaseGrad(cur, derivatives);
                        releaseGrad(d, derivatives);
                    }
                }
                releaseGrad(grad, derivatives);
                detachTape(val);
            }
        }
    } finally {
        backwardDepth--;
    }
}