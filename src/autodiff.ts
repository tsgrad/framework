import { Queue } from "./datastructures";

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

    noGrad: boolean;;
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
    data: number;
    accumulateDerivative(x: any): void;
    isLeaf(): boolean;
    isConstant(): boolean;
    parents(): Variable[];
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

export function backpropagate(start: Variable, dStart: number = 1): void{
    let order: Variable[] = topologicalSort(start);
    let derivatives: Map<number, number> = new Map<number, number>();
    derivatives.set(start.uniqueId, dStart);

    for (const val of order){
        if (val.isLeaf()){
            val.accumulateDerivative(derivatives.get(val.uniqueId)!);
        }
        else{
            let valDerivatives: [Variable, number][] = val.chainRule(derivatives.get(val.uniqueId));
            for (const [parent, d] of valDerivatives){
                let cur: number | undefined = derivatives.get(parent.uniqueId);
                if (cur === undefined){
                    derivatives.set(parent.uniqueId, d);
                }
                else{
                    derivatives.set(parent.uniqueId, cur + d);
                }
            }
        }
    }
}