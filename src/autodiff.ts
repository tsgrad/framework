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
        this.noGrad = false;
        this.savedValues = [];
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
    accumulateDerivative(x: any): void;
    isLeaf(): boolean;
    isConstant(): boolean;
    parents(): Variable[];
    chainRule(dOutput: any): [Variable, any][];
}

export function topologicalSort(variable: Variable): Variable[]{
    let seen: Set<Variable> = new Set<Variable>();
    let res: Variable[] = [];
    let queue = new Queue<Variable>();
    queue.push(variable);
    seen.add(variable);

    while (!queue.isEmpty()){
        let node: Variable = queue.pop()!;
        res.push(node);

        
        for (const val of node.history.inputs){
            if (!seen.has(val)){
                queue.push(val);
                seen.add(val);
            }
        }
    }
    return res.reverse();
}