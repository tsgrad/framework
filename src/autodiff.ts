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

    saveForBackward(values: any[]): void {
        if (this.noGrad)
            return;
        this.savedValues = values;
    }

    savedTensors(): any[]{
        return this.savedValues;
    }
}