export function centralDifference(f: Function, vals: number[], arg: number = 0, epsilon: number = 1e-6){
    let original = vals[arg];
    vals[arg] = original + epsilon;
    let v1 = f(...vals);
    vals[arg] = original - epsilon;
    let v2 = f(...vals);
    vals[arg] = original;

    return (v1 - v2) / (2 * epsilon);
}