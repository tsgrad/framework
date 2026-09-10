import { zipWith, sum, mul } from "./operators";

export type OutIndex = number[];
export type Shape = number[];
export type Stride = number[];
export type Index = number[];

export function indexToPosition(index: Index, strides: Stride): number{
    return sum(zipWith(mul)(index, strides));
}

export function toIndex(ordinal: number, shape: Shape, outIndex: OutIndex): void{
    for(let i = shape.length - 1; i >= 0; i--){
        outIndex[i] = ordinal % shape[i];
        ordinal = Math.trunc(ordinal / shape[i]);
    }
}