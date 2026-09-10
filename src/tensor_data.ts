import { zipWith, sum, mul } from "./operators";

export type OutIndex = number[];
export type Shape = number[];
export type Stride = number[];
export type Index = number[];
export type Storage = number[];

export function indexToPosition(index: Index, strides: Stride): number{
    return sum(zipWith(mul)(index, strides));
}

export function positionToIndex(ordinal: number, shape: Shape, outIndex: OutIndex): void{
    for(let i = shape.length - 1; i >= 0; i--){
        outIndex[i] = ordinal % shape[i];
        ordinal = Math.trunc(ordinal / shape[i]);
    }
}

class TensorData{
    _storage: Storage;
    _shape: Shape;
    _stride: Stride;
    constructor(storage: Storage, shape: Shape, stride: Stride){
        this._storage = storage;
        this._shape = shape;
        this._stride = stride;
    }

    permute(...order: number[]): TensorData{
        if (order.length != this._shape.length)
            throw "Must give a position to each dimension but no more";

        let freq: number[] = Array(order.length).fill(0);
        for (let i = 0; i < order.length; i++){
            if (order[i] < 0 || order[i] >= order.length)
                throw "Order value: " + order[i] + " outside bounds of shape.length :";
            freq[order[i]]++;
        }

        let newShape: number[] = Array(order.length).fill(0);
        let newStride: number[] = Array(order.length).fill(0);
        for (let i = 0; i < order.length; i++){
            if (freq[i] != 1)
                throw "Dimension " + i + " is referenced more or less than once";

            newShape[i] = this._shape[order[i]];
            newStride[i] = this._stride[order[i]];
        }
        return new TensorData(this._storage, newShape, newStride);
    }
}