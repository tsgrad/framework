import { zipWith, sum, mul, max } from "./operators";

export type OutIndex = number[];
export type Shape = number[];
export type Stride = number[];
export type Index = number[];
export type Storage = number[];

export type UserShape = number[];

export function indexToPosition(index: Index, strides: Stride): number{
    return sum(zipWith(mul)(index, strides));
}

export function positionToIndex(ordinal: number, shape: Shape, outIndex: OutIndex): void{
    for(let i = shape.length - 1; i >= 0; i--){
        outIndex[i] = ordinal % shape[i];
        ordinal = Math.trunc(ordinal / shape[i]);
    }
}

export function broadcastIndex(bigIndex: Index, bigShape: Shape, smallShape: Shape, smallIndex: OutIndex): void{
    // Turns bigIndex into smallIndex, modifies smallIndex in place
    smallIndex.length = 0; // Reset it just incase there was garbage in it before
    let m = bigShape.length;
    let n = smallShape.length;
    let diff = m - n;
    for (let i = 0; i < smallShape.length; i++){
        if (smallShape[i] === 1)
            smallIndex.push(0);
        else if (smallShape[i] !== bigShape[i + diff])
            throw "smallShape dimension " + i + " must be 1 or equal to bigShape dimension " + (i + diff);
        else
            smallIndex.push(smallShape[i]);
    }
}

export function shapeBroadcast(shape1: UserShape, shape2: UserShape): UserShape{
    let i = shape1.length - 1, j = shape2.length - 1;
    let res: UserShape = Array(max(shape1.length, shape2.length)).fill(0);
    let k = res.length - 1;
    for (; i >= 0 && j >= 0; i--, j--, k--){
        if (shape1[i] === 1)
            res[k] = shape2[j];
        else if (shape2[j] !== 1 && shape1[i] !== shape2[j])
            throw "shape1 dimension " + i + " must be 1 or equal to shape2 dimension " + j;
        else
            res[k] = shape1[i];
    }

    for(; i >= 0; i--, k--)
        res[k] = shape1[i];

    for(; j >= 0; j--, k--)
        res[k] = shape2[j];

    return res;
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