import { zipWith, sum, mul, max, prod } from "./operators.js";
import { getBackend } from "./backend/index.js";
import type { BackendStorage } from "./backend/device_backend.js";

export type OutIndex = number[];
export type Shape = number[];
export type Stride = number[];
export type Index = number[];
export type Storage = Float32Array;
export type TensorBuffer = BackendStorage;

export type UserShape = number[];
export type UserStrides = number[];

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
            smallIndex.push(bigIndex[i + diff]);
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

export function readStorage(storage: TensorBuffer): Float32Array {
    return getBackend().copyToHost(storage);
}

export function readStorageAsync(storage: TensorBuffer): Promise<Float32Array> {
    return getBackend().copyToHostAsync(storage);
}

export function writeStorage(storage: TensorBuffer, values: ArrayLike<number>): void {
    const src = values instanceof Float32Array ? values : Float32Array.from(values);
    const backend = getBackend();
    backend.copyFromHost(storage, src);
}

export class TensorData{
    _storage: TensorBuffer;
    _shape: Shape;
    _stride: Stride;
    strides: UserStrides;
    shape: UserShape;
    dims: number;
    size: number;
    private _released = false;

    constructor(storage: TensorBuffer, shape: Shape, strides: Stride = TensorData.calculateStrides(shape), retain = false){
        this._shape = shape;
        this._stride = strides;

        this.strides = strides;
        this.dims = strides.length;
        this.size = prod(shape);
        this.shape = shape;

        const backend = getBackend();
        if (storage instanceof Float32Array && backend.kind !== 'cpu') {
            if (storage.length !== this.size) throw "_storage.length not equal to this.size";
            const handle = backend.empty(this.size);
            if (this.size > 0) backend.copyFromHost(handle, storage);
            this._storage = handle;
        } else {
            if (!backend.isStorage(storage)) throw new Error(`Storage does not belong to the active ${backend.kind} backend`);
            if (storage instanceof Float32Array && storage.length !== this.size) throw "_storage.length not equal to this.size";
            if (retain) backend.retain(storage);
            this._storage = storage;
        }
    }

    static share(storage: TensorBuffer, shape: Shape, strides: Stride = TensorData.calculateStrides(shape)): TensorData {
        return new TensorData(storage, shape, strides, true);
    }

    static fill(shape: Shape, num: number = 0): TensorData{
        let size = prod(shape);
        return new TensorData(getBackend().alloc(size, num), shape, TensorData.calculateStrides(shape));
    }

    static empty(shape: Shape): TensorData{
        let size = prod(shape);
        return new TensorData(getBackend().empty(size), shape, TensorData.calculateStrides(shape));
    }

    release(): void {
        if (this._released) return;
        this._released = true;
        getBackend().release(this._storage);
    }

    static calculateStrides(shape: Shape): Stride{
        let res: number[] = new Array(shape.length);
        let stride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            res[i] = stride;
            stride *= shape[i]!;
        }
        return res;
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
        return TensorData.share(this._storage, newShape, newStride);
    }
}