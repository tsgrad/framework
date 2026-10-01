import { parentPort, workerData } from 'node:worker_threads';


// functions from tensor_data copied here since we can't import relative files here, also changed the implementations so we dont have to
// this is because a node worker has its own v8 isolate, javascript heap, and other things so the worker doesn't know what path we came from, thus no imports
export function indexToPosition(index: number[], strides: number[]): number{
    let res = 0;
    for (let i = 0; i < index.length; i++)
        res += index[i] * strides[i];
    return res;
}

export function positionToIndex(ordinal: number, shape: number[], outIndex: number[]): void{
    for(let i = shape.length - 1; i >= 0; i--){
        outIndex[i] = ordinal % shape[i];
        ordinal = Math.trunc(ordinal / shape[i]);
    }
}

export function broadcastIndex(bigIndex: number[], bigShape: number[], smallShape: number[], smallIndex: number[]): void{
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

interface MapTask {
    type: 'map';
    funcstring: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStrides: number[];
    inBuffer: SharedArrayBuffer;
    inShape: number[];
    inStrides: number[];
    aligned: boolean;
}

interface ZipTask {
    type: 'zip';
    funcstring: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStrides: number[];
    aBuffer: SharedArrayBuffer;
    aShape: number[];
    aStrides: number[];
    bBuffer: SharedArrayBuffer;
    bShape: number[];
    bStrides: number[];
    aligned: boolean;
}

interface ReduceTask {
    type: 'red';
    funcstring: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStrides: number[];
    inBuffer: SharedArrayBuffer;
    inShape: number[];
    inStrides: number[];
    reduceDim: number;
    reduceDimSize: number;
}