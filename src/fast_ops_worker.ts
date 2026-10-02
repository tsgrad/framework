import { parentPort, workerData } from 'node:worker_threads';


// functions from tensor_data copied here since I don't want to deal with import issues
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
    outStride: number[];
    inBuffer: SharedArrayBuffer;
    inShape: number[];
    inStride: number[];
    aligned: boolean;
}

interface ZipTask {
    type: 'zip';
    funcstring: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStride: number[];
    aBuffer: SharedArrayBuffer;
    aShape: number[];
    aStride: number[];
    bBuffer: SharedArrayBuffer;
    bShape: number[];
    bStride: number[];
    aligned: boolean;
}

interface ReduceTask {
    type: 'red';
    funcstring: string;
    start: number;
    end: number;
    outBuffer: SharedArrayBuffer;
    outShape: number[];
    outStride: number[];
    inBuffer: SharedArrayBuffer;
    inShape: number[];
    inStride: number[];
    reduceDim: number;
    reduceDimSize: number;
    base: number;
}

interface MatMulTask {
    type: 'mul',
    start: number,
    end: number,
    outBuffer: SharedArrayBuffer,
    outStride: number[],
    aBuffer: SharedArrayBuffer,
    aStride: number[],
    bBuffer: SharedArrayBuffer,
    bStride: number[],
    m: number,
    n: number,
    k: number
}

type Task = MapTask | ZipTask | ReduceTask | MatMulTask;

const { workerId, syncBuffer } = workerData as {
    workerId: number;
    syncBuffer: SharedArrayBuffer;
};
const syncArray = new Int32Array(syncBuffer);

function reconstructFn<T>(source: string): T {
    return new Function('return ' + source)();
}

function handleMap(task: MapTask): void{
    const func = reconstructFn<(x: number) => number>(task.funcstring);
    const outStorage = new Float32Array(task.outBuffer);
    const inStorage = new Float32Array(task.inBuffer);

    if (task.aligned){
        for (let i = task.start; i < task.end; i++)
            outStorage[i] = func(inStorage[i]);
        return;
    }

    const outIndex = new Array(task.outShape.length);
    const inIndex = new Array(task.inShape.length);

    for (let i = task.start; i < task.end; i++){
        positionToIndex(i, task.outShape, outIndex);
        const outPos = indexToPosition(outIndex, task.outStride);

        broadcastIndex(outIndex, task.outShape, task.inShape, inIndex);
        const inPos = indexToPosition(inIndex, task.inStride);

        outStorage[outPos] = func(inStorage[inPos]);
    }
}

function handleZip(task: ZipTask): void{
    const func = reconstructFn<(x: number, y: number) => number>(task.funcstring);
    const outStorage = new Float32Array(task.outBuffer);
    const aStorage = new Float32Array(task.aBuffer);
    const bStorage = new Float32Array(task.bBuffer);

    if (task.aligned){
        for (let i = task.start; i < task.end; i++)
            outStorage[i] = func(aStorage[i], bStorage[i]);
        return;
    }

    const outIndex = new Array(task.outShape.length);
    const aIndex = new Array(task.aShape.length);
    const bIndex = new Array(task.bShape.length);

    for (let i = task.start; i < task.end; i++){
        positionToIndex(i, task.outShape, outIndex);
        const outPos = indexToPosition(outIndex, task.outStride);

        broadcastIndex(outIndex, task.outShape, task.aShape, aIndex);
        const aPos = indexToPosition(aIndex, task.aStride);

        broadcastIndex(outIndex, task.outShape, task.bShape, bIndex);
        const bPos = indexToPosition(bIndex, task.bStride);

        outStorage[outPos] = func(aStorage[aPos], bStorage[bPos]);
    }
}

function handleReduce(task: ReduceTask): void{
    const func = reconstructFn<(x: number, y: number) => number>(task.funcstring);
    const outStorage = new Float32Array(task.outBuffer);
    const inStorage = new Float32Array(task.inBuffer);

    const outIndex = new Array(task.outShape.length);
    for(let i = task.start; i < task.end; i++){
        positionToIndex(i, task.outShape, outIndex);
        let cur = task.base;
        let temp = outIndex[task.reduceDim];
        for (let j = 0; j < task.inShape[task.reduceDim]; j++){
            
            outIndex[task.reduceDim] = j;
            cur = func(cur, inStorage[indexToPosition(outIndex, task.inStride)]);
            
        }
        outIndex[task.reduceDim] = temp;

        outStorage[indexToPosition(outIndex, task.outStride)] = cur;
    }
}

function handleMatMul(task: MatMulTask): void{
    const outBuffer = new Float32Array(task.outBuffer);
    const aStorage = new Float32Array(task.aBuffer);
    const bStorage = new Float32Array(task.bBuffer);

    for(let i = task.start; i < task.end; i++){
        const row = Math.floor(i / task.n);
        const col = i % task.n;

        let aPos = row * task.aStride[0];
        let bPos = col * task.bStride[1];

        let sum = 0;
        for (let j = 0; j < task.k; j++){
            sum += aStorage[aPos] * bStorage[bPos];
            aPos += task.aStride[1];
            bPos += task.bStride[0];
        }

        const outPos = row * task.outStride[0] + col * task.outStride[1];
        outBuffer[outPos] = sum;
    }
}

parentPort!.on('message', (task: Task) => {
    switch (task.type) {
        case 'map':    handleMap(task);    break;
        case 'zip':    handleZip(task);    break;
        case 'red':    handleReduce(task); break;
        case 'mul':    handleMatMul(task); break;
    }

    Atomics.store(syncArray, workerId, 1);
    Atomics.notify(syncArray, workerId);
    parentPort!.postMessage('done');
});