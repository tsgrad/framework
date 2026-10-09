import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync } from 'node:fs';

import { Storage, Shape, Stride, indexToPosition, broadcastIndex, positionToIndex} from "./tensor_data.js";
import { prod } from './operators.js';


const MAX_NUM_WORKERS = availableParallelism();
const THRESHOLD = 2048;

class WorkerPool {
    private workers: Worker[];
    private syncBuffer: SharedArrayBuffer;
    private syncArray: Int32Array;


    constructor(numWorkers: number) {
        this.syncBuffer = new SharedArrayBuffer(numWorkers * Int32Array.BYTES_PER_ELEMENT);
        this.syncArray = new Int32Array(this.syncBuffer);

        const currentDir = dirname(fileURLToPath(import.meta.url));
        const tsPath = join(currentDir, 'fast_ops_worker.ts');
        const jsPath = join(currentDir, 'fast_ops_worker.js');
        const workerPath = existsSync(tsPath) ? tsPath : jsPath;
        
        this.workers = Array.from({length: numWorkers}, (_, id) => {
            const worker = new Worker(workerPath, {workerData: {workerId: id, syncBuffer: this.syncBuffer}});
            worker.on('error', (err: any) => {console.error(`[fast_ops] Worker ${id} error:`, err)});
            return worker;
        });
    }

    parallelFor(size: number, taskFactory: (start: number, end: number) => object): void{
        const chunkSize = Math.ceil(size / this.workers.length);

        // Reset all of the workers to unused
        for (let i = 0; i < this.workers.length; i++)
            Atomics.store(this.syncArray, i, 0);

        // Dispatch all of the workers
        for (let i = 0; i < this.workers.length; i++){
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, size);

            // if we ran out of work for a worker to do
            if (start >= size)
                // Pat it on the back said it did a good job and let it have a break
                Atomics.store(this.syncArray, i, 1);
            else // else off to work it goes
                this.workers[i].postMessage(taskFactory(start, end));
        }

        // Wait for all the workers to be done
        for (let i = 0; i < this.workers.length; i++)
            // Wait for each worker in order
            while (Atomics.load(this.syncArray, i) === 0)
                Atomics.wait(this.syncArray, i, 0, 120000); // You got 2 minutes or we give up on you
    }

    terminate(): void{
        for (let i = 0; i < this.workers.length; i++)
            this.workers[i].terminate();
    }
};

let _pool : WorkerPool | null | undefined = undefined;

const PARALLEL_DISABLED = typeof process !== 'undefined' && (process.env['VITEST'] !== undefined || process.env['TSGRAD_DISABLE_PARALLEL'] !== undefined);

function getPool(): WorkerPool | null {
    if (PARALLEL_DISABLED) return null;
    if (_pool === undefined) {
        try {
            _pool = new WorkerPool(MAX_NUM_WORKERS);
        } catch {
            _pool = null;
        }
    }
    return _pool;
}

export function destroyPool(): void{
    if (_pool){
        _pool.terminate();
        _pool = undefined;
    }
}

function isShared(storage: Storage): boolean{
    return storage.buffer instanceof SharedArrayBuffer;
}

function shapeAligned(a: Shape, b: Shape){
    return a.length == b.length && a.every((val: number, i: number) => val === b[i]);
}

function strideAligned(a: Stride, b: Stride){
    return a.length == b.length && a.every((val: number, i: number) => val === b[i]);
}

export function fastTensorMap(func: (x: number) => number): (out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Shape, inStride: Stride) => void{
    function map(out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Shape, inStride: Stride): void{
        const size = prod(outShape);
        const aligned: boolean = shapeAligned(outShape, inShape) && strideAligned(outStride, inStride);
        const pool = getPool();

        // Will only be parallel if we have a worker pool, the tensors are big enough to make it worth doing, and both out and inStorage are shared
        if (pool && size >= THRESHOLD && isShared(out) && isShared(inStorage)){
            const funcstring = func.toString();
            pool.parallelFor(size, (start, end) => ({
                type: 'map',
                funcstring,
                start,
                end,
                outBuffer: out.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStride: Array.from(outStride),
                inBuffer: inStorage.buffer as SharedArrayBuffer,
                inShape: Array.from(inShape),
                inStride: Array.from(inStride),
                aligned,
            }));
            return;
        }

        if (aligned) {
            for (let i = 0; i < size; i++)
                out[i] = func(inStorage[i]!);
            return;
        }

        const outIndex = new Array(outShape.length);
        const inIndex = new Array(inShape.length);

        for (let i = 0; i < size; i++){
            positionToIndex(i, outShape, outIndex);
            const outPos = indexToPosition(outIndex, outStride);

            broadcastIndex(outIndex, outShape, inShape, inIndex);
            const inPos = indexToPosition(inIndex, inStride);

            out[outPos] = func(inStorage[inPos]);
        }
    }

    return map;
}

export function fastTensorZip(func: (x: number, y: number) => number): (out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride) => void{
    function zip(out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride): void{
        const size = prod(outShape);
        const aligned: boolean = shapeAligned(outShape, aShape) && strideAligned(outStride, aStride) && shapeAligned(outShape, bShape) && strideAligned(outStride, bStride);
        const pool = getPool();
        
        // Will only be parallel if we have a worker pool, the tensors are big enough to make it worth doing, and all 3 are shared
        if (pool && size >= THRESHOLD && isShared(out) && isShared(aStorage) && isShared(bStorage)){
            const funcstring = func.toString();
            pool.parallelFor(size, (start, end) => ({
                type: 'zip',
                funcstring,
                start,
                end,
                outBuffer: out.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStride: Array.from(outStride),
                aBuffer: aStorage.buffer as SharedArrayBuffer,
                aShape: Array.from(aShape),
                aStride: Array.from(aStride),
                bBuffer: bStorage.buffer as SharedArrayBuffer,
                bShape: Array.from(bShape),
                bStride: Array.from(bStride),
                aligned,
            }));
            return;
        }

        if (aligned) {
            for (let i = 0; i < size; i++)
                out[i] = func(aStorage[i], bStorage[i]);
            return;
        }
        
        const outIndex = new Array(outShape.length);
        const aIndex = new Array(aShape.length);
        const bIndex = new Array(bShape.length);
        for (let i = 0; i < size; i++){
            positionToIndex(i, outShape, outIndex);
            const outPos = indexToPosition(outIndex, outStride);

            broadcastIndex(outIndex, outShape, aShape, aIndex);
            const aPos = indexToPosition(aIndex, aStride);

            broadcastIndex(outIndex, outShape, bShape, bIndex);
            const bPos = indexToPosition(bIndex, bStride);

            out[outPos] = func(aStorage[aPos], bStorage[bPos]);
        }
    }
    return zip;
}

export function fastTensorReduce(func: (x: number, y: number) => number, base: number): (out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, reduceDim: number) => void {
    function reduce(out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, reduceDim: number): void{
        if (reduceDim < 0 || reduceDim >= aShape.length)
            throw "Invalid reduceDim, reduceDim must fall within bounds of aShape, got reduceDim: " + reduceDim + ", and aShape.length: " + aShape.length;

        const size = prod(outShape);
        const reduceDimSize = aShape[reduceDim];
        const pool = getPool();

        // Will only be parallel if we have a worker pool, the tensors are big enough to make it worth doing, and both are shared
        if (pool && size >= THRESHOLD && isShared(out) && isShared(aStorage)){
            const funcstring = func.toString();
            pool.parallelFor(size, (start, end) => ({
                type: 'red',
                funcstring,
                start,
                end,
                outBuffer: out.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStride: Array.from(outStride),
                inBuffer: aStorage.buffer as SharedArrayBuffer,
                inShape: Array.from(aShape),
                inStride: Array.from(aStride),
                reduceDim,
                reduceDimSize,
                base
            }));
            return;
        }

        let curpos: number[] = Array(outShape.length).fill(0);
        for(let i = 0; i < size; i++){
            let cur = base;
            let temp = curpos[reduceDim];
            for (let j = 0; j < aShape[reduceDim]; j++){
                
                curpos[reduceDim] = j;
                cur = func(cur, aStorage[indexToPosition(curpos, aStride)]);
                
            }
            curpos[reduceDim] = temp;

            out[indexToPosition(curpos, outStride)] = cur;
            let x = 0;
            curpos[0]++;
            while(x < curpos.length - 1 && curpos[x] >= outShape[x]){
                curpos[x + 1]++;
                curpos[x] = 0;
                x++;
            } 
        }
    }
    return reduce;
}

function shapesCanBeMul(aShape: Shape, bShape: Shape){
    if (aShape.length != bShape.length || aShape.length < 2) return false;
    
    for (let i = 0; i < aShape.length - 2; i++){
        if (aShape[i] != bShape[i])
            return false;
    }
    
    return aShape[aShape.length - 1] == bShape[bShape.length - 2];
}

export function fastMatMul(out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride){
    if (!shapesCanBeMul(aShape, bShape))
        throw `Incompatible shapes for matrix mul, a: ${aShape}, b: ${bShape}`;
    
    if (aShape.length !== 2)
        throw "Shapes length must be 2 for now";

    const m = aShape[0];
    const k = aShape[1];
    const bK = bShape[0];
    const n = bShape[1];

    const size = m * n * k;
    const outSize = m * n;

    

    if (size >= THRESHOLD && isShared(out) && isShared(aStorage) && isShared(bStorage)){
        const pool = getPool();
        if (pool){
            pool.parallelFor(outSize, (start, end) => ({
                type: 'mul',
                start,
                end,
                outBuffer: out.buffer as SharedArrayBuffer,
                outStride: Array.from(outStride),
                aBuffer: aStorage.buffer as SharedArrayBuffer,
                aStride: Array.from(aStride),
                bBuffer: bStorage.buffer as SharedArrayBuffer,
                bStride: Array.from(bStride),
                m,
                n,
                k
            }));
            return;
        }
    }

    for (let i = 0; i < outSize; i++){
        // left to right top to bottom
        const row = Math.floor(i / n);
        const column = i % n;

        let aPos = row * aStride[0];
        let bPos = column * bStride[1];
        let sum = 0;

        for (let j = 0; j < k; j++){
            sum += aStorage[aPos] * bStorage[bPos];
            aPos += aStride[1];
            bPos += bStride[0];
        }
        const outPos = row * outStride[0] + column * outStride[1];
        out[outPos] = sum;
    }
}

function fastConv1d(out: Storage, outShape: Shape, outStride: Stride, outSize: number,
    input: Storage, inputShape: Shape, inputStride: Stride,
    weight: Storage, weightShape: Shape, weightStride: Stride,
    reverse: boolean): void{
    /*
        Given input tensor shape of

        `batch, inChannels, width`

        and weight tensor shape of

        `outChannels, inChannels, kWidth`

        Computes padded output of

        `batch, outChannels, width`
    */

    if (inputShape.length !== 3) throw "unexpected inputShape size, expected 3 got: " + inputShape.length;

    const batch = inputShape[0];
    const inChannels = inputShape[1];
    const width = inputShape[2]; // width is the literal length of the input array
    const outChannels = weightShape[0];
    const kWidth = weightShape[2]; // k width is the length of the kernel

    if (inChannels !== weightShape[1]) throw "in channels must be same between both input shape and weight shape";
    if (outShape[0] !== batch || outShape[1] !== outChannels || outShape[2] !== width) throw "output shape is not the expected shape";


    const size = batch * outChannels * width;
    
    if (size >= THRESHOLD && isShared(out) && isShared(input) && isShared(weight)){
        const pool = getPool();
        if (pool){
            pool.parallelFor(size, (start, end) => ({
                type: 'conv1d',
                start,
                end,
                outBuffer: out.buffer as SharedArrayBuffer,
                outShape: Array.from(outShape),
                outStride: Array.from(outStride),
                inBuffer: input.buffer as SharedArrayBuffer,
                inStride: Array.from(inputStride),
                weightBuffer: weight.buffer as SharedArrayBuffer,
                weightStride: Array.from(weightStride),
                reverse,
                inChannels,
                kWidth
            }));
            return;
        }
    }

    let inpos = [0, 0, 0];
    let weightpos = [0, 0, 0];
    let outpos = [0, 0, 0];
    for (let b = 0; b < batch; b++){
        inpos[0] = b;
        outpos[0] = b;
        for (let oc = 0; oc < outChannels; oc++){
            outpos[1] = oc;
            weightpos[0] = oc;
            for (let w = 0; w < width; w++){
                outpos[2] = w;
                let total = 0.0;
                for (let ic = 0; ic < inChannels; ic++){
                    inpos[1] = ic;
                    weightpos[1] = ic;
                    for (let k = 0; k < kWidth; k++){
                        let x = reverse ? w - k : w + k;
                        inpos[2] = x;
                        weightpos[2] = k;
                        if (x >= 0 && x < width) // if inbounds add it, else add 0 (which is the same as doing nothing)
                            total += input[indexToPosition(inpos, inputStride)] * weight[indexToPosition(weightpos, weightStride)];
                    }
                }
                out[indexToPosition(outpos, outStride)] = total;
            }
        }
    }
}