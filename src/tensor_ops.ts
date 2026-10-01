import { Storage, Shape, Stride, indexToPosition, broadcastIndex, positionToIndex} from "./tensor_data.js";
import { id, isClose, reluBack, invBack, logBack, add, eq, exp, inv, leakyrelu, log, lt, mul, neg, relu, sigmoid, prod } from "./operators.js";
import { Tensor } from "./tensor.js";

export function tensorMap(func: (x: number) => number): (out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Shape, inStride: Stride) => void{
    function map(out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Shape, inStride: Stride): void{
        const outIndex = new Array(outShape.length);
        const inIndex = new Array(inShape.length);
        const size = prod(outShape);

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

export function tensorZip(func: (x: number, y: number) => number): (out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride) => void{
    function zip(out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride): void{
        const outIndex = new Array(outShape.length);
        const aIndex = new Array(aShape.length);
        const bIndex = new Array(bShape.length);
        const size = prod(outShape);
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

export function tensorReduce(func: (x: number, y: number) => number, base: number): (out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, reduceDim: number) => void {
    function reduce(out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, reduceDim: number): void{
        if (reduceDim < 0 || reduceDim >= aShape.length)
            throw "Invalid reduceDim, reduceDim must fall within bounds of aShape, got reduceDim: " + reduceDim + ", and aShape.length: " + aShape.length;

        let cells = prod(outShape);
        let curpos: number[] = Array(outShape.length).fill(0);
        for(let i = 0; i < cells; i++){
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