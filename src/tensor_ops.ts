import { Storage, Shape, Stride, indexToPosition, broadcastIndex} from "./tensor_data";
import { mul, prod } from "./operators";

export function tensorMap(func: (x: number) => number): (out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Storage, inStride: Stride) => void{
    function map(out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Shape, inStride: Stride): void{
        //simple
        if (outShape.length === inShape.length && outShape.every((dim, i) => dim === inShape[i])){
            let cells = prod(outShape);
            out.length = cells;
            let curpos: number[] = Array(outShape.length).fill(0);
            for(let i = 0; i < cells; i++){
                out[indexToPosition(curpos, outStride)] = func(inStorage[indexToPosition(curpos, inStride)]);
                let x = 0;
                curpos[0]++;
                while(x < curpos.length - 1 && curpos[x] >= outShape[x]){
                    curpos[x + 1]++;
                    curpos[x] = 0;
                    x++;
                } 
            }
        }
        else{ // broadcasted
            if (inShape.length > outShape.length || !inShape.every((dim, i) => (dim == 1 || dim == outShape[i + (outShape.length - inShape.length)])))
                throw "inShape's # of dimensions must be less than outShape, and inShape's values have to be 1 or equal to outShape's";

            let cells = prod(outShape);
            out.length = cells;
            let curpos: number[] = Array(outShape.length).fill(0);
            for(let i = 0; i < cells; i++){
                let smallIndex: Shape = [];
                broadcastIndex(curpos, outShape, inShape, smallIndex);
                out[indexToPosition(curpos, outStride)] = func(inStorage[indexToPosition(smallIndex, inStride)]);
                let x = 0;
                curpos[0]++;
                while(x < curpos.length - 1 && curpos[x] >= outShape[x]){
                    curpos[x + 1]++;
                    curpos[x] = 0;
                    x++;
                } 
            }
        }
    }
    return map;
}

export function tensorZip(func: (x: number, y: number) => number): (out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride) => void{
    function zip(out: Storage, outShape: Shape, outStride: Stride, aStorage: Storage, aShape: Shape, aStride: Stride, bStorage: Storage, bShape: Shape, bStride: Stride): void{
        //simple
        if (outShape.length === aShape.length && outShape.every((dim, i) => dim === aShape[i] && dim === bShape[i])){
            let cells = prod(outShape);
            out.length = cells;
            let curpos: number[] = Array(outShape.length).fill(0);
            for(let i = 0; i < cells; i++){
                out[indexToPosition(curpos, outStride)] = func(aStorage[indexToPosition(curpos, aStride)], bStorage[indexToPosition(curpos, bStride)]);
                let x = 0;
                curpos[0]++;
                while(x < curpos.length - 1 && curpos[x] >= outShape[x]){
                    curpos[x + 1]++;
                    curpos[x] = 0;
                    x++;
                } 
            }
        }
        else{ // broadcasted
            if ((aShape.length > outShape.length || !aShape.every((dim, i) => (dim == 1 || dim == outShape[i + (outShape.length - aShape.length)])))
            && (bShape.length > outShape.length || !bShape.every((dim, i) => (dim == 1 || dim == outShape[i + (outShape.length - bShape.length)]))))
                throw "aShape's # of dimensions must be less than outShape, and aShape's values have to be 1 or equal to outShape's";

            let cells = prod(outShape);
            out.length = cells;
            let curpos: number[] = Array(outShape.length).fill(0);
            for(let i = 0; i < cells; i++){
                let aIndex: Shape = [];
				let bIndex: Shape = [];
                broadcastIndex(curpos, outShape, aShape, aIndex);
                broadcastIndex(curpos, outShape, bShape, bIndex);
                out[indexToPosition(curpos, outStride)] = func(aStorage[indexToPosition(aIndex, aStride)], bStorage[indexToPosition(bIndex, bStride)]);
                let x = 0;
                curpos[0]++;
                while(x < curpos.length - 1 && curpos[x] >= outShape[x]){
                    curpos[x + 1]++;
                    curpos[x] = 0;
                    x++;
                } 
            }
        }
    }
    return zip;
}
