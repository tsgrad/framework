import { Storage, Shape, Stride, indexToPosition, broadcastIndex} from "./tensor_data";
import { mul, prod } from "./operators";

export function tensorMap(func: (x: number) => number): (out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Storage, inStride: Stride) => void{
    function map(out: Storage, outShape: Shape, outStride: Stride, inStorage: Storage, inShape: Storage, inStride: Stride): void{
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
