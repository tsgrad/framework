import { Context } from "./autodiff";

export class History{
    lastFn : Function | undefined;
    ctx: Context | undefined;
    inputs: Tensor[];

    constructor(lstFn: Function | undefined, ct: Context | undefined, inpts: Tensor[]){
        this.lastFn = lstFn;
        this.ctx = ct;
        this.inputs = inpts;
    }
}

export class Tensor{

}