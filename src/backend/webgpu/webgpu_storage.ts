export class WebGPUStorage {
    owners = 1;
    released = false;

    constructor(
        readonly buffer: GPUBuffer,
        readonly numel: number,
        readonly byteSize: number,
    ) {}
}
