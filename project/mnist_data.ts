import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";

export type Image = number[][];
export type OneHotLabel = number[];
export type ImageData = [Image[], OneHotLabel[]];

export async function loadMnist(directory: string, split: "train" | "t10k", start = 0, count = Infinity): Promise<ImageData> {
    const imageBuffer = gunzipSync(await readFile(join(directory, `${split}-images-idx3-ubyte.gz`)));
    const labelBuffer = gunzipSync(await readFile(join(directory, `${split}-labels-idx1-ubyte.gz`)));

    if (imageBuffer.readUInt32BE(0) !== 2051) throw "Invalid MNIST image file";
    if (labelBuffer.readUInt32BE(0) !== 2049) throw "Invalid MNIST label file";

    const imageCount = imageBuffer.readUInt32BE(4);
    const labelCount = labelBuffer.readUInt32BE(4);
    const rows = imageBuffer.readUInt32BE(8);
    const columns = imageBuffer.readUInt32BE(12);

    if (imageCount !== labelCount) throw "Image and label counts differ";
    if (rows !== 28 || columns !== 28) throw `Expected 28x28 images, received ${rows}x${columns}`;

    const end = Math.min(imageCount, start + count);
    const images: Image[] = [];
    const labels: OneHotLabel[] = [];

    for (let index = start; index < end; index++) {
        const imageStart = 16 + index * rows * columns;
        const image: Image = [];

        for (let row = 0; row < rows; row++) {
            const values: number[] = [];

            for (let column = 0; column < columns; column++) {
                const pixel = imageBuffer[imageStart + row * columns + column];

                // Normalize from 0–255 to 0–1.
                values.push(pixel / 255);
            }

            image.push(values);
        }

        const digit = labelBuffer[8 + index];
        const oneHot = Array<number>(10).fill(0);
        oneHot[digit] = 1;

        images.push(image);
        labels.push(oneHot);
    }

    return [images, labels];
}