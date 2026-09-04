/// <reference types="vite/client" />

import GIF from "gif.js";
import gifWorkerUrl from "gif.js/dist/gif.worker.js?url";
import Plotly from "plotly.js/dist/plotly.min.js";

import { datasets } from "../../src/datasets";
import { Scalar } from "../../src/scalar";
import { Network, ScalarTrain } from "../run_scalar";
import { animate, plotOut } from "./plots";
import { Network1, Network2, Network3, Network6} from "../test_scalar"
import { SGDMomentum } from "../../src/optim";


const plotElement = document.querySelector<HTMLDivElement>("#plot");
const exportButton =
    document.querySelector<HTMLButtonElement>("#export-gif");

if (plotElement === null || exportButton === null) {
    throw new Error("Required page elements were not found");
}

type NetworkFactory = () => Network;
type FrozenModel = ReturnType<typeof snapshotModel>;

function snapshotModel(source: ScalarTrain, createNetwork: NetworkFactory) {
    const frozen = new ScalarTrain();
    frozen.model = createNetwork();

    const sourceParameters = source.model.parameters();
    const frozenParameters = frozen.model.parameters();

    if (sourceParameters.length !== frozenParameters.length) {
        throw new Error("Snapshot network architecture does not match");
    }

    frozenParameters.forEach((parameter, index) => {
        parameter.update(
            new Scalar(sourceParameters[index].value.data),
        );
    });

    return (points: [number, number][]): number[] =>
        points.map(point => frozen.runOne(point)[0].data);
}

function calculateLoss(model: FrozenModel): number {
    let loss = 0;
    for (let index = 0; index < graph.n; index++) {
        const prediction = model([graph.x[index]])[0];
        const probability = Math.min(
            1 - 1e-8,
            Math.max(1e-8, prediction),
        );
        const target = graph.y[index];
        loss -=
            target * Math.log(probability) +
            (1 - target) * Math.log(1 - probability);
    }
    return loss / graph.n;
}

function loadImage(source: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = source;
    });
}

exportButton.addEventListener("click", async () => {
    exportButton.disabled = true;

    try {
        const gif = new GIF({
            workers: 2,
            quality: 10,
            width: 600,
            height: 600,
            workerScript: gifWorkerUrl,
        });

        for (let index = 0; index < models.length; index++) {
            exportButton.textContent =
                `Rendering ${index + 1}/${models.length}`;

            const figure = plotOut(graph, models[index], "", 40);

            await Plotly.react(
                plotElement,
                figure.data,
                {
                    ...figure.layout,
                    width: 600,
                    height: 600,
                    // Remove the title and outside frame spacing.
                    title: undefined,
                    margin: {
                        t: 0,
                        r: 0,
                        b: 0,
                        l: 0,
                        pad: 0,
                    },
                    paper_bgcolor: "#ffffff",
                    plot_bgcolor: "#ffffff",
                    showlegend: false,
                    xaxis: {
                        ...figure.layout.xaxis,
                        showline: false,
                        zeroline: false,
                        showgrid: false,
                        ticks: "",
                    },
                    yaxis: {
                        ...figure.layout.yaxis,
                        showline: false,
                        zeroline: false,
                        showgrid: false,
                        ticks: "",
                    },
                    annotations: [{
                        x: 0.975,
                        y: 0.025,
                        xref: "paper",
                        yref: "paper",
                        xanchor: "right",
                        yanchor: "bottom",
                        text:
                            `<b>Epoch:</b> ${epochs[index]}` +
                            `<br><b>Loss:</b> ${losses[index].toFixed(4)}`,
                        showarrow: false,
                        align: "left",
                        font: {
                            family: "Arial, sans-serif",
                            size: 18,
                            color: "#111111",
                        },
                        bgcolor: "rgba(255, 255, 255, 0.72)",
                        borderwidth: 0,
                        borderpad: 8,
                    }],
                },
                {
                    staticPlot: true,
                },
            );

            const imageUrl = await Plotly.toImage(plotElement, {
                format: "png",
                width: 600,
                height: 600,
                scale: 1,
            });

            const image = await loadImage(imageUrl);

            gif.addFrame(image, {
                delay: 60,
                copy: true,
            });
        }

        exportButton.textContent = "Encoding GIF...";

        const blob = await new Promise<Blob>(resolve => {
            gif.on("finished", finishedBlob => {
                resolve(finishedBlob);
            });

            gif.render();
        });

        const downloadUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = downloadUrl;
        link.download = "minitorch-training.gif";
        link.click();

        URL.revokeObjectURL(downloadUrl);
    } catch (error) {
        console.error(error);
        alert("GIF export failed. Check the browser console.");
    } finally {
        exportButton.disabled = false;
        exportButton.textContent = "Export GIF";

        await animate(plotElement, graph, models, epochs);
    }
});

const PTS = 200;
const epochCount = 500;
const learningRate = 0.5;
const beta = 0.9;
const noise = 0.5;
const graph = datasets.Moon(PTS, noise);

const createNetwork = (): Network => new Network6();
const trainer = new ScalarTrain();
trainer.model = createNetwork();
trainer.optim = new SGDMomentum(trainer.model.parameters(), learningRate, beta);
const initialModel = snapshotModel(trainer, createNetwork);

const models = [initialModel];
const epochs = [0];
const losses = [calculateLoss(initialModel)];

trainer.train(
    graph,
    learningRate,
    epochCount,
    (epoch: number) => {
        const checkpoint = snapshotModel(
            trainer,
            createNetwork,
        );
        models.push(checkpoint);
        epochs.push(epoch);
        losses.push(calculateLoss(checkpoint));
    },
);

console.table(
    trainer.model.namedParameters().map(([name, parameter]) => ({
        name,
        type: name.includes(".weight_") ? "weight" : "bias",
        value: parameter.value.data,
    })),
);

await animate(plotElement, graph, models, epochs);