import Plotly from "plotly.js/dist/plotly.min.js";
import type {
    Data,
    Layout,
    PlotlyHTMLElement,
    Root,
} from "plotly.js";

import { Graph } from "../../src/datasets";

interface GraphBounds {
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
}

function getGraphBounds(graph: Graph, paddingFraction = 0.05): GraphBounds {
    if (graph.x.length === 0) {
        return {
            xMin: 0,
            xMax: 1,
            yMin: 0,
            yMax: 1,
        };
    }

    const xs = graph.x.map(point => point[0]);
    const ys = graph.x.map(point => point[1]);

    const rawXMin = Math.min(...xs);
    const rawXMax = Math.max(...xs);
    const rawYMin = Math.min(...ys);
    const rawYMax = Math.max(...ys);

    const xSpan = rawXMax - rawXMin || 1;
    const ySpan = rawYMax - rawYMin || 1;

    const xPadding = xSpan * paddingFraction;
    const yPadding = ySpan * paddingFraction;

    return {
        xMin: rawXMin - xPadding,
        xMax: rawXMax + xPadding,
        yMin: rawYMin - yPadding,
        yMax: rawYMax + yPadding,
    };
}

function graphLayout(graph: Graph): Partial<Layout> {
    const bounds = getGraphBounds(graph);

    return {
        xaxis: {
            showgrid: false,
            zeroline: false,
            visible: false,
            range: [bounds.xMin, bounds.xMax],
        },
        yaxis: {
            showgrid: false,
            zeroline: false,
            visible: false,
            range: [bounds.yMin, bounds.yMax],
        },
    };
}

type Model = (points: [number, number][]) => number[];

export interface PlotSpec {
    data: Data[];
    layout: Partial<Layout>;
}

export function makeScatters(graph: Graph, model?: Model, size = 50): Data[] {
    const colorMap = ["#004cff", "#ff2020"];
    const symbolMap = ["circle-dot", "x"];

    const colors = graph.y.map(label => colorMap[label]);
    const symbols = graph.y.map(label => symbolMap[label]);
    const scatters: Data[] = [];

    if (model !== undefined) {
        const bounds = getGraphBounds(graph);
        const gridX = Array.from(
            { length: size + 1 },
            (_, index) =>
                bounds.xMin +
                (index / size) * (bounds.xMax - bounds.xMin),
        );
        const gridY = Array.from(
            { length: size + 1 },
            (_, index) =>
                bounds.yMin +
                (index / size) * (bounds.yMax - bounds.yMin),
        );
        const z = gridY.map(y =>
            model(gridX.map(x => [x, y] as [number, number])),
        );
        scatters.push({
            type: "heatmap",
            x: gridX,
            y: gridY,
            z,
            zmin: 0.1,
            zmax: 0.9,
            zmid: 0.5,
            colorscale: [
                [0.00, "#0000ff"],
                [0.25, "#3f6fff"],
                [0.40, "#dce7ff"],
                [0.48, "#fafcff"],
                [0.50, "#ffffff"],
                [0.52, "#fffafa"],
                [0.60, "#ffe0e0"],
                [0.75, "#ff5c5c"],
                [1.00, "#ff0000"],
            ],
            zsmooth: "best",
            opacity: 0.65,
            showscale: false,
            hoverinfo: "skip",
        });
    }

    scatters.push({
        type: "scatter",
        mode: "markers",
        x: graph.x.map(point => point[0]),
        y: graph.x.map(point => point[1]),
        marker: {
            symbol: symbols,
            color: colors,
            size: 15,
            line: {
                width: 3,
                color: "black",
            },
            opacity: 0.9,
        },
    });

    return scatters;
}

function makeOneD(graph: Graph, model?: Model, size = 50): Data[] {
    const colorMap = ["#004cff", "#ff2020"];
    const symbolMap = ["circle-dot", "x"];

    const colors = graph.y.map(label => colorMap[label]);
    const symbols = graph.y.map(label => symbolMap[label]);
    const scatters: Data[] = [];

    if (model !== undefined) {
        const points: [number, number][] = Array.from(
            { length: size + 1 },
            (_, j) => [j / (size + 1), 0],
        );

        scatters.push({
            type: "scatter",
            mode: "lines",
            x: points.map(point => point[0]),
            y: model(points),
        });
    }

    scatters.push({
        type: "scatter",
        mode: "markers",
        x: graph.x.map(point => point[0]),
        y: graph.y,
        marker: {
            symbol: symbols,
            color: colors,
            size: 15,
            line: {
                width: 3,
                color: "black",
            },
        },
    });

    return scatters;
}

export function animate(target: Root, graph: Graph, models: Model[], names: number[]): Promise<PlotlyHTMLElement> {
    if (models.length === 0) {
        throw new Error("animate requires at least one model");
    }

    if (models.length !== names.length) {
        throw new Error("models and names must have the same length");
    }

    const backgrounds = models.map((model, index) => ({
        ...makeScatters(graph, model)[0],
        visible: index === 0,
    }) as Data);

    const points = makeScatters(graph)[0];

    const steps = names.map((name, index) => {
        const visible = [
            ...backgrounds.map(() => false),
            true,
        ];

        visible[index] = true;

        return {
            method: "update",
            args: [{ visible }, {}],
            label: name.toString(),
        };
    });

    const layout: Partial<Layout> = {
        ...graphLayout(graph),
        sliders: [{
            active: 0,
            currentvalue: { prefix: "Epoch " },
            pad: { t: 50 },
            steps: steps as any,
        }],
    };

    return Plotly.newPlot(
        target,
        [...backgrounds, points],
        layout,
    );
}

export function plotOut(graph: Graph, model?: Model, _name = "", size = 50, oneD = false): PlotSpec {
    return {
        data: oneD
            ? makeOneD(graph, model, size)
            : makeScatters(graph, model, size),
        layout: graphLayout(graph),
    };
}