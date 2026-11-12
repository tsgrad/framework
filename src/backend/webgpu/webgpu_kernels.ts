import type { BinaryOp, ReduceOp, UnaryOp } from '../device_backend.js';

export const WORKGROUP_SIZE = 256;
export const MAX_DIMS = 16;

const unaryBody: Record<UnaryOp, string> = {
    neg: 'return -x;',
    inv: 'return 1.0 / x;',
    sigmoid: 'return select(exp(x) / (1.0 + exp(x)), 1.0 / (1.0 + exp(-x)), x >= 0.0);',
    relu: 'return max(x, 0.0);',
    log: 'return log(x);',
    exp: 'return exp(x);',
    id: 'return x;',
};

const binaryBody: Record<BinaryOp, string> = {
    add: 'return a + b;',
    sub: 'return a - b;',
    mul: 'return a * b;',
    div: 'return a / b;',
    lt: 'return select(0.0, 1.0, a < b);',
    eq: 'return select(0.0, 1.0, a == b);',
    isClose: 'return select(0.0, 1.0, abs(a - b) < 0.01);',
};

const reduceInfo: Record<ReduceOp, { body: string; identity: string }> = {
    sum: { body: 'return a + b;', identity: '0.0' },
    prod: { body: 'return a * b;', identity: '1.0' },
    max: { body: 'return max(a, b);', identity: '-3.402823466e+38' },
};

const INDEX_HELPERS = `
const MAX_DIMS: u32 = ${MAX_DIMS}u;

fn to_index(ordinal_in: u32, shape_offset: u32, rank: u32, out_index: ptr<function, array<u32, ${MAX_DIMS}>>) {
    var ordinal = ordinal_in;
    for (var rev: u32 = 0u; rev < rank; rev++) {
        let i = rank - 1u - rev;
        let size = params[shape_offset + i];
        (*out_index)[i] = ordinal % size;
        ordinal = ordinal / size;
    }
}

fn index_position(index: array<u32, ${MAX_DIMS}>, stride_offset: u32, rank: u32) -> u32 {
    var position = 0u;
    for (var i = 0u; i < rank; i++) {
        position += index[i] * params[stride_offset + i];
    }
    return position;
}

fn broadcast_position(big_index: array<u32, ${MAX_DIMS}>, big_rank: u32, shape_offset: u32, stride_offset: u32, rank: u32) -> u32 {
    let diff = big_rank - rank;
    var position = 0u;
    for (var i = 0u; i < rank; i++) {
        let index = select(big_index[i + diff], 0u, params[shape_offset + i] == 1u);
        position += index * params[stride_offset + i];
    }
    return position;
}
`;

export function alignedMapShader(op: UnaryOp): string {
    return `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
struct Params { size: u32 }
@group(0) @binding(2) var<uniform> p: Params;
fn apply(x: f32) -> f32 { ${unaryBody[op]} }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < p.size) { output_data[gid.x] = apply(input_data[gid.x]); }
}`;
}

export function stridedMapShader(op: UnaryOp): string {
    return `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
${INDEX_HELPERS}
fn apply(x: f32) -> f32 { ${unaryBody[op]} }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params[0]) { return; }
    let out_rank = params[1];
    let in_rank = params[2];
    var index: array<u32, ${MAX_DIMS}>;
    to_index(gid.x, 4u, out_rank, &index);
    let out_pos = index_position(index, ${4 + MAX_DIMS}u, out_rank);
    let in_pos = broadcast_position(index, out_rank, ${4 + MAX_DIMS * 2}u, ${4 + MAX_DIMS * 3}u, in_rank);
    output_data[out_pos] = apply(input_data[in_pos]);
}`;
}

export function alignedZipShader(op: BinaryOp): string {
    return `
@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
struct Params { size: u32 }
@group(0) @binding(3) var<uniform> p: Params;
fn apply(a: f32, b: f32) -> f32 { ${binaryBody[op]} }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < p.size) { output_data[gid.x] = apply(a_data[gid.x], b_data[gid.x]); }
}`;
}

export function stridedZipShader(op: BinaryOp): string {
    return `
@group(0) @binding(0) var<storage, read> a_data: array<f32>;
@group(0) @binding(1) var<storage, read> b_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
${INDEX_HELPERS}
fn apply(a: f32, b: f32) -> f32 { ${binaryBody[op]} }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params[0]) { return; }
    let out_rank = params[1];
    let a_rank = params[2];
    let b_rank = params[3];
    var index: array<u32, ${MAX_DIMS}>;
    to_index(gid.x, 4u, out_rank, &index);
    let out_pos = index_position(index, ${4 + MAX_DIMS}u, out_rank);
    let a_pos = broadcast_position(index, out_rank, ${4 + MAX_DIMS * 2}u, ${4 + MAX_DIMS * 3}u, a_rank);
    let b_pos = broadcast_position(index, out_rank, ${4 + MAX_DIMS * 4}u, ${4 + MAX_DIMS * 5}u, b_rank);
    output_data[out_pos] = apply(a_data[a_pos], b_data[b_pos]);
}`;
}

export function reduceShader(op: ReduceOp): string {
    const { body, identity } = reduceInfo[op];
    return `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
${INDEX_HELPERS}
var<workgroup> partial: array<f32, ${WORKGROUP_SIZE}>;
fn apply(a: f32, b: f32) -> f32 { ${body} }
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) tid: u32) {
    let out_ordinal = wid.x;
    if (out_ordinal >= params[0]) { return; }
    let rank = params[1];
    let dim = params[2];
    let dim_size = params[3];
    var index: array<u32, ${MAX_DIMS}>;
    to_index(out_ordinal, 4u, rank, &index);
    var value: f32 = ${identity};
    for (var j = tid; j < dim_size; j += ${WORKGROUP_SIZE}u) {
        index[dim] = j;
        value = apply(value, input_data[index_position(index, ${4 + MAX_DIMS * 3}u, rank)]);
    }
    partial[tid] = value;
    workgroupBarrier();
    for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { partial[tid] = apply(partial[tid], partial[tid + stride]); }
        workgroupBarrier();
    }
    if (tid == 0u) {
        index[dim] = 0u;
        output_data[index_position(index, ${4 + MAX_DIMS}u, rank)] = partial[0];
    }
}`;
}

export const FILL_SHADER = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
struct Params { size: u32, value: f32 }
@group(0) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < p.size) { data[gid.x] = p.value; }
}`;

export const MATMUL_SHADER = `
const TILE: u32 = 16u;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u) {
    let row = group.y * TILE + local.y;
    let col = group.x * TILE + local.x;
    let m = p[0]; let n = p[1]; let k = p[2];
    var total = 0.0;
    let tiles = (k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        let ak = t * TILE + local.x;
        let bk = t * TILE + local.y;
        let lane = local.y * TILE + local.x;
        tile_a[lane] = 0.0;
        tile_b[lane] = 0.0;
        if (row < m && ak < k) { tile_a[lane] = a[p[9] + row * p[5] + ak * p[6]]; }
        if (bk < k && col < n) { tile_b[lane] = b[p[10] + bk * p[7] + col * p[8]]; }
        workgroupBarrier();
        for (var j = 0u; j < TILE; j++) {
            total += tile_a[local.y * TILE + j] * tile_b[j * TILE + local.x];
        }
        workgroupBarrier();
    }
    if (row < m && col < n) { out[p[11] + row * p[3] + col * p[4]] = total; }
}`;

export const CONV1D_SHADER = `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let total_size = p[0] * p[1] * p[2];
    if (gid.x >= total_size) { return; }
    let ow = gid.x % p[2]; let oc = (gid.x / p[2]) % p[1]; let batch = gid.x / (p[1] * p[2]);
    var sum = 0.0;
    for (var ic = 0u; ic < p[3]; ic++) {
        for (var k = 0u; k < p[5]; k++) {
            let forward = ow + k;
            let valid_forward = forward < p[4];
            let valid_reverse = k <= ow && (ow - k) < p[4];
            if (select(valid_forward, valid_reverse, p[6] != 0u)) {
                let x = select(forward, ow - k, p[6] != 0u);
                sum += input_data[batch*p[10] + ic*p[11] + x*p[12]]
                     * weight[oc*p[13] + ic*p[14] + k*p[15]];
            }
        }
    }
    out[batch*p[7] + oc*p[8] + ow*p[9]] = sum;
}`;

export const CONV2D_SHADER = `
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let spatial = p[2] * p[3]; let total_size = p[0] * p[1] * spatial;
    if (gid.x >= total_size) { return; }
    let ow = gid.x % p[3]; let oh = (gid.x / p[3]) % p[2];
    let oc = (gid.x / spatial) % p[1]; let batch = gid.x / (p[1] * spatial);
    var sum = 0.0;
    for (var ic = 0u; ic < p[4]; ic++) {
        for (var kh = 0u; kh < p[7]; kh++) {
            for (var kw = 0u; kw < p[8]; kw++) {
                let fh = oh + kh; let fw = ow + kw;
                let vr = kh <= oh && kw <= ow && (oh-kh) < p[5] && (ow-kw) < p[6];
                let vf = fh < p[5] && fw < p[6];
                if (select(vf, vr, p[9] != 0u)) {
                    let ih = select(fh, oh-kh, p[9] != 0u); let iw = select(fw, ow-kw, p[9] != 0u);
                    sum += input_data[batch*p[14] + ic*p[15] + ih*p[16] + iw*p[17]]
                         * weight[oc*p[18] + ic*p[19] + kh*p[20] + kw*p[21]];
                }
            }
        }
    }
    out[batch*p[10] + oc*p[11] + oh*p[12] + ow*p[13]] = sum;
}`;

export const SGD_SHADER = `
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x < p[0]) { param[gid.x] -= bitcast<f32>(p[1]) * grad[gid.x]; }
}`;

export const SGD_MOMENTUM_SHADER = `
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= p[0]) { return; }
    let lr = bitcast<f32>(p[1]); let beta = bitcast<f32>(p[2]);
    let next = beta * velocity[gid.x] + (1.0 - beta) * grad[gid.x];
    velocity[gid.x] = next;
    param[gid.x] -= lr * next;
}`;
