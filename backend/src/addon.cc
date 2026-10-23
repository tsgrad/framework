#include "tsgrad/ops.h"

#include <napi.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

// Turns one JavaScript call into a tsgrad::View and calls the C++ function.
// Kernel math stays in the .cpp and .cu files.

namespace {

std::vector<int32_t> read_i32_array(Napi::Env env, const Napi::Value& value, const char* name) {
    if (!value.IsArray()) {
        throw Napi::TypeError::New(env, std::string(name) + " must be an array");
    }
    Napi::Array array = value.As<Napi::Array>();
    std::vector<int32_t> out(array.Length());
    for (uint32_t i = 0; i < array.Length(); i++) {
        Napi::Value element = array.Get(i);
        if (!element.IsNumber()) {
            throw Napi::TypeError::New(env, std::string(name) + " must contain numbers");
        }
        out[i] = element.As<Napi::Number>().Int32Value();
    }
    return out;
}

float* read_f32(Napi::Env env, const Napi::Value& value, const char* name) {
    if (!value.IsTypedArray() ||
        value.As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
        throw Napi::TypeError::New(env, std::string(name) + " must be a Float32Array");
    }
    return value.As<Napi::Float32Array>().Data();
}

const int32_t* data_or_null(const std::vector<int32_t>& values) {
    return values.empty() ? nullptr : values.data();
}

tsgrad::View make_view(float* data, const std::vector<int32_t>& shape, const std::vector<int32_t>& stride) {
    return tsgrad::View{
        data,
        data_or_null(shape),
        data_or_null(stride),
        static_cast<int32_t>(shape.size()),
    };
}

void check_rank(Napi::Env env, const std::vector<int32_t>& shape, const std::vector<int32_t>& stride, const char* name) {
    if (shape.size() != stride.size()) {
        throw Napi::TypeError::New(env, std::string(name) + " shape and stride must have the same length");
    }
}

using MapFn = void (*)(const tsgrad::View&, const tsgrad::View&);
using ZipFn = void (*)(const tsgrad::View&, const tsgrad::View&, const tsgrad::View&);
using ReduceFn = void (*)(const tsgrad::View&, const tsgrad::View&, int32_t);
using ConvFn = void (*)(const tsgrad::View&, const tsgrad::View&, const tsgrad::View&, bool);

template <typename Fn>
void call_native(Napi::Env env, Fn&& fn) {
    try {
        fn();
    } catch (const Napi::Error&) {
        throw;
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
}

Napi::Value bind_map(const Napi::CallbackInfo& info, MapFn fn) {
    Napi::Env env = info.Env();
    if (info.Length() != 6) {
        throw Napi::TypeError::New(env, "map expects (out, outShape, outStride, input, inShape, inStride)");
    }
    std::vector<int32_t> out_shape = read_i32_array(env, info[1], "outShape");
    std::vector<int32_t> out_stride = read_i32_array(env, info[2], "outStride");
    std::vector<int32_t> in_shape = read_i32_array(env, info[4], "inShape");
    std::vector<int32_t> in_stride = read_i32_array(env, info[5], "inStride");
    check_rank(env, out_shape, out_stride, "out");
    check_rank(env, in_shape, in_stride, "input");

    tsgrad::View out = make_view(read_f32(env, info[0], "out"), out_shape, out_stride);
    tsgrad::View in = make_view(read_f32(env, info[3], "input"), in_shape, in_stride);
    call_native(env, [&](){ fn(out, in); });
    return env.Undefined();
}

Napi::Value bind_zip(const Napi::CallbackInfo& info, ZipFn fn) {
    Napi::Env env = info.Env();
    if (info.Length() != 9) {
        throw Napi::TypeError::New(env, "zip expects (out, outShape, outStride, a, aShape, aStride, b, bShape, bStride)");
    }
    std::vector<int32_t> out_shape = read_i32_array(env, info[1], "outShape");
    std::vector<int32_t> out_stride = read_i32_array(env, info[2], "outStride");
    std::vector<int32_t> a_shape = read_i32_array(env, info[4], "aShape");
    std::vector<int32_t> a_stride = read_i32_array(env, info[5], "aStride");
    std::vector<int32_t> b_shape = read_i32_array(env, info[7], "bShape");
    std::vector<int32_t> b_stride = read_i32_array(env, info[8], "bStride");
    check_rank(env, out_shape, out_stride, "out");
    check_rank(env, a_shape, a_stride, "a");
    check_rank(env, b_shape, b_stride, "b");

    tsgrad::View out = make_view(read_f32(env, info[0], "out"), out_shape, out_stride);
    tsgrad::View a = make_view(read_f32(env, info[3], "a"), a_shape, a_stride);
    tsgrad::View b = make_view(read_f32(env, info[6], "b"), b_shape, b_stride);
    call_native(env, [&](){ fn(out, a, b); });
    return env.Undefined();
}

Napi::Value bind_reduce(const Napi::CallbackInfo& info, ReduceFn fn) {
    Napi::Env env = info.Env();
    if (info.Length() != 7) {
        throw Napi::TypeError::New(env, "reduce expects (out, outShape, outStride, input, inShape, inStride, dim)");
    }
    std::vector<int32_t> out_shape = read_i32_array(env, info[1], "outShape");
    std::vector<int32_t> out_stride = read_i32_array(env, info[2], "outStride");
    std::vector<int32_t> in_shape = read_i32_array(env, info[4], "inShape");
    std::vector<int32_t> in_stride = read_i32_array(env, info[5], "inStride");
    check_rank(env, out_shape, out_stride, "out");
    check_rank(env, in_shape, in_stride, "input");
    if (!info[6].IsNumber()) {
        throw Napi::TypeError::New(env, "dim must be a number");
    }
    int32_t dim = info[6].As<Napi::Number>().Int32Value();

    tsgrad::View out = make_view(read_f32(env, info[0], "out"), out_shape, out_stride);
    tsgrad::View in = make_view(read_f32(env, info[3], "input"), in_shape, in_stride);
    call_native(env, [&](){ fn(out, in, dim); });
    return env.Undefined();
}

void export_map(Napi::Env env, Napi::Object& exports, const char* name, MapFn fn) {
    exports.Set(name, Napi::Function::New(env, [fn](const Napi::CallbackInfo& info) {
        return bind_map(info, fn);
    }));
}

void export_zip(Napi::Env env, Napi::Object& exports, const char* name, ZipFn fn) {
    exports.Set(name, Napi::Function::New(env, [fn](const Napi::CallbackInfo& info) {
        return bind_zip(info, fn);
    }));
}

void export_reduce(Napi::Env env, Napi::Object& exports, const char* name, ReduceFn fn) {
    exports.Set(name, Napi::Function::New(env, [fn](const Napi::CallbackInfo& info) {
        return bind_reduce(info, fn);
    }));
}

Napi::Value bind_conv(const Napi::CallbackInfo& info, ConvFn fn) {
    Napi::Env env = info.Env();
    if (info.Length() != 10) {
        throw Napi::TypeError::New(env, "conv expects (out, outShape, outStride, input, inShape, inStride, weight, weightShape, weightStride, reverse)");
    }
    std::vector<int32_t> out_shape = read_i32_array(env, info[1], "outShape");
    std::vector<int32_t> out_stride = read_i32_array(env, info[2], "outStride");
    std::vector<int32_t> in_shape = read_i32_array(env, info[4], "inShape");
    std::vector<int32_t> in_stride = read_i32_array(env, info[5], "inStride");
    std::vector<int32_t> weight_shape = read_i32_array(env, info[7], "weightShape");
    std::vector<int32_t> weight_stride = read_i32_array(env, info[8], "weightStride");
    check_rank(env, out_shape, out_stride, "out");
    check_rank(env, in_shape, in_stride, "input");
    check_rank(env, weight_shape, weight_stride, "weight");
    if (!info[9].IsBoolean()) {
        throw Napi::TypeError::New(env, "reverse must be a boolean");
    }
    bool reverse = info[9].As<Napi::Boolean>().Value();

    tsgrad::View out = make_view(read_f32(env, info[0], "out"), out_shape, out_stride);
    tsgrad::View input = make_view(read_f32(env, info[3], "input"), in_shape, in_stride);
    tsgrad::View weight = make_view(read_f32(env, info[6], "weight"), weight_shape, weight_stride);
    call_native(env, [&](){ fn(out, input, weight, reverse); });
    return env.Undefined();
}

void export_conv(Napi::Env env, Napi::Object& exports, const char* name, ConvFn fn) {
    exports.Set(name, Napi::Function::New(env, [fn](const Napi::CallbackInfo& info) {
        return bind_conv(info, fn);
    }));
}

Napi::Object init(Napi::Env env, Napi::Object exports) {
    export_map(env, exports, "mapNeg", tsgrad::map_neg);
    export_map(env, exports, "mapInv", tsgrad::map_inv);
    export_map(env, exports, "mapSigmoid", tsgrad::map_sigmoid);
    export_map(env, exports, "mapRelu", tsgrad::map_relu);
    export_map(env, exports, "mapLog", tsgrad::map_log);
    export_map(env, exports, "mapExp", tsgrad::map_exp);
    export_map(env, exports, "mapId", tsgrad::map_id);

    export_zip(env, exports, "zipAdd", tsgrad::zip_add);
    export_zip(env, exports, "zipMul", tsgrad::zip_mul);
    export_zip(env, exports, "zipLt", tsgrad::zip_lt);
    export_zip(env, exports, "zipEq", tsgrad::zip_eq);
    export_zip(env, exports, "zipIsClose", tsgrad::zip_is_close);

    export_reduce(env, exports, "reduceMax", tsgrad::reduce_max);
    export_reduce(env, exports, "reduceSum", tsgrad::reduce_sum);
    export_reduce(env, exports, "reduceProd", tsgrad::reduce_prod);

    export_zip(env, exports, "matmul", tsgrad::matmul);
    export_conv(env, exports, "conv1d", tsgrad::conv1d);
    export_conv(env, exports, "conv2d", tsgrad::conv2d);
    return exports;
}

} // namespace

NODE_API_MODULE(tsgrad_native, init)
