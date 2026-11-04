#include "tsgrad/ops.h"
#include "tsgrad/storage.h"

#include <napi.h>

#include <cstdint>
#include <exception>
#include <memory>
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

class StorageWrap : public Napi::ObjectWrap<StorageWrap> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static tsgrad::Storage& require(Napi::Env env, const Napi::Value& value, const char* name);

    StorageWrap(const Napi::CallbackInfo& info);
    ~StorageWrap();
    std::shared_ptr<tsgrad::Storage> storage;

private:
    void account(Napi::Env env);
    void recycle(Napi::Env env);

    static Napi::FunctionReference constructor;
    static Napi::Value Alloc(const Napi::CallbackInfo& info);
    static Napi::Value Empty(const Napi::CallbackInfo& info);
    static Napi::Value Retain(const Napi::CallbackInfo& info);
    static Napi::Value Release(const Napi::CallbackInfo& info);
    static Napi::Value Fill(const Napi::CallbackInfo& info);
    static Napi::Value CopyToHost(const Napi::CallbackInfo& info);
    static Napi::Value CopyFromHost(const Napi::CallbackInfo& info);

    int owners = 0;
    bool accounted = false;
};

Napi::FunctionReference StorageWrap::constructor;

StorageWrap::StorageWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<StorageWrap>(info) {}

StorageWrap::~StorageWrap() {
    recycle(Env());
}

void StorageWrap::account(Napi::Env env) {
    if (accounted || !storage) return;
    const int64_t bytes = static_cast<int64_t>(storage->numel()) * static_cast<int64_t>(sizeof(float));
    if (bytes == 0) return;
    Napi::MemoryManagement::AdjustExternalMemory(env, bytes);
    accounted = true;
}

void StorageWrap::recycle(Napi::Env env) {
    if (!storage) return;
    const int64_t bytes = static_cast<int64_t>(storage->numel()) * static_cast<int64_t>(sizeof(float));
    storage.reset();
    owners = 0;
    if (accounted && bytes > 0) {
        Napi::MemoryManagement::AdjustExternalMemory(env, -bytes);
        accounted = false;
    }
}

Napi::Value Synchronize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
#if TSGRAD_HAS_CUDA
    try {
        tsgrad::cuda::Context::get().synchronize();
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
#endif
    return env.Undefined();
}

Napi::Value BackendInfo(const Napi::CallbackInfo& info) {
    Napi::Object result = Napi::Object::New(info.Env());
#if TSGRAD_HAS_CUDA
    result.Set("kind", "cuda");
#else
    result.Set("kind", "cpu");
#endif
    return result;
}

Napi::Object StorageWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "NativeStorage", {});
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("alloc", Napi::Function::New(env, &StorageWrap::Alloc));
    exports.Set("empty", Napi::Function::New(env, &StorageWrap::Empty));
    exports.Set("retain", Napi::Function::New(env, &StorageWrap::Retain));
    exports.Set("release", Napi::Function::New(env, &StorageWrap::Release));
    exports.Set("fill", Napi::Function::New(env, &StorageWrap::Fill));
    exports.Set("copyToHost", Napi::Function::New(env, &StorageWrap::CopyToHost));
    exports.Set("copyFromHost", Napi::Function::New(env, &StorageWrap::CopyFromHost));
    exports.Set("synchronize", Napi::Function::New(env, Synchronize));
    exports.Set("backendInfo", Napi::Function::New(env, BackendInfo));
    return exports;
}

tsgrad::Storage& StorageWrap::require(Napi::Env env, const Napi::Value& value, const char* name) {
    if (!value.IsObject() || !value.As<Napi::Object>().InstanceOf(constructor.Value())) {
        throw Napi::TypeError::New(env, std::string(name) + " must be a NativeStorage");
    }
    StorageWrap* wrap = Unwrap(value.As<Napi::Object>());
    if (wrap == nullptr || !wrap->storage) {
        throw Napi::Error::New(env, std::string(name) + " is empty");
    }
    return *wrap->storage;
}

Napi::Value StorageWrap::Alloc(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) {
        throw Napi::TypeError::New(env, "alloc expects (numel, fill?)");
    }
    const double raw = info[0].As<Napi::Number>().DoubleValue();
    if (!(raw >= 0.0) || raw > static_cast<double>(SIZE_MAX)) {
        throw Napi::RangeError::New(env, "numel is out of range");
    }
    const size_t numel = static_cast<size_t>(raw);
    float fill = 0.0F;
    if (info.Length() >= 2 && !info[1].IsUndefined()) {
        if (!info[1].IsNumber()) {
            throw Napi::TypeError::New(env, "fill must be a number");
        }
        fill = info[1].As<Napi::Number>().FloatValue();
    }

    try {
        Napi::Object obj = constructor.New({});
        StorageWrap* wrap = Unwrap(obj);
        wrap->storage = std::make_shared<tsgrad::Storage>(numel, true, fill);
        wrap->owners = 1;
        wrap->account(env);
        return obj;
    } catch (const Napi::Error&) {
        throw;
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
}

Napi::Value StorageWrap::Empty(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 1 || !info[0].IsNumber()) {
        throw Napi::TypeError::New(env, "empty expects (numel)");
    }
    const double raw = info[0].As<Napi::Number>().DoubleValue();
    if (!(raw >= 0.0) || raw > static_cast<double>(SIZE_MAX)) {
        throw Napi::RangeError::New(env, "numel is out of range");
    }
    const size_t numel = static_cast<size_t>(raw);

    try {
        Napi::Object obj = constructor.New({});
        StorageWrap* wrap = Unwrap(obj);
        wrap->storage = std::make_shared<tsgrad::Storage>(numel, false, 0.0F);
        wrap->owners = 1;
        wrap->account(env);
        return obj;
    } catch (const Napi::Error&) {
        throw;
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
}

Napi::Value StorageWrap::Retain(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 1) {
        throw Napi::TypeError::New(env, "retain expects (storage)");
    }
    require(env, info[0], "storage");
    StorageWrap* wrap = Unwrap(info[0].As<Napi::Object>());
    wrap->owners++;
    return env.Undefined();
}

Napi::Value StorageWrap::Release(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 1) {
        throw Napi::TypeError::New(env, "release expects (storage)");
    }
    require(env, info[0], "storage");
    StorageWrap* wrap = Unwrap(info[0].As<Napi::Object>());
    if (wrap->owners <= 0) {
        throw Napi::Error::New(env, "release on a storage with no owners");
    }
    wrap->owners--;
    if (wrap->owners == 0) wrap->recycle(env);
    return env.Undefined();
}

Napi::Value StorageWrap::Fill(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 2 || !info[1].IsNumber()) {
        throw Napi::TypeError::New(env, "fill expects (storage, value)");
    }
    tsgrad::Storage& storage = require(env, info[0], "storage");
    try {
        storage.fill(info[1].As<Napi::Number>().FloatValue());
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
    return env.Undefined();
}

Napi::Value StorageWrap::CopyToHost(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 1) {
        throw Napi::TypeError::New(env, "copyToHost expects (storage)");
    }
    tsgrad::Storage& storage = require(env, info[0], "storage");
    Napi::Float32Array out = Napi::Float32Array::New(env, storage.numel());
    try {
        storage.copy_to_host(out.Data());
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
    return out;
}

Napi::Value StorageWrap::CopyFromHost(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 2) {
        throw Napi::TypeError::New(env, "copyFromHost expects (storage, values)");
    }
    tsgrad::Storage& storage = require(env, info[0], "storage");
    if (!info[1].IsTypedArray() ||
        info[1].As<Napi::TypedArray>().TypedArrayType() != napi_float32_array) {
        throw Napi::TypeError::New(env, "values must be a Float32Array");
    }
    Napi::Float32Array values = info[1].As<Napi::Float32Array>();
    if (values.ElementLength() != storage.numel()) {
        throw Napi::RangeError::New(env, "copyFromHost length does not match storage");
    }
    try {
        storage.copy_from_host(read_f32(env, values, "values"));
    } catch (const std::exception& ex) {
        throw Napi::Error::New(env, ex.what());
    }
    return env.Undefined();
}

const int32_t* data_or_null(const std::vector<int32_t>& values) {
    return values.empty() ? nullptr : values.data();
}

tsgrad::View make_view(Napi::Env env, const Napi::Value& storage, const std::vector<int32_t>& shape, const std::vector<int32_t>& stride, const char* name) {
    return tsgrad::View{
        StorageWrap::require(env, storage, name).data(),
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

    tsgrad::View out = make_view(env, info[0], out_shape, out_stride, "out");
    tsgrad::View in = make_view(env, info[3], in_shape, in_stride, "input");
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

    tsgrad::View out = make_view(env, info[0], out_shape, out_stride, "out");
    tsgrad::View a = make_view(env, info[3], a_shape, a_stride, "a");
    tsgrad::View b = make_view(env, info[6], b_shape, b_stride, "b");
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

    tsgrad::View out = make_view(env, info[0], out_shape, out_stride, "out");
    tsgrad::View in = make_view(env, info[3], in_shape, in_stride, "input");
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

    tsgrad::View out = make_view(env, info[0], out_shape, out_stride, "out");
    tsgrad::View input = make_view(env, info[3], in_shape, in_stride, "input");
    tsgrad::View weight = make_view(env, info[6], weight_shape, weight_stride, "weight");
    call_native(env, [&](){ fn(out, input, weight, reverse); });
    return env.Undefined();
}

Napi::Value bind_sgd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 7) {
        throw Napi::TypeError::New(env, "sgd expects (param, shape, stride, grad, gradShape, gradStride, lr)");
    }
    std::vector<int32_t> shape = read_i32_array(env, info[1], "shape");
    std::vector<int32_t> stride = read_i32_array(env, info[2], "stride");
    std::vector<int32_t> grad_shape = read_i32_array(env, info[4], "gradShape");
    std::vector<int32_t> grad_stride = read_i32_array(env, info[5], "gradStride");
    check_rank(env, shape, stride, "param");
    check_rank(env, grad_shape, grad_stride, "grad");
    if (!info[6].IsNumber()) {
        throw Napi::TypeError::New(env, "lr must be a number");
    }
    float lr = info[6].As<Napi::Number>().FloatValue();
    tsgrad::View param = make_view(env, info[0], shape, stride, "param");
    tsgrad::View grad = make_view(env, info[3], grad_shape, grad_stride, "grad");
    call_native(env, [&](){ tsgrad::sgd(param, grad, lr); });
    return env.Undefined();
}

Napi::Value bind_sgd_momentum(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 11) {
        throw Napi::TypeError::New(env, "sgdMomentum expects (param, shape, stride, grad, gradShape, gradStride, velocity, velocityShape, velocityStride, lr, beta)");
    }
    std::vector<int32_t> shape = read_i32_array(env, info[1], "shape");
    std::vector<int32_t> stride = read_i32_array(env, info[2], "stride");
    std::vector<int32_t> grad_shape = read_i32_array(env, info[4], "gradShape");
    std::vector<int32_t> grad_stride = read_i32_array(env, info[5], "gradStride");
    std::vector<int32_t> velocity_shape = read_i32_array(env, info[7], "velocityShape");
    std::vector<int32_t> velocity_stride = read_i32_array(env, info[8], "velocityStride");
    check_rank(env, shape, stride, "param");
    check_rank(env, grad_shape, grad_stride, "grad");
    check_rank(env, velocity_shape, velocity_stride, "velocity");
    if (!info[9].IsNumber() || !info[10].IsNumber()) {
        throw Napi::TypeError::New(env, "lr and beta must be numbers");
    }
    float lr = info[9].As<Napi::Number>().FloatValue();
    float beta = info[10].As<Napi::Number>().FloatValue();
    tsgrad::View param = make_view(env, info[0], shape, stride, "param");
    tsgrad::View grad = make_view(env, info[3], grad_shape, grad_stride, "grad");
    tsgrad::View velocity = make_view(env, info[6], velocity_shape, velocity_stride, "velocity");
    call_native(env, [&](){ tsgrad::sgd_momentum(param, grad, velocity, lr, beta); });
    return env.Undefined();
}

void export_conv(Napi::Env env, Napi::Object& exports, const char* name, ConvFn fn) {
    exports.Set(name, Napi::Function::New(env, [fn](const Napi::CallbackInfo& info) {
        return bind_conv(info, fn);
    }));
}

Napi::Object init(Napi::Env env, Napi::Object exports) {
    StorageWrap::Init(env, exports);
    export_map(env, exports, "mapNeg", tsgrad::map_neg);
    export_map(env, exports, "mapInv", tsgrad::map_inv);
    export_map(env, exports, "mapSigmoid", tsgrad::map_sigmoid);
    export_map(env, exports, "mapRelu", tsgrad::map_relu);
    export_map(env, exports, "mapLog", tsgrad::map_log);
    export_map(env, exports, "mapExp", tsgrad::map_exp);
    export_map(env, exports, "mapId", tsgrad::map_id);

    export_zip(env, exports, "zipAdd", tsgrad::zip_add);
    export_zip(env, exports, "zipSub", tsgrad::zip_sub);
    export_zip(env, exports, "zipMul", tsgrad::zip_mul);
    export_zip(env, exports, "zipDiv", tsgrad::zip_div);
    export_zip(env, exports, "zipLt", tsgrad::zip_lt);
    export_zip(env, exports, "zipEq", tsgrad::zip_eq);
    export_zip(env, exports, "zipIsClose", tsgrad::zip_is_close);

    export_reduce(env, exports, "reduceMax", tsgrad::reduce_max);
    export_reduce(env, exports, "reduceSum", tsgrad::reduce_sum);
    export_reduce(env, exports, "reduceProd", tsgrad::reduce_prod);

    export_zip(env, exports, "matmul", tsgrad::matmul);
    export_conv(env, exports, "conv1d", tsgrad::conv1d);
    export_conv(env, exports, "conv2d", tsgrad::conv2d);
    exports.Set("sgd", Napi::Function::New(env, bind_sgd));
    exports.Set("sgdMomentum", Napi::Function::New(env, bind_sgd_momentum));
    return exports;
}

} // namespace

NODE_API_MODULE(tsgrad_native, init)
