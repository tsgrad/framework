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
    static Napi::Value Uniform(const Napi::CallbackInfo& info);
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

Napi::Object StorageWrap::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "NativeStorage", {});
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("alloc", Napi::Function::New(env, &StorageWrap::Alloc));
    exports.Set("empty", Napi::Function::New(env, &StorageWrap::Empty));
    exports.Set("retain", Napi::Function::New(env, &StorageWrap::Retain));
    exports.Set("release", Napi::Function::New(env, &StorageWrap::Release));
    exports.Set("fill", Napi::Function::New(env, &StorageWrap::Fill));
    exports.Set("uniform", Napi::Function::New(env, &StorageWrap::Uniform));
    exports.Set("copyToHost", Napi::Function::New(env, &StorageWrap::CopyToHost));
    exports.Set("copyFromHost", Napi::Function::New(env, &StorageWrap::CopyFromHost));
    exports.Set("synchronize", Napi::Function::New(env, Synchronize));
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

Napi::Value StorageWrap::Uniform(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 3 || !info[1].IsNumber() || !info[2].IsNumber()) {
        throw Napi::TypeError::New(env, "uniform expects (storage, low, high)");
    }
    tsgrad::Storage& storage = require(env, info[0], "storage");
    float low = info[1].As<Napi::Number>().FloatValue();
    float high = info[2].As<Napi::Number>().FloatValue();
    try {
        tsgrad::uniform(storage.data(), storage.numel(), low, high);
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

struct Owned {
    std::vector<int32_t> shape;
    std::vector<int32_t> stride;
};

tsgrad::View take_view(Napi::Env env, const Napi::CallbackInfo& info, size_t at, Owned& owned, const char* name) {
    owned.shape = read_i32_array(env, info[at + 1], name);
    owned.stride = read_i32_array(env, info[at + 2], name);
    check_rank(env, owned.shape, owned.stride, name);
    return make_view(env, info[at], owned.shape, owned.stride, name);
}

int32_t take_i32(Napi::Env env, const Napi::Value& value, const char* name) {
    if (!value.IsNumber()) throw Napi::TypeError::New(env, std::string(name) + " must be a number");
    return value.As<Napi::Number>().Int32Value();
}

float take_f32(Napi::Env env, const Napi::Value& value, const char* name) {
    if (!value.IsNumber()) throw Napi::TypeError::New(env, std::string(name) + " must be a number");
    return value.As<Napi::Number>().FloatValue();
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

Napi::Value bind_softmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 7) throw Napi::TypeError::New(env, "softmax expects (out, outShape, outStride, input, inShape, inStride, dim)");
    Owned out_a, in_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View in = take_view(env, info, 3, in_a, "input");
    int32_t dim = take_i32(env, info[6], "dim");
    call_native(env, [&](){ tsgrad::softmax(out, in, dim); });
    return env.Undefined();
}

Napi::Value bind_softmax_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 10) throw Napi::TypeError::New(env, "softmaxBackward expects (out, outShape, outStride, softmax, softmaxShape, softmaxStride, grad, gradShape, gradStride, dim)");
    Owned out_a, s_a, g_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View softmax = take_view(env, info, 3, s_a, "softmax");
    tsgrad::View grad = take_view(env, info, 6, g_a, "grad");
    int32_t dim = take_i32(env, info[9], "dim");
    call_native(env, [&](){ tsgrad::softmax_backward(out, softmax, grad, dim); });
    return env.Undefined();
}

Napi::Value bind_log_softmax(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 7) throw Napi::TypeError::New(env, "logSoftmax expects (out, outShape, outStride, input, inShape, inStride, dim)");
    Owned out_a, in_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View in = take_view(env, info, 3, in_a, "input");
    int32_t dim = take_i32(env, info[6], "dim");
    call_native(env, [&](){ tsgrad::log_softmax(out, in, dim); });
    return env.Undefined();
}

Napi::Value bind_log_softmax_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 10) throw Napi::TypeError::New(env, "logSoftmaxBackward expects (out, outShape, outStride, logSoftmax, logSoftmaxShape, logSoftmaxStride, grad, gradShape, gradStride, dim)");
    Owned out_a, l_a, g_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View log_softmax = take_view(env, info, 3, l_a, "logSoftmax");
    tsgrad::View grad = take_view(env, info, 6, g_a, "grad");
    int32_t dim = take_i32(env, info[9], "dim");
    call_native(env, [&](){ tsgrad::log_softmax_backward(out, log_softmax, grad, dim); });
    return env.Undefined();
}

Napi::Value bind_softmax_cross_entropy(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 13) throw Napi::TypeError::New(env, "softmaxCrossEntropy expects (loss, lossShape, lossStride, softmax, softmaxShape, softmaxStride, logits, logitsShape, logitsStride, target, targetShape, targetStride, dim)");
    Owned loss_a, s_a, x_a, t_a;
    tsgrad::View loss = take_view(env, info, 0, loss_a, "loss");
    tsgrad::View softmax = take_view(env, info, 3, s_a, "softmax");
    tsgrad::View logits = take_view(env, info, 6, x_a, "logits");
    tsgrad::View target = take_view(env, info, 9, t_a, "target");
    int32_t dim = take_i32(env, info[12], "dim");
    call_native(env, [&](){ tsgrad::softmax_cross_entropy(loss, softmax, logits, target, dim); });
    return env.Undefined();
}

Napi::Value bind_softmax_cross_entropy_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 10) throw Napi::TypeError::New(env, "softmaxCrossEntropyBackward expects (gradLogits, gradLogitsShape, gradLogitsStride, softmax, softmaxShape, softmaxStride, target, targetShape, targetStride, dim)");
    Owned g_a, s_a, t_a;
    tsgrad::View grad = take_view(env, info, 0, g_a, "gradLogits");
    tsgrad::View softmax = take_view(env, info, 3, s_a, "softmax");
    tsgrad::View target = take_view(env, info, 6, t_a, "target");
    int32_t dim = take_i32(env, info[9], "dim");
    call_native(env, [&](){ tsgrad::softmax_cross_entropy_backward(grad, softmax, target, dim); });
    return env.Undefined();
}

Napi::Value bind_relu_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 9) throw Napi::TypeError::New(env, "reluBackward expects (out, outShape, outStride, grad, gradShape, gradStride, input, inShape, inStride)");
    Owned out_a, g_a, in_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View grad = take_view(env, info, 3, g_a, "grad");
    tsgrad::View input = take_view(env, info, 6, in_a, "input");
    call_native(env, [&](){ tsgrad::relu_backward(out, grad, input); });
    return env.Undefined();
}

Napi::Value bind_sigmoid_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 9) throw Napi::TypeError::New(env, "sigmoidBackward expects (out, outShape, outStride, grad, gradShape, gradStride, output, outputShape, outputStride)");
    Owned out_a, g_a, y_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View grad = take_view(env, info, 3, g_a, "grad");
    tsgrad::View output = take_view(env, info, 6, y_a, "output");
    call_native(env, [&](){ tsgrad::sigmoid_backward(out, grad, output); });
    return env.Undefined();
}

Napi::Value bind_expand(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 6) throw Napi::TypeError::New(env, "expand expects (out, outShape, outStride, input, inShape, inStride)");
    Owned out_a, in_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View input = take_view(env, info, 3, in_a, "input");
    call_native(env, [&](){ tsgrad::expand(out, input); });
    return env.Undefined();
}

Napi::Value bind_bias_act(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 10) throw Napi::TypeError::New(env, "biasAct expects (out, outShape, outStride, input, inShape, inStride, bias, biasShape, biasStride, act)");
    Owned out_a, in_a, b_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View input = take_view(env, info, 3, in_a, "input");
    tsgrad::View bias = take_view(env, info, 6, b_a, "bias");
    int32_t act = take_i32(env, info[9], "act");
    call_native(env, [&](){ tsgrad::bias_act(out, input, bias, act); });
    return env.Undefined();
}

Napi::Value bind_bias_act_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 16) throw Napi::TypeError::New(env, "biasActBackward expects (gradInput, gradInputShape, gradInputStride, gradBias, gradBiasShape, gradBiasStride, grad, gradShape, gradStride, saved, savedShape, savedStride, bias, biasShape, biasStride, act)");
    Owned gi_a, gb_a, g_a, s_a, b_a;
    tsgrad::View grad_input = take_view(env, info, 0, gi_a, "gradInput");
    tsgrad::View grad_bias = take_view(env, info, 3, gb_a, "gradBias");
    tsgrad::View grad = take_view(env, info, 6, g_a, "grad");
    tsgrad::View saved = take_view(env, info, 9, s_a, "saved");
    tsgrad::View bias = take_view(env, info, 12, b_a, "bias");
    int32_t act = take_i32(env, info[15], "act");
    call_native(env, [&](){ tsgrad::bias_act_backward(grad_input, grad_bias, grad, saved, bias, act); });
    return env.Undefined();
}

Napi::Value bind_layer_norm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 19) throw Napi::TypeError::New(env, "layerNorm expects (out, outShape, outStride, mean, meanShape, meanStride, rstd, rstdShape, rstdStride, input, inShape, inStride, weight, weightShape, weightStride, bias, biasShape, biasStride, eps)");
    Owned out_a, mean_a, rstd_a, in_a, w_a, b_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View mean = take_view(env, info, 3, mean_a, "mean");
    tsgrad::View rstd = take_view(env, info, 6, rstd_a, "rstd");
    tsgrad::View input = take_view(env, info, 9, in_a, "input");
    tsgrad::View weight = take_view(env, info, 12, w_a, "weight");
    tsgrad::View bias = take_view(env, info, 15, b_a, "bias");
    float eps = take_f32(env, info[18], "eps");
    call_native(env, [&](){ tsgrad::layer_norm(out, mean, rstd, input, weight, bias, eps); });
    return env.Undefined();
}

Napi::Value bind_layer_norm_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 24) throw Napi::TypeError::New(env, "layerNormBackward expects (gradInput, gradInputShape, gradInputStride, gradWeight, gradWeightShape, gradWeightStride, gradBias, gradBiasShape, gradBiasStride, grad, gradShape, gradStride, input, inShape, inStride, mean, meanShape, meanStride, rstd, rstdShape, rstdStride, weight, weightShape, weightStride)");
    Owned gi_a, gw_a, gb_a, g_a, in_a, mean_a, rstd_a, w_a;
    tsgrad::View grad_input = take_view(env, info, 0, gi_a, "gradInput");
    tsgrad::View grad_weight = take_view(env, info, 3, gw_a, "gradWeight");
    tsgrad::View grad_bias = take_view(env, info, 6, gb_a, "gradBias");
    tsgrad::View grad = take_view(env, info, 9, g_a, "grad");
    tsgrad::View input = take_view(env, info, 12, in_a, "input");
    tsgrad::View mean = take_view(env, info, 15, mean_a, "mean");
    tsgrad::View rstd = take_view(env, info, 18, rstd_a, "rstd");
    tsgrad::View weight = take_view(env, info, 21, w_a, "weight");
    call_native(env, [&](){ tsgrad::layer_norm_backward(grad_input, grad_weight, grad_bias, grad, input, mean, rstd, weight); });
    return env.Undefined();
}

Napi::Value bind_max_pool2d(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 8) throw Napi::TypeError::New(env, "maxPool2d expects (out, outShape, outStride, input, inShape, inStride, kh, kw)");
    Owned out_a, in_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View input = take_view(env, info, 3, in_a, "input");
    int32_t kh = take_i32(env, info[6], "kh");
    int32_t kw = take_i32(env, info[7], "kw");
    call_native(env, [&](){ tsgrad::max_pool2d(out, input, kh, kw); });
    return env.Undefined();
}

Napi::Value bind_max_pool2d_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 14) throw Napi::TypeError::New(env, "maxPool2dBackward expects (gradInput, gradInputShape, gradInputStride, grad, gradShape, gradStride, input, inShape, inStride, output, outputShape, outputStride, kh, kw)");
    Owned gi_a, g_a, in_a, o_a;
    tsgrad::View grad_input = take_view(env, info, 0, gi_a, "gradInput");
    tsgrad::View grad = take_view(env, info, 3, g_a, "grad");
    tsgrad::View input = take_view(env, info, 6, in_a, "input");
    tsgrad::View output = take_view(env, info, 9, o_a, "output");
    int32_t kh = take_i32(env, info[12], "kh");
    int32_t kw = take_i32(env, info[13], "kw");
    call_native(env, [&](){ tsgrad::max_pool2d_backward(grad_input, grad, input, output, kh, kw); });
    return env.Undefined();
}

Napi::Value bind_avg_pool2d(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 8) throw Napi::TypeError::New(env, "avgPool2d expects (out, outShape, outStride, input, inShape, inStride, kh, kw)");
    Owned out_a, in_a;
    tsgrad::View out = take_view(env, info, 0, out_a, "out");
    tsgrad::View input = take_view(env, info, 3, in_a, "input");
    int32_t kh = take_i32(env, info[6], "kh");
    int32_t kw = take_i32(env, info[7], "kw");
    call_native(env, [&](){ tsgrad::avg_pool2d(out, input, kh, kw); });
    return env.Undefined();
}

Napi::Value bind_avg_pool2d_backward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() != 8) throw Napi::TypeError::New(env, "avgPool2dBackward expects (gradInput, gradInputShape, gradInputStride, grad, gradShape, gradStride, kh, kw)");
    Owned gi_a, g_a;
    tsgrad::View grad_input = take_view(env, info, 0, gi_a, "gradInput");
    tsgrad::View grad = take_view(env, info, 3, g_a, "grad");
    int32_t kh = take_i32(env, info[6], "kh");
    int32_t kw = take_i32(env, info[7], "kw");
    call_native(env, [&](){ tsgrad::avg_pool2d_backward(grad_input, grad, kh, kw); });
    return env.Undefined();
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
    exports.Set("softmax", Napi::Function::New(env, bind_softmax));
    exports.Set("softmaxBackward", Napi::Function::New(env, bind_softmax_backward));
    exports.Set("logSoftmax", Napi::Function::New(env, bind_log_softmax));
    exports.Set("logSoftmaxBackward", Napi::Function::New(env, bind_log_softmax_backward));
    exports.Set("softmaxCrossEntropy", Napi::Function::New(env, bind_softmax_cross_entropy));
    exports.Set("softmaxCrossEntropyBackward", Napi::Function::New(env, bind_softmax_cross_entropy_backward));
    exports.Set("reluBackward", Napi::Function::New(env, bind_relu_backward));
    exports.Set("sigmoidBackward", Napi::Function::New(env, bind_sigmoid_backward));
    exports.Set("expand", Napi::Function::New(env, bind_expand));
    exports.Set("biasAct", Napi::Function::New(env, bind_bias_act));
    exports.Set("biasActBackward", Napi::Function::New(env, bind_bias_act_backward));
    exports.Set("layerNorm", Napi::Function::New(env, bind_layer_norm));
    exports.Set("layerNormBackward", Napi::Function::New(env, bind_layer_norm_backward));
    exports.Set("maxPool2d", Napi::Function::New(env, bind_max_pool2d));
    exports.Set("maxPool2dBackward", Napi::Function::New(env, bind_max_pool2d_backward));
    exports.Set("avgPool2d", Napi::Function::New(env, bind_avg_pool2d));
    exports.Set("avgPool2dBackward", Napi::Function::New(env, bind_avg_pool2d_backward));
    return exports;
}

} // namespace

NODE_API_MODULE(tsgrad_native, init)
