/**
 * Napi bindings for unary tensor operations
 *
 * Covers: activations, element-wise math, normalization,
 * shape operations, and clamping.
 *
 * Note: Only wraps functions that exist in the C API (ts_torch.h)
 */

#include <napi.h>
#include "ts_torch.h"
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Shared helpers (defined in napi_bindings.cpp, declared here for linkage)
// ---------------------------------------------------------------------------

extern ts_Tensor* GetTensorHandle(const Napi::Value& val);
extern Napi::Value WrapTensorHandle(Napi::Env env, ts_Tensor* handle);
extern bool CheckAndThrowError(Napi::Env env, const ts_Error& err,
                               const char* funcName);

// ============================================================================
// Macro helpers to reduce boilerplate
// ============================================================================

/**
 * UNARY_OP: tensor-only, returns tensor.  f(tensor) -> tensor
 */
#define NAPI_UNARY_OP(NapiFn, CFn, Name)                                     \
Napi::Value NapiFn(const Napi::CallbackInfo& info) {                          \
  Napi::Env env = info.Env();                                                 \
  if (info.Length() < 1) {                                                    \
    throw Napi::Error::New(env, Name " requires 1 argument");                 \
  }                                                                           \
  ts_Tensor* tensor = GetTensorHandle(info[0]);                               \
  if (!tensor) {                                                              \
    throw Napi::Error::New(env, "Invalid tensor handle");                     \
  }                                                                           \
  ts_Error err = {0, ""};                                                     \
  ts_Tensor* result = CFn(tensor, &err);                                      \
  if (CheckAndThrowError(env, err, Name)) return env.Null();                  \
  return WrapTensorHandle(env, result);                                       \
}

/**
 * UNARY_DIM_OP: tensor + int64 dim.  f(tensor, dim) -> tensor
 */
#define NAPI_UNARY_DIM_OP(NapiFn, CFn, Name)                                 \
Napi::Value NapiFn(const Napi::CallbackInfo& info) {                          \
  Napi::Env env = info.Env();                                                 \
  if (info.Length() < 2) {                                                    \
    throw Napi::Error::New(env, Name " requires 2 arguments");                \
  }                                                                           \
  ts_Tensor* tensor = GetTensorHandle(info[0]);                               \
  if (!tensor) {                                                              \
    throw Napi::Error::New(env, "Invalid tensor handle");                     \
  }                                                                           \
  int64_t dim = info[1].As<Napi::Number>().Int64Value();                      \
  ts_Error err = {0, ""};                                                     \
  ts_Tensor* result = CFn(tensor, dim, &err);                                 \
  if (CheckAndThrowError(env, err, Name)) return env.Null();                  \
  return WrapTensorHandle(env, result);                                       \
}

// ============================================================================
// 1. Activation Functions (in C API)
// ============================================================================

NAPI_UNARY_OP(NapiTensorRelu,    ts_tensor_relu,    "ts_tensor_relu")
NAPI_UNARY_OP(NapiTensorSigmoid, ts_tensor_sigmoid, "ts_tensor_sigmoid")
NAPI_UNARY_OP(NapiTensorTanh,    ts_tensor_tanh,    "ts_tensor_tanh")

// ============================================================================
// 2. Element-wise Math Operations (in C API)
// ============================================================================

NAPI_UNARY_OP(NapiTensorExp,   ts_tensor_exp,   "ts_tensor_exp")
NAPI_UNARY_OP(NapiTensorLog,   ts_tensor_log,   "ts_tensor_log")
NAPI_UNARY_OP(NapiTensorSqrt,  ts_tensor_sqrt,  "ts_tensor_sqrt")
NAPI_UNARY_OP(NapiTensorNeg,   ts_tensor_neg,   "ts_tensor_neg")

// ============================================================================
// 3. Normalization Operations
// ============================================================================

// softmax(tensor, dim) -> tensor
NAPI_UNARY_DIM_OP(NapiTensorSoftmax, ts_tensor_softmax, "ts_tensor_softmax")

// log_softmax(tensor, dim) -> tensor
NAPI_UNARY_DIM_OP(NapiTensorLogSoftmax, ts_tensor_log_softmax, "ts_tensor_log_softmax")

// layer_norm(tensor, normalized_shape, weight, bias, eps)
Napi::Value NapiTensorLayerNorm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 5) {
    throw Napi::Error::New(env, "ts_tensor_layer_norm requires 5 arguments");
  }

  ts_Tensor* input = GetTensorHandle(info[0]);
  if (!input) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  // normalized_shape as BigInt64Array
  Napi::TypedArray shape_arr = info[1].As<Napi::TypedArray>();
  int64_t* normalized_shape = static_cast<int64_t*>(
    shape_arr.ArrayBuffer().Data()) +
    shape_arr.ByteOffset() / sizeof(int64_t);
  size_t normalized_shape_len = shape_arr.ElementLength();

  // weight and bias can be null
  ts_Tensor* weight = info[2].IsNull() || info[2].IsUndefined()
    ? nullptr : GetTensorHandle(info[2]);
  ts_Tensor* bias = info[3].IsNull() || info[3].IsUndefined()
    ? nullptr : GetTensorHandle(info[3]);

  double eps = info[4].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_layer_norm(
    input, normalized_shape, normalized_shape_len, weight, bias, eps, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_layer_norm")) return env.Null();
  return WrapTensorHandle(env, result);
}

// ============================================================================
// 4. Shape Operations
// ============================================================================

// transpose(tensor, dim0, dim1)
Napi::Value NapiTensorTranspose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_transpose requires 3 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t dim0 = info[1].As<Napi::Number>().Int64Value();
  int64_t dim1 = info[2].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_transpose(tensor, dim0, dim1, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_transpose")) return env.Null();
  return WrapTensorHandle(env, result);
}

// reshape(tensor, shape_typed_array)
Napi::Value NapiTensorReshape(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_reshape requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  Napi::TypedArray shape_arr = info[1].As<Napi::TypedArray>();
  int64_t* shape = static_cast<int64_t*>(
    shape_arr.ArrayBuffer().Data()) +
    shape_arr.ByteOffset() / sizeof(int64_t);
  size_t ndim = shape_arr.ElementLength();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_reshape(tensor, shape, ndim, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_reshape")) return env.Null();
  return WrapTensorHandle(env, result);
}

// ============================================================================
// 5. Clamping Operations
// ============================================================================

// clamp(tensor, min, max)
Napi::Value NapiTensorClamp(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_clamp requires 3 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  double min_val = info[1].As<Napi::Number>().DoubleValue();
  double max_val = info[2].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_clamp(tensor, min_val, max_val, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_clamp")) return env.Null();
  return WrapTensorHandle(env, result);
}

// clamp_min(tensor, min)
Napi::Value NapiTensorClampMin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_clamp_min requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  double min_val = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_clamp_min(tensor, min_val, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_clamp_min")) return env.Null();
  return WrapTensorHandle(env, result);
}

// clamp_max(tensor, max)
Napi::Value NapiTensorClampMax(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_clamp_max requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  double max_val = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_clamp_max(tensor, max_val, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_clamp_max")) return env.Null();
  return WrapTensorHandle(env, result);
}

// ============================================================================
// Registration Helper
// ============================================================================

/**
 * RegisterTensorUnaryOps: Register all unary operation wrappers
 *
 * Called from Init() in napi_bindings.cpp to populate the exports object
 * with all unary function bindings.
 */
void RegisterTensorUnaryOps(Napi::Env env, Napi::Object exports) {
  // Activation functions
  exports.Set(Napi::String::New(env, "ts_tensor_relu"),
              Napi::Function::New(env, NapiTensorRelu));
  exports.Set(Napi::String::New(env, "ts_tensor_sigmoid"),
              Napi::Function::New(env, NapiTensorSigmoid));
  exports.Set(Napi::String::New(env, "ts_tensor_tanh"),
              Napi::Function::New(env, NapiTensorTanh));

  // Math operations
  exports.Set(Napi::String::New(env, "ts_tensor_exp"),
              Napi::Function::New(env, NapiTensorExp));
  exports.Set(Napi::String::New(env, "ts_tensor_log"),
              Napi::Function::New(env, NapiTensorLog));
  exports.Set(Napi::String::New(env, "ts_tensor_sqrt"),
              Napi::Function::New(env, NapiTensorSqrt));
  exports.Set(Napi::String::New(env, "ts_tensor_neg"),
              Napi::Function::New(env, NapiTensorNeg));

  // Normalization
  exports.Set(Napi::String::New(env, "ts_tensor_softmax"),
              Napi::Function::New(env, NapiTensorSoftmax));
  exports.Set(Napi::String::New(env, "ts_tensor_log_softmax"),
              Napi::Function::New(env, NapiTensorLogSoftmax));
  exports.Set(Napi::String::New(env, "ts_tensor_layer_norm"),
              Napi::Function::New(env, NapiTensorLayerNorm));

  // Shape operations
  exports.Set(Napi::String::New(env, "ts_tensor_transpose"),
              Napi::Function::New(env, NapiTensorTranspose));
  exports.Set(Napi::String::New(env, "ts_tensor_reshape"),
              Napi::Function::New(env, NapiTensorReshape));

  // Clamping
  exports.Set(Napi::String::New(env, "ts_tensor_clamp"),
              Napi::Function::New(env, NapiTensorClamp));
  exports.Set(Napi::String::New(env, "ts_tensor_clamp_min"),
              Napi::Function::New(env, NapiTensorClampMin));
  exports.Set(Napi::String::New(env, "ts_tensor_clamp_max"),
              Napi::Function::New(env, NapiTensorClampMax));
}
