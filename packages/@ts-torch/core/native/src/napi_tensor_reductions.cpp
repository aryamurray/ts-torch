/**
 * Napi bindings for tensor reduction operations
 *
 * Covers: sum, mean, variance, argmax, softmax, log_softmax, and related reductions
 *
 * All wrappers follow the same pattern:
 *   1. Extract tensor handle(s) from Napi::External
 *   2. Extract scalar/shape params from Napi::Number / TypedArray
 *   3. Call the corresponding ts_tensor_* C API function
 *   4. Check error, wrap result handle, return
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
 * REDUCTION_OP: tensor-only, returns tensor.  f(tensor) -> tensor
 */
#define NAPI_REDUCTION_OP(NapiFn, CFn, Name)                                  \
Napi::Value NapiFn(const Napi::CallbackInfo& info) {                           \
  Napi::Env env = info.Env();                                                  \
  if (info.Length() < 1) {                                                     \
    throw Napi::Error::New(env, Name " requires 1 argument");                  \
  }                                                                            \
  ts_Tensor* tensor = GetTensorHandle(info[0]);                                \
  if (!tensor) {                                                               \
    throw Napi::Error::New(env, "Invalid tensor handle");                      \
  }                                                                            \
  ts_Error err = {0, ""};                                                      \
  ts_Tensor* result = CFn(tensor, &err);                                       \
  if (CheckAndThrowError(env, err, Name)) return env.Null();                   \
  return WrapTensorHandle(env, result);                                        \
}

/**
 * REDUCTION_DIM_OP: tensor + int64 dim, bool keepdim.  f(tensor, dim, keepdim) -> tensor
 */
#define NAPI_REDUCTION_DIM_OP(NapiFn, CFn, Name)                              \
Napi::Value NapiFn(const Napi::CallbackInfo& info) {                           \
  Napi::Env env = info.Env();                                                  \
  if (info.Length() < 3) {                                                     \
    throw Napi::Error::New(env, Name " requires 3 arguments");                 \
  }                                                                            \
  ts_Tensor* tensor = GetTensorHandle(info[0]);                                \
  if (!tensor) {                                                               \
    throw Napi::Error::New(env, "Invalid tensor handle");                      \
  }                                                                            \
  int64_t dim = info[1].As<Napi::Number>().Int64Value();                       \
  int keepdim = info[2].As<Napi::Boolean>().Value() ? 1 : 0;                   \
  ts_Error err = {0, ""};                                                      \
  ts_Tensor* result = CFn(tensor, dim, keepdim, &err);                         \
  if (CheckAndThrowError(env, err, Name)) return env.Null();                   \
  return WrapTensorHandle(env, result);                                        \
}

/**
 * REDUCTION_DIM_OP_NO_KEEPDIM: tensor + int64 dim (no keepdim).  f(tensor, dim) -> tensor
 */
#define NAPI_REDUCTION_DIM_OP_NO_KEEPDIM(NapiFn, CFn, Name)                   \
Napi::Value NapiFn(const Napi::CallbackInfo& info) {                           \
  Napi::Env env = info.Env();                                                  \
  if (info.Length() < 2) {                                                     \
    throw Napi::Error::New(env, Name " requires 2 arguments");                 \
  }                                                                            \
  ts_Tensor* tensor = GetTensorHandle(info[0]);                                \
  if (!tensor) {                                                               \
    throw Napi::Error::New(env, "Invalid tensor handle");                      \
  }                                                                            \
  int64_t dim = info[1].As<Napi::Number>().Int64Value();                       \
  ts_Error err = {0, ""};                                                      \
  ts_Tensor* result = CFn(tensor, dim, &err);                                  \
  if (CheckAndThrowError(env, err, Name)) return env.Null();                   \
  return WrapTensorHandle(env, result);                                        \
}

// ============================================================================
// 1. Sum Reduction Operations
// ============================================================================

// sum(tensor) -> scalar tensor
NAPI_REDUCTION_OP(NapiTensorSum, ts_tensor_sum, "ts_tensor_sum")

// sum_dim(tensor, dim, keepdim) -> tensor
NAPI_REDUCTION_DIM_OP(NapiTensorSumDim, ts_tensor_sum_dim, "ts_tensor_sum_dim")

// ============================================================================
// 2. Mean Reduction Operations
// ============================================================================

// mean(tensor) -> scalar tensor
NAPI_REDUCTION_OP(NapiTensorMean, ts_tensor_mean, "ts_tensor_mean")

// mean_dim(tensor, dim, keepdim) -> tensor
NAPI_REDUCTION_DIM_OP(NapiTensorMeanDim, ts_tensor_mean_dim, "ts_tensor_mean_dim")

// ============================================================================
// 3. Variance and Statistics
// ============================================================================

// var(tensor, dim, unbiased, keepdim) -> tensor
Napi::Value NapiTensorVar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_var requires 4 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  int unbiased = info[2].As<Napi::Boolean>().Value() ? 1 : 0;
  int keepdim = info[3].As<Napi::Boolean>().Value() ? 1 : 0;

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_var(tensor, dim, unbiased, keepdim, &err);
  if (CheckAndThrowError(env, err, "ts_tensor_var")) return env.Null();
  return WrapTensorHandle(env, result);
}

// ============================================================================
// 4. Argmax Operation
// ============================================================================

// argmax(tensor, dim, keepdim) -> tensor (indices)
NAPI_REDUCTION_DIM_OP(NapiTensorArgmax, ts_tensor_argmax, "ts_tensor_argmax")

// Note: softmax and log_softmax are in napi_tensor_unary_ops.cpp (not reduction-specific)

// ============================================================================
// Registration Helper
// ============================================================================

/**
 * RegisterTensorReductions: Register all reduction operation wrappers
 *
 * Called from Init() in napi_bindings.cpp to populate the exports object
 * with all reduction function bindings.
 */
void RegisterTensorReductions(Napi::Env env, Napi::Object exports) {
  // Sum operations
  exports.Set(Napi::String::New(env, "ts_tensor_sum"),
              Napi::Function::New(env, NapiTensorSum));
  exports.Set(Napi::String::New(env, "ts_tensor_sum_dim"),
              Napi::Function::New(env, NapiTensorSumDim));

  // Mean operations
  exports.Set(Napi::String::New(env, "ts_tensor_mean"),
              Napi::Function::New(env, NapiTensorMean));
  exports.Set(Napi::String::New(env, "ts_tensor_mean_dim"),
              Napi::Function::New(env, NapiTensorMeanDim));

  // Variance
  exports.Set(Napi::String::New(env, "ts_tensor_var"),
              Napi::Function::New(env, NapiTensorVar));

  // Argmax
  exports.Set(Napi::String::New(env, "ts_tensor_argmax"),
              Napi::Function::New(env, NapiTensorArgmax));

}
