/**
 * Napi bindings for ts-torch
 * Thin wrapper layer over C API using Node-API (Napi)
 *
 * This module exports C functions as JavaScript-callable methods.
 * The Napi layer handles:
 * - V8 value marshalling (JS objects ↔ C pointers)
 * - Memory management (automatic cleanup via finalizers)
 * - Error handling (C errors → JavaScript exceptions)
 */

#include <napi.h>
#include "ts_torch.h"
#include <cstring>
#include <string>

// ============================================================================
// Utility Helpers
// ============================================================================

/**
 * Extract opaque tensor handle from Napi::External
 */
static ts_Tensor* GetTensorHandle(const Napi::Value& val) {
  if (!val.IsExternal()) {
    return nullptr;
  }
  return static_cast<ts_Tensor*>(val.As<Napi::External<void>>().Data());
}

/**
 * Extract opaque scope handle from Napi::External
 */
static ts_Scope* GetScopeHandle(const Napi::Value& val) {
  if (!val.IsExternal()) {
    return nullptr;
  }
  return static_cast<ts_Scope*>(val.As<Napi::External<void>>().Data());
}

/**
 * Wrap tensor handle in Napi::External with automatic cleanup finalizer
 */
static Napi::Value WrapTensorHandle(Napi::Env env, ts_Tensor* handle) {
  if (!handle) {
    return env.Null();
  }
  return Napi::External<void>::New(env, handle,
    [](Napi::Env, void* ptr) {
      if (ptr) {
        ts_tensor_delete(static_cast<ts_Tensor*>(ptr));
      }
    });
}

/**
 * Wrap scope handle in Napi::External (no cleanup - managed by C layer)
 */
static Napi::Value WrapScopeHandle(Napi::Env env, ts_Scope* handle) {
  if (!handle) {
    return env.Null();
  }
  return Napi::External<void>::New(env, static_cast<void*>(handle));
}

/**
 * Check error and throw JavaScript exception if needed
 */
static bool CheckAndThrowError(Napi::Env env, const ts_Error& err,
                                 const char* funcName) {
  if (err.code != 0) {
    std::string msg = funcName;
    msg += ": ";
    msg += err.message;
    throw Napi::Error::New(env, msg);
    return true;
  }
  return false;
}

// ============================================================================
// Version
// ============================================================================

Napi::Value NapiVersion(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  const char* ver = ts_version();
  return Napi::String::New(env, ver ? ver : "unknown");
}

Napi::Value NapiCudaIsAvailable(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  bool available = ts_cuda_is_available();
  return Napi::Boolean::New(env, available);
}

// ============================================================================
// Tensor Factory Functions
// ============================================================================

Napi::Value NapiTensorZeros(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_zeros requires 4 arguments");
  }

  // Extract shape array - use TypedArray
  Napi::TypedArray shape_arr = info[0].As<Napi::TypedArray>();
  int64_t* shape = static_cast<int64_t*>(shape_arr.ArrayBuffer().Data()) + shape_arr.ByteOffset() / sizeof(int64_t);
  size_t ndim = shape_arr.ElementLength();

  ts_DType dtype = static_cast<ts_DType>(info[1].As<Napi::Number>().Int32Value());
  ts_DeviceType device = static_cast<ts_DeviceType>(info[2].As<Napi::Number>().Int32Value());
  int device_index = info[3].As<Napi::Number>().Int32Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_zeros(shape, ndim, dtype, device, device_index, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_zeros")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorOnes(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_ones requires 4 arguments");
  }

  Napi::TypedArray shape_arr = info[0].As<Napi::TypedArray>();
  int64_t* shape = static_cast<int64_t*>(shape_arr.ArrayBuffer().Data()) + shape_arr.ByteOffset() / sizeof(int64_t);
  size_t ndim = shape_arr.ElementLength();

  ts_DType dtype = static_cast<ts_DType>(info[1].As<Napi::Number>().Int32Value());
  ts_DeviceType device = static_cast<ts_DeviceType>(info[2].As<Napi::Number>().Int32Value());
  int device_index = info[3].As<Napi::Number>().Int32Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_ones(shape, ndim, dtype, device, device_index, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_ones")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorRandn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_randn requires 4 arguments");
  }

  Napi::TypedArray shape_arr = info[0].As<Napi::TypedArray>();
  int64_t* shape = static_cast<int64_t*>(shape_arr.ArrayBuffer().Data()) + shape_arr.ByteOffset() / sizeof(int64_t);
  size_t ndim = shape_arr.ElementLength();

  ts_DType dtype = static_cast<ts_DType>(info[1].As<Napi::Number>().Int32Value());
  ts_DeviceType device = static_cast<ts_DeviceType>(info[2].As<Napi::Number>().Int32Value());
  int device_index = info[3].As<Napi::Number>().Int32Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_randn(shape, ndim, dtype, device, device_index, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_randn")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorEmpty(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_empty requires 4 arguments");
  }

  Napi::TypedArray shape_arr = info[0].As<Napi::TypedArray>();
  int64_t* shape = static_cast<int64_t*>(shape_arr.ArrayBuffer().Data()) + shape_arr.ByteOffset() / sizeof(int64_t);
  size_t ndim = shape_arr.ElementLength();

  ts_DType dtype = static_cast<ts_DType>(info[1].As<Napi::Number>().Int32Value());
  ts_DeviceType device = static_cast<ts_DeviceType>(info[2].As<Napi::Number>().Int32Value());
  int device_index = info[3].As<Napi::Number>().Int32Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_empty(shape, ndim, dtype, device, device_index, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_empty")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

// ============================================================================
// Tensor Properties
// ============================================================================

Napi::Value NapiTensorNdim(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_ndim requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  int64_t ndim = ts_tensor_ndim(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_ndim")) {
    return env.Null();
  }

  return Napi::Number::New(env, ndim);
}

Napi::Value NapiTensorDtype(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_dtype requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_DType dtype = ts_tensor_dtype(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_dtype")) {
    return env.Null();
  }

  return Napi::Number::New(env, static_cast<int>(dtype));
}

Napi::Value NapiTensorNumel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_numel requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  int64_t numel = ts_tensor_numel(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_numel")) {
    return env.Null();
  }

  return Napi::Number::New(env, numel);
}

// ============================================================================
// Binary Operations
// ============================================================================

Napi::Value NapiTensorAdd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_add requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);

  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_add(a, b, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_add")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorMul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_mul requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);

  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_mul(a, b, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_mul")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorMatmul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_matmul requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);

  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_matmul(a, b, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_matmul")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

// ============================================================================
// Unary Operations
// ============================================================================

Napi::Value NapiTensorRelu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_relu requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_relu(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_relu")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorSigmoid(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_sigmoid requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_sigmoid(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_sigmoid")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorTanh(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_tanh requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_tanh(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_tanh")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorTranspose(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_transpose requires 3 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t dim1 = info[1].As<Napi::Number>().Int64Value();
  int64_t dim2 = info[2].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_transpose(tensor, dim1, dim2, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_transpose")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

// ============================================================================
// Reductions
// ============================================================================

Napi::Value NapiTensorSum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_sum requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_sum(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_sum")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorMean(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_mean requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_mean(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_mean")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

// ============================================================================
// Autograd
// ============================================================================

Napi::Value NapiTensorBackward(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_backward requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error error;
  error.code = 0;
  ts_tensor_backward(tensor, &error);

  if (error.code != 0) {
    throw Napi::Error::New(env, error.message);
  }

  return env.Undefined();
}

Napi::Value NapiTensorGrad(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_grad requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error error;
  error.code = 0;
  ts_Tensor* grad = ts_tensor_grad(tensor, &error);

  if (error.code != 0) {
    throw Napi::Error::New(env, error.message);
  }

  return WrapTensorHandle(env, grad);
}

// ============================================================================
// Memory Management
// ============================================================================

Napi::Value NapiTensorDelete(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_delete requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    return env.Undefined();
  }

  ts_tensor_delete(tensor);
  return env.Undefined();
}

// ============================================================================
// Scope Management
// ============================================================================

Napi::Value NapiScopeBegin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Scope* scope = ts_scope_begin();
  return WrapScopeHandle(env, scope);
}

Napi::Value NapiScopeEnd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_scope_end requires 1 argument");
  }

  ts_Scope* scope = GetScopeHandle(info[0]);
  if (!scope) {
    return env.Undefined();
  }

  ts_scope_end(scope);
  return env.Undefined();
}

Napi::Value NapiScopeRegisterTensor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_scope_register_tensor requires 2 arguments");
  }

  ts_Scope* scope = GetScopeHandle(info[0]);
  ts_Tensor* tensor = GetTensorHandle(info[1]);

  if (!scope || !tensor) {
    return env.Undefined();
  }

  ts_scope_register_tensor(scope, tensor);
  return env.Undefined();
}

Napi::Value NapiScopeEscapeTensor(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_scope_escape_tensor requires 2 arguments");
  }

  ts_Scope* scope = GetScopeHandle(info[0]);
  ts_Tensor* tensor = GetTensorHandle(info[1]);

  if (!scope || !tensor) {
    return env.Null();
  }

  ts_Tensor* result = ts_scope_escape_tensor(scope, tensor);
  return WrapTensorHandle(env, result);
}

// ============================================================================
// Module Initialization
// ============================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Version
  exports.Set(Napi::String::New(env, "ts_version"),
    Napi::Function::New(env, NapiVersion));

  // Device info
  exports.Set(Napi::String::New(env, "ts_cuda_is_available"),
    Napi::Function::New(env, NapiCudaIsAvailable));

  // Tensor factories
  exports.Set(Napi::String::New(env, "ts_tensor_zeros"),
    Napi::Function::New(env, NapiTensorZeros));
  exports.Set(Napi::String::New(env, "ts_tensor_ones"),
    Napi::Function::New(env, NapiTensorOnes));
  exports.Set(Napi::String::New(env, "ts_tensor_randn"),
    Napi::Function::New(env, NapiTensorRandn));
  exports.Set(Napi::String::New(env, "ts_tensor_empty"),
    Napi::Function::New(env, NapiTensorEmpty));

  // Tensor properties
  exports.Set(Napi::String::New(env, "ts_tensor_ndim"),
    Napi::Function::New(env, NapiTensorNdim));
  exports.Set(Napi::String::New(env, "ts_tensor_dtype"),
    Napi::Function::New(env, NapiTensorDtype));
  exports.Set(Napi::String::New(env, "ts_tensor_numel"),
    Napi::Function::New(env, NapiTensorNumel));

  // Binary operations
  exports.Set(Napi::String::New(env, "ts_tensor_add"),
    Napi::Function::New(env, NapiTensorAdd));
  exports.Set(Napi::String::New(env, "ts_tensor_mul"),
    Napi::Function::New(env, NapiTensorMul));
  exports.Set(Napi::String::New(env, "ts_tensor_matmul"),
    Napi::Function::New(env, NapiTensorMatmul));

  // Unary operations
  exports.Set(Napi::String::New(env, "ts_tensor_relu"),
    Napi::Function::New(env, NapiTensorRelu));
  exports.Set(Napi::String::New(env, "ts_tensor_sigmoid"),
    Napi::Function::New(env, NapiTensorSigmoid));
  exports.Set(Napi::String::New(env, "ts_tensor_tanh"),
    Napi::Function::New(env, NapiTensorTanh));
  exports.Set(Napi::String::New(env, "ts_tensor_transpose"),
    Napi::Function::New(env, NapiTensorTranspose));

  // Reductions
  exports.Set(Napi::String::New(env, "ts_tensor_sum"),
    Napi::Function::New(env, NapiTensorSum));
  exports.Set(Napi::String::New(env, "ts_tensor_mean"),
    Napi::Function::New(env, NapiTensorMean));

  // Autograd
  exports.Set(Napi::String::New(env, "ts_tensor_backward"),
    Napi::Function::New(env, NapiTensorBackward));
  exports.Set(Napi::String::New(env, "ts_tensor_grad"),
    Napi::Function::New(env, NapiTensorGrad));

  // Memory management
  exports.Set(Napi::String::New(env, "ts_tensor_delete"),
    Napi::Function::New(env, NapiTensorDelete));

  // Scope management
  exports.Set(Napi::String::New(env, "ts_scope_begin"),
    Napi::Function::New(env, NapiScopeBegin));
  exports.Set(Napi::String::New(env, "ts_scope_end"),
    Napi::Function::New(env, NapiScopeEnd));
  exports.Set(Napi::String::New(env, "ts_scope_register_tensor"),
    Napi::Function::New(env, NapiScopeRegisterTensor));
  exports.Set(Napi::String::New(env, "ts_scope_escape_tensor"),
    Napi::Function::New(env, NapiScopeEscapeTensor));

  return exports;
}

NODE_API_MODULE(ts_torch, Init)
