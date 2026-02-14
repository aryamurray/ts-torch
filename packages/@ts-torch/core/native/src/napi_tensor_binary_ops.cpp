/**
 * Napi wrapper implementations for binary tensor operations.
 *
 * Covers:
 *   - Arithmetic: add, sub, mul, div, matmul, bmm, chain_matmul, pow, remainder, fmod
 *   - Scalar arithmetic: add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar
 *   - Comparisons: eq, ne, lt, le, gt, ge
 *   - Min/Max: minimum, maximum
 *   - Advanced: where, gather, scatter, scatter_add, index_select,
 *               masked_fill, masked_select
 *   - In-place: add_, sub_, mul_, div_, mul_scalar_, div_scalar_,
 *               add_alpha_, pow_
 *   - Output variants: add_out, sub_out, mul_out, div_out, matmul_out
 */

#include "napi_bindings.h"

// ============================================================================
// Arithmetic Operations (tensor, tensor) -> tensor
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
  CheckAndThrowError(env, err, "ts_tensor_add");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorSub(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_sub requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_sub(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_sub");
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
  CheckAndThrowError(env, err, "ts_tensor_mul");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorDiv(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_div requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_div(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_div");
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
  CheckAndThrowError(env, err, "ts_tensor_matmul");
  return WrapTensorHandle(env, result);
}

// bmm(a, b) -> tensor  [B, M, K] @ [B, K, N] -> [B, M, N]
Napi::Value NapiTensorBmm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_bmm requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_bmm(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_bmm");
  return WrapTensorHandle(env, result);
}

// chain_matmul(tensors[]) -> tensor
Napi::Value NapiTensorChainMatmul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_tensor_chain_matmul requires 1 argument (array of tensor handles)");
  }

  // Expect a JS array of external tensor handles
  Napi::Array arr = info[0].As<Napi::Array>();
  size_t count = arr.Length();
  if (count < 2) {
    throw Napi::Error::New(env, "ts_tensor_chain_matmul requires at least 2 tensors");
  }

  std::vector<ts_TensorHandle> handles(count);
  for (size_t i = 0; i < count; i++) {
    handles[i] = GetTensorHandle(arr.Get(static_cast<uint32_t>(i)));
    if (!handles[i]) {
      throw Napi::Error::New(env, "Invalid tensor handle in array");
    }
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_chain_matmul(handles.data(), count, &err);
  CheckAndThrowError(env, err, "ts_tensor_chain_matmul");
  return WrapTensorHandle(env, result);
}

// pow(tensor, tensor) -> tensor
Napi::Value NapiTensorPow(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_pow requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_pow(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_pow");
  return WrapTensorHandle(env, result);
}

// pow(tensor, scalar) -> tensor
Napi::Value NapiTensorPowScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_pow_scalar requires 2 arguments (tensor, scalar)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double exponent = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_pow_scalar(tensor, exponent, &err);
  CheckAndThrowError(env, err, "ts_tensor_pow_scalar");
  return WrapTensorHandle(env, result);
}

// ============================================================================
// Scalar Arithmetic Operations (tensor, scalar) -> tensor
// ============================================================================

Napi::Value NapiTensorAddScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_add_scalar requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double scalar = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_add_scalar(tensor, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_add_scalar");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorSubScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_sub_scalar requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double scalar = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_sub_scalar(tensor, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_sub_scalar");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorMulScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_mul_scalar requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double scalar = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_mul_scalar(tensor, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_mul_scalar");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorDivScalar(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_div_scalar requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double scalar = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_div_scalar(tensor, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_div_scalar");
  return WrapTensorHandle(env, result);
}

// ============================================================================
// Comparison Operations (tensor, tensor) -> bool tensor
// ============================================================================

Napi::Value NapiTensorEq(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_eq requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_eq(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_eq");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorNe(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_ne requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_ne(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_ne");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorLt(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_lt requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_lt(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_lt");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorLe(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_le requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_le(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_le");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorGt(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_gt requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_gt(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_gt");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorGe(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_ge requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_ge(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_ge");
  return WrapTensorHandle(env, result);
}

// ============================================================================
// Min/Max Operations (tensor, tensor) -> tensor
// ============================================================================

Napi::Value NapiTensorMinimum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_minimum requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_minimum(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_minimum");
  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorMaximum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_maximum requires 2 arguments");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  if (!a || !b) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_maximum(a, b, &err);
  CheckAndThrowError(env, err, "ts_tensor_maximum");
  return WrapTensorHandle(env, result);
}

// ============================================================================
// Advanced Binary / Multi-Tensor Operations
// ============================================================================

// where(condition, x, y) -> tensor
Napi::Value NapiTensorWhere(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_where requires 3 arguments (condition, x, y)");
  }

  ts_Tensor* condition = GetTensorHandle(info[0]);
  ts_Tensor* x = GetTensorHandle(info[1]);
  ts_Tensor* y = GetTensorHandle(info[2]);
  if (!condition || !x || !y) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_where(condition, x, y, &err);
  CheckAndThrowError(env, err, "ts_tensor_where");
  return WrapTensorHandle(env, result);
}

// gather(input, dim, index) -> tensor
Napi::Value NapiTensorGather(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_gather requires 3 arguments (input, dim, index)");
  }

  ts_Tensor* input = GetTensorHandle(info[0]);
  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  ts_Tensor* index = GetTensorHandle(info[2]);
  if (!input || !index) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_gather(input, dim, index, &err);
  CheckAndThrowError(env, err, "ts_tensor_gather");
  return WrapTensorHandle(env, result);
}

// scatter(input, dim, index, src) -> tensor
Napi::Value NapiTensorScatter(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_scatter requires 4 arguments (input, dim, index, src)");
  }

  ts_Tensor* input = GetTensorHandle(info[0]);
  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  ts_Tensor* index = GetTensorHandle(info[2]);
  ts_Tensor* src = GetTensorHandle(info[3]);
  if (!input || !index || !src) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_scatter(input, dim, index, src, &err);
  CheckAndThrowError(env, err, "ts_tensor_scatter");
  return WrapTensorHandle(env, result);
}

// scatter_add(input, dim, index, src) -> tensor
Napi::Value NapiTensorScatterAdd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "ts_tensor_scatter_add requires 4 arguments (input, dim, index, src)");
  }

  ts_Tensor* input = GetTensorHandle(info[0]);
  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  ts_Tensor* index = GetTensorHandle(info[2]);
  ts_Tensor* src = GetTensorHandle(info[3]);
  if (!input || !index || !src) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_scatter_add(input, dim, index, src, &err);
  CheckAndThrowError(env, err, "ts_tensor_scatter_add");
  return WrapTensorHandle(env, result);
}

// index_select(tensor, dim, index) -> tensor
Napi::Value NapiTensorIndexSelect(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_index_select requires 3 arguments (tensor, dim, index)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  ts_Tensor* index = GetTensorHandle(info[2]);
  if (!tensor || !index) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_index_select(tensor, dim, index, &err);
  CheckAndThrowError(env, err, "ts_tensor_index_select");
  return WrapTensorHandle(env, result);
}

// masked_fill(tensor, mask, value) -> tensor
Napi::Value NapiTensorMaskedFill(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_masked_fill requires 3 arguments (tensor, mask, value)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* mask = GetTensorHandle(info[1]);
  double value = info[2].As<Napi::Number>().DoubleValue();
  if (!tensor || !mask) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_masked_fill(tensor, mask, value, &err);
  CheckAndThrowError(env, err, "ts_tensor_masked_fill");
  return WrapTensorHandle(env, result);
}

// masked_select(tensor, mask) -> 1-D tensor
Napi::Value NapiTensorMaskedSelect(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_masked_select requires 2 arguments (tensor, mask)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* mask = GetTensorHandle(info[1]);
  if (!tensor || !mask) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_masked_select(tensor, mask, &err);
  CheckAndThrowError(env, err, "ts_tensor_masked_select");
  return WrapTensorHandle(env, result);
}

// index(tensor, indices...) - advanced indexing
// ts_torch.h does not expose generic advanced indexing; placeholder.
Napi::Value NapiTensorIndex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  throw Napi::Error::New(env, "ts_tensor_index (advanced indexing) not yet exposed in C API");
}

// ============================================================================
// In-Place Operations (tensor, tensor) -> void
// ============================================================================

// add_(tensor, other) -> undefined  (tensor += other)
Napi::Value NapiTensorAdd_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_add_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_add_(tensor, other, &err);
  CheckAndThrowError(env, err, "ts_tensor_add_");
  return env.Undefined();
}

// sub_(tensor, other) -> undefined  (tensor -= other)
Napi::Value NapiTensorSub_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_sub_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_sub_(tensor, other, &err);
  CheckAndThrowError(env, err, "ts_tensor_sub_");
  return env.Undefined();
}

// mul_(tensor, other) -> undefined  (tensor *= other)
Napi::Value NapiTensorMul_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_mul_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_mul_(tensor, other, &err);
  CheckAndThrowError(env, err, "ts_tensor_mul_");
  return env.Undefined();
}

// div_(tensor, other) -> undefined  (tensor /= other)
Napi::Value NapiTensorDiv_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_div_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_div_(tensor, other, &err);
  CheckAndThrowError(env, err, "ts_tensor_div_");
  return env.Undefined();
}

// mul_scalar_(tensor, scalar) -> undefined  (tensor *= scalar)
Napi::Value NapiTensorMulScalar_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_mul_scalar_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double scalar = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_tensor_mul_scalar_(tensor, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_mul_scalar_");
  return env.Undefined();
}

// div_scalar_(tensor, scalar) -> undefined  (tensor /= scalar)
Napi::Value NapiTensorDivScalar_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_div_scalar_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }
  double scalar = info[1].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_tensor_div_scalar_(tensor, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_div_scalar_");
  return env.Undefined();
}

// add_alpha_(tensor, other, alpha) -> undefined  (tensor += alpha * other)
Napi::Value NapiTensorAddAlpha_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_add_alpha_ requires 3 arguments (tensor, other, alpha)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  double alpha = info[2].As<Napi::Number>().DoubleValue();
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_add_alpha_(tensor, other, alpha, &err);
  CheckAndThrowError(env, err, "ts_tensor_add_alpha_");
  return env.Undefined();
}

// optim_add_(tensor, other, alpha) -> undefined
// Optimizer-only variant that bypasses autograd via .data()
Napi::Value NapiTensorOptimAdd_(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_optim_add_ requires 3 arguments (tensor, other, alpha)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  double alpha = info[2].As<Napi::Number>().DoubleValue();
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_optim_add_(tensor, other, alpha, &err);
  CheckAndThrowError(env, err, "ts_tensor_optim_add_");
  return env.Undefined();
}

// sub_inplace(tensor, other) -> undefined  (legacy API name)
Napi::Value NapiTensorSubInplace(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "ts_tensor_sub_inplace requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_sub_inplace(tensor, other, &err);
  CheckAndThrowError(env, err, "ts_tensor_sub_inplace");
  return env.Undefined();
}

// add_scaled_inplace(tensor, other, scalar) -> undefined
// Legacy name for tensor += scalar * other
Napi::Value NapiTensorAddScaledInplace(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_add_scaled_inplace requires 3 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  ts_Tensor* other = GetTensorHandle(info[1]);
  double scalar = info[2].As<Napi::Number>().DoubleValue();
  if (!tensor || !other) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_add_scaled_inplace(tensor, other, scalar, &err);
  CheckAndThrowError(env, err, "ts_tensor_add_scaled_inplace");
  return env.Undefined();
}

// ============================================================================
// Output-Variant Operations (a, b, out) -> void
// ============================================================================

Napi::Value NapiTensorAddOut(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_add_out requires 3 arguments (a, b, out)");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  ts_Tensor* out = GetTensorHandle(info[2]);
  if (!a || !b || !out) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_add_out(a, b, out, &err);
  CheckAndThrowError(env, err, "ts_tensor_add_out");
  return env.Undefined();
}

Napi::Value NapiTensorSubOut(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_sub_out requires 3 arguments (a, b, out)");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  ts_Tensor* out = GetTensorHandle(info[2]);
  if (!a || !b || !out) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_sub_out(a, b, out, &err);
  CheckAndThrowError(env, err, "ts_tensor_sub_out");
  return env.Undefined();
}

Napi::Value NapiTensorMulOut(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_mul_out requires 3 arguments (a, b, out)");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  ts_Tensor* out = GetTensorHandle(info[2]);
  if (!a || !b || !out) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_mul_out(a, b, out, &err);
  CheckAndThrowError(env, err, "ts_tensor_mul_out");
  return env.Undefined();
}

Napi::Value NapiTensorDivOut(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_div_out requires 3 arguments (a, b, out)");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  ts_Tensor* out = GetTensorHandle(info[2]);
  if (!a || !b || !out) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_div_out(a, b, out, &err);
  CheckAndThrowError(env, err, "ts_tensor_div_out");
  return env.Undefined();
}

Napi::Value NapiTensorMatmulOut(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "ts_tensor_matmul_out requires 3 arguments (a, b, out)");
  }

  ts_Tensor* a = GetTensorHandle(info[0]);
  ts_Tensor* b = GetTensorHandle(info[1]);
  ts_Tensor* out = GetTensorHandle(info[2]);
  if (!a || !b || !out) {
    throw Napi::Error::New(env, "Invalid tensor handles");
  }

  ts_Error err = {0, ""};
  ts_tensor_matmul_out(a, b, out, &err);
  CheckAndThrowError(env, err, "ts_tensor_matmul_out");
  return env.Undefined();
}

// ============================================================================
// Registration Helper
// ============================================================================

/**
 * RegisterTensorBinaryOps: Register all binary operation wrappers
 *
 * Called from Init() in napi_bindings.cpp to populate the exports object
 * with all binary function bindings.
 */
void RegisterTensorBinaryOps(Napi::Env env, Napi::Object exports) {
  // Arithmetic operations
  exports.Set(Napi::String::New(env, "ts_tensor_add"),
              Napi::Function::New(env, NapiTensorAdd));
  exports.Set(Napi::String::New(env, "ts_tensor_sub"),
              Napi::Function::New(env, NapiTensorSub));
  exports.Set(Napi::String::New(env, "ts_tensor_mul"),
              Napi::Function::New(env, NapiTensorMul));
  exports.Set(Napi::String::New(env, "ts_tensor_div"),
              Napi::Function::New(env, NapiTensorDiv));
  exports.Set(Napi::String::New(env, "ts_tensor_matmul"),
              Napi::Function::New(env, NapiTensorMatmul));
  exports.Set(Napi::String::New(env, "ts_tensor_bmm"),
              Napi::Function::New(env, NapiTensorBmm));
  exports.Set(Napi::String::New(env, "ts_tensor_chain_matmul"),
              Napi::Function::New(env, NapiTensorChainMatmul));
  exports.Set(Napi::String::New(env, "ts_tensor_pow"),
              Napi::Function::New(env, NapiTensorPow));
  exports.Set(Napi::String::New(env, "ts_tensor_pow_scalar"),
              Napi::Function::New(env, NapiTensorPowScalar));

  // Scalar arithmetic variants
  exports.Set(Napi::String::New(env, "ts_tensor_add_scalar"),
              Napi::Function::New(env, NapiTensorAddScalar));
  exports.Set(Napi::String::New(env, "ts_tensor_sub_scalar"),
              Napi::Function::New(env, NapiTensorSubScalar));
  exports.Set(Napi::String::New(env, "ts_tensor_mul_scalar"),
              Napi::Function::New(env, NapiTensorMulScalar));
  exports.Set(Napi::String::New(env, "ts_tensor_div_scalar"),
              Napi::Function::New(env, NapiTensorDivScalar));

  // Comparison operations
  exports.Set(Napi::String::New(env, "ts_tensor_eq"),
              Napi::Function::New(env, NapiTensorEq));
  exports.Set(Napi::String::New(env, "ts_tensor_ne"),
              Napi::Function::New(env, NapiTensorNe));
  exports.Set(Napi::String::New(env, "ts_tensor_lt"),
              Napi::Function::New(env, NapiTensorLt));
  exports.Set(Napi::String::New(env, "ts_tensor_le"),
              Napi::Function::New(env, NapiTensorLe));
  exports.Set(Napi::String::New(env, "ts_tensor_gt"),
              Napi::Function::New(env, NapiTensorGt));
  exports.Set(Napi::String::New(env, "ts_tensor_ge"),
              Napi::Function::New(env, NapiTensorGe));

  // Min/Max operations
  exports.Set(Napi::String::New(env, "ts_tensor_minimum"),
              Napi::Function::New(env, NapiTensorMinimum));
  exports.Set(Napi::String::New(env, "ts_tensor_maximum"),
              Napi::Function::New(env, NapiTensorMaximum));

  // Advanced operations
  exports.Set(Napi::String::New(env, "ts_tensor_where"),
              Napi::Function::New(env, NapiTensorWhere));
  exports.Set(Napi::String::New(env, "ts_tensor_gather"),
              Napi::Function::New(env, NapiTensorGather));
  exports.Set(Napi::String::New(env, "ts_tensor_scatter"),
              Napi::Function::New(env, NapiTensorScatter));
  exports.Set(Napi::String::New(env, "ts_tensor_scatter_add"),
              Napi::Function::New(env, NapiTensorScatterAdd));
  exports.Set(Napi::String::New(env, "ts_tensor_index_select"),
              Napi::Function::New(env, NapiTensorIndexSelect));
  exports.Set(Napi::String::New(env, "ts_tensor_masked_fill"),
              Napi::Function::New(env, NapiTensorMaskedFill));
  exports.Set(Napi::String::New(env, "ts_tensor_masked_select"),
              Napi::Function::New(env, NapiTensorMaskedSelect));
  exports.Set(Napi::String::New(env, "ts_tensor_index"),
              Napi::Function::New(env, NapiTensorIndex));

  // In-place operations
  exports.Set(Napi::String::New(env, "ts_tensor_add_"),
              Napi::Function::New(env, NapiTensorAdd_));
  exports.Set(Napi::String::New(env, "ts_tensor_sub_"),
              Napi::Function::New(env, NapiTensorSub_));
  exports.Set(Napi::String::New(env, "ts_tensor_mul_"),
              Napi::Function::New(env, NapiTensorMul_));
  exports.Set(Napi::String::New(env, "ts_tensor_div_"),
              Napi::Function::New(env, NapiTensorDiv_));
  exports.Set(Napi::String::New(env, "ts_tensor_mul_scalar_"),
              Napi::Function::New(env, NapiTensorMulScalar_));
  exports.Set(Napi::String::New(env, "ts_tensor_div_scalar_"),
              Napi::Function::New(env, NapiTensorDivScalar_));
  exports.Set(Napi::String::New(env, "ts_tensor_add_alpha_"),
              Napi::Function::New(env, NapiTensorAddAlpha_));
  exports.Set(Napi::String::New(env, "ts_tensor_optim_add_"),
              Napi::Function::New(env, NapiTensorOptimAdd_));
  exports.Set(Napi::String::New(env, "ts_tensor_sub_inplace"),
              Napi::Function::New(env, NapiTensorSubInplace));
  exports.Set(Napi::String::New(env, "ts_tensor_add_scaled_inplace"),
              Napi::Function::New(env, NapiTensorAddScaledInplace));

  // Output variants
  exports.Set(Napi::String::New(env, "ts_tensor_add_out"),
              Napi::Function::New(env, NapiTensorAddOut));
  exports.Set(Napi::String::New(env, "ts_tensor_sub_out"),
              Napi::Function::New(env, NapiTensorSubOut));
  exports.Set(Napi::String::New(env, "ts_tensor_mul_out"),
              Napi::Function::New(env, NapiTensorMulOut));
  exports.Set(Napi::String::New(env, "ts_tensor_div_out"),
              Napi::Function::New(env, NapiTensorDivOut));
  exports.Set(Napi::String::New(env, "ts_tensor_matmul_out"),
              Napi::Function::New(env, NapiTensorMatmulOut));
}
