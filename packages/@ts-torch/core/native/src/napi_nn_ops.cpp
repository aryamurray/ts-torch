/**
 * NAPI bindings for neural network tensor operations
 *
 * Wraps linear layers, convolutions, pooling, normalization,
 * embedding, and loss functions. Operations that have C API wrappers
 * call through the C layer; operations without C API wrappers call
 * LibTorch directly via the internal ts_Tensor struct.
 */

#include "napi_bindings.h"
#include "ts_torch/internal.h"

// ============================================================================
// Helper: get optional tensor (nullptr/null -> empty torch::Tensor)
// ============================================================================

static torch::Tensor OptionalTensor(const Napi::Value& val) {
  if (val.IsNull() || val.IsUndefined()) return {};
  ts_Tensor* t = GetTensorHandle(val);
  return t ? t->tensor : torch::Tensor();
}

static ts_Tensor* RequireTensor(Napi::Env env, const Napi::Value& val,
                                const char* name) {
  ts_Tensor* t = GetTensorHandle(val);
  if (!t) {
    throw Napi::Error::New(env,
      std::string(name) + ": invalid tensor handle");
  }
  return t;
}

static Napi::Value WrapResult(Napi::Env env, torch::Tensor result) {
  auto* handle = new ts_Tensor(std::move(result));
  register_in_scope(handle);
  return WrapTensorHandle(env, handle);
}

// ============================================================================
// Linear Layers
// ============================================================================

// linear(input, weight, bias?) -> x @ W^T + b
Napi::Value NapiTensorLinear(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "linear");
  auto* weight = RequireTensor(env, info[1], "linear");
  torch::Tensor bias = (info.Length() > 2) ? OptionalTensor(info[2])
                                           : torch::Tensor();
  try {
    auto result = torch::linear(input->tensor, weight->tensor, bias);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("linear: ") + e.what());
  }
}

// linear_relu(input, weight, bias?) -> relu(x @ W^T + b)
Napi::Value NapiTensorLinearRelu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input = RequireTensor(env, info[0], "linear_relu");
  ts_Tensor* weight = RequireTensor(env, info[1], "linear_relu");
  ts_Tensor* bias = (info.Length() > 2 && info[2].IsExternal())
                      ? GetTensorHandle(info[2]) : nullptr;

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_linear_relu(input, weight, bias, &err);
  if (CheckAndThrowError(env, err, "linear_relu")) return env.Null();
  return WrapTensorHandle(env, result);
}

// linear_sigmoid(input, weight, bias?) -> sigmoid(x @ W^T + b)
Napi::Value NapiTensorLinearSigmoid(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input = RequireTensor(env, info[0], "linear_sigmoid");
  ts_Tensor* weight = RequireTensor(env, info[1], "linear_sigmoid");
  ts_Tensor* bias = (info.Length() > 2 && info[2].IsExternal())
                      ? GetTensorHandle(info[2]) : nullptr;

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_linear_sigmoid(input, weight, bias, &err);
  if (CheckAndThrowError(env, err, "linear_sigmoid")) return env.Null();
  return WrapTensorHandle(env, result);
}

// linear_tanh(input, weight, bias?) -> tanh(x @ W^T + b)
Napi::Value NapiTensorLinearTanh(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input = RequireTensor(env, info[0], "linear_tanh");
  ts_Tensor* weight = RequireTensor(env, info[1], "linear_tanh");
  ts_Tensor* bias = (info.Length() > 2 && info[2].IsExternal())
                      ? GetTensorHandle(info[2]) : nullptr;

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_linear_tanh(input, weight, bias, &err);
  if (CheckAndThrowError(env, err, "linear_tanh")) return env.Null();
  return WrapTensorHandle(env, result);
}

// linear_in_place_add(output, input, weight, bias?) -> output += x @ W^T + b
Napi::Value NapiTensorLinearInPlaceAdd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* output = RequireTensor(env, info[0], "linear_in_place_add");
  auto* input  = RequireTensor(env, info[1], "linear_in_place_add");
  auto* weight = RequireTensor(env, info[2], "linear_in_place_add");
  torch::Tensor bias = (info.Length() > 3) ? OptionalTensor(info[3])
                                           : torch::Tensor();
  try {
    auto linear_out = torch::linear(input->tensor, weight->tensor, bias);
    output->tensor.add_(linear_out);
    return env.Undefined();
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("linear_in_place_add: ") + e.what());
  }
}

// bilinear(input1, input2, weight, bias?) -> bilinear transform
Napi::Value NapiTensorBilinear(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input1 = RequireTensor(env, info[0], "bilinear");
  auto* input2 = RequireTensor(env, info[1], "bilinear");
  auto* weight = RequireTensor(env, info[2], "bilinear");
  torch::Tensor bias = (info.Length() > 3) ? OptionalTensor(info[3])
                                           : torch::Tensor();
  try {
    auto result = torch::bilinear(input1->tensor, input2->tensor,
                                  weight->tensor, bias);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("bilinear: ") + e.what());
  }
}

// ============================================================================
// Convolutions
// Args: input, weight, bias?, stride[], padding[], dilation[], groups
// ============================================================================

// Helper to extract int64 array from JS TypedArray or Array
static std::vector<int64_t> GetIntArray(const Napi::Value& val, size_t expect) {
  std::vector<int64_t> out;
  if (val.IsTypedArray()) {
    auto arr = val.As<Napi::TypedArray>();
    auto buf = static_cast<int64_t*>(arr.ArrayBuffer().Data());
    size_t off = arr.ByteOffset() / sizeof(int64_t);
    size_t len = arr.ElementLength();
    out.assign(buf + off, buf + off + len);
  } else if (val.IsArray()) {
    auto arr = val.As<Napi::Array>();
    out.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); i++) {
      out.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
    }
  } else if (val.IsNumber()) {
    out.resize(expect, val.As<Napi::Number>().Int64Value());
  }
  return out;
}

// conv1d(input, weight, bias?, stride, padding, dilation, groups)
Napi::Value NapiTensorConv1d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "conv1d");
  auto* weight = RequireTensor(env, info[1], "conv1d");
  torch::Tensor bias = OptionalTensor(info[2]);
  auto stride   = GetIntArray(info[3], 1);
  auto padding  = GetIntArray(info[4], 1);
  auto dilation = GetIntArray(info[5], 1);
  int64_t groups = info[6].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::conv1d(input->tensor, weight->tensor, bias,
                                stride, padding, dilation, groups);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("conv1d: ") + e.what());
  }
}

// conv2d - delegates to C API
Napi::Value NapiTensorConv2d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input  = RequireTensor(env, info[0], "conv2d");
  ts_Tensor* weight = RequireTensor(env, info[1], "conv2d");
  ts_Tensor* bias   = (info[2].IsNull() || info[2].IsUndefined())
                        ? nullptr : GetTensorHandle(info[2]);
  int64_t stride_h   = info[3].As<Napi::Number>().Int64Value();
  int64_t stride_w   = info[4].As<Napi::Number>().Int64Value();
  int64_t padding_h  = info[5].As<Napi::Number>().Int64Value();
  int64_t padding_w  = info[6].As<Napi::Number>().Int64Value();
  int64_t dilation_h = info[7].As<Napi::Number>().Int64Value();
  int64_t dilation_w = info[8].As<Napi::Number>().Int64Value();
  int64_t groups     = info[9].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_conv2d(input, weight, bias,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w, groups, &err);
  if (CheckAndThrowError(env, err, "conv2d")) return env.Null();
  return WrapTensorHandle(env, result);
}

// conv3d(input, weight, bias?, stride, padding, dilation, groups)
Napi::Value NapiTensorConv3d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "conv3d");
  auto* weight = RequireTensor(env, info[1], "conv3d");
  torch::Tensor bias = OptionalTensor(info[2]);
  auto stride   = GetIntArray(info[3], 3);
  auto padding  = GetIntArray(info[4], 3);
  auto dilation = GetIntArray(info[5], 3);
  int64_t groups = info[6].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::conv3d(input->tensor, weight->tensor, bias,
                                stride, padding, dilation, groups);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("conv3d: ") + e.what());
  }
}

// conv_transpose1d(input, weight, bias?, stride, padding, output_padding, groups, dilation)
Napi::Value NapiTensorConvTranspose1d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "conv_transpose1d");
  auto* weight = RequireTensor(env, info[1], "conv_transpose1d");
  torch::Tensor bias = OptionalTensor(info[2]);
  auto stride   = GetIntArray(info[3], 1);
  auto padding  = GetIntArray(info[4], 1);
  auto out_pad  = GetIntArray(info[5], 1);
  int64_t groups = info[6].As<Napi::Number>().Int64Value();
  auto dilation = GetIntArray(info[7], 1);
  try {
    auto result = torch::conv_transpose1d(input->tensor, weight->tensor, bias,
                                          stride, padding, out_pad, groups,
                                          dilation);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("conv_transpose1d: ") + e.what());
  }
}

// conv_transpose2d(input, weight, bias?, stride, padding, output_padding, groups, dilation)
Napi::Value NapiTensorConvTranspose2d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "conv_transpose2d");
  auto* weight = RequireTensor(env, info[1], "conv_transpose2d");
  torch::Tensor bias = OptionalTensor(info[2]);
  auto stride   = GetIntArray(info[3], 2);
  auto padding  = GetIntArray(info[4], 2);
  auto out_pad  = GetIntArray(info[5], 2);
  int64_t groups = info[6].As<Napi::Number>().Int64Value();
  auto dilation = GetIntArray(info[7], 2);
  try {
    auto result = torch::conv_transpose2d(input->tensor, weight->tensor, bias,
                                          stride, padding, out_pad, groups,
                                          dilation);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("conv_transpose2d: ") + e.what());
  }
}

// conv_transpose3d(input, weight, bias?, stride, padding, output_padding, groups, dilation)
Napi::Value NapiTensorConvTranspose3d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "conv_transpose3d");
  auto* weight = RequireTensor(env, info[1], "conv_transpose3d");
  torch::Tensor bias = OptionalTensor(info[2]);
  auto stride   = GetIntArray(info[3], 3);
  auto padding  = GetIntArray(info[4], 3);
  auto out_pad  = GetIntArray(info[5], 3);
  int64_t groups = info[6].As<Napi::Number>().Int64Value();
  auto dilation = GetIntArray(info[7], 3);
  try {
    auto result = torch::conv_transpose3d(input->tensor, weight->tensor, bias,
                                          stride, padding, out_pad, groups,
                                          dilation);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("conv_transpose3d: ") + e.what());
  }
}

// ============================================================================
// Pooling Operations
// Args: input, kernel_size, stride, padding
// ============================================================================

// max_pool1d(input, kernel, stride, padding)
Napi::Value NapiTensorMaxPool1d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "max_pool1d");
  int64_t kernel  = info[1].As<Napi::Number>().Int64Value();
  int64_t stride  = info[2].As<Napi::Number>().Int64Value();
  int64_t padding = info[3].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::max_pool1d(input->tensor, {kernel}, {stride},
                                    {padding});
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("max_pool1d: ") + e.what());
  }
}

// max_pool2d - delegates to C API
Napi::Value NapiTensorMaxPool2d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input = RequireTensor(env, info[0], "max_pool2d");
  int64_t kh = info[1].As<Napi::Number>().Int64Value();
  int64_t kw = info[2].As<Napi::Number>().Int64Value();
  int64_t sh = info[3].As<Napi::Number>().Int64Value();
  int64_t sw = info[4].As<Napi::Number>().Int64Value();
  int64_t ph = info[5].As<Napi::Number>().Int64Value();
  int64_t pw = info[6].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_max_pool2d(input, kh, kw, sh, sw, ph, pw, &err);
  if (CheckAndThrowError(env, err, "max_pool2d")) return env.Null();
  return WrapTensorHandle(env, result);
}

// max_pool3d(input, kernel[3], stride[3], padding[3])
Napi::Value NapiTensorMaxPool3d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "max_pool3d");
  auto kernel  = GetIntArray(info[1], 3);
  auto stride  = GetIntArray(info[2], 3);
  auto padding = GetIntArray(info[3], 3);
  try {
    auto result = torch::max_pool3d(input->tensor, kernel, stride, padding);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("max_pool3d: ") + e.what());
  }
}

// avg_pool1d(input, kernel, stride, padding)
Napi::Value NapiTensorAvgPool1d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "avg_pool1d");
  int64_t kernel  = info[1].As<Napi::Number>().Int64Value();
  int64_t stride  = info[2].As<Napi::Number>().Int64Value();
  int64_t padding = info[3].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::avg_pool1d(input->tensor, {kernel}, {stride},
                                    {padding});
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("avg_pool1d: ") + e.what());
  }
}

// avg_pool2d - delegates to C API
Napi::Value NapiTensorAvgPool2d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input = RequireTensor(env, info[0], "avg_pool2d");
  int64_t kh = info[1].As<Napi::Number>().Int64Value();
  int64_t kw = info[2].As<Napi::Number>().Int64Value();
  int64_t sh = info[3].As<Napi::Number>().Int64Value();
  int64_t sw = info[4].As<Napi::Number>().Int64Value();
  int64_t ph = info[5].As<Napi::Number>().Int64Value();
  int64_t pw = info[6].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_avg_pool2d(input, kh, kw, sh, sw, ph, pw, &err);
  if (CheckAndThrowError(env, err, "avg_pool2d")) return env.Null();
  return WrapTensorHandle(env, result);
}

// avg_pool3d(input, kernel[3], stride[3], padding[3])
Napi::Value NapiTensorAvgPool3d(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "avg_pool3d");
  auto kernel  = GetIntArray(info[1], 3);
  auto stride  = GetIntArray(info[2], 3);
  auto padding = GetIntArray(info[3], 3);
  try {
    auto result = torch::avg_pool3d(input->tensor, kernel, stride, padding);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("avg_pool3d: ") + e.what());
  }
}

// adaptive_avg_pool(input, output_size[])
Napi::Value NapiTensorAdaptiveAvgPool(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "adaptive_avg_pool");
  auto output_size = GetIntArray(info[1], 0);
  try {
    torch::Tensor result;
    if (output_size.size() == 1) {
      result = torch::adaptive_avg_pool1d(input->tensor, output_size);
    } else if (output_size.size() == 2) {
      result = torch::adaptive_avg_pool2d(input->tensor, output_size);
    } else {
      result = torch::adaptive_avg_pool3d(input->tensor, output_size);
    }
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("adaptive_avg_pool: ") + e.what());
  }
}

// adaptive_max_pool(input, output_size[])
Napi::Value NapiTensorAdaptiveMaxPool(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "adaptive_max_pool");
  auto output_size = GetIntArray(info[1], 0);
  try {
    torch::Tensor result;
    if (output_size.size() == 1) {
      auto [values, indices] =
        torch::adaptive_max_pool1d(input->tensor, output_size);
      result = values;
    } else if (output_size.size() == 2) {
      auto [values, indices] =
        torch::adaptive_max_pool2d(input->tensor, output_size);
      result = values;
    } else {
      auto [values, indices] =
        torch::adaptive_max_pool3d(input->tensor, output_size);
      result = values;
    }
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("adaptive_max_pool: ") + e.what());
  }
}

// ============================================================================
// Normalization
// ============================================================================

// batch_norm - delegates to C API
Napi::Value NapiTensorBatchNorm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input = RequireTensor(env, info[0], "batch_norm");
  ts_Tensor* weight =
    (info[1].IsNull() || info[1].IsUndefined()) ? nullptr : GetTensorHandle(info[1]);
  ts_Tensor* bias =
    (info[2].IsNull() || info[2].IsUndefined()) ? nullptr : GetTensorHandle(info[2]);
  ts_Tensor* running_mean =
    (info[3].IsNull() || info[3].IsUndefined()) ? nullptr : GetTensorHandle(info[3]);
  ts_Tensor* running_var =
    (info[4].IsNull() || info[4].IsUndefined()) ? nullptr : GetTensorHandle(info[4]);
  int training    = info[5].As<Napi::Number>().Int32Value();
  double momentum = info[6].As<Napi::Number>().DoubleValue();
  double eps      = info[7].As<Napi::Number>().DoubleValue();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_batch_norm(
    input, weight, bias, running_mean, running_var,
    training, momentum, eps, &err);
  if (CheckAndThrowError(env, err, "batch_norm")) return env.Null();
  return WrapTensorHandle(env, result);
}

// instance_norm(input, weight?, bias?, running_mean?, running_var?,
//               training, momentum, eps)
Napi::Value NapiTensorInstanceNorm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "instance_norm");
  torch::Tensor weight       = OptionalTensor(info[1]);
  torch::Tensor bias         = OptionalTensor(info[2]);
  torch::Tensor running_mean = OptionalTensor(info[3]);
  torch::Tensor running_var  = OptionalTensor(info[4]);
  bool training   = info[5].As<Napi::Boolean>().Value();
  double momentum = info[6].As<Napi::Number>().DoubleValue();
  double eps      = info[7].As<Napi::Number>().DoubleValue();
  try {
    auto result = torch::instance_norm(input->tensor, weight, bias,
                                       running_mean, running_var,
                                       training, momentum, eps, true);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("instance_norm: ") + e.what());
  }
}

// layer_norm - delegates to C API
// Note: LayerNorm is implemented in napi_tensor_unary_ops.cpp

// group_norm(input, num_groups, weight?, bias?, eps)
Napi::Value NapiTensorGroupNorm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "group_norm");
  int64_t num_groups = info[1].As<Napi::Number>().Int64Value();
  torch::Tensor weight = OptionalTensor(info[2]);
  torch::Tensor bias   = OptionalTensor(info[3]);
  double eps = info[4].As<Napi::Number>().DoubleValue();
  try {
    auto result = torch::group_norm(input->tensor, num_groups,
                                    weight, bias, eps);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("group_norm: ") + e.what());
  }
}

// local_response_norm(input, size, alpha, beta, k)
Napi::Value NapiTensorLocalResponseNorm(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "local_response_norm");
  int64_t size = info[1].As<Napi::Number>().Int64Value();
  double alpha = info[2].As<Napi::Number>().DoubleValue();
  double beta  = info[3].As<Napi::Number>().DoubleValue();
  double k     = info[4].As<Napi::Number>().DoubleValue();
  try {
    auto result = torch::nn::functional::local_response_norm(
      input->tensor,
      torch::nn::functional::LocalResponseNormFuncOptions(size)
        .alpha(alpha).beta(beta).k(k));
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("local_response_norm: ") + e.what());
  }
}

// ============================================================================
// Embedding
// ============================================================================

// embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)
Napi::Value NapiTensorEmbedding(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* weight  = RequireTensor(env, info[0], "embedding");
  auto* indices = RequireTensor(env, info[1], "embedding");
  int64_t padding_idx = info[2].As<Napi::Number>().Int64Value();
  bool scale_grad     = info[3].As<Napi::Boolean>().Value();
  bool sparse         = info[4].As<Napi::Boolean>().Value();
  try {
    auto result = torch::embedding(weight->tensor, indices->tensor,
                                   padding_idx, scale_grad, sparse);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("embedding: ") + e.what());
  }
}

// embedding_bag(weight, indices, offsets, scale_grad, mode, sparse, per_sample_weights?)
Napi::Value NapiTensorEmbeddingBag(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* weight  = RequireTensor(env, info[0], "embedding_bag");
  auto* indices = RequireTensor(env, info[1], "embedding_bag");
  auto* offsets = RequireTensor(env, info[2], "embedding_bag");
  bool scale_grad = info[3].As<Napi::Boolean>().Value();
  int64_t mode    = info[4].As<Napi::Number>().Int64Value(); // 0=sum,1=mean,2=max
  bool sparse     = info[5].As<Napi::Boolean>().Value();
  torch::Tensor per_sample_weights =
    (info.Length() > 6) ? OptionalTensor(info[6]) : torch::Tensor();
  try {
    auto [result, offset_out, bag_size, max_indices] =
      torch::embedding_bag(weight->tensor, indices->tensor, offsets->tensor,
                           scale_grad, mode, sparse, per_sample_weights);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("embedding_bag: ") + e.what());
  }
}

// ============================================================================
// Loss Functions
// ============================================================================

// cross_entropy_loss - delegates to C API
Napi::Value NapiTensorCrossEntropyLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* logits  = RequireTensor(env, info[0], "cross_entropy_loss");
  ts_Tensor* targets = RequireTensor(env, info[1], "cross_entropy_loss");

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_cross_entropy_loss(logits, targets, &err);
  if (CheckAndThrowError(env, err, "cross_entropy_loss")) return env.Null();
  return WrapTensorHandle(env, result);
}

// nll_loss - delegates to C API
Napi::Value NapiTensorNllLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* log_probs = RequireTensor(env, info[0], "nll_loss");
  ts_Tensor* targets   = RequireTensor(env, info[1], "nll_loss");

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_nll_loss(log_probs, targets, &err);
  if (CheckAndThrowError(env, err, "nll_loss")) return env.Null();
  return WrapTensorHandle(env, result);
}

// mse_loss - delegates to C API
Napi::Value NapiTensorMseLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  ts_Tensor* input  = RequireTensor(env, info[0], "mse_loss");
  ts_Tensor* target = RequireTensor(env, info[1], "mse_loss");

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_mse_loss(input, target, &err);
  if (CheckAndThrowError(env, err, "mse_loss")) return env.Null();
  return WrapTensorHandle(env, result);
}

// l1_loss(input, target, reduction)
// reduction: 0=none, 1=mean, 2=sum
Napi::Value NapiTensorL1Loss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "l1_loss");
  auto* target = RequireTensor(env, info[1], "l1_loss");
  int64_t reduction = info[2].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::l1_loss(input->tensor, target->tensor,
                                 reduction);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("l1_loss: ") + e.what());
  }
}

// smooth_l1_loss(input, target, reduction, beta)
Napi::Value NapiTensorSmoothL1Loss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "smooth_l1_loss");
  auto* target = RequireTensor(env, info[1], "smooth_l1_loss");
  int64_t reduction = info[2].As<Napi::Number>().Int64Value();
  double beta = info[3].As<Napi::Number>().DoubleValue();
  try {
    auto result = torch::smooth_l1_loss(input->tensor, target->tensor,
                                        reduction, beta);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("smooth_l1_loss: ") + e.what());
  }
}

// huber_loss(input, target, reduction, delta)
Napi::Value NapiTensorHuberLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "huber_loss");
  auto* target = RequireTensor(env, info[1], "huber_loss");
  int64_t reduction = info[2].As<Napi::Number>().Int64Value();
  double delta = info[3].As<Napi::Number>().DoubleValue();
  try {
    auto result = torch::huber_loss(input->tensor, target->tensor,
                                    reduction, delta);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("huber_loss: ") + e.what());
  }
}

// bce_loss(input, target, weight?, reduction)
Napi::Value NapiTensorBceLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "bce_loss");
  auto* target = RequireTensor(env, info[1], "bce_loss");
  torch::Tensor weight = OptionalTensor(info[2]);
  int64_t reduction = info[3].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::binary_cross_entropy(
      input->tensor, target->tensor, weight, reduction);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("bce_loss: ") + e.what());
  }
}

// bce_with_logits_loss(input, target, weight?, pos_weight?, reduction)
Napi::Value NapiTensorBceWithLogitsLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "bce_with_logits_loss");
  auto* target = RequireTensor(env, info[1], "bce_with_logits_loss");
  torch::Tensor weight     = OptionalTensor(info[2]);
  torch::Tensor pos_weight = OptionalTensor(info[3]);
  int64_t reduction = info[4].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::binary_cross_entropy_with_logits(
      input->tensor, target->tensor, weight, pos_weight, reduction);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("bce_with_logits_loss: ") + e.what());
  }
}

// kl_div_loss(input, target, reduction, log_target)
Napi::Value NapiTensorKlDivLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input  = RequireTensor(env, info[0], "kl_div_loss");
  auto* target = RequireTensor(env, info[1], "kl_div_loss");
  int64_t reduction = info[2].As<Napi::Number>().Int64Value();
  bool log_target = info[3].As<Napi::Boolean>().Value();
  try {
    auto result = torch::kl_div(input->tensor, target->tensor,
                                reduction, log_target);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("kl_div_loss: ") + e.what());
  }
}

// margin_ranking_loss(input1, input2, target, margin, reduction)
Napi::Value NapiTensorMarginRankingLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input1 = RequireTensor(env, info[0], "margin_ranking_loss");
  auto* input2 = RequireTensor(env, info[1], "margin_ranking_loss");
  auto* target = RequireTensor(env, info[2], "margin_ranking_loss");
  double margin = info[3].As<Napi::Number>().DoubleValue();
  int64_t reduction = info[4].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::margin_ranking_loss(
      input1->tensor, input2->tensor, target->tensor, margin, reduction);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("margin_ranking_loss: ") + e.what());
  }
}

// triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction)
Napi::Value NapiTensorTripletMarginLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* anchor   = RequireTensor(env, info[0], "triplet_margin_loss");
  auto* positive = RequireTensor(env, info[1], "triplet_margin_loss");
  auto* negative = RequireTensor(env, info[2], "triplet_margin_loss");
  double margin  = info[3].As<Napi::Number>().DoubleValue();
  double p       = info[4].As<Napi::Number>().DoubleValue();
  double eps     = info[5].As<Napi::Number>().DoubleValue();
  bool swap      = info[6].As<Napi::Boolean>().Value();
  int64_t reduction = info[7].As<Napi::Number>().Int64Value();
  try {
    auto result = torch::triplet_margin_loss(
      anchor->tensor, positive->tensor, negative->tensor,
      margin, p, eps, swap, reduction);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("triplet_margin_loss: ") + e.what());
  }
}

// ============================================================================
// Other Operations
// ============================================================================

// grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)
Napi::Value NapiTensorGridSampler(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* input = RequireTensor(env, info[0], "grid_sampler");
  auto* grid  = RequireTensor(env, info[1], "grid_sampler");
  int64_t interp_mode  = info[2].As<Napi::Number>().Int64Value();
  int64_t padding_mode = info[3].As<Napi::Number>().Int64Value();
  bool align_corners   = info[4].As<Napi::Boolean>().Value();
  try {
    auto result = torch::grid_sampler(input->tensor, grid->tensor,
                                      interp_mode, padding_mode,
                                      align_corners);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("grid_sampler: ") + e.what());
  }
}

// affine_grid_generator(theta, size[], align_corners)
Napi::Value NapiTensorAffineGridGenerator(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* theta = RequireTensor(env, info[0], "affine_grid_generator");
  auto size = GetIntArray(info[1], 0);
  bool align_corners = info[2].As<Napi::Boolean>().Value();
  try {
    auto result = torch::affine_grid_generator(theta->tensor, size,
                                               align_corners);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env,
      std::string("affine_grid_generator: ") + e.what());
  }
}

// ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
Napi::Value NapiTensorCtcLoss(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  auto* log_probs      = RequireTensor(env, info[0], "ctc_loss");
  auto* targets        = RequireTensor(env, info[1], "ctc_loss");
  auto* input_lengths  = RequireTensor(env, info[2], "ctc_loss");
  auto* target_lengths = RequireTensor(env, info[3], "ctc_loss");
  int64_t blank        = info[4].As<Napi::Number>().Int64Value();
  int64_t reduction    = info[5].As<Napi::Number>().Int64Value();
  bool zero_infinity   = info[6].As<Napi::Boolean>().Value();
  try {
    auto result = torch::ctc_loss(log_probs->tensor, targets->tensor,
                                  input_lengths->tensor,
                                  target_lengths->tensor,
                                  blank, reduction, zero_infinity);
    return WrapResult(env, std::move(result));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("ctc_loss: ") + e.what());
  }
}

// ============================================================================
// Registration: call from Init() in napi_bindings.cpp
// ============================================================================

void RegisterNNOps(Napi::Env env, Napi::Object exports) {
  // Linear layers
  exports.Set("ts_tensor_linear",
    Napi::Function::New(env, NapiTensorLinear));
  exports.Set("ts_tensor_linear_relu",
    Napi::Function::New(env, NapiTensorLinearRelu));
  exports.Set("ts_tensor_linear_sigmoid",
    Napi::Function::New(env, NapiTensorLinearSigmoid));
  exports.Set("ts_tensor_linear_tanh",
    Napi::Function::New(env, NapiTensorLinearTanh));
  exports.Set("ts_tensor_linear_in_place_add",
    Napi::Function::New(env, NapiTensorLinearInPlaceAdd));
  exports.Set("ts_tensor_bilinear",
    Napi::Function::New(env, NapiTensorBilinear));

  // Convolutions
  exports.Set("ts_tensor_conv1d",
    Napi::Function::New(env, NapiTensorConv1d));
  exports.Set("ts_tensor_conv2d",
    Napi::Function::New(env, NapiTensorConv2d));
  exports.Set("ts_tensor_conv3d",
    Napi::Function::New(env, NapiTensorConv3d));
  exports.Set("ts_tensor_conv_transpose1d",
    Napi::Function::New(env, NapiTensorConvTranspose1d));
  exports.Set("ts_tensor_conv_transpose2d",
    Napi::Function::New(env, NapiTensorConvTranspose2d));
  exports.Set("ts_tensor_conv_transpose3d",
    Napi::Function::New(env, NapiTensorConvTranspose3d));

  // Pooling
  exports.Set("ts_tensor_max_pool1d",
    Napi::Function::New(env, NapiTensorMaxPool1d));
  exports.Set("ts_tensor_max_pool2d",
    Napi::Function::New(env, NapiTensorMaxPool2d));
  exports.Set("ts_tensor_max_pool3d",
    Napi::Function::New(env, NapiTensorMaxPool3d));
  exports.Set("ts_tensor_avg_pool1d",
    Napi::Function::New(env, NapiTensorAvgPool1d));
  exports.Set("ts_tensor_avg_pool2d",
    Napi::Function::New(env, NapiTensorAvgPool2d));
  exports.Set("ts_tensor_avg_pool3d",
    Napi::Function::New(env, NapiTensorAvgPool3d));
  exports.Set("ts_tensor_adaptive_avg_pool",
    Napi::Function::New(env, NapiTensorAdaptiveAvgPool));
  exports.Set("ts_tensor_adaptive_max_pool",
    Napi::Function::New(env, NapiTensorAdaptiveMaxPool));

  // Normalization
  exports.Set("ts_tensor_batch_norm",
    Napi::Function::New(env, NapiTensorBatchNorm));
  exports.Set("ts_tensor_instance_norm",
    Napi::Function::New(env, NapiTensorInstanceNorm));
  // Note: ts_tensor_layer_norm is registered in RegisterTensorUnaryOps
  exports.Set("ts_tensor_group_norm",
    Napi::Function::New(env, NapiTensorGroupNorm));
  exports.Set("ts_tensor_local_response_norm",
    Napi::Function::New(env, NapiTensorLocalResponseNorm));

  // Embedding
  exports.Set("ts_tensor_embedding",
    Napi::Function::New(env, NapiTensorEmbedding));
  exports.Set("ts_tensor_embedding_bag",
    Napi::Function::New(env, NapiTensorEmbeddingBag));

  // Loss functions
  exports.Set("ts_tensor_cross_entropy_loss",
    Napi::Function::New(env, NapiTensorCrossEntropyLoss));
  exports.Set("ts_tensor_nll_loss",
    Napi::Function::New(env, NapiTensorNllLoss));
  exports.Set("ts_tensor_mse_loss",
    Napi::Function::New(env, NapiTensorMseLoss));
  exports.Set("ts_tensor_l1_loss",
    Napi::Function::New(env, NapiTensorL1Loss));
  exports.Set("ts_tensor_smooth_l1_loss",
    Napi::Function::New(env, NapiTensorSmoothL1Loss));
  exports.Set("ts_tensor_huber_loss",
    Napi::Function::New(env, NapiTensorHuberLoss));
  exports.Set("ts_tensor_bce_loss",
    Napi::Function::New(env, NapiTensorBceLoss));
  exports.Set("ts_tensor_bce_with_logits_loss",
    Napi::Function::New(env, NapiTensorBceWithLogitsLoss));
  exports.Set("ts_tensor_kl_div_loss",
    Napi::Function::New(env, NapiTensorKlDivLoss));
  exports.Set("ts_tensor_margin_ranking_loss",
    Napi::Function::New(env, NapiTensorMarginRankingLoss));
  exports.Set("ts_tensor_triplet_margin_loss",
    Napi::Function::New(env, NapiTensorTripletMarginLoss));

  // Other
  exports.Set("ts_tensor_grid_sampler",
    Napi::Function::New(env, NapiTensorGridSampler));
  exports.Set("ts_tensor_affine_grid_generator",
    Napi::Function::New(env, NapiTensorAffineGridGenerator));
  exports.Set("ts_tensor_ctc_loss",
    Napi::Function::New(env, NapiTensorCtcLoss));
}
