/**
 * Napi bindings for remaining tensor operations
 *
 * Covers: batch ops, device management, threading, utilities,
 * tensor properties, autograd advanced, and data access.
 *
 * Follows the same conventions as napi_bindings.cpp:
 * - GetTensorHandle / WrapTensorHandle for handle marshalling
 * - CheckAndThrowError for C error -> JS exception conversion
 * - ts_Error struct for C API calls; direct try/catch for inline LibTorch calls
 */

#include "napi_bindings.h"

// Internal header needed for direct LibTorch calls (item, strides, etc.)
#include "ts_torch/internal.h"

// ============================================================================
// Batch Operations
// ============================================================================

static ts_Batch* GetBatchHandle(const Napi::Value& val) {
  if (!val.IsExternal()) {
    return nullptr;
  }
  return static_cast<ts_Batch*>(val.As<Napi::External<void>>().Data());
}

Napi::Value NapiBatchBegin(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  ts_Error err = {0, ""};
  ts_BatchHandle batch = ts_batch_begin(&err);

  if (CheckAndThrowError(env, err, "ts_batch_begin")) {
    return env.Null();
  }

  // No destructor - batch lifetime is managed by batch_end / batch_abort
  return Napi::External<void>::New(env, static_cast<void*>(batch));
}

Napi::Value NapiBatchEnd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_batch_end requires 1 argument");
  }

  ts_Batch* batch = GetBatchHandle(info[0]);
  if (!batch) {
    throw Napi::Error::New(env, "Invalid batch handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_batch_end(batch, &err);

  if (CheckAndThrowError(env, err, "ts_batch_end")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiBatchIsRecording(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Boolean::New(env, ts_batch_is_recording() != 0);
}

Napi::Value NapiBatchAbort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_batch_abort requires 1 argument");
  }

  ts_Batch* batch = GetBatchHandle(info[0]);
  if (batch) {
    ts_batch_abort(batch);
  }

  return env.Undefined();
}

// ============================================================================
// Device Operations
// ============================================================================

Napi::Value NapiCudaDeviceCount(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, ts_cuda_device_count());
}

Napi::Value NapiCudaSynchronize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  try {
#ifdef USE_CUDA
    torch::cuda::synchronize();
#endif
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("cuda_synchronize: ") + e.what());
  }
  return env.Undefined();
}

Napi::Value NapiCudaEmptyCache(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  try {
#ifdef USE_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("cuda_empty_cache: ") + e.what());
  }
  return env.Undefined();
}

Napi::Value NapiCudaResetPeakMemory(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  try {
#ifdef USE_CUDA
    int device = 0;
    if (info.Length() > 0) {
      device = info[0].As<Napi::Number>().Int32Value();
    }
    c10::cuda::CUDACachingAllocator::resetPeakStats(device);
#endif
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("cuda_reset_peak_memory: ") + e.what());
  }
  return env.Undefined();
}

Napi::Value NapiIsAvailable(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "is_available requires 1 argument (device_type)");
  }

  int device_type = info[0].As<Napi::Number>().Int32Value();

  switch (device_type) {
    case TS_DEVICE_CPU:
      return Napi::Boolean::New(env, true);
    case TS_DEVICE_CUDA:
      return Napi::Boolean::New(env, ts_cuda_is_available() != 0);
    case TS_DEVICE_MPS:
#ifdef USE_MPS
      return Napi::Boolean::New(env, torch::mps::is_available());
#else
      return Napi::Boolean::New(env, false);
#endif
    default:
      return Napi::Boolean::New(env, false);
  }
}

Napi::Value NapiCurrentDevice(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  try {
    int device = 0;
#ifdef USE_CUDA
    if (torch::cuda::is_available()) {
      device = static_cast<int>(c10::cuda::current_device());
    }
#endif
    return Napi::Number::New(env, device);
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("current_device: ") + e.what());
  }
}

Napi::Value NapiSetDevice(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "set_device requires 1 argument");
  }

  int device = info[0].As<Napi::Number>().Int32Value();

  try {
#ifdef USE_CUDA
    c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));
#else
    (void)device;  // suppress unused warning on non-CUDA builds
#endif
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("set_device: ") + e.what());
  }
  return env.Undefined();
}

// ============================================================================
// Threading
// ============================================================================

Napi::Value NapiGetNumThreads(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Number::New(env, ts_get_num_threads());
}

Napi::Value NapiSetNumThreads(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "ts_set_num_threads requires 1 argument");
  }

  int num_threads = info[0].As<Napi::Number>().Int32Value();
  ts_set_num_threads(num_threads);

  return env.Undefined();
}

Napi::Value NapiInParallelRegion(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Boolean::New(env, at::in_parallel_region());
}

// ============================================================================
// Utilities: RNG & Default Dtype
// ============================================================================

Napi::Value NapiManualSeed(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "manual_seed requires 1 argument");
  }

  int64_t seed = info[0].As<Napi::Number>().Int64Value();

  try {
    torch::manual_seed(seed);
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("manual_seed: ") + e.what());
  }
  return env.Undefined();
}

Napi::Value NapiSeed(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  try {
    // seed() returns the new seed value after seeding from random source
    auto gen = at::detail::getDefaultCPUGenerator();
    uint64_t s = gen.seed();
    return Napi::Number::New(env, static_cast<double>(s));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("seed: ") + e.what());
  }
}

Napi::Value NapiGetSeed(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  try {
    auto gen = at::detail::getDefaultCPUGenerator();
    uint64_t s = gen.current_seed();
    return Napi::Number::New(env, static_cast<double>(s));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("get_seed: ") + e.what());
  }
}

Napi::Value NapiGetDefaultDtype(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  torch::ScalarType st = torch::typeMetaToScalarType(torch::get_default_dtype());
  return Napi::Number::New(env, static_cast<int>(scalar_type_to_dtype(st)));
}

Napi::Value NapiSetDefaultDtype(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "set_default_dtype requires 1 argument");
  }

  ts_DType dtype = static_cast<ts_DType>(info[0].As<Napi::Number>().Int32Value());
  torch::ScalarType st = dtype_to_scalar_type(dtype);

  try {
    torch::set_default_dtype(torch::scalarTypeToTypeMeta(st));
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("set_default_dtype: ") + e.what());
  }
  return env.Undefined();
}

// ============================================================================
// Tensor Properties: is_contiguous, contiguous, strides, storage_offset
// ============================================================================

Napi::Value NapiTensorIsContiguous(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "is_contiguous requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  return Napi::Boolean::New(env, tensor->tensor.is_contiguous());
}

Napi::Value NapiTensorContiguous(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "contiguous requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  try {
    auto result = tensor->tensor.contiguous();
    auto* handle = new ts_Tensor(std::move(result));
    register_in_scope(handle);
    return WrapTensorHandle(env, handle);
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("contiguous: ") + e.what());
  }
}

Napi::Value NapiTensorStrides(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "strides requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  auto strides = tensor->tensor.strides();
  Napi::Array result = Napi::Array::New(env, strides.size());
  for (size_t i = 0; i < strides.size(); i++) {
    result.Set(static_cast<uint32_t>(i), Napi::Number::New(env, strides[i]));
  }
  return result;
}

Napi::Value NapiTensorStorageOffset(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "storage_offset requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  return Napi::Number::New(env, static_cast<double>(tensor->tensor.storage_offset()));
}

// ============================================================================
// Advanced Autograd: requires_grad, set_requires_grad_, retain_grad, is_leaf
// ============================================================================

Napi::Value NapiTensorRequiresGrad(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "requires_grad requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  int result = ts_tensor_requires_grad(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_requires_grad")) {
    return env.Null();
  }

  return Napi::Boolean::New(env, result != 0);
}

Napi::Value NapiTensorSetRequiresGrad(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "set_requires_grad_ requires 2 arguments");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int requires_grad = info[1].As<Napi::Boolean>().Value() ? 1 : 0;

  ts_Error err = {0, ""};
  ts_tensor_set_requires_grad(tensor, requires_grad, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_set_requires_grad")) {
    return env.Null();
  }

  return env.Undefined();
}

Napi::Value NapiTensorRetainGrad(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "retain_grad requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  try {
    tensor->tensor.retain_grad();
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("retain_grad: ") + e.what());
  }
  return env.Undefined();
}

Napi::Value NapiTensorIsLeaf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "is_leaf requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  int result = ts_tensor_is_leaf(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_is_leaf")) {
    return env.Null();
  }

  return Napi::Boolean::New(env, result != 0);
}

Napi::Value NapiTensorZeroGrad(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "zero_grad requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_tensor_zero_grad(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_zero_grad")) {
    return env.Null();
  }

  return env.Undefined();
}

Napi::Value NapiTensorDetach(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "detach requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_detach(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_detach")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

// ============================================================================
// Scalar Extraction: item()
// ============================================================================

Napi::Value NapiTensorItem(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "item requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  try {
    double value = tensor->tensor.item<double>();
    return Napi::Number::New(env, value);
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("item: ") + e.what());
  }
}

// ============================================================================
// Data Access: data_ptr, copy_to_buffer, copy_from_buffer
// ============================================================================

Napi::Value NapiTensorDataPtr(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "data_ptr requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  void* ptr = ts_tensor_data_ptr(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_data_ptr")) {
    return env.Null();
  }

  // Return as an ArrayBuffer wrapping the tensor's data (no copy, zero-copy view)
  // The buffer is only valid while the tensor is alive
  size_t byte_size = tensor->tensor.numel() * tensor->tensor.element_size();
  return Napi::ArrayBuffer::New(env, ptr, byte_size);
}

Napi::Value NapiTensorCopyToBuffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "copy_to_buffer requires 2 arguments (tensor, buffer)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  Napi::ArrayBuffer buf = info[1].As<Napi::ArrayBuffer>();
  void* buffer = buf.Data();
  size_t buffer_size = buf.ByteLength();

  ts_Error err = {0, ""};
  ts_tensor_copy_to_buffer(tensor, buffer, buffer_size, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_copy_to_buffer")) {
    return env.Null();
  }

  return env.Undefined();
}

Napi::Value NapiTensorCopyFromBuffer(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "copy_from_buffer requires 2 arguments (tensor, buffer)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  Napi::ArrayBuffer buf = info[1].As<Napi::ArrayBuffer>();
  const void* src = buf.Data();
  size_t src_size = buf.ByteLength();

  try {
    size_t tensor_bytes = tensor->tensor.numel() * tensor->tensor.element_size();

    if (src_size < tensor_bytes) {
      throw Napi::Error::New(env, "Source buffer too small for tensor");
    }

    if (!tensor->tensor.is_contiguous()) {
      throw Napi::Error::New(env, "copy_from_buffer: tensor must be contiguous");
    }

    std::memcpy(tensor->tensor.data_ptr(), src, tensor_bytes);
  } catch (const Napi::Error&) {
    throw;  // re-throw Napi errors as-is
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("copy_from_buffer: ") + e.what());
  }

  return env.Undefined();
}

// ============================================================================
// Tensor Clone / Cat / Narrow / Triu
// ============================================================================

Napi::Value NapiTensorClone(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "clone requires 1 argument");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_clone(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_clone")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorCat(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "cat requires 2 arguments (tensors array, dim)");
  }

  // First arg is a JS array of tensor handles (External<void>)
  if (!info[0].IsArray()) {
    throw Napi::Error::New(env, "cat: first argument must be an array of tensor handles");
  }
  Napi::Array arr = info[0].As<Napi::Array>();
  size_t num_tensors = arr.Length();
  if (num_tensors == 0) {
    throw Napi::Error::New(env, "cat: tensor array must not be empty");
  }

  std::vector<ts_Tensor*> handles(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    ts_Tensor* t = GetTensorHandle(arr.Get(static_cast<uint32_t>(i)));
    if (!t) {
      throw Napi::Error::New(env, "cat: invalid tensor handle at index " + std::to_string(i));
    }
    handles[i] = t;
  }

  int64_t dim = info[1].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_cat(handles.data(), num_tensors, dim, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_cat")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorNarrow(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "narrow requires 4 arguments (tensor, dim, start, length)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  int64_t start = info[2].As<Napi::Number>().Int64Value();
  int64_t length = info[3].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_narrow(tensor, dim, start, length, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_narrow")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorTriu(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "triu requires 2 arguments (tensor, diagonal)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t diagonal = info[1].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_triu(tensor, diagonal, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_triu")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorTopk(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 5) {
    throw Napi::Error::New(env, "topk requires 5 arguments (tensor, k, dim, largest, sorted)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t k = info[1].As<Napi::Number>().Int64Value();
  int64_t dim = info[2].As<Napi::Number>().Int64Value();
  int largest = info[3].As<Napi::Boolean>().Value() ? 1 : 0;
  int sorted = info[4].As<Napi::Boolean>().Value() ? 1 : 0;

  ts_Tensor* indices_out = nullptr;
  ts_Error err = {0, ""};
  ts_Tensor* values = ts_tensor_topk(tensor, k, dim, largest, sorted, &indices_out, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_topk")) {
    return env.Null();
  }

  Napi::Array result = Napi::Array::New(env, 2);
  result.Set(static_cast<uint32_t>(0), WrapTensorHandle(env, values));
  result.Set(static_cast<uint32_t>(1), WrapTensorHandle(env, indices_out));
  return result;
}

Napi::Value NapiTensorSort(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "sort requires 3 arguments (tensor, dim, descending)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t dim = info[1].As<Napi::Number>().Int64Value();
  int descending = info[2].As<Napi::Boolean>().Value() ? 1 : 0;

  ts_Tensor* indices_out = nullptr;
  ts_Error err = {0, ""};
  ts_Tensor* values = ts_tensor_sort(tensor, dim, descending, &indices_out, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_sort")) {
    return env.Null();
  }

  Napi::Array result = Napi::Array::New(env, 2);
  result.Set(static_cast<uint32_t>(0), WrapTensorHandle(env, values));
  result.Set(static_cast<uint32_t>(1), WrapTensorHandle(env, indices_out));
  return result;
}

Napi::Value NapiTensorEinsum(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "einsum requires 2 arguments (equation, tensors array)");
  }

  std::string equation = info[0].As<Napi::String>().Utf8Value();

  if (!info[1].IsArray()) {
    throw Napi::Error::New(env, "einsum: second argument must be an array of tensor handles");
  }
  Napi::Array arr = info[1].As<Napi::Array>();
  size_t num_tensors = arr.Length();
  if (num_tensors == 0) {
    throw Napi::Error::New(env, "einsum: tensor array must not be empty");
  }

  std::vector<ts_Tensor*> handles(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    ts_Tensor* t = GetTensorHandle(arr.Get(static_cast<uint32_t>(i)));
    if (!t) {
      throw Napi::Error::New(env, "einsum: invalid tensor handle at index " + std::to_string(i));
    }
    handles[i] = t;
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_einsum(equation.c_str(), handles.data(), num_tensors, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_einsum")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorTril(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "tril requires 2 arguments (tensor, diagonal)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  int64_t diagonal = info[1].As<Napi::Number>().Int64Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_tril(tensor, diagonal, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_tril")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorToDevice(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 3) {
    throw Napi::Error::New(env, "to_device requires 3 arguments (tensor, device, device_index)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_DeviceType device = static_cast<ts_DeviceType>(info[1].As<Napi::Number>().Int32Value());
  int device_index = info[2].As<Napi::Number>().Int32Value();

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_to_device(tensor, device, device_index, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_to_device")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorMlpForward(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 4) {
    throw Napi::Error::New(env, "mlp_forward requires 4 arguments (input, weights, biases, apply_relu_except_last)");
  }

  ts_Tensor* input = GetTensorHandle(info[0]);
  if (!input) {
    throw Napi::Error::New(env, "Invalid input tensor handle");
  }

  // Extract weights array
  if (!info[1].IsArray()) {
    throw Napi::Error::New(env, "mlp_forward: weights must be an array of tensor handles");
  }
  Napi::Array weights_arr = info[1].As<Napi::Array>();
  size_t num_layers = weights_arr.Length();
  if (num_layers == 0) {
    throw Napi::Error::New(env, "mlp_forward: weights array must not be empty");
  }

  std::vector<ts_Tensor*> weights(num_layers);
  for (size_t i = 0; i < num_layers; i++) {
    ts_Tensor* t = GetTensorHandle(weights_arr.Get(static_cast<uint32_t>(i)));
    if (!t) {
      throw Napi::Error::New(env, "mlp_forward: invalid weight tensor handle at index " + std::to_string(i));
    }
    weights[i] = t;
  }

  // Extract biases array (allow null entries)
  if (!info[2].IsArray()) {
    throw Napi::Error::New(env, "mlp_forward: biases must be an array of tensor handles or nulls");
  }
  Napi::Array biases_arr = info[2].As<Napi::Array>();
  if (biases_arr.Length() != num_layers) {
    throw Napi::Error::New(env, "mlp_forward: biases array must have same length as weights array");
  }

  std::vector<ts_Tensor*> biases(num_layers);
  for (size_t i = 0; i < num_layers; i++) {
    Napi::Value val = biases_arr.Get(static_cast<uint32_t>(i));
    if (val.IsNull() || val.IsUndefined()) {
      biases[i] = nullptr;
    } else {
      ts_Tensor* t = GetTensorHandle(val);
      if (!t) {
        throw Napi::Error::New(env, "mlp_forward: invalid bias tensor handle at index " + std::to_string(i));
      }
      biases[i] = t;
    }
  }

  int apply_relu_except_last = info[3].As<Napi::Boolean>().Value() ? 1 : 0;

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_mlp_forward(
    input, weights.data(), biases.data(), num_layers, apply_relu_except_last, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_mlp_forward")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorRepeat(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "repeat requires 2 arguments (tensor, repeats)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  // Extract repeats from TypedArray (BigInt64Array)
  Napi::TypedArray typed = info[1].As<Napi::TypedArray>();
  int64_t* repeats = reinterpret_cast<int64_t*>(
    static_cast<uint8_t*>(typed.ArrayBuffer().Data()) + typed.ByteOffset());
  int num_dims = static_cast<int>(typed.ElementLength());

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_repeat(tensor, repeats, num_dims, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_repeat")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorExpand(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 2) {
    throw Napi::Error::New(env, "expand requires 2 arguments (tensor, sizes)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  // Extract sizes from TypedArray (BigInt64Array)
  Napi::TypedArray typed = info[1].As<Napi::TypedArray>();
  int64_t* sizes = reinterpret_cast<int64_t*>(
    static_cast<uint8_t*>(typed.ArrayBuffer().Data()) + typed.ByteOffset());
  int num_dims = static_cast<int>(typed.ElementLength());

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_expand(tensor, sizes, num_dims, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_expand")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

Napi::Value NapiTensorNonzero(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1) {
    throw Napi::Error::New(env, "nonzero requires 1 argument (tensor)");
  }

  ts_Tensor* tensor = GetTensorHandle(info[0]);
  if (!tensor) {
    throw Napi::Error::New(env, "Invalid tensor handle");
  }

  ts_Error err = {0, ""};
  ts_Tensor* result = ts_tensor_nonzero(tensor, &err);

  if (CheckAndThrowError(env, err, "ts_tensor_nonzero")) {
    return env.Null();
  }

  return WrapTensorHandle(env, result);
}

// ============================================================================
// Module Registration
// ============================================================================

void InitRemainingOps(Napi::Env env, Napi::Object exports) {
  // Batch operations
  exports.Set("ts_batch_begin",
    Napi::Function::New(env, NapiBatchBegin));
  exports.Set("ts_batch_end",
    Napi::Function::New(env, NapiBatchEnd));
  exports.Set("ts_batch_is_recording",
    Napi::Function::New(env, NapiBatchIsRecording));
  exports.Set("ts_batch_abort",
    Napi::Function::New(env, NapiBatchAbort));

  // Device management
  exports.Set("ts_cuda_device_count",
    Napi::Function::New(env, NapiCudaDeviceCount));
  exports.Set("ts_cuda_synchronize",
    Napi::Function::New(env, NapiCudaSynchronize));
  exports.Set("ts_cuda_empty_cache",
    Napi::Function::New(env, NapiCudaEmptyCache));
  exports.Set("ts_cuda_reset_peak_memory",
    Napi::Function::New(env, NapiCudaResetPeakMemory));
  exports.Set("ts_is_available",
    Napi::Function::New(env, NapiIsAvailable));
  exports.Set("ts_current_device",
    Napi::Function::New(env, NapiCurrentDevice));
  exports.Set("ts_set_device",
    Napi::Function::New(env, NapiSetDevice));

  // Threading
  exports.Set("ts_get_num_threads",
    Napi::Function::New(env, NapiGetNumThreads));
  exports.Set("ts_set_num_threads",
    Napi::Function::New(env, NapiSetNumThreads));
  exports.Set("ts_in_parallel_region",
    Napi::Function::New(env, NapiInParallelRegion));

  // Utilities
  exports.Set("ts_manual_seed",
    Napi::Function::New(env, NapiManualSeed));
  exports.Set("ts_seed",
    Napi::Function::New(env, NapiSeed));
  exports.Set("ts_get_seed",
    Napi::Function::New(env, NapiGetSeed));
  exports.Set("ts_get_default_dtype",
    Napi::Function::New(env, NapiGetDefaultDtype));
  exports.Set("ts_set_default_dtype",
    Napi::Function::New(env, NapiSetDefaultDtype));

  // Tensor properties
  exports.Set("ts_tensor_is_contiguous",
    Napi::Function::New(env, NapiTensorIsContiguous));
  exports.Set("ts_tensor_contiguous",
    Napi::Function::New(env, NapiTensorContiguous));
  exports.Set("ts_tensor_strides",
    Napi::Function::New(env, NapiTensorStrides));
  exports.Set("ts_tensor_storage_offset",
    Napi::Function::New(env, NapiTensorStorageOffset));

  // Autograd advanced
  exports.Set("ts_tensor_requires_grad",
    Napi::Function::New(env, NapiTensorRequiresGrad));
  exports.Set("ts_tensor_set_requires_grad",
    Napi::Function::New(env, NapiTensorSetRequiresGrad));
  exports.Set("ts_tensor_retain_grad",
    Napi::Function::New(env, NapiTensorRetainGrad));
  exports.Set("ts_tensor_is_leaf",
    Napi::Function::New(env, NapiTensorIsLeaf));
  exports.Set("ts_tensor_zero_grad",
    Napi::Function::New(env, NapiTensorZeroGrad));
  exports.Set("ts_tensor_detach",
    Napi::Function::New(env, NapiTensorDetach));
  exports.Set("ts_tensor_item",
    Napi::Function::New(env, NapiTensorItem));

  // Data access
  exports.Set("ts_tensor_data_ptr",
    Napi::Function::New(env, NapiTensorDataPtr));
  exports.Set("ts_tensor_copy_to_buffer",
    Napi::Function::New(env, NapiTensorCopyToBuffer));
  exports.Set("ts_tensor_copy_from_buffer",
    Napi::Function::New(env, NapiTensorCopyFromBuffer));

  // Clone / Cat / Narrow / Triu
  exports.Set("ts_tensor_clone",
    Napi::Function::New(env, NapiTensorClone));
  exports.Set("ts_tensor_cat",
    Napi::Function::New(env, NapiTensorCat));
  exports.Set("ts_tensor_narrow",
    Napi::Function::New(env, NapiTensorNarrow));
  exports.Set("ts_tensor_triu",
    Napi::Function::New(env, NapiTensorTriu));

  // Topk / Sort / Einsum / Tril / ToDevice / MlpForward / Repeat / Expand / Nonzero
  exports.Set("ts_tensor_topk",
    Napi::Function::New(env, NapiTensorTopk));
  exports.Set("ts_tensor_sort",
    Napi::Function::New(env, NapiTensorSort));
  exports.Set("ts_tensor_einsum",
    Napi::Function::New(env, NapiTensorEinsum));
  exports.Set("ts_tensor_tril",
    Napi::Function::New(env, NapiTensorTril));
  exports.Set("ts_tensor_to_device",
    Napi::Function::New(env, NapiTensorToDevice));
  exports.Set("ts_tensor_mlp_forward",
    Napi::Function::New(env, NapiTensorMlpForward));
  exports.Set("ts_tensor_repeat",
    Napi::Function::New(env, NapiTensorRepeat));
  exports.Set("ts_tensor_expand",
    Napi::Function::New(env, NapiTensorExpand));
  exports.Set("ts_tensor_nonzero",
    Napi::Function::New(env, NapiTensorNonzero));
}
