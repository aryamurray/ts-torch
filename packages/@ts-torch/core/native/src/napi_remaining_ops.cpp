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
    // Ensure tensor is contiguous and on CPU for memcpy
    auto cpu_tensor = tensor->tensor.contiguous();
    size_t tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size();

    if (src_size < tensor_bytes) {
      throw Napi::Error::New(env, "Source buffer too small for tensor");
    }

    std::memcpy(cpu_tensor.data_ptr(), src, tensor_bytes);
  } catch (const Napi::Error&) {
    throw;  // re-throw Napi errors as-is
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("copy_from_buffer: ") + e.what());
  }

  return env.Undefined();
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
}
