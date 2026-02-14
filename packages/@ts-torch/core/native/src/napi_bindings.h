#pragma once

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
static inline ts_Tensor* GetTensorHandle(const Napi::Value& val) {
  if (!val.IsExternal()) {
    return nullptr;
  }
  return static_cast<ts_Tensor*>(val.As<Napi::External<void>>().Data());
}

/**
 * Extract opaque scope handle from Napi::External
 */
static inline ts_Scope* GetScopeHandle(const Napi::Value& val) {
  if (!val.IsExternal()) {
    return nullptr;
  }
  return static_cast<ts_Scope*>(val.As<Napi::External<void>>().Data());
}

/**
 * Wrap tensor handle in Napi::External
 *
 * No GC finalizer — tensor lifetime is managed by the scope system
 * (ts_scope_begin / ts_scope_end) and explicit ts_tensor_delete calls.
 * Adding a GC finalizer would cause double-free with scope cleanup.
 */
static inline Napi::Value WrapTensorHandle(Napi::Env env, ts_Tensor* handle) {
  if (!handle) {
    return env.Null();
  }
  return Napi::External<void>::New(env, handle);
}

/**
 * Wrap scope handle in Napi::External (no cleanup - managed by C layer)
 */
static inline Napi::Value WrapScopeHandle(Napi::Env env, ts_Scope* handle) {
  if (!handle) {
    return env.Null();
  }
  return Napi::External<void>::New(env, static_cast<void*>(handle));
}

/**
 * Check error and throw JavaScript exception if needed
 */
static inline bool CheckAndThrowError(Napi::Env env, const ts_Error& err,
                                       const char* funcName) {
  if (err.code != 0) {
    std::string msg = funcName;
    msg += ": ";
    msg += err.message;
    throw Napi::Error::New(env, msg);
  }
  return false;
}

// ============================================================================
// Forward Declarations
// ============================================================================

// version.cpp
Napi::Value NapiVersion(const Napi::CallbackInfo& info);
Napi::Value NapiCudaIsAvailable(const Napi::CallbackInfo& info);

// error.cpp
Napi::Value NapiErrorClear(const Napi::CallbackInfo& info);
Napi::Value NapiErrorOccurred(const Napi::CallbackInfo& info);

// tensor_factory.cpp
Napi::Value NapiTensorZeros(const Napi::CallbackInfo& info);
Napi::Value NapiTensorOnes(const Napi::CallbackInfo& info);
Napi::Value NapiTensorRandn(const Napi::CallbackInfo& info);
Napi::Value NapiTensorRand(const Napi::CallbackInfo& info);
Napi::Value NapiTensorEmpty(const Napi::CallbackInfo& info);
Napi::Value NapiTensorZerosI32(const Napi::CallbackInfo& info);
Napi::Value NapiTensorOnesI32(const Napi::CallbackInfo& info);
Napi::Value NapiTensorRandnI32(const Napi::CallbackInfo& info);
Napi::Value NapiTensorRandI32(const Napi::CallbackInfo& info);
Napi::Value NapiTensorEmptyI32(const Napi::CallbackInfo& info);
Napi::Value NapiTensorFromBuffer(const Napi::CallbackInfo& info);

// tensor_properties.cpp
Napi::Value NapiTensorNdim(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSize(const Napi::CallbackInfo& info);
Napi::Value NapiTensorShape(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDtype(const Napi::CallbackInfo& info);
Napi::Value NapiTensorNumel(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDeviceType(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDeviceIndex(const Napi::CallbackInfo& info);
Napi::Value NapiTensorIsContiguous(const Napi::CallbackInfo& info);
Napi::Value NapiTensorRequiresGrad(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSetRequiresGrad(const Napi::CallbackInfo& info);

// tensor_memory.cpp
Napi::Value NapiTensorDelete(const Napi::CallbackInfo& info);
Napi::Value NapiTensorClone(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDetach(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDataPtr(const Napi::CallbackInfo& info);
Napi::Value NapiTensorCopyToBuffer(const Napi::CallbackInfo& info);

// napi_tensor_binary_ops.cpp -- Arithmetic
Napi::Value NapiTensorAdd(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSub(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMul(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDiv(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMatmul(const Napi::CallbackInfo& info);
Napi::Value NapiTensorBmm(const Napi::CallbackInfo& info);
Napi::Value NapiTensorChainMatmul(const Napi::CallbackInfo& info);
Napi::Value NapiTensorPow(const Napi::CallbackInfo& info);
Napi::Value NapiTensorPowScalar(const Napi::CallbackInfo& info);
// napi_tensor_binary_ops.cpp -- Scalar arithmetic
Napi::Value NapiTensorAddScalar(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSubScalar(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMulScalar(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDivScalar(const Napi::CallbackInfo& info);
// napi_tensor_binary_ops.cpp -- Comparisons
Napi::Value NapiTensorEq(const Napi::CallbackInfo& info);
Napi::Value NapiTensorNe(const Napi::CallbackInfo& info);
Napi::Value NapiTensorLt(const Napi::CallbackInfo& info);
Napi::Value NapiTensorLe(const Napi::CallbackInfo& info);
Napi::Value NapiTensorGt(const Napi::CallbackInfo& info);
Napi::Value NapiTensorGe(const Napi::CallbackInfo& info);
// napi_tensor_binary_ops.cpp -- Min/Max
Napi::Value NapiTensorMinimum(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMaximum(const Napi::CallbackInfo& info);
// napi_tensor_binary_ops.cpp -- Advanced
Napi::Value NapiTensorWhere(const Napi::CallbackInfo& info);
Napi::Value NapiTensorGather(const Napi::CallbackInfo& info);
Napi::Value NapiTensorScatter(const Napi::CallbackInfo& info);
Napi::Value NapiTensorScatterAdd(const Napi::CallbackInfo& info);
Napi::Value NapiTensorIndex(const Napi::CallbackInfo& info);
Napi::Value NapiTensorIndexSelect(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMaskedFill(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMaskedSelect(const Napi::CallbackInfo& info);
// napi_tensor_binary_ops.cpp -- In-place
Napi::Value NapiTensorAdd_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSub_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMul_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDiv_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMulScalar_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDivScalar_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorAddAlpha_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorOptimAdd_(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSubInplace(const Napi::CallbackInfo& info);
Napi::Value NapiTensorAddScaledInplace(const Napi::CallbackInfo& info);
// napi_tensor_binary_ops.cpp -- Out= variants
Napi::Value NapiTensorAddOut(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSubOut(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMulOut(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDivOut(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMatmulOut(const Napi::CallbackInfo& info);

// tensor_unary_ops.cpp
Napi::Value NapiTensorRelu(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSigmoid(const Napi::CallbackInfo& info);
Napi::Value NapiTensorTanh(const Napi::CallbackInfo& info);
Napi::Value NapiTensorExp(const Napi::CallbackInfo& info);
Napi::Value NapiTensorLog(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSqrt(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSoftmax(const Napi::CallbackInfo& info);
Napi::Value NapiTensorLogSoftmax(const Napi::CallbackInfo& info);
Napi::Value NapiTensorTranspose(const Napi::CallbackInfo& info);
Napi::Value NapiTensorReshape(const Napi::CallbackInfo& info);
Napi::Value NapiTensorNeg(const Napi::CallbackInfo& info);
Napi::Value NapiTensorAbs(const Napi::CallbackInfo& info);
Napi::Value NapiTensorClamp(const Napi::CallbackInfo& info);
Napi::Value NapiTensorDropout(const Napi::CallbackInfo& info);

// tensor_reductions.cpp
Napi::Value NapiTensorSum(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMean(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMax(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMin(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSumDim(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMeanDim(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMaxDim(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMinDim(const Napi::CallbackInfo& info);
Napi::Value NapiTensorArgmax(const Napi::CallbackInfo& info);
Napi::Value NapiTensorArgmin(const Napi::CallbackInfo& info);
Napi::Value NapiTensorAny(const Napi::CallbackInfo& info);
Napi::Value NapiTensorAll(const Napi::CallbackInfo& info);
Napi::Value NapiTensorNorm(const Napi::CallbackInfo& info);

// tensor_autograd.cpp
Napi::Value NapiTensorBackward(const Napi::CallbackInfo& info);
Napi::Value NapiTensorGrad(const Napi::CallbackInfo& info);
Napi::Value NapiTensorZeroGrad(const Napi::CallbackInfo& info);
Napi::Value NapiTensorRetainGrad(const Napi::CallbackInfo& info);
Napi::Value NapiTensorNoGrad(const Napi::CallbackInfo& info);
Napi::Value NapiTensorEnableGrad(const Napi::CallbackInfo& info);

// nn_ops.cpp
Napi::Value NapiTensorLinear(const Napi::CallbackInfo& info);
Napi::Value NapiTensorConv2d(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMaxPool2d(const Napi::CallbackInfo& info);
Napi::Value NapiTensorAvgPool2d(const Napi::CallbackInfo& info);
Napi::Value NapiTensorBatchNorm(const Napi::CallbackInfo& info);
Napi::Value NapiTensorLayerNorm(const Napi::CallbackInfo& info);
Napi::Value NapiTensorEmbedding(const Napi::CallbackInfo& info);
Napi::Value NapiTensorCrossEntropyLoss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorNllLoss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorMseLoss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorL1Loss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorSmoothL1Loss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorKlDivLoss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorBceLoss(const Napi::CallbackInfo& info);
Napi::Value NapiTensorBceWithLogitsLoss(const Napi::CallbackInfo& info);

// scope.cpp
Napi::Value NapiScopeBegin(const Napi::CallbackInfo& info);
Napi::Value NapiScopeEnd(const Napi::CallbackInfo& info);
Napi::Value NapiScopeRegisterTensor(const Napi::CallbackInfo& info);
Napi::Value NapiScopeEscapeTensor(const Napi::CallbackInfo& info);

// init.cpp
Napi::Object Init(Napi::Env env, Napi::Object exports);
