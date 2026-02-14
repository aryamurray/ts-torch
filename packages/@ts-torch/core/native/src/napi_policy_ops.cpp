/**
 * NAPI Wrappers for Policy Fused Operations
 *
 * Wraps ts_policy_forward and ts_backward_and_clip for JavaScript consumption.
 */

#include <napi.h>
#include "../include/ts_torch.h"
#include "napi_bindings.h"

// ============================================================================
// ts_policy_forward wrapper
// ============================================================================

Napi::Value NapiPolicyForward(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 8) {
        throw Napi::Error::New(env,
            "ts_policy_forward requires 8 arguments: "
            "observations, actions, batchSize, obsSize, nActions, "
            "piParams, vfParams, activationType");
    }

    // Extract Float32Array buffers
    auto observations = info[0].As<Napi::Float32Array>();
    auto actions = info[1].As<Napi::Float32Array>();
    int batchSize = info[2].As<Napi::Number>().Int32Value();
    int obsSize = info[3].As<Napi::Number>().Int32Value();
    int nActions = info[4].As<Napi::Number>().Int32Value();

    // Extract parameter arrays
    auto piParamsArray = info[5].As<Napi::Array>();
    auto vfParamsArray = info[6].As<Napi::Array>();
    int activationType = info[7].As<Napi::Number>().Int32Value();

    // Convert parameter arrays to handle vectors
    size_t nPiParams = piParamsArray.Length();
    std::vector<ts_TensorHandle> piParams(nPiParams);
    for (size_t i = 0; i < nPiParams; i++) {
        Napi::Value val = piParamsArray[static_cast<uint32_t>(i)];
        piParams[i] = GetTensorHandle(val);
    }

    size_t nVfParams = vfParamsArray.Length();
    std::vector<ts_TensorHandle> vfParams(nVfParams);
    for (size_t i = 0; i < nVfParams; i++) {
        Napi::Value val = vfParamsArray[static_cast<uint32_t>(i)];
        vfParams[i] = GetTensorHandle(val);
    }

    ts_Error err = {0, ""};
    ts_PolicyForwardResult result = ts_policy_forward(
        observations.Data(),
        actions.Data(),
        batchSize,
        obsSize,
        nActions,
        piParams.data(),
        static_cast<int>(nPiParams),
        vfParams.data(),
        static_cast<int>(nVfParams),
        activationType,
        &err
    );

    if (CheckAndThrowError(env, err, "ts_policy_forward")) {
        return env.Undefined();
    }

    // Return object with three tensor handles
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("actionLogProbs", WrapTensorHandle(env, result.action_log_probs));
    obj.Set("entropy", WrapTensorHandle(env, result.entropy));
    obj.Set("values", WrapTensorHandle(env, result.values));

    return obj;
}

// ============================================================================
// ts_backward_and_clip wrapper
// ============================================================================

Napi::Value NapiBackwardAndClip(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        throw Napi::Error::New(env,
            "ts_backward_and_clip requires 3 arguments: "
            "loss, parameters (array), maxGradNorm");
    }

    ts_TensorHandle loss = GetTensorHandle(info[0]);
    auto paramsArray = info[1].As<Napi::Array>();
    double maxGradNorm = info[2].As<Napi::Number>().DoubleValue();

    size_t numParams = paramsArray.Length();
    std::vector<ts_TensorHandle> params(numParams);
    for (size_t i = 0; i < numParams; i++) {
        Napi::Value val = paramsArray[static_cast<uint32_t>(i)];
        params[i] = GetTensorHandle(val);
    }

    ts_Error err = {0, ""};
    double totalNorm = ts_backward_and_clip(
        loss,
        params.data(),
        numParams,
        maxGradNorm,
        &err
    );

    if (CheckAndThrowError(env, err, "ts_backward_and_clip")) {
        return env.Undefined();
    }

    return Napi::Number::New(env, totalNorm);
}

// ============================================================================
// Registration
// ============================================================================

void RegisterPolicyOps(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "ts_policy_forward"),
                Napi::Function::New(env, NapiPolicyForward));
    exports.Set(Napi::String::New(env, "ts_backward_and_clip"),
                Napi::Function::New(env, NapiBackwardAndClip));
}
