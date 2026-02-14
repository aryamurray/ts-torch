/**
 * NAPI Wrappers for RL Fused Operations
 */

#include <napi.h>
#include "../include/ts_torch.h"
#include "napi_bindings.h"

// ============================================================================
// ts_compute_gae wrapper
// ============================================================================

Napi::Value NapiComputeGae(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 11) {
        throw Napi::Error::New(env,
            "ts_compute_gae requires 11 arguments: "
            "rewards, values, episodeStarts, lastValues, lastDones, "
            "bufferSize, nEnvs, gamma, gaeLambda, advantagesOut, returnsOut");
    }

    // Extract TypedArray buffers
    auto rewards = info[0].As<Napi::Float32Array>();
    auto values = info[1].As<Napi::Float32Array>();
    auto episodeStarts = info[2].As<Napi::Uint8Array>();
    auto lastValues = info[3].As<Napi::Float32Array>();
    auto lastDones = info[4].As<Napi::Uint8Array>();
    int bufferSize = info[5].As<Napi::Number>().Int32Value();
    int nEnvs = info[6].As<Napi::Number>().Int32Value();
    double gamma = info[7].As<Napi::Number>().DoubleValue();
    double gaeLambda = info[8].As<Napi::Number>().DoubleValue();
    auto advantagesOut = info[9].As<Napi::Float32Array>();
    auto returnsOut = info[10].As<Napi::Float32Array>();

    ts_Error err = {0, ""};
    ts_compute_gae(
        rewards.Data(),
        values.Data(),
        episodeStarts.Data(),
        lastValues.Data(),
        lastDones.Data(),
        bufferSize,
        nEnvs,
        gamma,
        gaeLambda,
        advantagesOut.Data(),
        returnsOut.Data(),
        &err
    );

    if (CheckAndThrowError(env, err, "ts_compute_gae")) {
        return env.Undefined();
    }

    return env.Undefined();
}

// ============================================================================
// ts_clip_grad_norm_ wrapper
// ============================================================================

Napi::Value NapiClipGradNorm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        throw Napi::Error::New(env,
            "ts_clip_grad_norm_ requires 2 arguments: parameters (array), maxNorm");
    }

    auto paramsArray = info[0].As<Napi::Array>();
    double maxNorm = info[1].As<Napi::Number>().DoubleValue();

    size_t numParams = paramsArray.Length();
    std::vector<ts_TensorHandle> params(numParams);

    for (size_t i = 0; i < numParams; i++) {
        Napi::Value val = paramsArray[static_cast<uint32_t>(i)];
        params[i] = GetTensorHandle(val);
    }

    ts_Error err = {0, ""};
    double totalNorm = ts_clip_grad_norm_(params.data(), numParams, maxNorm, &err);

    if (CheckAndThrowError(env, err, "ts_clip_grad_norm_")) {
        return env.Undefined();
    }

    return Napi::Number::New(env, totalNorm);
}

// ============================================================================
// ts_normalize_inplace wrapper
// ============================================================================

Napi::Value NapiNormalizeInplace(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        throw Napi::Error::New(env,
            "ts_normalize_inplace requires 1 argument: data (Float32Array)");
    }

    auto data = info[0].As<Napi::Float32Array>();

    ts_Error err = {0, ""};
    ts_normalize_inplace(data.Data(), data.ElementLength(), &err);

    if (CheckAndThrowError(env, err, "ts_normalize_inplace")) {
        return env.Undefined();
    }

    return env.Undefined();
}

// ============================================================================
// Registration
// ============================================================================

void RegisterRLOps(Napi::Env env, Napi::Object exports) {
    exports.Set(Napi::String::New(env, "ts_compute_gae"),
                Napi::Function::New(env, NapiComputeGae));
    exports.Set(Napi::String::New(env, "ts_clip_grad_norm_"),
                Napi::Function::New(env, NapiClipGradNorm));
    exports.Set(Napi::String::New(env, "ts_normalize_inplace"),
                Napi::Function::New(env, NapiNormalizeInplace));
}
