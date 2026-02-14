/**
 * Batched Operations for Reduced FFI Overhead (Phase 2)
 *
 * This file implements:
 * 1. Native recording mode with placeholder handles
 * 2. Direct batched operations (chain_matmul, mlp_forward)
 * 3. Thread controls (set_num_threads, get_num_threads)
 *
 * The recording mode allows chaining operations like:
 *   input.matmul(w1).relu().matmul(w2)
 * in a single FFI round-trip by returning placeholder handles
 * that are resolved when batch_end is called.
 */

#include "ts_torch/internal.h"

// For dlsym (Unix) / GetProcAddress (Windows) to resolve BLAS thread APIs at runtime
#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif
#include <vector>
#include <memory>

// ============================================================================
// Recording Mode Implementation
// ============================================================================

/**
 * Maximum number of inputs per operation
 */
static constexpr size_t MAX_OP_INPUTS = 4;

/**
 * Recorded operation structure
 */
struct RecordedOp {
    ts_OpCode opcode;

    // Input tracking: either placeholder id (>=0) or real tensor handle
    int32_t input_ids[MAX_OP_INPUTS];         // Placeholder IDs (-1 for real tensors)
    ts_TensorHandle real_inputs[MAX_OP_INPUTS]; // Real tensor handles (nullptr for placeholders)
    size_t num_inputs;

    // Optional arguments for ops that need them
    double scalar_arg;
    int64_t dim_arg;
    std::vector<int64_t> shape_arg;  // For reshape

    // Output placeholder ID
    int32_t output_id;
};

/**
 * Batch context for recording operations
 */
struct ts_Batch {
    std::vector<RecordedOp> ops;
    std::vector<std::unique_ptr<ts_Tensor>> placeholders;  // Placeholder tensors
    int32_t next_placeholder_id = 0;
    int32_t last_output_id = -1;

    /**
     * Create a placeholder tensor handle for batch recording
     */
    ts_TensorHandle make_placeholder() {
        int32_t id = next_placeholder_id++;
        last_output_id = id;
        // Create placeholder using the special constructor
        auto placeholder = std::make_unique<ts_Tensor>(id);
        ts_TensorHandle handle = placeholder.get();
        placeholders.push_back(std::move(placeholder));
        return handle;
    }
};

/**
 * Thread-local active batch (nullptr when not recording)
 */
thread_local ts_Batch* g_active_batch = nullptr;

/**
 * Execute a single recorded operation
 */
static torch::Tensor execute_op(
    const RecordedOp& op,
    const std::vector<torch::Tensor>& results
) {
    // Resolve inputs
    std::vector<torch::Tensor> inputs;
    inputs.reserve(op.num_inputs);

    for (size_t i = 0; i < op.num_inputs; i++) {
        if (op.input_ids[i] >= 0) {
            // Placeholder - get from results
            inputs.push_back(results[op.input_ids[i]]);
        } else {
            // Real tensor
            inputs.push_back(op.real_inputs[i]->tensor);
        }
    }

    // Execute based on opcode
    switch (op.opcode) {
        case TS_OP_ADD:
            return inputs[0] + inputs[1];

        case TS_OP_SUB:
            return inputs[0] - inputs[1];

        case TS_OP_MUL:
            return inputs[0] * inputs[1];

        case TS_OP_DIV:
            return inputs[0] / inputs[1];

        case TS_OP_MATMUL:
            return torch::matmul(inputs[0], inputs[1]);

        case TS_OP_RELU:
            return torch::relu(inputs[0]);

        case TS_OP_SIGMOID:
            return torch::sigmoid(inputs[0]);

        case TS_OP_SOFTMAX:
            return torch::softmax(inputs[0], op.dim_arg);

        case TS_OP_TANH:
            return torch::tanh(inputs[0]);

        case TS_OP_LINEAR:
            if (op.num_inputs == 3 && op.real_inputs[2] != nullptr) {
                return torch::linear(inputs[0], inputs[1], inputs[2]);
            } else {
                return torch::linear(inputs[0], inputs[1]);
            }

        case TS_OP_SUM:
            return inputs[0].sum();

        case TS_OP_MEAN:
            return inputs[0].mean();

        case TS_OP_TRANSPOSE:
            return inputs[0].transpose(op.dim_arg, static_cast<int64_t>(op.scalar_arg));

        case TS_OP_RESHAPE:
            return inputs[0].reshape(op.shape_arg);

        default:
            throw std::runtime_error("Unknown opcode in batch execution");
    }
}

// ============================================================================
// Batch API Implementation
// ============================================================================

ts_BatchHandle ts_batch_begin(ts_Error* error) {
    if (g_active_batch) {
        set_error(error, 1, "Nested batching not supported");
        return nullptr;
    }

    g_active_batch = new ts_Batch();
    return g_active_batch;
}

ts_TensorHandle ts_batch_end(ts_BatchHandle batch, ts_Error* error) {
    if (batch != g_active_batch) {
        set_error(error, 1, "Invalid batch handle");
        return nullptr;
    }

    if (batch->ops.empty()) {
        set_error(error, 1, "No operations recorded in batch");
        g_active_batch = nullptr;
        delete batch;
        return nullptr;
    }

    try {
        // Execute all recorded operations
        std::vector<torch::Tensor> results(batch->next_placeholder_id);

        for (const auto& op : batch->ops) {
            torch::Tensor result = execute_op(op, results);
            results[op.output_id] = std::move(result);
        }

        // Get final result
        torch::Tensor final_result = results[batch->last_output_id];

        // Cleanup batch
        g_active_batch = nullptr;
        delete batch;

        // Create result tensor and register in scope
        auto* handle = new ts_Tensor(std::move(final_result));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        g_active_batch = nullptr;
        delete batch;
        set_error(error, 1, e.what());
        return nullptr;
    }
}

void ts_batch_abort(ts_BatchHandle batch) {
    if (batch == g_active_batch) {
        g_active_batch = nullptr;
    }
    delete batch;
}

int ts_batch_is_recording(void) {
    return g_active_batch != nullptr ? 1 : 0;
}

// ============================================================================
// Helper to record binary operations
// ============================================================================

static ts_TensorHandle record_binary_op(
    ts_OpCode opcode,
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    RecordedOp op;
    op.opcode = opcode;
    op.num_inputs = 2;
    op.scalar_arg = 0;
    op.dim_arg = 0;

    // Record input a
    op.input_ids[0] = a->is_placeholder() ? a->batch_id : -1;
    op.real_inputs[0] = a->is_placeholder() ? nullptr : a;

    // Record input b
    op.input_ids[1] = b->is_placeholder() ? b->batch_id : -1;
    op.real_inputs[1] = b->is_placeholder() ? nullptr : b;

    // Create placeholder for output
    auto* placeholder = g_active_batch->make_placeholder();
    op.output_id = placeholder->batch_id;

    g_active_batch->ops.push_back(std::move(op));
    return placeholder;
}

static ts_TensorHandle record_unary_op(
    ts_OpCode opcode,
    ts_TensorHandle a,
    ts_Error* error
) {
    RecordedOp op;
    op.opcode = opcode;
    op.num_inputs = 1;
    op.scalar_arg = 0;
    op.dim_arg = 0;

    // Record input
    op.input_ids[0] = a->is_placeholder() ? a->batch_id : -1;
    op.real_inputs[0] = a->is_placeholder() ? nullptr : a;

    // Create placeholder for output
    auto* placeholder = g_active_batch->make_placeholder();
    op.output_id = placeholder->batch_id;

    g_active_batch->ops.push_back(std::move(op));
    return placeholder;
}

// ============================================================================
// Direct Batched Operations (No recording overhead)
// ============================================================================

ts_TensorHandle ts_tensor_chain_matmul(
    ts_TensorHandle* tensors,
    size_t count,
    ts_Error* error
) {
    try {
        if (!tensors || count < 2) {
            set_error(error, 1, "chain_matmul requires at least 2 tensors");
            return nullptr;
        }

        // Check all handles are valid
        for (size_t i = 0; i < count; i++) {
            if (!tensors[i]) {
                set_error(error, 1, "Null tensor in chain_matmul");
                return nullptr;
            }
        }

        // Execute chain multiplication
        torch::Tensor result = tensors[0]->tensor;
        for (size_t i = 1; i < count; i++) {
            result = torch::matmul(result, tensors[i]->tensor);
        }

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mlp_forward(
    ts_TensorHandle input,
    ts_TensorHandle* weights,
    ts_TensorHandle* biases,
    size_t num_layers,
    int apply_relu_except_last,
    ts_Error* error
) {
    try {
        if (!input || !weights || num_layers == 0) {
            set_error(error, 1, "Invalid arguments to mlp_forward");
            return nullptr;
        }

        // Validate all weight handles
        for (size_t i = 0; i < num_layers; i++) {
            if (!weights[i]) {
                set_error(error, 1, "Null weight tensor in mlp_forward");
                return nullptr;
            }
        }

        torch::Tensor x = input->tensor;

        for (size_t i = 0; i < num_layers; i++) {
            // Linear: x @ W^T + b
            if (biases && biases[i]) {
                x = torch::linear(x, weights[i]->tensor, biases[i]->tensor);
            } else {
                x = torch::linear(x, weights[i]->tensor);
            }

            // Apply ReLU to all but last layer (if requested)
            if (apply_relu_except_last && i < num_layers - 1) {
                x = torch::relu(x);
            }
        }

        auto* handle = new ts_Tensor(std::move(x));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Thread Controls (Phase 6)
// ============================================================================

void ts_set_num_threads(int num_threads) {
    if (num_threads <= 0) {
        // Auto mode - let LibTorch decide
        at::set_num_threads(at::get_num_threads());
        return;
    }

    // ATen parallel_for + OpenMP (at::set_num_threads calls omp_set_num_threads internally)
    at::set_num_threads(num_threads);

    // MKL and OpenBLAS have their own thread pools separate from OpenMP.
    // Resolve their runtime APIs via dlsym to avoid hard link dependencies.
#if !defined(_WIN32)
    // mkl_set_num_threads(int) — Intel MKL
    using MklSetThreads = void(*)(int);
    static auto mkl_set = reinterpret_cast<MklSetThreads>(dlsym(RTLD_DEFAULT, "MKL_Set_Num_Threads"));
    if (mkl_set) mkl_set(num_threads);

    // openblas_set_num_threads(int) — OpenBLAS
    using BlasSetThreads = void(*)(int);
    static auto blas_set = reinterpret_cast<BlasSetThreads>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    if (blas_set) blas_set(num_threads);
#else
    // Windows: GetProcAddress against loaded modules
    HMODULE mkl_mod = GetModuleHandleA("mkl_rt.dll");
    if (!mkl_mod) mkl_mod = GetModuleHandleA("mkl_rt.2.dll");
    if (mkl_mod) {
        using MklSetThreads = void(*)(int);
        auto mkl_set = reinterpret_cast<MklSetThreads>(GetProcAddress(mkl_mod, "MKL_Set_Num_Threads"));
        if (mkl_set) mkl_set(num_threads);
    }
    HMODULE blas_mod = GetModuleHandleA("libopenblas.dll");
    if (blas_mod) {
        using BlasSetThreads = void(*)(int);
        auto blas_set = reinterpret_cast<BlasSetThreads>(GetProcAddress(blas_mod, "openblas_set_num_threads"));
        if (blas_set) blas_set(num_threads);
    }
#endif
}

int ts_get_num_threads(void) {
    return at::get_num_threads();
}
