import { vi } from 'vitest';

/**
 * Mock FFI interface for testing without native bindings
 */

export interface MockFFISymbols {
  // Tensor creation
  torch_zeros: ReturnType<typeof vi.fn>;
  torch_ones: ReturnType<typeof vi.fn>;
  torch_randn: ReturnType<typeof vi.fn>;
  torch_from_blob: ReturnType<typeof vi.fn>;
  torch_arange: ReturnType<typeof vi.fn>;
  torch_linspace: ReturnType<typeof vi.fn>;

  // Tensor info
  torch_tensor_shape: ReturnType<typeof vi.fn>;
  torch_tensor_ndim: ReturnType<typeof vi.fn>;
  torch_tensor_numel: ReturnType<typeof vi.fn>;
  torch_tensor_dtype: ReturnType<typeof vi.fn>;
  torch_tensor_device: ReturnType<typeof vi.fn>;
  torch_tensor_requires_grad: ReturnType<typeof vi.fn>;

  // Tensor operations
  torch_tensor_add: ReturnType<typeof vi.fn>;
  torch_tensor_sub: ReturnType<typeof vi.fn>;
  torch_tensor_mul: ReturnType<typeof vi.fn>;
  torch_tensor_div: ReturnType<typeof vi.fn>;
  torch_tensor_matmul: ReturnType<typeof vi.fn>;
  torch_tensor_sum: ReturnType<typeof vi.fn>;
  torch_tensor_mean: ReturnType<typeof vi.fn>;
  torch_tensor_max: ReturnType<typeof vi.fn>;
  torch_tensor_min: ReturnType<typeof vi.fn>;

  // Activation functions
  torch_relu: ReturnType<typeof vi.fn>;
  torch_sigmoid: ReturnType<typeof vi.fn>;
  torch_tanh: ReturnType<typeof vi.fn>;
  torch_softmax: ReturnType<typeof vi.fn>;
  torch_log_softmax: ReturnType<typeof vi.fn>;

  // Autograd
  torch_tensor_backward: ReturnType<typeof vi.fn>;
  torch_tensor_grad: ReturnType<typeof vi.fn>;
  torch_tensor_zero_grad: ReturnType<typeof vi.fn>;
  torch_tensor_detach: ReturnType<typeof vi.fn>;
  torch_tensor_requires_grad_: ReturnType<typeof vi.fn>;

  // Memory management
  torch_tensor_clone: ReturnType<typeof vi.fn>;
  torch_tensor_to: ReturnType<typeof vi.fn>;
  torch_tensor_data_ptr: ReturnType<typeof vi.fn>;
  torch_tensor_free: ReturnType<typeof vi.fn>;

  // Reshaping
  torch_tensor_view: ReturnType<typeof vi.fn>;
  torch_tensor_reshape: ReturnType<typeof vi.fn>;
  torch_tensor_transpose: ReturnType<typeof vi.fn>;
  torch_tensor_permute: ReturnType<typeof vi.fn>;
  torch_tensor_squeeze: ReturnType<typeof vi.fn>;
  torch_tensor_unsqueeze: ReturnType<typeof vi.fn>;

  // Indexing
  torch_tensor_slice: ReturnType<typeof vi.fn>;
  torch_tensor_index_select: ReturnType<typeof vi.fn>;
  torch_tensor_masked_select: ReturnType<typeof vi.fn>;

  // Loss functions
  torch_mse_loss: ReturnType<typeof vi.fn>;
  torch_cross_entropy_loss: ReturnType<typeof vi.fn>;
  torch_nll_loss: ReturnType<typeof vi.fn>;
  torch_bce_loss: ReturnType<typeof vi.fn>;

  // NN operations
  torch_linear: ReturnType<typeof vi.fn>;
  torch_conv2d: ReturnType<typeof vi.fn>;
  torch_max_pool2d: ReturnType<typeof vi.fn>;
  torch_dropout: ReturnType<typeof vi.fn>;
  torch_batch_norm: ReturnType<typeof vi.fn>;

  // Utilities
  torch_set_num_threads: ReturnType<typeof vi.fn>;
  torch_get_num_threads: ReturnType<typeof vi.fn>;
  torch_manual_seed: ReturnType<typeof vi.fn>;
}

export interface MockFFI {
  symbols: MockFFISymbols;
  reset: () => void;
}

/**
 * Create a mock FFI object with all torch functions mocked
 */
export function createMockFFI(): MockFFI {
  const symbols: MockFFISymbols = {
    // Tensor creation
    torch_zeros: vi.fn(),
    torch_ones: vi.fn(),
    torch_randn: vi.fn(),
    torch_from_blob: vi.fn(),
    torch_arange: vi.fn(),
    torch_linspace: vi.fn(),

    // Tensor info
    torch_tensor_shape: vi.fn(),
    torch_tensor_ndim: vi.fn(),
    torch_tensor_numel: vi.fn(),
    torch_tensor_dtype: vi.fn(),
    torch_tensor_device: vi.fn(),
    torch_tensor_requires_grad: vi.fn(),

    // Tensor operations
    torch_tensor_add: vi.fn(),
    torch_tensor_sub: vi.fn(),
    torch_tensor_mul: vi.fn(),
    torch_tensor_div: vi.fn(),
    torch_tensor_matmul: vi.fn(),
    torch_tensor_sum: vi.fn(),
    torch_tensor_mean: vi.fn(),
    torch_tensor_max: vi.fn(),
    torch_tensor_min: vi.fn(),

    // Activation functions
    torch_relu: vi.fn(),
    torch_sigmoid: vi.fn(),
    torch_tanh: vi.fn(),
    torch_softmax: vi.fn(),
    torch_log_softmax: vi.fn(),

    // Autograd
    torch_tensor_backward: vi.fn(),
    torch_tensor_grad: vi.fn(),
    torch_tensor_zero_grad: vi.fn(),
    torch_tensor_detach: vi.fn(),
    torch_tensor_requires_grad_: vi.fn(),

    // Memory management
    torch_tensor_clone: vi.fn(),
    torch_tensor_to: vi.fn(),
    torch_tensor_data_ptr: vi.fn(),
    torch_tensor_free: vi.fn(),

    // Reshaping
    torch_tensor_view: vi.fn(),
    torch_tensor_reshape: vi.fn(),
    torch_tensor_transpose: vi.fn(),
    torch_tensor_permute: vi.fn(),
    torch_tensor_squeeze: vi.fn(),
    torch_tensor_unsqueeze: vi.fn(),

    // Indexing
    torch_tensor_slice: vi.fn(),
    torch_tensor_index_select: vi.fn(),
    torch_tensor_masked_select: vi.fn(),

    // Loss functions
    torch_mse_loss: vi.fn(),
    torch_cross_entropy_loss: vi.fn(),
    torch_nll_loss: vi.fn(),
    torch_bce_loss: vi.fn(),

    // NN operations
    torch_linear: vi.fn(),
    torch_conv2d: vi.fn(),
    torch_max_pool2d: vi.fn(),
    torch_dropout: vi.fn(),
    torch_batch_norm: vi.fn(),

    // Utilities
    torch_set_num_threads: vi.fn(),
    torch_get_num_threads: vi.fn(),
    torch_manual_seed: vi.fn(),
  };

  const reset = () => {
    Object.values(symbols).forEach((fn) => fn.mockReset());
  };

  return {
    symbols,
    reset,
  };
}
