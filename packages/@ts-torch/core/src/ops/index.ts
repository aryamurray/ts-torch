/**
 * Optimized operations for reduced FFI overhead
 *
 * This module provides:
 * - batch(): Record and execute multiple ops in a single FFI call
 * - chainMatmul(): Chain matrix multiplication in one call
 * - mlpForward(): Complete MLP forward pass in one call
 * - Thread controls for tuning LibTorch parallelism
 *
 * @module ops
 */

export {
  // Recording mode
  batch,
  isBatchRecording,
  // Direct batched operations
  chainMatmul,
  mlpForward,
  // Thread controls
  setNumThreads,
  getNumThreads,
} from './batch.js'
