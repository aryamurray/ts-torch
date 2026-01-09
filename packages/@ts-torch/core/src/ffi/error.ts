/**
 * Error handling utilities for FFI calls
 * Manages error struct allocation and error checking
 */

import { ptr, type Pointer } from "bun:ffi";

/**
 * C error struct layout:
 * struct Error {
 *   int32_t code;      // 4 bytes
 *   char message[256]; // 256 bytes
 * }
 * Total: 260 bytes
 */
export const ERROR_STRUCT_SIZE = 260;

/**
 * Error codes returned by native library
 */
export enum ErrorCode {
  OK = 0,
  NULL_POINTER = 1,
  INVALID_SHAPE = 2,
  INVALID_DTYPE = 3,
  DIMENSION_MISMATCH = 4,
  OUT_OF_MEMORY = 5,
  CUDA_ERROR = 6,
  GRAD_ERROR = 7,
  SCOPE_ERROR = 8,
  UNKNOWN = 99,
}

/**
 * Error code to message mapping
 */
const ERROR_MESSAGES: Record<ErrorCode, string> = {
  [ErrorCode.OK]: "Success",
  [ErrorCode.NULL_POINTER]: "Null pointer encountered",
  [ErrorCode.INVALID_SHAPE]: "Invalid tensor shape",
  [ErrorCode.INVALID_DTYPE]: "Invalid data type",
  [ErrorCode.DIMENSION_MISMATCH]: "Tensor dimension mismatch",
  [ErrorCode.OUT_OF_MEMORY]: "Out of memory",
  [ErrorCode.CUDA_ERROR]: "CUDA error",
  [ErrorCode.GRAD_ERROR]: "Gradient computation error",
  [ErrorCode.SCOPE_ERROR]: "Memory scope error",
  [ErrorCode.UNKNOWN]: "Unknown error",
};

/**
 * Custom error class for torch operations
 */
export class TorchError extends Error {
  public readonly code: ErrorCode;
  public readonly nativeMessage: string;

  constructor(code: ErrorCode, nativeMessage: string) {
    const baseMessage = ERROR_MESSAGES[code] || ERROR_MESSAGES[ErrorCode.UNKNOWN];
    const fullMessage = nativeMessage ? `${baseMessage}: ${nativeMessage}` : baseMessage;

    super(fullMessage);
    this.name = "TorchError";
    this.code = code;
    this.nativeMessage = nativeMessage;

    // Maintain proper stack trace for V8
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, TorchError);
    }
  }

  /**
   * Check if error is a specific type
   */
  public is(code: ErrorCode): boolean {
    return this.code === code;
  }

  /**
   * Convert to JSON representation
   */
  public toJSON(): object {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      nativeMessage: this.nativeMessage,
    };
  }
}

/**
 * Create a new error struct buffer
 * Allocates memory for error reporting from FFI calls
 *
 * @returns Pointer to allocated error buffer
 */
export function createError(): Pointer {
  // Allocate buffer for error struct
  const buffer = new ArrayBuffer(ERROR_STRUCT_SIZE);
  const view = new DataView(buffer);

  // Initialize error code to 0 (OK)
  view.setInt32(0, 0, true); // little-endian

  // Return pointer to buffer
  return ptr(buffer);
}

/**
 * Check if an error occurred and throw if necessary
 * Reads the error struct and throws TorchError if code != 0
 *
 * @param errorPtr - Pointer to error struct
 * @throws TorchError if error code is not 0
 */
export function checkError(errorPtr: Pointer): void {
  // Read error code (first 4 bytes)
  const codeView = new DataView(errorPtr as unknown as ArrayBuffer, 0, 4);
  const code = codeView.getInt32(0, true) as ErrorCode;

  if (code === ErrorCode.OK) {
    return; // No error
  }

  // Read error message (next 256 bytes)
  const messageBytes = new Uint8Array(errorPtr as unknown as ArrayBuffer, 4, 256);

  // Find null terminator
  let messageLength = 0;
  for (let i = 0; i < messageBytes.length; i++) {
    if (messageBytes[i] === 0) {
      messageLength = i;
      break;
    }
  }

  // Decode message string
  const decoder = new TextDecoder("utf-8");
  const message = decoder.decode(messageBytes.subarray(0, messageLength));

  throw new TorchError(code, message);
}

/**
 * Wrapper for FFI calls with automatic error handling
 * Simplifies error checking pattern
 *
 * @param fn - Function that takes error pointer and returns result
 * @returns Result from function
 * @throws TorchError if error occurred
 *
 * @example
 * const result = withError(err => lib.symbols.ts_tensor_zeros(shape, ndim, dtype, false, err));
 */
export function withError<T>(fn: (errorPtr: Pointer) => T): T {
  const errorPtr = createError();
  try {
    const result = fn(errorPtr);
    checkError(errorPtr);
    return result;
  } finally {
    // Error buffer is automatically freed when it goes out of scope
    // Bun's GC will handle cleanup
  }
}

/**
 * Check if pointer is null
 * @param ptr - Pointer to check
 * @throws TorchError if pointer is null
 */
export function checkNull(ptr: Pointer | null, message = "Unexpected null pointer"): void {
  if (ptr === null || ptr === 0) {
    throw new TorchError(ErrorCode.NULL_POINTER, message);
  }
}

/**
 * Validate tensor shape
 * @param shape - Shape array to validate
 * @throws TorchError if shape is invalid
 */
export function validateShape(shape: number[] | bigint[]): void {
  if (!Array.isArray(shape) || shape.length === 0) {
    throw new TorchError(ErrorCode.INVALID_SHAPE, "Shape must be non-empty array");
  }

  for (let i = 0; i < shape.length; i++) {
    const rawDim = shape[i];
    if (rawDim === undefined) continue;
    const dim = typeof rawDim === "bigint" ? Number(rawDim) : rawDim;
    if (!Number.isInteger(dim) || dim < 0) {
      throw new TorchError(
        ErrorCode.INVALID_SHAPE,
        `Shape dimension ${i} must be non-negative integer, got ${rawDim}`,
      );
    }
  }
}

/**
 * Validate dtype value
 * @param dtype - Data type enum value
 * @throws TorchError if dtype is invalid
 */
export function validateDtype(dtype: number): void {
  if (!Number.isInteger(dtype) || dtype < 0 || dtype > 3) {
    throw new TorchError(
      ErrorCode.INVALID_DTYPE,
      `Invalid dtype ${dtype}, must be 0 (f32), 1 (f64), 2 (i32), or 3 (i64)`,
    );
  }
}
