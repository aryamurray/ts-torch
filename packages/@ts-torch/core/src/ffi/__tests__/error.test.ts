/**
 * Tests for error handling utilities
 */

import { describe, test, expect } from "bun:test";
import {
  TorchError,
  ErrorCode,
  createError,
  checkError,
  withError,
  checkNull,
  validateShape,
  validateDtype,
  ERROR_STRUCT_SIZE,
} from "../error.js";
import { ptr } from "bun:ffi";

describe("FFI Error Handling", () => {
  describe("ERROR_STRUCT_SIZE", () => {
    test("should be 260 bytes (4 + 256)", () => {
      expect(ERROR_STRUCT_SIZE).toBe(260);
    });
  });

  describe("TorchError", () => {
    test("should extend Error", () => {
      const err = new TorchError(ErrorCode.INVALID_SHAPE, "test");
      expect(err).toBeInstanceOf(Error);
      expect(err).toBeInstanceOf(TorchError);
    });

    test("should store error code and message", () => {
      const err = new TorchError(ErrorCode.NULL_POINTER, "null ptr");
      expect(err.code).toBe(ErrorCode.NULL_POINTER);
      expect(err.nativeMessage).toBe("null ptr");
      expect(err.message).toContain("Null pointer");
      expect(err.message).toContain("null ptr");
    });

    test("should have correct name", () => {
      const err = new TorchError(ErrorCode.OK, "");
      expect(err.name).toBe("TorchError");
    });

    test("is() should check error code", () => {
      const err = new TorchError(ErrorCode.CUDA_ERROR, "cuda failed");
      expect(err.is(ErrorCode.CUDA_ERROR)).toBe(true);
      expect(err.is(ErrorCode.NULL_POINTER)).toBe(false);
    });

    test("toJSON() should serialize error", () => {
      const err = new TorchError(ErrorCode.DIMENSION_MISMATCH, "shape error");
      const json = err.toJSON();

      expect(json).toMatchObject({
        name: "TorchError",
        code: ErrorCode.DIMENSION_MISMATCH,
        nativeMessage: "shape error",
      });
    });
  });

  describe("createError", () => {
    test("should allocate error buffer", () => {
      const err = createError();
      expect(err).toBeTruthy();
      expect(typeof err).toBe("object");
    });

    test("should initialize with code 0 (OK)", () => {
      const err = createError();
      expect(() => checkError(err)).not.toThrow();
    });
  });

  describe("checkError", () => {
    test("should not throw for OK error", () => {
      const buffer = new ArrayBuffer(ERROR_STRUCT_SIZE);
      const view = new DataView(buffer);
      view.setInt32(0, ErrorCode.OK, true);

      const errorPtr = ptr(buffer);
      expect(() => checkError(errorPtr)).not.toThrow();
    });

    test("should throw TorchError for non-zero code", () => {
      const buffer = new ArrayBuffer(ERROR_STRUCT_SIZE);
      const view = new DataView(buffer);

      // Set error code
      view.setInt32(0, ErrorCode.NULL_POINTER, true);

      // Set error message
      const message = "Test error message";
      const encoder = new TextEncoder();
      const messageBytes = encoder.encode(message);
      const uint8View = new Uint8Array(buffer, 4, 256);
      uint8View.set(messageBytes);

      const errorPtr = ptr(buffer);

      expect(() => checkError(errorPtr)).toThrow(TorchError);

      try {
        checkError(errorPtr);
      } catch (err) {
        expect(err).toBeInstanceOf(TorchError);
        expect((err as TorchError).code).toBe(ErrorCode.NULL_POINTER);
        expect((err as TorchError).nativeMessage).toBe(message);
      }
    });

    test("should handle empty error message", () => {
      const buffer = new ArrayBuffer(ERROR_STRUCT_SIZE);
      const view = new DataView(buffer);
      view.setInt32(0, ErrorCode.UNKNOWN, true);

      const errorPtr = ptr(buffer);

      expect(() => checkError(errorPtr)).toThrow(TorchError);

      try {
        checkError(errorPtr);
      } catch (err) {
        expect((err as TorchError).code).toBe(ErrorCode.UNKNOWN);
        expect((err as TorchError).nativeMessage).toBe("");
      }
    });
  });

  describe("withError", () => {
    test("should automatically check error", () => {
      // Function that succeeds
      const result = withError((err) => {
        // Return success (don't modify error struct)
        return 42;
      });

      expect(result).toBe(42);
    });

    test("should throw if error is set", () => {
      expect(() => {
        withError((err) => {
          // Simulate error
          const buffer = err as unknown as ArrayBuffer;
          const view = new DataView(buffer);
          view.setInt32(0, ErrorCode.OUT_OF_MEMORY, true);

          return null;
        });
      }).toThrow(TorchError);
    });

    test("should return function result", () => {
      const obj = { value: 123 };
      const result = withError(() => obj);
      expect(result).toBe(obj);
    });
  });

  describe("checkNull", () => {
    test("should throw for null pointer", () => {
      expect(() => checkNull(null as any)).toThrow(TorchError);
      expect(() => checkNull(0 as any)).toThrow(TorchError);
      expect(() => checkNull(0n as any)).toThrow(TorchError);
    });

    test("should not throw for valid pointer", () => {
      const buffer = new ArrayBuffer(8);
      const validPtr = ptr(buffer);

      expect(() => checkNull(validPtr)).not.toThrow();
    });

    test("should use custom message", () => {
      try {
        checkNull(null as any, "Custom error");
      } catch (err) {
        expect(err).toBeInstanceOf(TorchError);
        expect((err as TorchError).nativeMessage).toBe("Custom error");
      }
    });
  });

  describe("validateShape", () => {
    test("should accept valid shapes", () => {
      expect(() => validateShape([1, 2, 3])).not.toThrow();
      expect(() => validateShape([0])).not.toThrow();
      expect(() => validateShape([1n, 2n, 3n])).not.toThrow();
    });

    test("should reject empty array", () => {
      expect(() => validateShape([])).toThrow(TorchError);
    });

    test("should reject non-array", () => {
      expect(() => validateShape(null as any)).toThrow(TorchError);
      expect(() => validateShape(undefined as any)).toThrow(TorchError);
      expect(() => validateShape(42 as any)).toThrow(TorchError);
    });

    test("should reject negative dimensions", () => {
      expect(() => validateShape([1, -2, 3])).toThrow(TorchError);
      expect(() => validateShape([-1n])).toThrow(TorchError);
    });

    test("should reject non-integer dimensions", () => {
      expect(() => validateShape([1.5, 2])).toThrow(TorchError);
      expect(() => validateShape([NaN])).toThrow(TorchError);
    });
  });

  describe("validateDtype", () => {
    test("should accept valid dtypes", () => {
      expect(() => validateDtype(0)).not.toThrow(); // f32
      expect(() => validateDtype(1)).not.toThrow(); // f64
      expect(() => validateDtype(2)).not.toThrow(); // i32
      expect(() => validateDtype(3)).not.toThrow(); // i64
    });

    test("should reject invalid dtype values", () => {
      expect(() => validateDtype(-1)).toThrow(TorchError);
      expect(() => validateDtype(4)).toThrow(TorchError);
      expect(() => validateDtype(99)).toThrow(TorchError);
    });

    test("should reject non-integer dtype", () => {
      expect(() => validateDtype(1.5)).toThrow(TorchError);
      expect(() => validateDtype(NaN)).toThrow(TorchError);
    });
  });
});
