/**
 * Tests for scoped memory management
 */

import { describe, test, expect, beforeEach } from "bun:test";
import {
  run,
  runAsync,
  inScope,
  scopeDepth,
  currentScopeId,
  scopeTensorCount,
  registerTensor,
  escapeTensor,
  type ScopedTensor,
} from "../scope";
import type { Pointer } from "bun:ffi";

// Mock tensor for testing
class MockTensor implements ScopedTensor {
  handle: Pointer = 0 as unknown as Pointer;
  escaped = false;
  freed = false;

  constructor(handle: number) {
    this.handle = handle as unknown as Pointer;
  }

  markEscaped(): void {
    this.escaped = true;
  }

  free(): void {
    this.freed = true;
  }
}

describe("Scope Management", () => {
  describe("inScope", () => {
    test("returns false outside scope", () => {
      expect(inScope()).toBe(false);
    });

    test("returns true inside scope", () => {
      run(() => {
        expect(inScope()).toBe(true);
      });
    });

    test("returns false after scope exits", () => {
      run(() => {
        expect(inScope()).toBe(true);
      });
      expect(inScope()).toBe(false);
    });
  });

  describe("scopeDepth", () => {
    test("returns 0 outside scope", () => {
      expect(scopeDepth()).toBe(0);
    });

    test("returns 1 in single scope", () => {
      run(() => {
        expect(scopeDepth()).toBe(1);
      });
    });

    test("tracks nested scopes correctly", () => {
      run(() => {
        expect(scopeDepth()).toBe(1);
        run(() => {
          expect(scopeDepth()).toBe(2);
          run(() => {
            expect(scopeDepth()).toBe(3);
          });
          expect(scopeDepth()).toBe(2);
        });
        expect(scopeDepth()).toBe(1);
      });
      expect(scopeDepth()).toBe(0);
    });
  });

  describe("currentScopeId", () => {
    test("returns -1 outside scope", () => {
      expect(currentScopeId()).toBe(-1);
    });

    test("returns valid ID inside scope", () => {
      run(() => {
        const id = currentScopeId();
        expect(id).toBeGreaterThanOrEqual(0);
      });
    });

    test("different scopes have different IDs", () => {
      const ids: number[] = [];
      run(() => {
        ids.push(currentScopeId());
      });
      run(() => {
        ids.push(currentScopeId());
      });
      expect(ids[0]).not.toBe(ids[1]);
    });
  });

  describe("run", () => {
    test("executes function and returns result", () => {
      const result = run(() => {
        return 42;
      });
      expect(result).toBe(42);
    });

    test("preserves return type", () => {
      const obj = run(() => {
        return { value: 100, nested: { key: "test" } };
      });
      expect(obj.value).toBe(100);
      expect(obj.nested.key).toBe("test");
    });

    test("handles exceptions", () => {
      expect(() => {
        run(() => {
          throw new Error("Test error");
        });
      }).toThrow("Test error");
    });

    test("cleans up scope after exception", () => {
      expect(() => {
        run(() => {
          throw new Error("Test error");
        });
      }).toThrow();

      expect(inScope()).toBe(false);
      expect(scopeDepth()).toBe(0);
    });
  });

  describe("runAsync", () => {
    test("executes async function and returns result", async () => {
      const result = await runAsync(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        return 42;
      });
      expect(result).toBe(42);
    });

    test("maintains scope during async operations", async () => {
      const checks: boolean[] = [];
      await runAsync(async () => {
        checks.push(inScope());
        await new Promise((resolve) => setTimeout(resolve, 10));
        checks.push(inScope());
      });
      expect(checks).toEqual([true, true]);
      expect(inScope()).toBe(false);
    });

    test("handles async exceptions", async () => {
      await expect(
        runAsync(async () => {
          await new Promise((resolve) => setTimeout(resolve, 10));
          throw new Error("Async error");
        }),
      ).rejects.toThrow("Async error");

      expect(inScope()).toBe(false);
    });
  });

  describe("registerTensor", () => {
    test("does nothing outside scope", () => {
      const tensor = new MockTensor(1);
      expect(() => registerTensor(tensor)).not.toThrow();
    });

    test("tracks tensor inside scope", () => {
      run(() => {
        const tensor = new MockTensor(1);
        registerTensor(tensor);
        expect(scopeTensorCount()).toBe(1);
      });
    });

    test("tracks multiple tensors", () => {
      run(() => {
        for (let i = 0; i < 5; i++) {
          const tensor = new MockTensor(i);
          registerTensor(tensor);
        }
        expect(scopeTensorCount()).toBe(5);
      });
    });

    test("nested scopes track tensors separately", () => {
      run(() => {
        registerTensor(new MockTensor(1));
        expect(scopeTensorCount()).toBe(1);

        run(() => {
          registerTensor(new MockTensor(2));
          registerTensor(new MockTensor(3));
          expect(scopeTensorCount()).toBe(2);
        });

        expect(scopeTensorCount()).toBe(1);
      });
    });
  });

  describe("escapeTensor", () => {
    test("throws outside scope", () => {
      const tensor = new MockTensor(1);
      expect(() => escapeTensor(tensor)).toThrow("not currently in a scope");
    });

    test("marks tensor as escaped", () => {
      run(() => {
        const tensor = new MockTensor(1);
        registerTensor(tensor);
        expect(tensor.escaped).toBe(false);

        escapeTensor(tensor);
        expect(tensor.escaped).toBe(true);
      });
    });

    test("returns the same tensor for chaining", () => {
      run(() => {
        const tensor = new MockTensor(1);
        registerTensor(tensor);
        const escaped = escapeTensor(tensor);
        expect(escaped).toBe(tensor);
      });
    });
  });

  describe("scopeTensorCount", () => {
    test("returns 0 outside scope", () => {
      expect(scopeTensorCount()).toBe(0);
    });

    test("counts registered tensors", () => {
      run(() => {
        expect(scopeTensorCount()).toBe(0);
        registerTensor(new MockTensor(1));
        expect(scopeTensorCount()).toBe(1);
        registerTensor(new MockTensor(2));
        expect(scopeTensorCount()).toBe(2);
      });
    });

    test("resets after scope exit", () => {
      run(() => {
        registerTensor(new MockTensor(1));
        registerTensor(new MockTensor(2));
        expect(scopeTensorCount()).toBe(2);
      });
      expect(scopeTensorCount()).toBe(0);
    });
  });

  describe("Integration scenarios", () => {
    test("simple scope with escape", () => {
      let escapedTensor: MockTensor | null = null;

      run(() => {
        const temp1 = new MockTensor(1);
        const temp2 = new MockTensor(2);
        const keep = new MockTensor(3);

        registerTensor(temp1);
        registerTensor(temp2);
        registerTensor(keep);

        escapedTensor = escapeTensor(keep);

        expect(scopeTensorCount()).toBe(3);
        expect(escapedTensor.escaped).toBe(true);
        expect(temp1.escaped).toBe(false);
        expect(temp2.escaped).toBe(false);
      });

      expect(escapedTensor).not.toBeNull();
      expect(escapedTensor!.escaped).toBe(true);
    });

    test("nested scopes with selective escaping", () => {
      const results: MockTensor[] = [];

      run(() => {
        const outer1 = new MockTensor(1);
        registerTensor(outer1);

        run(() => {
          const inner1 = new MockTensor(2);
          const inner2 = new MockTensor(3);
          registerTensor(inner1);
          registerTensor(inner2);

          results.push(escapeTensor(inner1));
        });

        results.push(escapeTensor(outer1));
      });

      expect(results).toHaveLength(2);
      expect(results[0]!.escaped).toBe(true);
      expect(results[1]!.escaped).toBe(true);
    });

    test("async scope with tensor lifecycle", async () => {
      const tensor = await runAsync(async () => {
        const t = new MockTensor(1);
        registerTensor(t);

        await new Promise((resolve) => setTimeout(resolve, 10));

        return escapeTensor(t);
      });

      expect(tensor.escaped).toBe(true);
    });
  });
});
