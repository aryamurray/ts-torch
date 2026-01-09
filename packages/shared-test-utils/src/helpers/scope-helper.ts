import type { Tensor } from '@ts-torch/core';

/**
 * Helper class for managing tensor scopes in tests
 */
export class ScopeTestHelper {
  private tensors: Set<Tensor> = new Set();
  private freed: boolean = false;

  /**
   * Register a tensor to be freed when the scope ends
   */
  register(tensor: Tensor): Tensor {
    if (this.freed) {
      throw new Error('Cannot register tensors on a freed scope');
    }
    this.tensors.add(tensor);
    return tensor;
  }

  /**
   * Register multiple tensors
   */
  registerAll(...tensors: Tensor[]): void {
    tensors.forEach((t) => this.register(t));
  }

  /**
   * Manually free all registered tensors
   */
  free(): void {
    if (!this.freed) {
      this.tensors.forEach((tensor) => {
        if (typeof tensor.free === 'function') {
          tensor.free();
        }
      });
      this.tensors.clear();
      this.freed = true;
    }
  }

  /**
   * Check if the scope has been freed
   */
  isFreed(): boolean {
    return this.freed;
  }

  /**
   * Get the number of registered tensors
   */
  count(): number {
    return this.tensors.size;
  }
}

/**
 * Scoped test wrapper that manages tensor lifecycle
 * Uses torch.run() for automatic cleanup
 */
export function scopedTest<T>(
  fn: (scope: ScopeTestHelper) => T
): T {
  const scope = new ScopeTestHelper();

  try {
    // If torch.run is available, use it
    if (typeof (globalThis as any).torch?.run === 'function') {
      return (globalThis as any).torch.run(() => fn(scope));
    }

    // Otherwise, manual cleanup
    return fn(scope);
  } finally {
    scope.free();
  }
}

/**
 * Async scoped test wrapper that manages tensor lifecycle
 * Uses torch.runAsync() for automatic cleanup
 */
export async function scopedTestAsync<T>(
  fn: (scope: ScopeTestHelper) => Promise<T>
): Promise<T> {
  const scope = new ScopeTestHelper();

  try {
    // If torch.runAsync is available, use it
    if (typeof (globalThis as any).torch?.runAsync === 'function') {
      return await (globalThis as any).torch.runAsync(() => fn(scope));
    }

    // Otherwise, manual cleanup
    return await fn(scope);
  } finally {
    scope.free();
  }
}

/**
 * Create a test scope manually
 */
export function createTestScope(): ScopeTestHelper {
  return new ScopeTestHelper();
}
