/**
 * Scoped memory management for automatic tensor cleanup
 *
 * Implements PyTorch-style scope-based memory management using a stack
 * of scope contexts. All tensors created within a scope are automatically
 * freed when the scope exits, unless explicitly escaped.
 *
 * @example
 * ```ts
 * import { run } from '@ts-torch/core/memory';
 * import { zeros } from '@ts-torch/core';
 *
 * const result = run(() => {
 *   const a = zeros([100, 100]); // Auto-freed on scope exit
 *   const b = zeros([100, 100]); // Auto-freed on scope exit
 *   const c = a.add(b);
 *   return c.escape(); // Prevent c from being freed
 * });
 * // a and b are freed here, c lives on
 * ```
 */

import { getLib } from "../ffi/loader.js";
import type { Pointer } from "bun:ffi";

/**
 * Tensor interface for scope management
 * We use a minimal interface to avoid circular dependencies
 */
export interface ScopedTensor {
  readonly handle: Pointer;
  readonly escaped: boolean;
  markEscaped(): void;
}

/**
 * Scope context tracking tensors and parent scope
 */
interface ScopeContext {
  readonly id: number;
  readonly tensors: Set<ScopedTensor>;
  readonly parent: ScopeContext | null;
}

/**
 * Current scope context (thread-local-like via closure)
 */
let currentScope: ScopeContext | null = null;

/**
 * Execute code with scoped tensor memory management.
 * All tensors created within are automatically freed when scope exits.
 * Use tensor.escape() to keep a tensor alive after scope.
 *
 * @template T - Return type of the function
 * @param fn - Function to execute within the scope
 * @returns The result of the function
 *
 * @example
 * ```ts
 * const result = run(() => {
 *   const x = zeros([10, 10]);
 *   const y = ones([10, 10]);
 *   const sum = x.add(y);
 *   return sum.escape(); // Keep sum alive
 * });
 * // x and y are freed, sum persists
 * ```
 */
export function run<T>(fn: () => T): T {
  const lib = getLib();

  // Begin native scope
  lib.symbols.ts_scope_begin();

  // Create JS scope context
  const newScope: ScopeContext = {
    id: Date.now(), // Use timestamp as unique scope ID
    tensors: new Set(),
    parent: currentScope,
  };

  const previousScope = currentScope;
  currentScope = newScope;

  try {
    const result = fn();
    return result;
  } finally {
    // Cleanup tensors that weren't escaped
    for (const tensor of newScope.tensors) {
      if (!tensor.escaped) {
        // The tensor will be freed by native ts_scope_end
        // We just need to track which ones should be freed
      }
    }

    // End native scope
    lib.symbols.ts_scope_end();

    // Restore previous scope
    currentScope = previousScope;
  }
}

/**
 * Async version of run() for async operations.
 * Enables scoped memory management with async/await code.
 *
 * @template T - Return type of the async function
 * @param fn - Async function to execute within the scope
 * @returns Promise of the function result
 *
 * @example
 * ```ts
 * const result = await runAsync(async () => {
 *   const data = await fetchTensorData();
 *   const tensor = fromBuffer(data);
 *   const processed = await processAsync(tensor);
 *   return processed.escape();
 * });
 * ```
 */
export async function runAsync<T>(fn: () => Promise<T>): Promise<T> {
  const lib = getLib();

  lib.symbols.ts_scope_begin();

  const newScope: ScopeContext = {
    id: Date.now(), // Use timestamp as unique scope ID
    tensors: new Set(),
    parent: currentScope,
  };

  const previousScope = currentScope;
  currentScope = newScope;

  try {
    const result = await fn();
    return result;
  } finally {
    // Cleanup tensors that weren't escaped
    for (const tensor of newScope.tensors) {
      if (!tensor.escaped) {
        // The tensor will be freed by native ts_scope_end
      }
    }

    lib.symbols.ts_scope_end();

    currentScope = previousScope;
  }
}

/**
 * Register a tensor with current scope (called by Tensor constructor).
 * If no scope is active, tensor must be manually freed.
 *
 * @param tensor - Tensor to register
 *
 * @internal
 */
export function registerTensor(tensor: ScopedTensor): void {
  if (currentScope !== null) {
    currentScope.tensors.add(tensor);

    // Register with native scope
    const lib = getLib();
    lib.symbols.ts_scope_register_tensor(tensor.handle);
  }
}

/**
 * Escape tensor from current scope (called by tensor.escape()).
 * Prevents the tensor from being freed when the scope exits.
 *
 * @template T - Tensor type
 * @param tensor - Tensor to escape
 * @returns The same tensor for chaining
 *
 * @throws Error if not currently in a scope
 *
 * @example
 * ```ts
 * const tensor = run(() => {
 *   const x = zeros([10, 10]);
 *   return escapeTensor(x);
 * });
 * // tensor is not freed
 * ```
 */
export function escapeTensor<T extends ScopedTensor>(tensor: T): T {
  if (currentScope === null) {
    throw new Error(
      "Cannot escape tensor: not currently in a scope. " +
        "Use torch.run(() => { ... }) to create a scope.",
    );
  }

  // Mark as escaped in JS
  tensor.markEscaped();

  // Remove from native scope tracking
  const lib = getLib();
  lib.symbols.ts_scope_escape_tensor(tensor.handle);

  return tensor;
}

/**
 * Check if currently inside a scope.
 *
 * @returns True if inside a scope, false otherwise
 *
 * @example
 * ```ts
 * console.log(inScope()); // false
 * run(() => {
 *   console.log(inScope()); // true
 * });
 * ```
 */
export function inScope(): boolean {
  return currentScope !== null;
}

/**
 * Get current scope depth (for debugging).
 * Returns 0 if not in a scope.
 *
 * @returns Number of nested scopes
 *
 * @example
 * ```ts
 * console.log(scopeDepth()); // 0
 * run(() => {
 *   console.log(scopeDepth()); // 1
 *   run(() => {
 *     console.log(scopeDepth()); // 2
 *   });
 * });
 * ```
 */
export function scopeDepth(): number {
  let depth = 0;
  let scope = currentScope;
  while (scope !== null) {
    depth++;
    scope = scope.parent;
  }
  return depth;
}

/**
 * Get current scope ID (for debugging).
 * Returns -1 if not in a scope.
 *
 * @returns Scope ID or -1
 *
 * @internal
 */
export function currentScopeId(): number {
  return currentScope?.id ?? -1;
}

/**
 * Get number of tensors in current scope (for debugging).
 * Returns 0 if not in a scope.
 *
 * @returns Number of tracked tensors
 *
 * @example
 * ```ts
 * run(() => {
 *   const a = zeros([10, 10]);
 *   const b = ones([10, 10]);
 *   console.log(scopeTensorCount()); // 2
 * });
 * ```
 */
export function scopeTensorCount(): number {
  return currentScope?.tensors.size ?? 0;
}
