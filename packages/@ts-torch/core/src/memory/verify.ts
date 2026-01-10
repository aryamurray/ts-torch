/**
 * Type-check verification for memory management system
 * This file ensures all types compile correctly without needing the native library
 */

import type { Pointer } from '../ffi/error.js'
import type { Shape } from '../types/shape.js'
import type { DTypeName } from '../types/dtype.js'

// Import types to verify they compile
import type { ScopedTensor } from './scope.js'

import type { PoolableTensor, PoolStats } from './pool.js'

// Verify function signatures
import {
  run,
  runAsync,
  registerTensor,
  escapeTensor,
  inScope,
  scopeDepth,
  currentScopeId,
  scopeTensorCount,
} from './scope.js'

import { TensorPool, globalTensorPool } from './pool.js'

// Type tests
type AssertTrue<T extends true> = T
type AssertExtends<T, U> = T extends U ? true : false

// Test scope function types
type RunType = typeof run
type RunAsyncType = typeof runAsync
type _TestRun = AssertTrue<AssertExtends<RunType, <T>(fn: () => T) => T>>
type _TestRunAsync = AssertTrue<AssertExtends<RunAsyncType, <T>(fn: () => Promise<T>) => Promise<T>>>

// Test utility function types
type InScopeType = typeof inScope
type ScopeDepthType = typeof scopeDepth
type _TestInScope = AssertTrue<AssertExtends<InScopeType, () => boolean>>
type _TestScopeDepth = AssertTrue<AssertExtends<ScopeDepthType, () => number>>

// Test ScopedTensor interface
type _TestScopedTensor = AssertTrue<
  AssertExtends<ScopedTensor, { readonly handle: Pointer; readonly escaped: boolean; markEscaped(): void }>
>

// Test PoolableTensor interface
type _TestPoolableTensor = AssertTrue<
  AssertExtends<PoolableTensor, { readonly shape: Shape; readonly dtype: DTypeName; readonly handle: unknown }>
>

// Test PoolStats interface
type _TestPoolStats = AssertTrue<
  AssertExtends<
    PoolStats,
    {
      readonly size: number
      readonly hitCount: number
      readonly missCount: number
      readonly hitRate: number
      readonly pools: ReadonlyMap<string, number>
    }
  >
>

// Test TensorPool methods exist
type _TestPoolAcquire = AssertTrue<
  AssertExtends<
    TensorPool['acquire'],
    <S extends Shape, D extends DTypeName>(shape: S, dtype: D) => PoolableTensor | null
  >
>

type _TestPoolRelease = AssertTrue<AssertExtends<TensorPool['release'], (tensor: PoolableTensor) => void>>

type _TestPoolStatsMethod = AssertTrue<AssertExtends<TensorPool['stats'], () => PoolStats>>

// Mock implementations for compile-time testing
class MockTensor implements ScopedTensor {
  handle: Pointer = 0 as unknown as Pointer
  escaped = false
  markEscaped(): void {
    this.escaped = true
  }
}

class MockPoolable implements PoolableTensor {
  shape: readonly number[] = [10, 10]
  dtype: DTypeName = 'float32'
  handle: unknown = 0
}

// Test usage patterns compile
function testBasicScope(): void {
  const result: number = run(() => {
    return 42
  })
  console.log(result)
}

async function testAsyncScope(): Promise<void> {
  const result: string = await runAsync(async () => {
    await Promise.resolve()
    return 'done'
  })
  console.log(result)
}

function testNestedScopes(): void {
  run(() => {
    const depth1: number = scopeDepth()
    run(() => {
      const depth2: number = scopeDepth()
      console.log(depth1, depth2)
    })
  })
}

function testTensorRegistration(): void {
  run(() => {
    const tensor = new MockTensor()
    registerTensor(tensor)
    const count: number = scopeTensorCount()
    console.log(count)
  })
}

function testEscaping(): void {
  run(() => {
    const tensor = new MockTensor()
    registerTensor(tensor)
    const escaped: MockTensor = escapeTensor(tensor)
    console.log(escaped.escaped)
  })
}

function testPool(): void {
  const pool = new TensorPool(16)
  const tensor: PoolableTensor | null = pool.acquire([10, 10], 'float32')

  if (tensor) {
    pool.release(tensor)
  }

  const stats: PoolStats = pool.stats()
  const hitRate: number = stats.hitRate
  const size: number = stats.size

  console.log(hitRate, size)
}

function testGlobalPool(): void {
  const tensor = new MockPoolable()
  globalTensorPool.release(tensor)
  const acquired = globalTensorPool.acquire([10, 10], 'float32')
  console.log(acquired)
}

function testPoolOperations(): void {
  const pool = new TensorPool()

  pool.clear()
  pool.clearKey([10, 10], 'float32')
  const keySize: number = pool.getKeySize([10, 10], 'float32')
  const keys: string[] = pool.getKeys()
  pool.prune(100)

  console.log(keySize, keys)
}

function testInScope(): void {
  const outside: boolean = inScope()
  run(() => {
    const inside: boolean = inScope()
    console.log(outside, inside)
  })
}

function testScopeId(): void {
  const outsideId: number = currentScopeId()
  run(() => {
    const insideId: number = currentScopeId()
    console.log(outsideId, insideId)
  })
}

// Export to prevent tree-shaking
export {
  testBasicScope,
  testAsyncScope,
  testNestedScopes,
  testTensorRegistration,
  testEscaping,
  testPool,
  testGlobalPool,
  testPoolOperations,
  testInScope,
  testScopeId,
}

console.log('✓ All types compile successfully')
