/**
 * Examples demonstrating the scoped memory management system
 *
 * These examples show how to use torch.run() for automatic memory cleanup
 * and the tensor pool for performance optimization.
 */

import { run, runAsync, TensorPool, globalTensorPool } from './index.js'

// Note: These examples use a hypothetical Tensor API
// In practice, you would use the actual ts-torch Tensor class

/**
 * Example 1: Basic scoped memory management
 *
 * All tensors created within the scope are automatically freed when it exits.
 */
export function basicScopeExample() {
  // Outside scope - tensors must be manually freed
  console.log('Before scope:')
  console.log('In scope:', false)

  const result = run(() => {
    console.log('\nInside scope:')
    console.log('In scope:', true)

    // These tensors will be auto-freed when scope exits
    // const a = torch.zeros([100, 100]);
    // const b = torch.ones([100, 100]);
    // const c = a.add(b);

    // Return the result - it will be escaped automatically
    // return c.escape();
    return 'result'
  })

  console.log('\nAfter scope:')
  console.log('In scope:', false)
  console.log('Result:', result)
  // a and b have been freed
  // c persists because it was escaped
}

/**
 * Example 2: Nested scopes
 *
 * Scopes can be nested for fine-grained control over memory lifetimes.
 */
export function nestedScopesExample() {
  console.log('Outer scope:')

  const outerResult = run(() => {
    // const x = torch.randn([50, 50]);

    console.log('  Inner scope:')
    const innerResult = run(() => {
      // const y = torch.randn([50, 50]);
      // const z = x.matmul(y);
      // return z.escape();
      return 'inner'
    })
    // y is freed, z persists

    // const result = x.add(innerResult);
    // return result.escape();
    return innerResult
  })
  // x is freed, result persists

  console.log('Result:', outerResult)
}

/**
 * Example 3: Training loop with automatic cleanup
 *
 * Use scopes in training loops to prevent memory leaks from intermediate tensors.
 */
export function trainingLoopExample() {
  console.log('Training loop with scoped memory:')

  // const model = createModel();
  // const optimizer = new SGD(model.parameters());

  for (let epoch = 0; epoch < 10; epoch++) {
    let totalLoss = 0

    // Scope per epoch
    run(() => {
      for (let batch = 0; batch < 100; batch++) {
        // Scope per batch - automatic cleanup after each iteration
        const loss = run(() => {
          // const input = getBatch();
          // const output = model.forward(input);
          // const loss = criterion(output, target);

          // Only the loss value escapes
          // return loss.item();
          return Math.random()
        })

        totalLoss += loss

        // All intermediate tensors (input, output) are freed here
      }
    })

    console.log(`Epoch ${epoch}: Loss = ${totalLoss / 100}`)
  }
}

/**
 * Example 4: Async operations with scoped memory
 *
 * Use runAsync() for async operations while maintaining memory safety.
 */
export async function asyncScopeExample() {
  console.log('Async scope example:')

  const result = await runAsync(async () => {
    console.log('  Loading data...')
    // Simulate async data loading
    await new Promise((resolve) => setTimeout(resolve, 100))

    // const data = await fetchTensorData();
    // const tensor = torch.fromBuffer(data);

    console.log('  Processing...')
    await new Promise((resolve) => setTimeout(resolve, 100))

    // const processed = await processAsync(tensor);
    // return processed.escape();
    return 'processed'
  })

  console.log('Result:', result)
}

/**
 * Example 5: Tensor pooling for performance
 *
 * Reuse tensor allocations to reduce overhead in hot paths.
 */
export function tensorPoolExample() {
  console.log('Tensor pool example:')

  const pool = new TensorPool()

  // Training loop with pooling
  for (let i = 0; i < 1000; i++) {
    run(() => {
      // Try to acquire from pool, create if not available
      // const grad = pool.acquire([256, 256], "float32") ?? torch.zeros([256, 256]);
      // Use gradient...
      // optimizer.step(grad);
      // Return to pool for reuse
      // pool.release(grad);
    })
  }

  const stats = pool.stats()
  console.log('Pool stats:')
  console.log(`  Total tensors: ${stats.size}`)
  console.log(`  Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`)
  console.log(`  Hits: ${stats.hitCount}, Misses: ${stats.missCount}`)
}

/**
 * Example 6: Memory management in inference
 *
 * Efficient memory usage during model inference.
 */
export function inferenceExample() {
  console.log('Inference with scoped memory:')

  // const model = loadModel();

  const results = []

  for (let i = 0; i < 100; i++) {
    const prediction = run(() => {
      // const input = preprocessInput(data[i]);
      // const output = model.forward(input);
      // return output.argmax().item();
      return i % 10
    })

    results.push(prediction)
    // All intermediate tensors are freed after each inference
  }

  console.log(`Processed ${results.length} inputs`)
}

/**
 * Example 7: Selective escaping
 *
 * Escape only the tensors you need to keep alive.
 */
export function selectiveEscapingExample() {
  console.log('Selective escaping:')

  const { mean, variance } = run(() => {
    // const data = torch.randn([1000, 100]);

    // These will be freed
    // const centered = data.sub(data.mean(0));
    // const squared = centered.pow(2);

    // Only escape what we need
    return {
      // mean: data.mean(0).escape(),
      // variance: squared.mean(0).escape(),
      mean: 'mean',
      variance: 'variance',
    }
  })

  // Only mean and variance persist
  console.log('Statistics computed:', { mean, variance })
}

/**
 * Example 8: Global tensor pool usage
 *
 * Use the global pool for convenience.
 */
export function globalPoolExample() {
  console.log('Global tensor pool:')

  for (let i = 0; i < 100; i++) {
    run(() => {
      // Try global pool first
      // const temp = globalTensorPool.acquire([10, 10], "float32") ?? torch.zeros([10, 10]);
      // Use tensor...
      // Return to global pool
      // globalTensorPool.release(temp);
    })
  }

  const stats = globalTensorPool.stats()
  console.log('Global pool stats:', stats)
}

/**
 * Example 9: Mixed manual and automatic memory management
 *
 * Combine scoped memory with manual control where needed.
 */
export function mixedMemoryManagementExample() {
  console.log('Mixed memory management:')

  // Manual allocation outside scope
  // const weights = torch.randn([100, 100]);

  // Use scoped memory for intermediate computations
  for (let i = 0; i < 10; i++) {
    run(() => {
      // const input = torch.randn([100, 100]);
      // const output = weights.matmul(input);
      // const loss = output.sum();
      // Update weights (escaped from scope)
      // weights.sub_(loss.grad());
      // input, output, loss are freed here
    })
  }

  // Manual cleanup when done
  // weights.free();
}

/**
 * Example 10: Error handling with scopes
 *
 * Scopes ensure cleanup even when exceptions occur.
 */
export function errorHandlingExample() {
  console.log('Error handling with scopes:')

  try {
    run(() => {
      // const tensor = torch.zeros([100, 100]);

      // Simulate an error
      throw new Error('Something went wrong!')

      // Even though this code doesn't execute,
      // the scope will still clean up 'tensor'
    })
  } catch (error) {
    console.log('Error caught:', (error as Error).message)
    console.log('Tensor was still cleaned up!')
  }
}

/**
 * Run all examples
 */
export function runAllExamples() {
  console.log('='.repeat(60))
  console.log('Memory Management Examples')
  console.log('='.repeat(60))

  basicScopeExample()
  console.log('\n' + '-'.repeat(60) + '\n')

  nestedScopesExample()
  console.log('\n' + '-'.repeat(60) + '\n')

  trainingLoopExample()
  console.log('\n' + '-'.repeat(60) + '\n')

  asyncScopeExample().then(() => {
    console.log('\n' + '-'.repeat(60) + '\n')
    tensorPoolExample()
    console.log('\n' + '-'.repeat(60) + '\n')

    inferenceExample()
    console.log('\n' + '-'.repeat(60) + '\n')

    selectiveEscapingExample()
    console.log('\n' + '-'.repeat(60) + '\n')

    globalPoolExample()
    console.log('\n' + '-'.repeat(60) + '\n')

    mixedMemoryManagementExample()
    console.log('\n' + '-'.repeat(60) + '\n')

    errorHandlingExample()
    console.log('\n' + '='.repeat(60))
  })
}

// Uncomment to run examples:
// runAllExamples();
