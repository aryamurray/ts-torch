/**
 * Examples demonstrating type-safe .pipe() chaining and shape inference
 *
 * This file showcases the advanced TypeScript features of the ts-torch nn module:
 * - Compile-time shape checking
 * - Type-safe .pipe() composition
 * - Shape inference through complex pipelines
 * - Type errors for shape mismatches
 */

import { Linear, ReLU, Sigmoid, Softmax, Sequential, sequential } from '../index.js'
import type { Tensor } from '../module.js'

// ============================================================================
// Example 1: Basic .pipe() chaining
// ============================================================================

/**
 * Simple feed-forward network using .pipe()
 *
 * Type inference tracks shapes through the entire pipeline!
 */
function example1_SimplePipe() {
  // Build a model: 784 -> 128 -> 64 -> 10
  const model = new Linear(784, 128)
    .pipe(new ReLU())
    .pipe(new Linear(128, 64))
    .pipe(new ReLU())
    .pipe(new Linear(64, 10))
    .pipe(new Softmax(-1))

  // Type: PipedModule<readonly [number, 784], readonly [number, 10]>

  // Type-safe forward pass
  const input = {} as Tensor<readonly [32, 784]>
  const output = model.forward(input)
  void output // Type of output: Tensor<readonly [32, 10]>

  return model
}

// ============================================================================
// Example 2: Type errors for shape mismatches
// ============================================================================

/**
 * This example shows how TypeScript catches shape mismatches at compile time
 */
function example2_TypeErrors() {
  const layer1 = new Linear(784, 128)
  const layer2 = new Linear(128, 64)
  const layer3 = new Linear(256, 10) // Note: expects 256 inputs, not 64!
  void layer3 // Intentionally unused - demonstrates incompatible shape

  // This works: 784 -> 128 -> 64
  const validPipeline = layer1.pipe(new ReLU()).pipe(layer2)

  // This would be a TYPE ERROR:
  // const invalidPipeline = validPipeline.pipe(layer3);
  // Error: Type 'readonly [number, 64]' is not assignable to type 'readonly [number, 256]'
  //
  // TypeScript catches that layer3 expects 256 inputs but only gets 64!

  return validPipeline
}

// ============================================================================
// Example 3: Sequential container
// ============================================================================

/**
 * Using Sequential for module composition
 */
function example3_Sequential() {
  // Explicit type annotation for full type safety
  const model = new Sequential<readonly [number, 784], readonly [number, 10]>(
    new Linear(784, 256),
    new ReLU(),
    new Linear(256, 128),
    new ReLU(),
    new Linear(128, 10),
    new Softmax(-1),
  )

  const input = {} as Tensor<readonly [32, 784]>
  const output = model.forward(input)
  void output // Type: Tensor<readonly [32, 10]>

  return model
}

// ============================================================================
// Example 4: Sequential builder with full type inference
// ============================================================================

/**
 * Using the sequential() builder for automatic shape inference
 */
function example4_SequentialBuilder() {
  // The builder tracks shapes at each step!
  const model = sequential<readonly [number, 784]>()
    .add(new Linear(784, 256))
    .add(new ReLU())
    .add(new Linear(256, 128))
    .add(new ReLU())
    .add(new Linear(128, 10))
    .add(new Softmax(-1))
    .build()

  // Type: Sequential<readonly [number, 784], readonly [number, 10]>

  const input = {} as Tensor<readonly [32, 784]>
  const output = model.forward(input)
  void output // Type: Tensor<readonly [32, 10]>

  return model
}

// ============================================================================
// Example 5: Deep networks with intermediate shapes
// ============================================================================

/**
 * Building a deeper network with carefully tracked shapes
 */
function example5_DeepNetwork() {
  // Encoder: 784 -> 512 -> 256 -> 128
  const encoder = new Linear(784, 512)
    .pipe(new ReLU())
    .pipe(new Linear(512, 256))
    .pipe(new ReLU())
    .pipe(new Linear(256, 128))

  // Decoder: 128 -> 256 -> 512 -> 784
  const decoder = new Linear(128, 256)
    .pipe(new ReLU())
    .pipe(new Linear(256, 512))
    .pipe(new ReLU())
    .pipe(new Linear(512, 784))
    .pipe(new Sigmoid()) // Output in [0, 1] for reconstruction

  // Autoencoder: combine encoder and decoder
  const autoencoder = encoder.pipe(decoder)
  // Type: PipedModule<readonly [number, 784], readonly [number, 784]>
  // Input and output have same shape!

  const input = {} as Tensor<readonly [32, 784]>
  const reconstruction = autoencoder.forward(input)
  void reconstruction // Type: Tensor<readonly [32, 784]>

  return { encoder, decoder, autoencoder }
}

// ============================================================================
// Example 6: Classifier with explicit feature dimensions
// ============================================================================

/**
 * Image classifier with explicit dimensions
 */
function example6_ImageClassifier() {
  const NUM_CLASSES = 10 as const
  const IMAGE_SIZE = 784 as const // 28x28 flattened

  const classifier = new Linear(IMAGE_SIZE, 512)
    .pipe(new ReLU())
    .pipe(new Linear(512, 256))
    .pipe(new ReLU())
    .pipe(new Linear(256, NUM_CLASSES))
    .pipe(new Softmax(-1))

  // Type system knows exact dimensions!
  // InputShape: readonly [number, 784]
  // OutputShape: readonly [number, 10]

  const images = {} as Tensor<readonly [64, 784]>
  const predictions = classifier.forward(images)
  void predictions // Type: Tensor<readonly [64, 10]>

  return classifier
}

// ============================================================================
// Example 7: Composing pre-built modules
// ============================================================================

/**
 * Building complex models from pre-built components
 */
function example7_ModularConstruction() {
  // Define reusable building blocks
  const hiddenLayer = (inFeatures: number, outFeatures: number) => new Linear(inFeatures, outFeatures).pipe(new ReLU())

  const outputLayer = (inFeatures: number, numClasses: number) =>
    new Linear(inFeatures, numClasses).pipe(new Softmax(-1))

  // Compose them
  const model = hiddenLayer(784, 512).pipe(hiddenLayer(512, 256)).pipe(hiddenLayer(256, 128)).pipe(outputLayer(128, 10))

  return model
}

// ============================================================================
// Example 8: Training mode and parameter management
// ============================================================================

/**
 * Using training/eval modes and accessing parameters
 */
function example8_TrainingMode() {
  const model = new Linear(784, 128).pipe(new ReLU()).pipe(new Linear(128, 10))

  // Switch to training mode
  model.train()
  console.log('Training:', model.training) // true

  // Get all parameters for optimizer
  const params = model.parameters()
  console.log('Number of parameters:', params.length)

  // Get named parameters
  const namedParams = model.namedParameters()
  for (const [name, param] of namedParams) {
    console.log(`Parameter: ${name}, requires_grad: ${param.requiresGrad}`)
  }

  // Switch to evaluation mode
  model.eval()
  console.log('Training:', model.training) // false

  return model
}

// ============================================================================
// Export examples for testing
// ============================================================================

export const examples = {
  example1_SimplePipe,
  example2_TypeErrors,
  example3_Sequential,
  example4_SequentialBuilder,
  example5_DeepNetwork,
  example6_ImageClassifier,
  example7_ModularConstruction,
  example8_TrainingMode,
}

/**
 * Type-level tests to ensure shape inference works correctly
 * These types will fail to compile if shape inference is broken
 */
export namespace TypeTests {
  // Test 1: Simple pipe preserves exact types
  const model1 = new Linear(10, 20).pipe(new ReLU())
  void model1 // Used for type testing only
  // model1.forward: Tensor<readonly [number, 10]> -> Tensor<readonly [number, 20]>

  // Test 2: Long chain preserves end-to-end types
  const model2 = new Linear(10, 20).pipe(new ReLU()).pipe(new Linear(20, 30)).pipe(new ReLU())
  void model2 // Used for type testing only
  // model2.forward: Tensor<readonly [number, 10]> -> Tensor<readonly [number, 30]>

  // Test 3: Sequential has correct types
  const model3 = new Sequential<readonly [number, 10], readonly [number, 30]>(new Linear(10, 20), new Linear(20, 30))
  void model3 // Used for type testing only
  // model3.forward: Tensor<readonly [number, 10]> -> Tensor<readonly [number, 30]>
}
