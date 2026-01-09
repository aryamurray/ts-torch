// @ts-nocheck
/**
 * MNIST Classification Example
 *
 * This example demonstrates the TARGET API for training a simple MLP on MNIST
 * using the ts-torch library with type-safe tensor operations.
 *
 * NOTE: This is a design reference showing the intended API. The underlying
 * FFI bindings to libtorch are not yet connected, so this won't run until
 * the native library is built and linked.
 */

// ===============================
// Imports
// ===============================

import { torch, Tensor, float32, int64, Device } from "@ts-torch/core";
import { Module, Linear, ReLU, Sequential } from "@ts-torch/nn";
import { SGD, crossEntropyLoss } from "@ts-torch/optim";
import { Mnist } from "@ts-torch/datasets";

// ===============================
// Typed Dimensions
// ===============================

// Using type aliases for semantic clarity
type Batch = number; // Dynamic batch size
type Pixels = 784; // 28x28 flattened
type Hidden = 128; // Hidden layer size
type Classes = 10; // Digit classes 0-9

// ===============================
// Dataset Setup
// ===============================

// Load MNIST and prepare data pipeline
const train = Mnist.train()
  .map((sample) => ({
    // Flatten 28x28 images to 784-dim vectors
    image: sample.image.reshape<[Pixels]>([784] as const),
    label: sample.label,
  }))
  .batch<Batch>(64)
  .shuffle();

const test = Mnist.test()
  .map((sample) => ({
    image: sample.image.reshape<[Pixels]>([784] as const),
    label: sample.label,
  }))
  .batch<Batch>(1000);

// ===============================
// Model Definition
// ===============================

/**
 * Simple Multi-Layer Perceptron for MNIST classification.
 *
 * Architecture:
 *   Input: [Batch, 784]
 *     -> Linear(784, 128) -> ReLU
 *     -> Linear(128, 10)
 *   Output: [Batch, 10] (logits)
 */
class MLP extends Module<
  Tensor<[Batch, Pixels], typeof float32>,
  Tensor<[Batch, Classes], typeof float32>
> {
  // Layer definitions with typed dimensions
  private fc1 = new Linear<Pixels, Hidden>(784, 128);
  private fc2 = new Linear<Hidden, Classes>(128, 10);
  private relu = new ReLU();

  forward(x: Tensor<[Batch, Pixels], typeof float32>): Tensor<[Batch, Classes], typeof float32> {
    // Type-safe forward pass using pipe
    return x.pipe(this.fc1).pipe(this.relu).pipe(this.fc2);
  }

  parameters() {
    return [...this.fc1.parameters(), ...this.fc2.parameters()];
  }

  namedParameters() {
    const params = new Map<string, Tensor<readonly number[], typeof float32>>();
    for (const [name, param] of this.fc1.namedParameters()) {
      params.set(`fc1.${name}`, param);
    }
    for (const [name, param] of this.fc2.namedParameters()) {
      params.set(`fc2.${name}`, param);
    }
    return params;
  }
}

// Alternative: Using Sequential for simpler architectures
const mlpSequential = new Sequential([
  new Linear<Pixels, Hidden>(784, 128),
  new ReLU(),
  new Linear<Hidden, Classes>(128, 10),
]);

// ===============================
// Training Setup
// ===============================

// Select device (CPU or CUDA if available)
const device = torch.cuda.isAvailable() ? Device.cuda(0) : Device.cpu();

console.log(`Using device: ${device}`);

// Instantiate model and move to device
const model = new MLP().to(device);

// Create optimizer with momentum
const optimizer = new SGD(model.parameters(), {
  lr: 0.01,
  momentum: 0.9,
});

// ===============================
// Training Loop
// ===============================

const NUM_EPOCHS = 5;

console.log("Starting training...\n");

for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
  let epochLoss = 0;
  let numBatches = 0;

  for (const batch of train) {
    // Use torch.run() for automatic memory management
    // All tensors created inside are freed when scope exits
    torch.run(() => {
      // Move data to device
      const x = batch.image.to(device); // [B, 784]
      const y = batch.label.to(device); // [B]

      // Forward pass
      const logits = model.forward(x); // [B, 10]

      // Compute cross-entropy loss
      const loss = crossEntropyLoss(logits, y);

      // Backward pass
      loss.backward();

      // Update parameters
      optimizer.step();
      optimizer.zeroGrad();

      // Accumulate loss (escape scalar to read outside scope)
      epochLoss += loss.item();
      numBatches++;
    });
  }

  const avgLoss = epochLoss / numBatches;
  console.log(`Epoch ${epoch + 1}/${NUM_EPOCHS} | Loss: ${avgLoss.toFixed(4)}`);
}

// ===============================
// Evaluation
// ===============================

console.log("\nEvaluating on test set...\n");

let correct = 0;
let total = 0;

// Set model to eval mode (affects dropout, batchnorm, etc.)
model.eval();

for (const batch of test) {
  torch.run(() => {
    const x = batch.image.to(device);
    const y = batch.label.to(device);

    // Forward pass (no gradient tracking needed)
    const logits = model.forward(x);

    // Get predictions (argmax along class dimension)
    const predictions = logits.argmax(1); // [B]

    // Count correct predictions
    const matches = predictions.eq(y); // [B] boolean tensor
    correct += matches.sum().item();
    total += y.shape[0];
  });
}

const accuracy = (correct / total) * 100;
console.log(`Test Accuracy: ${accuracy.toFixed(2)}%`);
console.log(`Correct: ${correct} / ${total}`);

// ===============================
// Type Safety Demonstrations
// ===============================

// The following would cause compile-time errors:

// Shape mismatch in Linear layer
// const badLayer = new Linear<256, 10>(256, 10);
// model.forward(x).pipe(badLayer);
// Error: Tensor<[Batch, 10]> not assignable to Tensor<[Batch, 256]>

// Invalid matmul dimensions
// const a = torch.randn([32, 128] as const);
// const b = torch.randn([64, 128] as const);
// a.matmul(b);
// Error: Inner dimensions must match (128 vs 64)

// Invalid reshape
// const t = torch.randn([2, 3, 4] as const); // 24 elements
// t.reshape([5, 5] as const); // 25 elements
// Error: Cannot reshape 24 elements to [5, 5]

console.log("\nDone!");
