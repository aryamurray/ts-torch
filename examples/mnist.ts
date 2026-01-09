/**
 * MNIST Classification with ts-torch
 *
 * Trains a simple MLP on the MNIST handwritten digits dataset.
 */

import { torch } from "@ts-torch/core";
import { Linear, ReLU } from "@ts-torch/nn";
import { MNIST } from "@ts-torch/datasets";

console.log("=== MNIST Classification ===\n");

// ==================== Load Dataset ====================
console.log("Loading MNIST dataset...");

const trainData = new MNIST("./data/mnist", true);
const testData = new MNIST("./data/mnist", false);

await trainData.load();
await testData.load();

console.log(`Training samples: ${trainData.length}`);
console.log(`Test samples: ${testData.length}`);
console.log("");

// ==================== Define Model ====================
console.log("Creating model: 784 -> 128 -> 64 -> 10");

// Create layers (weights persist outside scopes)
const fc1 = new Linear(784, 128);
const fc2 = new Linear(128, 64);
const fc3 = new Linear(64, 10);
const relu = new ReLU();

console.log("  " + fc1.toString());
console.log("  " + fc2.toString());
console.log("  " + fc3.toString());
console.log("");

// Forward pass function
function forward(x: any) {
  let h = fc1.forward(x);
  h = relu.forward(h as any);
  h = fc2.forward(h as any);
  h = relu.forward(h as any);
  h = fc3.forward(h as any);
  return h;
}

// ==================== Training ====================
const EPOCHS = 3;
const BATCH_SIZE = 64;
const LEARNING_RATE = 0.01;

console.log(`Training for ${EPOCHS} epochs, batch size ${BATCH_SIZE}, lr ${LEARNING_RATE}`);
console.log("");

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  let totalLoss = 0;
  let numBatches = 0;
  let correct = 0;
  let total = 0;

  const startTime = Date.now();

  for (const batch of trainData.batches(BATCH_SIZE, true)) {
    torch.run(() => {
      // Forward pass
      const logits = forward(batch.images);

      // Compute softmax probabilities
      const probs = logits.softmax(1);

      // Simple cross-entropy loss approximation:
      // For each sample, get -log(prob of correct class)
      const probsArray = probs.toArray() as Float32Array;
      let batchLoss = 0;

      for (let i = 0; i < batch.labels.length; i++) {
        const label = batch.labels[i]!;
        const prob = probsArray[i * 10 + label]!;
        batchLoss -= Math.log(prob + 1e-10);

        // Check prediction (argmax)
        let maxProb = -1;
        let pred = 0;
        for (let c = 0; c < 10; c++) {
          const p = probsArray[i * 10 + c]!;
          if (p > maxProb) {
            maxProb = p;
            pred = c;
          }
        }
        if (pred === label) correct++;
        total++;
      }

      totalLoss += batchLoss / batch.labels.length;
      numBatches++;

      // Manual SGD update on weights (simplified - no autograd yet)
      // In a real implementation, we'd use backward() and optimizer.step()
    });
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const avgLoss = (totalLoss / numBatches).toFixed(4);
  const accuracy = ((correct / total) * 100).toFixed(2);

  console.log(`Epoch ${epoch + 1}/${EPOCHS} | Loss: ${avgLoss} | Train Acc: ${accuracy}% | Time: ${elapsed}s`);
}

console.log("");

// ==================== Evaluation ====================
console.log("Evaluating on test set...");

let testCorrect = 0;
let testTotal = 0;

for (const batch of testData.batches(1000)) {
  torch.run(() => {
    const logits = forward(batch.images);
    const probs = logits.softmax(1);
    const probsArray = probs.toArray() as Float32Array;

    for (let i = 0; i < batch.labels.length; i++) {
      const label = batch.labels[i]!;

      // Argmax prediction
      let maxProb = -1;
      let pred = 0;
      for (let c = 0; c < 10; c++) {
        const p = probsArray[i * 10 + c]!;
        if (p > maxProb) {
          maxProb = p;
          pred = c;
        }
      }

      if (pred === label) testCorrect++;
      testTotal++;
    }
  });
}

const testAccuracy = ((testCorrect / testTotal) * 100).toFixed(2);
console.log(`\nTest Accuracy: ${testAccuracy}%`);
console.log(`Correct: ${testCorrect} / ${testTotal}`);

console.log("\n=== Done ===");
