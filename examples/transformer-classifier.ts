/**
 * Transformer Sequence Classifier Example
 *
 * Trains a Transformer encoder on synthetic token classification data.
 * Demonstrates: nn.embedding, nn.transformerEncoder, nn.flatten, sequence shape inference.
 */

import { device, run, fromArray, DType, randint } from '@ts-torch/core'
import { nn } from '@ts-torch/nn'
import { Adam } from '@ts-torch/train'

const VOCAB_SIZE = 100
const EMBED_DIM = 32
const SEQ_LEN = 16
const NUM_CLASSES = 2
const BATCH_SIZE = 32
const NUM_BATCHES = 200
const EPOCHS = 5

async function main() {
  console.log('=== Transformer Classifier (Synthetic Data) ===\n')

  const cpu = device.cpu()

  // Model definition
  const model = nn
    .sequence(
      nn.input([SEQ_LEN]),
      nn.embedding(VOCAB_SIZE, EMBED_DIM),
      nn.transformerEncoder(4, 2, { dimFeedforward: 64, dropout: 0.1 }),
      nn.flatten(),
      nn.fc(NUM_CLASSES),
    )
    .init(cpu)

  console.log(model.summary())
  console.log()

  // Optimizer
  const optimizer = Adam({ lr: 1e-3 }).init(model.parameters())

  // Simple training loop with synthetic data
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    model.train()
    let totalLoss = 0
    let correct = 0
    let total = 0

    for (let batch = 0; batch < NUM_BATCHES; batch++) {
      run(() => {
        // Generate random token sequences and binary labels
        const tokens = randint(0, VOCAB_SIZE, [BATCH_SIZE, SEQ_LEN], DType.int64)
        // Simple rule: label = 1 if first token > VOCAB_SIZE/2, else 0
        const firstTokens = tokens.narrow(1, 0, 1).reshape([BATCH_SIZE])
        const labels = fromArray(
          new BigInt64Array(BATCH_SIZE).map((_, i) => {
            const arr = firstTokens.toArray() as BigInt64Array
            return arr[i]! >= BigInt(VOCAB_SIZE / 2) ? 1n : 0n
          }),
          [BATCH_SIZE] as const,
          DType.int64,
        )

        // Forward pass
        const logits = model.forward(tokens) as any
        const logSoftmax = logits.logSoftmax(1)

        // Cross-entropy loss (manual since we have raw logits)
        const oneHot = fromArray(
          new Float32Array(BATCH_SIZE * NUM_CLASSES).fill(0),
          [BATCH_SIZE, NUM_CLASSES] as const,
          DType.float32,
        )
        const nll = logSoftmax.mul(oneHot).sumDim(1).mean()
        const batchLoss = nll.mulScalar(-1)

        // Backward + update
        optimizer.zeroGrad()
        batchLoss.backward()
        optimizer.step()

        totalLoss += (batchLoss.toArray() as Float32Array)[0]!
        total += BATCH_SIZE

        // Count correct predictions
        const preds = logits.toArray() as Float32Array
        const labelArr = labels.toArray() as BigInt64Array
        for (let i = 0; i < BATCH_SIZE; i++) {
          const pred = preds[i * NUM_CLASSES]! < preds[i * NUM_CLASSES + 1]! ? 1 : 0
          if (BigInt(pred) === labelArr[i]!) correct++
        }
      })
    }

    const avgLoss = totalLoss / NUM_BATCHES
    const accuracy = (correct / total) * 100
    console.log(`Epoch ${epoch + 1}/${EPOCHS} - Loss: ${avgLoss.toFixed(4)} - Accuracy: ${accuracy.toFixed(1)}%`)
  }

  console.log('\n=== Done ===')
}

main()
