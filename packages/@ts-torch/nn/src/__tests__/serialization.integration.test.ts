/**
 * Integration tests for model serialization:
 * - stateDict() roundtrip
 * - safetensors roundtrip
 * - loadStateDict
 * - _config field survival
 * - model.save / nn.load directory-based save/load
 * - config.load from directory
 *
 * Uses REAL FFI-backed tensors — no mocks.
 *
 * NOTE: Models are created outside run() scopes for async tests so that
 * parameter tensors are not freed prematurely by scope cleanup.
 * For sync-only tests, everything is wrapped in run() for proper cleanup.
 */

import { describe, test, expect, afterEach, vi } from 'vitest'
import { device, run } from '@ts-torch/core'
import { nn } from '../builders.js'
import { Linear } from '../modules/linear.js'
import { ReLU } from '../modules/activation.js'
import { Sequential } from '../modules/container.js'
import { Module, Parameter } from '../module.js'
import { encodeSafetensors, decodeSafetensors } from '../safetensors.js'
import { mkdtemp, rm, readFile, access } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

const cpu = device.cpu()

// Temp dir management
let tempDirs: string[] = []

async function createTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'ts-torch-test-'))
  tempDirs.push(dir)
  return dir
}

afterEach(async () => {
  for (const dir of tempDirs) {
    await rm(dir, { recursive: true, force: true }).catch(() => {})
  }
  tempDirs = []
})

describe('stateDict roundtrip', () => {
  test('stateDict() extracts real tensor data', () => {
    run(() => {
      const model = new Sequential(new Linear(4, 3))
      const state = model.stateDict()

      const keys = Object.keys(state)
      expect(keys.length).toBeGreaterThanOrEqual(1)

      // Weight should have shape [3, 4] for Linear(4, 3)
      const weightKey = keys.find(k => k.includes('weight'))!
      expect(weightKey).toBeDefined()
      expect(state[weightKey]!.shape).toEqual([3, 4])
      expect(state[weightKey]!.dtype).toBe('float32')
      expect(state[weightKey]!.data).toBeInstanceOf(Float32Array)
      expect((state[weightKey]!.data as Float32Array).length).toBe(12)
    })
  })

  test('stateDict → encodeSafetensors → decodeSafetensors roundtrips', () => {
    run(() => {
      const model = new Sequential(
        new Linear(6, 3),
        new ReLU(),
        new Linear(3, 2),
      )

      const originalState = model.stateDict()

      // Encode as safetensors and decode
      const encoded = encodeSafetensors(originalState)
      const { tensors: decoded } = decodeSafetensors(encoded)

      const originalKeys = Object.keys(originalState).sort()
      const decodedKeys = Object.keys(decoded).sort()
      expect(decodedKeys).toEqual(originalKeys)

      for (const key of originalKeys) {
        const orig = originalState[key]!
        const dec = decoded[key]!

        expect(dec.shape).toEqual(orig.shape)
        expect(dec.dtype).toBe(orig.dtype)

        const origData = Array.from(orig.data as Float32Array)
        const decData = Array.from(dec.data as Float32Array)
        expect(decData).toEqual(origData)
      }
    })
  })

  test('loadStateDict restores weights correctly', () => {
    // No run() — loadStateDict frees/creates tensors internally;
    // wrapping in run() causes double-free on scope exit
    const model1 = new Sequential(new Linear(4, 3))
    const state1 = model1.stateDict()

    // Create a second model with same architecture (different random weights)
    const model2 = new Sequential(new Linear(4, 3))

    // Load state from model1 into model2
    model2.loadStateDict(state1)

    const state2 = model2.stateDict()

    // After loading, states should match
    for (const key of Object.keys(state1)) {
      const origData = Array.from(state1[key]!.data as Float32Array)
      const loadedData = Array.from(state2[key]!.data as Float32Array)
      expect(loadedData).toEqual(origData)
    }
  })
})

describe('stateDict() error handling', () => {
  test('throws when parameter tensor has no toArray() method', () => {
    const model = new Module()
    const fakeTensor = { shape: [2, 3], dtype: { name: 'float32' } } as any
    ;(model as any)._parameters.set('bad', new Parameter(fakeTensor, false))

    expect(() => model.stateDict()).toThrow('Cannot serialize parameter "bad": tensor has no toArray() method')
  })

  test('throws when parameter tensor has no dtype.name', () => {
    const model = new Module()
    const fakeTensor = {
      shape: [2, 3],
      dtype: {},
      toArray: () => new Float32Array(6),
    } as any
    ;(model as any)._parameters.set('bad', new Parameter(fakeTensor, false))

    expect(() => model.stateDict()).toThrow('Cannot serialize parameter "bad": tensor has no dtype.name')
  })
})

describe('model.save() and model.loadWeights()', () => {
  test('saves and loads model via directory format', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'my-model')

    const config = nn.sequence(
      nn.input(6),
      nn.fc(3).relu(),
      nn.fc(2),
    )

    // Create models outside run() for async operations
    const model = config.init(cpu)
    const originalState = model.stateDict()

    await model.save(modelDir, { epoch: 5, loss: 0.01 })

    // Create new model with same architecture and load weights
    const model2 = config.init(cpu)
    const metadata = await model2.loadWeights(modelDir)

    expect(metadata).toEqual({ epoch: 5, loss: 0.01 })

    const loadedState = model2.stateDict()
    for (const key of Object.keys(originalState)) {
      const origData = Array.from(originalState[key]!.data as Float32Array)
      const loadedData = Array.from(loadedState[key]!.data as Float32Array)
      expect(loadedData).toEqual(origData)
    }
  })
})

describe('config.load() from directory', () => {
  test('saves and loads model via config.load()', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'my-model')

    const config = nn.sequence(
      nn.input(8),
      nn.fc(4).relu(),
      nn.fc(2),
    )

    // Create model outside run() so tensors survive across async boundaries
    const model = config.init(cpu)
    const originalState = model.stateDict()

    // Save to directory
    await model.save(modelDir)

    // Load via config.load()
    const { model: loadedModel, metadata } = await config.load(cpu, modelDir)
    const loadedState = loadedModel.stateDict()

    expect(metadata).toEqual({})

    // Compare weights
    for (const key of Object.keys(originalState)) {
      const origData = Array.from(originalState[key]!.data as Float32Array)
      const loadedData = Array.from(loadedState[key]!.data as Float32Array)
      expect(loadedData).toEqual(origData)
    }
  })
})

describe('_config field', () => {
  test('config.init() sets _config on Sequential', () => {
    // No run() needed — only checking metadata, not using tensors after this
    const config = nn.sequence(
      nn.input(10),
      nn.fc(5).relu(),
      nn.fc(3),
    )
    const model = config.init(cpu)

    expect(model._config).toBeDefined()
    const cfg = model._config as any
    expect(cfg.format).toBe('ts-torch-sequence')
    expect(cfg.version).toBe(1)
    expect(cfg.input.shape).toEqual([10])
    expect(cfg.blocks).toHaveLength(2)
  })

  test('_config survives loadStateDict', () => {
    // No run() — loadStateDict frees/creates tensors internally;
    // wrapping in run() causes double-free on scope exit
    const config = nn.sequence(
      nn.input(10),
      nn.fc(5).relu(),
      nn.fc(3),
    )

    const model1 = config.init(cpu)
    const state = model1.stateDict()

    const model2 = config.init(cpu)
    model2.loadStateDict(state)

    // _config should still be present after loadStateDict
    expect(model2._config).toBeDefined()
    const cfg = model2._config as any
    expect(cfg.format).toBe('ts-torch-sequence')
    expect(cfg.blocks).toHaveLength(2)
  })
})

describe('directory save/load', () => {
  test('model.save writes config.json and model.safetensors', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'my-model')

    const config = nn.sequence(
      nn.input(8),
      nn.fc(4).relu(),
      nn.fc(2),
    )

    const model = config.init(cpu)
    await model.save(modelDir)

    // Check files exist and are valid
    const configJson = JSON.parse(await readFile(join(modelDir, 'config.json'), 'utf-8'))
    expect(configJson.format).toBe('ts-torch-sequence')
    expect(configJson.blocks).toHaveLength(2)

    // model.safetensors should exist and be non-empty
    const sfBuffer = await readFile(join(modelDir, 'model.safetensors'))
    expect(sfBuffer.byteLength).toBeGreaterThan(8)
  })

  test('nn.load() reads from directory', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'my-model')

    const config = nn.sequence(
      nn.input(8),
      nn.fc(4).relu(),
      nn.fc(2),
    )

    const model = config.init(cpu)
    const originalState = model.stateDict()
    await model.save(modelDir)

    // Load via nn.load
    const { model: loadedModel } = await nn.load(cpu, modelDir)
    const loadedState = loadedModel.stateDict()

    for (const key of Object.keys(originalState)) {
      const origData = Array.from(originalState[key]!.data as Float32Array)
      const loadedData = Array.from(loadedState[key]!.data as Float32Array)
      expect(loadedData).toEqual(origData)
    }
  })

  test('model.save throws if model has no _config', async () => {
    const dir = await createTempDir()

    // Create model directly (not via config.init), so _config won't be set
    const model = new Sequential(new Linear(4, 3))
    await expect(model.save(join(dir, 'no-config'))).rejects.toThrow()
  })

  test('model.save with metadata roundtrips through nn.load', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'my-model')

    const config = nn.sequence(
      nn.input(8),
      nn.fc(4).relu(),
      nn.fc(2),
    )

    const model = config.init(cpu)
    await model.save(modelDir, { epoch: 10, loss: 0.05, note: 'best model' })

    const { metadata } = await nn.load(cpu, modelDir)
    expect(metadata.epoch).toBe(10)
    expect(metadata.loss).toBe(0.05)
    expect(metadata.note).toBe('best model')
  })
})

describe('directory roundtrip', () => {
  test('save → config.load → save → nn.load roundtrip', async () => {
    const dir = await createTempDir()
    const dir1 = join(dir, 'step1')
    const dir2 = join(dir, 'step2')

    const config = nn.sequence(
      nn.input(6),
      nn.fc(4).relu(),
      nn.fc(2),
    )

    // 1. Init and save to directory
    const model1 = config.init(cpu)
    const originalState = model1.stateDict()
    await model1.save(dir1)

    // 2. Load from directory via config.load
    const { model: model2 } = await config.load(cpu, dir1)

    // 3. Save again
    await model2.save(dir2)

    // 4. Load from directory via nn.load
    const { model: model3 } = await nn.load(cpu, dir2)
    const finalState = model3.stateDict()

    // Verify weights match through the full roundtrip
    for (const key of Object.keys(originalState)) {
      const origData = Array.from(originalState[key]!.data as Float32Array)
      const finalData = Array.from(finalState[key]!.data as Float32Array)
      expect(finalData).toEqual(origData)
    }
  })
})

describe('atomic directory writes', () => {
  test('failed save does not leave a partial target directory', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'atomic-test')

    const config = nn.sequence(
      nn.input(4),
      nn.fc(3),
    )

    const model = config.init(cpu)

    // Sabotage stateDict to throw mid-save (after config.json is written to tmp)
    const origStateDict = model.stateDict.bind(model)
    vi.spyOn(model, 'stateDict').mockImplementation(() => {
      throw new Error('sabotaged')
    })

    await expect(model.save(modelDir)).rejects.toThrow('sabotaged')

    // Target directory should not exist
    await expect(access(modelDir)).rejects.toThrow()

    vi.restoreAllMocks()
  })

  test('saving twice to same directory overwrites cleanly', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'overwrite-test')

    const config = nn.sequence(
      nn.input(4),
      nn.fc(3),
    )

    const model1 = config.init(cpu)
    await model1.save(modelDir, { epoch: 1 })

    const model2 = config.init(cpu)
    await model2.save(modelDir, { epoch: 2 })

    // Load and verify the second save's metadata
    const { metadata } = await nn.load(cpu, modelDir)
    expect(metadata.epoch).toBe(2)
  })
})
