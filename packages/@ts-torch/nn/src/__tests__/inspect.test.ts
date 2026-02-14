/**
 * Tests for nn.inspect() and inspectSafetensorsHeader()
 *
 * Uses REAL FFI-backed tensors — no mocks.
 */

import { describe, test, expect, afterEach } from 'vitest'
import { device } from '@ts-torch/core'
import { nn } from '../builders.js'
import { inspectSafetensorsHeader } from '../safetensors.js'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

const cpu = device.cpu()

let tempDirs: string[] = []

async function createTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'ts-torch-inspect-'))
  tempDirs.push(dir)
  return dir
}

afterEach(async () => {
  for (const dir of tempDirs) {
    await rm(dir, { recursive: true, force: true }).catch(() => {})
  }
  tempDirs = []
})

describe('inspectSafetensorsHeader()', () => {
  test('reads header without loading tensor data', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'model')

    const config = nn.sequence(
      nn.input(4),
      nn.fc(3).relu(),
      nn.fc(2),
    )

    const model = config.init(cpu)
    await model.save(modelDir)

    const result = await inspectSafetensorsHeader(join(modelDir, 'model.safetensors'))

    // Check parameters are listed
    const paramNames = Object.keys(result.parameters).sort()
    expect(paramNames.length).toBeGreaterThanOrEqual(4) // 2 layers × (weight + bias)

    // Check a weight entry has shape and dtype
    const weightKey = paramNames.find(k => k.includes('weight'))!
    expect(weightKey).toBeDefined()
    expect(result.parameters[weightKey]!.dtype).toBe('float32')
    expect(Array.isArray(result.parameters[weightKey]!.shape)).toBe(true)

    // Metadata should have framework key
    expect(result.metadata.framework).toBe('ts-torch')
  })
})

describe('nn.inspect()', () => {
  test('inspects saved model directory', async () => {
    const dir = await createTempDir()
    const modelDir = join(dir, 'model')

    const config = nn.sequence(
      nn.input(8),
      nn.fc(4).relu(),
      nn.fc(2),
    )

    const model = config.init(cpu)
    await model.save(modelDir, { epoch: 3, lr: 0.001 })

    const result = await nn.inspect(modelDir)

    // Config should match
    expect((result.config as any).format).toBe('ts-torch-sequence')
    expect((result.config as any).blocks).toHaveLength(2)

    // Parameters should be listed
    const paramNames = Object.keys(result.parameters)
    expect(paramNames.length).toBeGreaterThanOrEqual(4)

    // Metadata should contain user metadata (not framework)
    expect(result.metadata.epoch).toBe(3)
    expect(result.metadata.lr).toBe(0.001)

    // File size should be reported
    expect(result.fileSizeBytes).toBeGreaterThan(0)
    expect(typeof result.fileSize).toBe('string')
    expect(result.fileSize).toMatch(/\d/)
  })
})
