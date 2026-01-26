/**
 * Tests for CIFAR datasets
 */

import { afterEach, describe, it, expect } from 'vitest'
import { mkdir, rm, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { CIFAR10, CIFAR100 } from '../cifar.js'

let tempRoot: string | null = null

async function createTempDir(): Promise<string> {
  const root = join(tmpdir(), `ts-torch-cifar-${Date.now()}`)
  await mkdir(root, { recursive: true })
  tempRoot = root
  return root
}

afterEach(async () => {
  if (tempRoot) {
    await rm(tempRoot, { recursive: true, force: true })
    tempRoot = null
  }
})

function createCifarRecord(label: number, extraLabel?: number): Buffer {
  const imageSize = 32 * 32 * 3
  const header = extraLabel === undefined ? 1 : 2
  const buffer = Buffer.alloc(header + imageSize)
  buffer[0] = label
  if (extraLabel !== undefined) {
    buffer[1] = extraLabel
  }
  for (let i = 0; i < imageSize; i++) {
    buffer[header + i] = i % 256
  }
  return buffer
}

describe('CIFAR10', () => {
  it('loads binary batches from disk', async () => {
    const root = await createTempDir()
    const cifarDir = join(root, 'cifar-10-batches-bin')
    await mkdir(cifarDir, { recursive: true })

    const files = [
      'data_batch_1.bin',
      'data_batch_2.bin',
      'data_batch_3.bin',
      'data_batch_4.bin',
      'data_batch_5.bin',
    ]
    for (const [idx, file] of files.entries()) {
      await writeFile(join(cifarDir, file), createCifarRecord(idx))
    }

    const dataset = new CIFAR10(root, true)
    await dataset.init()

    expect(dataset.length).toBe(5)
    const [image, label] = dataset.getItem(2)
    expect(image.shape).toEqual([3, 32, 32])
    expect(label).toBe(2)
  })
})

describe('CIFAR100', () => {
  it('loads binary train file from disk', async () => {
    const root = await createTempDir()
    const cifarDir = join(root, 'cifar-100-binary')
    await mkdir(cifarDir, { recursive: true })

    await writeFile(join(cifarDir, 'train.bin'), createCifarRecord(1, 7))

    const dataset = new CIFAR100(root, true, undefined, false, true)
    await dataset.init()

    const [image, label] = dataset.getItem(0)
    expect(image.shape).toEqual([3, 32, 32])
    expect(label).toBe(7)
  })
})
