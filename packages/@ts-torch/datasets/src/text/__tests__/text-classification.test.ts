/**
 * Tests for text classification datasets
 */

import { afterEach, describe, it, expect } from 'vitest'
import { mkdir, rm, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { CSVTextDataset } from '../text-classification.js'

let tempRoot: string | null = null

async function createTempDir(): Promise<string> {
  const root = join(tmpdir(), `ts-torch-text-${Date.now()}`)
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

describe('CSVTextDataset', () => {
  it('loads text and labels from csv', async () => {
    const root = await createTempDir()
    const csvPath = join(root, 'data.csv')
    await writeFile(csvPath, 'text,label\n"hello",1\n"world",0\n')

    const dataset = new CSVTextDataset(csvPath)
    await dataset.init()

    expect(dataset.length).toBe(2)
    expect(dataset.getItem(0)).toEqual(['hello', 1])
    expect(dataset.getItem(1)).toEqual(['world', 0])
  })
})
