/**
 * Tests for ImageFolder dataset
 */

import { afterEach, describe, it, expect } from 'vitest'
import { mkdir, rm, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { PNG } from 'pngjs'
import { ImageFolder } from '../image-folder.js'

let tempRoot: string | null = null

async function createTempDir(): Promise<string> {
  const root = join(tmpdir(), `ts-torch-image-folder-${Date.now()}`)
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

function createPngBuffer(): Buffer {
  const png = new PNG({ width: 2, height: 2 })
  png.data[0] = 255
  png.data[1] = 0
  png.data[2] = 0
  png.data[3] = 255
  png.data[4] = 0
  png.data[5] = 255
  png.data[6] = 0
  png.data[7] = 255
  png.data[8] = 0
  png.data[9] = 0
  png.data[10] = 255
  png.data[11] = 255
  png.data[12] = 255
  png.data[13] = 255
  png.data[14] = 255
  png.data[15] = 255
  return PNG.sync.write(png)
}

describe('ImageFolder', () => {
  it('loads images from class directories', async () => {
    const root = await createTempDir()
    const classA = join(root, 'class-a')
    const classB = join(root, 'class-b')
    await mkdir(classA, { recursive: true })
    await mkdir(classB, { recursive: true })

    await writeFile(join(classA, 'sample.png'), createPngBuffer())
    await writeFile(join(classB, 'sample.png'), createPngBuffer())

    const dataset = new ImageFolder(root, undefined, ['.png'])
    await dataset.init()

    expect(dataset.length).toBe(2)
    expect(dataset.getClasses()).toEqual(['class-a', 'class-b'])

    const [image, label] = await dataset.getItem(0)
    expect(image.shape).toEqual([3, 2, 2])
    expect(label).toBe(0)
  })
})
