/**
 * ImageFolder dataset for loading images from a directory structure
 */

import type { Tensor } from '@ts-torch/core'
import { BaseDataset } from '../dataset.js'
import type { Transform } from '../transforms.js'

/**
 * ImageFolder dataset
 *
 * Loads images from a directory where subdirectories represent classes.
 *
 * Expected structure:
 * ```
 * root/
 *   dog/
 *     xxx.png
 *     xxy.png
 *   cat/
 *     123.png
 *     nsdf3.png
 * ```
 */
export class ImageFolder extends BaseDataset<[Tensor, number]> {
  private samples: Array<[string, number]> = []
  private classToIdx: Map<string, number> = new Map()
  private classes: string[] = []

  constructor(
    _root: string,
    _transform?: Transform<Tensor, Tensor>,
    _extensions: string[] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
  ) {
    super()
  }

  async init(): Promise<void> {
    // TODO: Scan directory structure and build samples list
    // const classes = await readdir(this.root);
    // for (const [idx, className] of classes.entries()) {
    //   this.classToIdx.set(className, idx);
    //   this.classes.push(className);
    //
    //   const classDir = join(this.root, className);
    //   const files = await readdir(classDir);
    //
    //   for (const file of files) {
    //     if (this.extensions.some(ext => file.endsWith(ext))) {
    //       this.samples.push([join(classDir, file), idx]);
    //     }
    //   }
    // }
  }

  async getItem(index: number): Promise<[Tensor, number]> {
    if (index < 0 || index >= this.length) {
      throw new Error(`Index ${index} out of bounds`)
    }

    const [_path, _label] = this.samples[index]!

    // TODO: Load image and convert to tensor
    // let image = await loadImage(path);
    // if (this.transform) {
    //   image = await this.transform(image);
    // }
    // return [image, label];

    throw new Error('ImageFolder.getItem not yet implemented')
  }

  get length(): number {
    return this.samples.length
  }

  getClasses(): string[] {
    return this.classes
  }

  getClassToIdx(): Map<string, number> {
    return this.classToIdx
  }
}
