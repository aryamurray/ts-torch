/**
 * Device namespace - Declarative device context factory
 *
 * @example
 * ```ts
 * import { device } from '@ts-torch/core'
 *
 * const cpu = device.cpu()
 * const cuda = device.cuda(0)
 * const mps = device.mps()
 *
 * // Create tensors directly on device
 * const x = cuda.zeros([784, 128])
 * const y = cuda.randn([128, 10])
 * ```
 */

import { DeviceContext } from './context.js'

export { DeviceContext }

/**
 * Device namespace for creating device contexts
 *
 * Use this instead of the Device class for the declarative API.
 */
export const device = {
  /**
   * Create a CPU device context
   *
   * @example
   * ```ts
   * const cpu = device.cpu()
   * const x = cpu.zeros([2, 3])
   * ```
   */
  cpu: DeviceContext.cpu,

  /**
   * Create a CUDA device context
   *
   * @param index - GPU index (default: 0)
   *
   * @example
   * ```ts
   * const cuda = device.cuda(0)
   * const x = cuda.zeros([784, 128])
   * ```
   */
  cuda: DeviceContext.cuda,

  /**
   * Create an MPS device context (Apple Silicon)
   *
   * @example
   * ```ts
   * const mps = device.mps()
   * const x = mps.zeros([2, 3])
   * ```
   */
  mps: DeviceContext.mps,
}
