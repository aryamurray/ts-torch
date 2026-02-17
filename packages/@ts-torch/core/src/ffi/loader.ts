/**
 * Napi native library loader for ts-torch
 * Loads pre-compiled Node.js addon (.node module)
 *
 * Architecture:
 * - All symbols bound at compile time (no dynamic FFI)
 * - No per-operation FFI overhead
 * - Memory managed by scope system (run() / escape()), not GC finalizers
 *
 * The .node module is built by cmake-js and contains all C functions
 * directly callable from JavaScript with minimal overhead.
 *
 * IMPORTANT - Napi TypedArray vs ArrayBuffer convention:
 * - Napi expects TypedArray objects directly (Float32Array, BigInt64Array, etc.)
 * - The underlying ArrayBuffer and ByteOffset are extracted automatically by Napi
 * - This convention applies to all buffer parameters (shape, data, weights, etc.)
 */

import { createRequire } from 'module'
import { existsSync, readFileSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { Logger } from '../logger.js'
import { applyConfig } from '../config.js'

/**
 * TypedArray type for shape buffers
 * Accept any typed array or buffer-like type
 */
type ShapeBuffer = any

/**
 * Type definition for the loaded Napi module
 * Maps to C function names exported from napi_bindings.cpp
 *
 * Napi convention:
 * - Shape buffers are TypedArrays (ndim extracted automatically via ElementLength)
 * - Errors are thrown as JS exceptions (no errBuffer parameter)
 * - Tensor handles are opaque Napi::External<void> objects
 */
export type NativeModule = {
  // Utility functions
  ts_version: () => string
  ts_cuda_is_available: () => number
  ts_cuda_device_count: () => number
  ts_set_num_threads: (numThreads: number) => void
  ts_get_num_threads: () => number
  ts_manual_seed: (seed: number) => void

  // Tensor property queries
  ts_tensor_ndim: (tensor: unknown) => number
  ts_tensor_size: (tensor: unknown, dim: number) => number

  // Tensor factories - take (shapeBuffer, dtype, device, deviceIndex)
  ts_tensor_zeros: (shapeBuffer: ShapeBuffer, dtype: number, device: number, deviceIndex: number) => unknown
  ts_tensor_ones: (shapeBuffer: ShapeBuffer, dtype: number, device: number, deviceIndex: number) => unknown
  ts_tensor_randn: (shapeBuffer: ShapeBuffer, dtype: number, device: number, deviceIndex: number) => unknown
  ts_tensor_rand: (shapeBuffer: ShapeBuffer, dtype: number, device: number, deviceIndex: number) => unknown
  ts_tensor_empty: (shapeBuffer: ShapeBuffer, dtype: number, device: number, deviceIndex: number) => unknown
  ts_tensor_from_buffer: (
    dataBuffer: Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array,
    shapeBuffer: ShapeBuffer,
    dtype: number,
    device: number,
    deviceIndex: number,
  ) => unknown

  // Tensor properties and manipulation
  ts_tensor_shape: (tensor: unknown, outShape: Int32Array) => number
  ts_tensor_requires_grad: (tensor: unknown) => boolean
  ts_tensor_set_requires_grad: (tensor: unknown, value: boolean) => void
  ts_tensor_dtype: (tensor: unknown) => number
  ts_tensor_clone: (tensor: unknown) => unknown
  ts_tensor_detach: (tensor: unknown) => unknown
  ts_tensor_zero_grad: (tensor: unknown) => void
  ts_tensor_delete: (tensor: unknown) => void

  // Binary operations (Napi throws on error, no errBuffer)
  ts_tensor_add: (a: unknown, b: unknown) => unknown
  ts_tensor_sub: (a: unknown, b: unknown) => unknown
  ts_tensor_mul: (a: unknown, b: unknown) => unknown
  ts_tensor_div: (a: unknown, b: unknown) => unknown
  ts_tensor_matmul: (a: unknown, b: unknown) => unknown
  ts_tensor_minimum: (a: unknown, b: unknown) => unknown
  ts_tensor_maximum: (a: unknown, b: unknown) => unknown

  // Unary operations
  ts_tensor_relu: (tensor: unknown) => unknown
  ts_tensor_sigmoid: (tensor: unknown) => unknown
  ts_tensor_tanh: (tensor: unknown) => unknown
  ts_tensor_exp: (tensor: unknown) => unknown
  ts_tensor_log: (tensor: unknown) => unknown
  ts_tensor_sqrt: (tensor: unknown) => unknown
  ts_tensor_neg: (tensor: unknown) => unknown
  ts_tensor_softmax: (tensor: unknown, dim: number) => unknown
  ts_tensor_log_softmax: (tensor: unknown, dim: number) => unknown
  ts_tensor_transpose: (tensor: unknown, dim0: number, dim1: number) => unknown
  ts_tensor_reshape: (tensor: unknown, shapeBuffer: ShapeBuffer) => unknown
  ts_tensor_clamp: (tensor: unknown, min: number, max: number) => unknown

  // Reduction operations
  ts_tensor_sum: (tensor: unknown) => unknown
  ts_tensor_mean: (tensor: unknown) => unknown
  ts_tensor_sum_dim: (tensor: unknown, dim: number, keepdim: boolean) => unknown
  ts_tensor_mean_dim: (tensor: unknown, dim: number, keepdim: boolean) => unknown

  // Scalar operations
  ts_tensor_add_scalar: (tensor: unknown, scalar: number) => unknown
  ts_tensor_sub_scalar: (tensor: unknown, scalar: number) => unknown
  ts_tensor_mul_scalar: (tensor: unknown, scalar: number) => unknown
  ts_tensor_div_scalar: (tensor: unknown, scalar: number) => unknown

  // Out variants (write result to existing tensor)
  ts_tensor_add_out: (a: unknown, b: unknown, out: unknown) => void
  ts_tensor_sub_out: (a: unknown, b: unknown, out: unknown) => void
  ts_tensor_mul_out: (a: unknown, b: unknown, out: unknown) => void
  ts_tensor_div_out: (a: unknown, b: unknown, out: unknown) => void
  ts_tensor_matmul_out: (a: unknown, b: unknown, out: unknown) => void

  // In-place variants (modify tensor in place)
  ts_tensor_add_: (a: unknown, b: unknown) => void
  ts_tensor_sub_inplace: (a: unknown, b: unknown) => void
  ts_tensor_mul_: (a: unknown, b: unknown) => void
  ts_tensor_div_: (a: unknown, b: unknown) => void
  ts_tensor_mul_scalar_: (tensor: unknown, scalar: number) => void
  ts_tensor_div_scalar_: (tensor: unknown, scalar: number) => void
  ts_tensor_add_scaled_inplace: (tensor: unknown, other: unknown, scalar: number) => void
  ts_tensor_optim_add_: (tensor: unknown, other: unknown, alpha: number) => void
  ts_tensor_zero_grad_: (tensor: unknown) => void

  // Buffer operations
  ts_tensor_copy_to_buffer: (tensor: unknown, buffer: ArrayBufferLike) => void

  // Autograd operations
  ts_tensor_backward: (tensor: unknown) => void

  // Scope operations (memory management)
  ts_scope_begin: () => unknown
  ts_scope_end: (scope: unknown) => void
  ts_scope_escape_tensor: (scope: unknown, tensor: unknown) => void

  // Batch operations (graph compilation)
  ts_batch_begin: () => unknown
  ts_batch_end: (batch: unknown) => void
  ts_batch_abort: (batch: unknown) => void
  ts_batch_is_recording: () => number
  ts_tensor_chain_matmul: (tensors: unknown[]) => unknown
  ts_tensor_mlp_forward: (
    input: unknown,
    weights: unknown[],
    biases: unknown[],
    applyReluExceptLast: boolean,
  ) => unknown

  // Additional tensor operations
  ts_tensor_cat: (tensors: unknown[], dim: number) => unknown
  ts_tensor_einsum: (equation: string, tensors: unknown[]) => unknown
  ts_tensor_clamp_min: (tensor: unknown, min: number) => unknown
  ts_tensor_clamp_max: (tensor: unknown, max: number) => unknown
  ts_tensor_grad: (tensor: unknown) => unknown
  ts_tensor_to_device: (tensor: unknown, device: number, deviceIndex: number) => unknown

  // Reshaping operations
  ts_tensor_flatten: (tensor: unknown, startDim: number, endDim: number) => unknown
  ts_tensor_view: (tensor: unknown, shapeBuffer: ShapeBuffer) => unknown
  ts_tensor_unsqueeze: (tensor: unknown, dim: number) => unknown
  ts_tensor_squeeze: (tensor: unknown, dim: number) => unknown
  ts_tensor_flatten_from_index: (tensor: unknown, startIdx: number) => unknown

  // Padding and advanced reshaping
  ts_tensor_pad: (tensor: unknown, padSizes: ShapeBuffer) => unknown

  // Reduction with dimension
  ts_tensor_max_dim: (tensor: unknown, dim: number, keepdim: boolean) => unknown
  ts_tensor_min_dim: (tensor: unknown, dim: number, keepdim: boolean) => unknown
  ts_tensor_argmin: (tensor: unknown, dim: number, keepdim: boolean) => unknown

  // Indexing and selection
  ts_tensor_index_select: (tensor: unknown, dim: number, indices: unknown) => unknown
  ts_tensor_argmax: (tensor: unknown, dim: number, keepdim: boolean) => unknown
  ts_tensor_narrow: (tensor: unknown, dim: number, start: number, length: number) => unknown
  ts_tensor_topk: (tensor: unknown, k: number, dim: number, largest: boolean, sorted: boolean) => unknown
  ts_tensor_sort: (tensor: unknown, dim: number, descending: boolean) => unknown

  // Triangle operations
  ts_tensor_triu: (tensor: unknown, diagonal: number) => unknown
  ts_tensor_tril: (tensor: unknown, diagonal: number) => unknown

  // Masking and advanced operations
  ts_tensor_masked_fill: (tensor: unknown, mask: unknown, value: number) => unknown
  ts_tensor_bmm: (a: unknown, b: unknown) => unknown
  ts_tensor_gather: (tensor: unknown, dim: number, indices: unknown) => unknown
  ts_tensor_scatter: (tensor: unknown, dim: number, indices: unknown, src: unknown) => unknown
  ts_tensor_where: (condition: unknown, x: unknown, y: unknown) => unknown
  ts_tensor_nonzero: (tensor: unknown) => unknown

  // Repetition and expansion (take buffer and length)
  ts_tensor_repeat: (tensor: unknown, repeats: ShapeBuffer, numRepeats: number) => unknown
  ts_tensor_expand: (tensor: unknown, sizes: ShapeBuffer, numSizes: number) => unknown

  // Loss functions (take just input and target, return scalar)
  ts_tensor_cross_entropy_loss: (input: unknown, target: unknown) => unknown
  ts_tensor_nll_loss: (input: unknown, target: unknown) => unknown
  ts_tensor_mse_loss: (input: unknown, target: unknown) => unknown

  // Neural network operations
  ts_tensor_conv2d: (
    input: unknown,
    weight: unknown,
    bias: unknown | null,
    strideH: number,
    strideW: number,
    paddingH: number,
    paddingW: number,
    dilationH: number,
    dilationW: number,
    groups: number,
  ) => unknown
  ts_tensor_max_pool2d: (
    input: unknown,
    kernelH: number,
    kernelW: number,
    strideH: number,
    strideW: number,
    paddingH: number,
    paddingW: number,
  ) => unknown
  ts_tensor_avg_pool2d: (
    input: unknown,
    kernelH: number,
    kernelW: number,
    strideH: number,
    strideW: number,
    paddingH: number,
    paddingW: number,
  ) => unknown
  ts_tensor_dropout: (tensor: unknown, p: number, training: number) => unknown
  ts_tensor_batch_norm: (
    input: unknown,
    weight: unknown | null,
    bias: unknown | null,
    runningMean: unknown | null,
    runningVar: unknown | null,
    training: number,
    momentum: number,
    eps: number,
  ) => unknown
  ts_tensor_layer_norm: (
    input: unknown,
    normalizedShape: ShapeBuffer,
    weight: unknown | null,
    bias: unknown | null,
    eps: number,
  ) => unknown

  // Comparison operations
  ts_tensor_eq: (a: unknown, b: unknown) => unknown

  // Fused linear operations
  ts_tensor_linear_relu: (input: unknown, weight: unknown, bias: unknown | null) => unknown
  ts_tensor_linear_sigmoid: (input: unknown, weight: unknown, bias: unknown | null) => unknown
  ts_tensor_linear_tanh: (input: unknown, weight: unknown, bias: unknown | null) => unknown

  // Fused add operations
  ts_tensor_add_relu: (a: unknown, b: unknown) => unknown

  // Policy fused operations
  ts_policy_forward: (
    observations: Float32Array,
    actions: Float32Array,
    batchSize: number,
    obsSize: number,
    nActions: number,
    piParams: unknown[],
    vfParams: unknown[],
    activationType: number,
  ) => { actionLogProbs: unknown; entropy: unknown; values: unknown }
  ts_backward_and_clip: (loss: unknown, parameters: unknown[], maxGradNorm: number) => number

  // RL fused operations
  ts_compute_gae: (
    rewards: Float32Array,
    values: Float32Array,
    episodeStarts: Uint8Array,
    lastValues: Float32Array,
    lastDones: Uint8Array,
    bufferSize: number,
    nEnvs: number,
    gamma: number,
    gaeLambda: number,
    advantagesOut: Float32Array,
    returnsOut: Float32Array,
  ) => void
  ts_clip_grad_norm_: (parameters: unknown[], maxNorm: number) => number
  ts_normalize_inplace: (data: Float32Array) => void

  // Other operations not explicitly typed - use flexible signature
  [key: string]: (...args: any[]) => any
}

/**
 * Build metadata for CUDA packages
 */
interface BuildMeta {
  cuda: string
  libtorch: string
  platform: string
  builtAt: string
}

/**
 * Cached module instance
 */
let libInstance: NativeModule | null = null

/**
 * Platform-specific library information
 */
interface PlatformInfo {
  packageName: string
  libraryName: string
}

/**
 * Get platform-specific package name and library name
 */
export function getPlatformPackage(): PlatformInfo {
  const platform = process.platform
  const arch = process.arch

  switch (platform) {
    case 'darwin':
      switch (arch) {
        case 'arm64':
          return {
            packageName: '@ts-torch-platform/darwin-arm64',
            libraryName: 'libts_torch',
          }
        case 'x64':
          return {
            packageName: '@ts-torch-platform/darwin-x64',
            libraryName: 'libts_torch',
          }
        default:
          throw new Error(`Unsupported macOS architecture: ${arch}`)
      }

    case 'linux':
      switch (arch) {
        case 'x64':
          return {
            packageName: '@ts-torch-platform/linux-x64',
            libraryName: 'libts_torch',
          }
        default:
          throw new Error(`Unsupported Linux architecture: ${arch}`)
      }

    case 'win32':
      switch (arch) {
        case 'x64':
          return {
            packageName: '@ts-torch-platform/win32-x64',
            libraryName: 'ts_torch',
          }
        default:
          throw new Error(`Unsupported Windows architecture: ${arch}`)
      }

    default:
      throw new Error(`Unsupported platform: ${platform}`)
  }
}

/**
 * CUDA versions to check, in order of preference (latest first)
 */
const CUDA_VERSIONS = ['cu124', 'cu121', 'cu118']

/**
 * Cached CUDA library path (if found)
 */
let cudaLibPath: string | null = null

/**
 * Cached workspace root path
 */
let workspaceRoot: string | null = null
let workspaceRootSearched = false

/**
 * Find the workspace root directory using multiple detection strategies
 * Caches the result for performance
 */
function findWorkspaceRoot(): string | null {
  if (workspaceRootSearched) {
    return workspaceRoot
  }
  workspaceRootSearched = true

  const loaderDir = import.meta.dirname
  if (!loaderDir) {
    Logger.debug('Cannot determine loader directory, workspace detection skipped')
    return null
  }

  const root = process.platform === 'win32' ? loaderDir.split(':')[0] + ':\\' : '/'

  let dir = loaderDir
  while (dir !== root) {
    const pkgPath = join(dir, 'package.json')

    if (existsSync(pkgPath)) {
      try {
        const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'))
        if (pkg.workspaces) {
          const hasWorkspaces = Array.isArray(pkg.workspaces) || typeof pkg.workspaces === 'object'
          if (hasWorkspaces) {
            workspaceRoot = dir
            Logger.debug(`Found workspace root via workspaces field: ${dir}`)
            return workspaceRoot
          }
        }
      } catch {
        // Invalid JSON, continue searching
      }
    }

    if (existsSync(join(dir, 'pnpm-workspace.yaml'))) {
      workspaceRoot = dir
      Logger.debug(`Found workspace root via pnpm-workspace.yaml: ${dir}`)
      return workspaceRoot
    }

    if (existsSync(join(dir, 'lerna.json'))) {
      workspaceRoot = dir
      Logger.debug(`Found workspace root via lerna.json: ${dir}`)
      return workspaceRoot
    }

    dir = dirname(dir)
  }

  // Second pass: look for common monorepo directory patterns
  dir = loaderDir
  while (dir !== root) {
    const pkgPath = join(dir, 'package.json')
    if (existsSync(pkgPath)) {
      const hasPackagesDir = existsSync(join(dir, 'packages'))
      const hasAppsDir = existsSync(join(dir, 'apps'))

      if (hasPackagesDir || hasAppsDir) {
        workspaceRoot = dir
        Logger.debug(`Found workspace root via directory pattern (${hasPackagesDir ? 'packages/' : 'apps/'}): ${dir}`)
        return workspaceRoot
      }
    }
    dir = dirname(dir)
  }

  // Third pass: use git root as fallback
  dir = loaderDir
  while (dir !== root) {
    if (existsSync(join(dir, '.git')) && existsSync(join(dir, 'package.json'))) {
      workspaceRoot = dir
      Logger.debug(`Found workspace root via .git directory: ${dir}`)
      return workspaceRoot
    }
    dir = dirname(dir)
  }

  Logger.debug('No workspace root found')
  return null
}

/**
 * Get paths relative to workspace root for development
 */
function getDevPaths(platformDir: string): string[] {
  const paths: string[] = []
  const root = findWorkspaceRoot()

  if (root) {
    paths.push(
      join(root, 'packages', '@ts-torch-platform', platformDir, 'lib', 'ts_torch.node'),
      join(root, 'packages', '@ts-torch', 'core', 'native', 'build', 'Release', 'ts_torch.node'),
      join(root, 'packages', '@ts-torch', 'core', 'native', 'build', 'ts_torch.node'),
    )
  }

  const cwd = process.cwd()
  if (cwd !== root) {
    paths.push(
      join(cwd, 'packages', '@ts-torch-platform', platformDir, 'lib', 'ts_torch.node'),
      join(cwd, 'native', 'build', 'Release', 'ts_torch.node'),
      join(cwd, 'native', 'build', 'ts_torch.node'),
    )
  }

  return paths
}

/**
 * Get libtorch paths for development
 */
function getLibtorchDevPaths(subdir: string): string[] {
  const paths: string[] = []
  const root = findWorkspaceRoot()

  if (root) {
    paths.push(join(root, subdir, 'lib'))
  }

  const cwd = process.cwd()
  if (cwd !== root) {
    paths.push(join(cwd, subdir, 'lib'))
  }

  return paths
}

/**
 * Check if a CUDA build is valid by verifying .build-meta.json
 */
function isValidCudaBuild(pkgDir: string, libFileName: string, expectedCuda: string): boolean {
  const libPath = join(pkgDir, 'lib', libFileName)
  const metaPath = join(pkgDir, 'lib', '.build-meta.json')

  if (!existsSync(libPath) || !existsSync(metaPath)) {
    return false
  }

  try {
    const meta: BuildMeta = JSON.parse(readFileSync(metaPath, 'utf-8'))
    if (meta.cuda !== expectedCuda) {
      Logger.debug(`Build meta mismatch: expected ${expectedCuda}, got ${meta.cuda}`)
      return false
    }
    const expectedPlatform = `${process.platform}-${process.arch}`
    if (meta.platform !== expectedPlatform) {
      Logger.debug(`Platform mismatch: expected ${expectedPlatform}, got ${meta.platform}`)
      return false
    }
    return true
  } catch {
    Logger.debug(`Invalid .build-meta.json in ${pkgDir}`)
    return false
  }
}

/**
 * Try to find a valid CUDA library
 */
function findCudaLibrary(): string | null {
  if (process.platform !== 'linux' && process.platform !== 'win32') {
    return null
  }

  const platform = process.platform === 'win32' ? 'win32' : 'linux'
  const libFileName = 'ts_torch.node'

  for (const cudaVer of CUDA_VERSIONS) {
    const pkg = `@ts-torch-cuda/${platform}-x64-${cudaVer}`
    try {
      const require_ = createRequire(import.meta.url)
      const pkgPath = require_.resolve(`${pkg}/package.json`)
      const pkgDir = dirname(pkgPath)

      if (isValidCudaBuild(pkgDir, libFileName, cudaVer)) {
        Logger.info(`Using CUDA ${cudaVer} module`)
        cudaLibPath = join(pkgDir, 'lib', libFileName)
        return cudaLibPath
      }
    } catch {
      // Package not installed, continue to next version
    }
  }

  const root = findWorkspaceRoot()
  if (root) {
    for (const cudaVer of CUDA_VERSIONS) {
      const devPath = join(root, 'packages', '@ts-torch-cuda', `${platform}-x64-${cudaVer}`)
      if (existsSync(devPath) && isValidCudaBuild(devPath, libFileName, cudaVer)) {
        Logger.info(`Using CUDA ${cudaVer} module (dev)`)
        cudaLibPath = join(devPath, 'lib', libFileName)
        return cudaLibPath
      }
    }
  }

  return null
}

/**
 * Get the full path to the native module (.node file)
 *
 * Resolution order:
 * 1. TS_TORCH_LIB environment variable (for custom builds)
 * 2. CUDA packages (if installed and valid)
 * 3. CPU platform-specific package (production)
 * 4. Local development paths (workspace monorepo)
 *
 * @throws Error if module cannot be found
 */
export function getLibraryPath(): string {
  // 1. Check environment variable override
  const envPath = process.env.TS_TORCH_LIB
  if (envPath) {
    if (existsSync(envPath)) {
      return resolve(envPath)
    }
    Logger.debug(`TS_TORCH_LIB set but file not found: ${envPath}`)
  }

  // 2. Check for CUDA packages (prefer GPU over CPU)
  const cudaPath = findCudaLibrary()
  if (cudaPath) {
    return cudaPath
  }

  const { packageName } = getPlatformPackage()
  const require_ = createRequire(import.meta.url)

  // 3. Try to resolve from installed CPU platform package
  try {
    const packagePath = require_.resolve(`${packageName}/package.json`)
    const packageDir = packagePath.replace(/[/\\]package\.json$/, '')
    const libPath = join(packageDir, 'lib', 'ts_torch.node')

    if (existsSync(libPath)) {
      return libPath
    }
  } catch {
    // Package not found, try development paths
  }

  // 4. Try local development paths (workspace monorepo structure)
  const platformDir = packageName.replace('@ts-torch-platform/', '')
  const possiblePaths = getDevPaths(platformDir)

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return resolve(path)
    }
  }

  // Module not found - provide helpful setup instructions
  const isCudaSupported = process.platform === 'linux' || process.platform === 'win32'
  const cudaInstructions = isCudaSupported ? `\nFor GPU/CUDA support:\n` + `  bun run setup:cuda\n\n` : ''

  throw new Error(
    `Could not find ts-torch native module for ${process.platform}-${process.arch}.\n\n` +
      `Quick Setup:\n` +
      `  bun run setup          # Download LibTorch and build (CPU)\n` +
      cudaInstructions +
      `Manual Installation:\n` +
      `  bun add ${packageName}\n\n` +
      `Or set TS_TORCH_LIB environment variable to point to your .node file.\n\n` +
      `For debugging, set TS_TORCH_DEBUG=1 to see detailed resolution info.\n\n` +
      `Searched paths:\n${possiblePaths.map((p) => `  - ${p}`).join('\n')}`,
  )
}

/**
 * Find the libtorch library directory
 * Used to add DLL dependencies to the search path on Windows
 */
function findLibtorchPath(): string | null {
  const envPaths = [
    process.env.LIBTORCH ? join(process.env.LIBTORCH, 'lib') : '',
    process.env.LIBTORCH_PATH ? join(process.env.LIBTORCH_PATH, 'lib') : '',
  ].filter(Boolean)

  for (const p of envPaths) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  for (const p of getLibtorchDevPaths('libtorch')) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  return null
}

/**
 * Find libtorch path from CUDA package (if using CUDA)
 */
function findCudaLibtorchPath(): string | null {
  const envPaths = [
    process.env.LIBTORCH_CUDA ? join(process.env.LIBTORCH_CUDA, 'lib') : '',
    process.env.LIBTORCH_CUDA_PATH ? join(process.env.LIBTORCH_CUDA_PATH, 'lib') : '',
  ].filter(Boolean)

  for (const p of envPaths) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  if (cudaLibPath) {
    const libDir = dirname(cudaLibPath)
    const libtorchLib = join(libDir, 'libtorch', 'lib')

    if (existsSync(libtorchLib)) {
      return libtorchLib
    }
  }

  for (const p of getLibtorchDevPaths('libtorch-cuda')) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  return null
}

/**
 * Setup DLL search path for Windows
 */
function setupDllSearchPath(): void {
  if (process.platform !== 'win32') {
    return
  }

  const cudaLibtorchLib = findCudaLibtorchPath()
  if (cudaLibtorchLib) {
    const currentPath = process.env.PATH || ''
    if (!currentPath.includes(cudaLibtorchLib)) {
      process.env.PATH = `${cudaLibtorchLib};${currentPath}`
    }
    return
  }

  const libtorchLib = findLibtorchPath()
  if (libtorchLib) {
    const currentPath = process.env.PATH || ''
    if (!currentPath.includes(libtorchLib)) {
      process.env.PATH = `${libtorchLib};${currentPath}`
    }
  }
}

/**
 * Load the native module and return Napi bindings
 * Uses lazy loading and caching for performance
 *
 * @returns Module instance with typed bindings
 * @throws Error if module cannot be loaded
 */
export function getLib(): NativeModule {
  if (libInstance !== null) {
    return libInstance
  }

  // Setup DLL search path before loading (Windows only)
  setupDllSearchPath()

  const modulePath = getLibraryPath()

  try {
    const require_ = createRequire(import.meta.url)
    libInstance = require_(modulePath) as NativeModule
    applyConfig(libInstance)
    return libInstance
  } catch (err) {
    const libtorchPath = findLibtorchPath()
    throw new Error(
      `Failed to load ts-torch native module from: ${modulePath}\n` +
        `Error: ${err instanceof Error ? err.message : String(err)}\n\n` +
        `LibTorch path: ${libtorchPath || 'NOT FOUND'}\n\n` +
        `This may indicate:\n` +
        `  - Module architecture mismatch\n` +
        `  - Missing system dependencies (libtorch, CUDA, etc.)\n` +
        `  - Corrupted module file\n\n` +
        `Ensure libtorch is at project root: ts-torch/libtorch/\n` +
        `Or run "bun run setup" to download and build automatically.\n` +
        `Or set LIBTORCH or LIBTORCH_PATH environment variable.`,
    )
  }
}

/**
 * Close the module and release resources
 */
export function closeLib(): void {
  libInstance = null
  cudaLibPath = null
  workspaceRoot = null
  workspaceRootSearched = false
}

/**
 * Auto-cleanup on process exit
 */
if (typeof process !== 'undefined') {
  process.on('exit', () => {
    closeLib()
  })
}
