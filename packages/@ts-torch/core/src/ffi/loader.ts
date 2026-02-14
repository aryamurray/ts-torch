/**
 * Napi native library loader for ts-torch
 * Loads pre-compiled Node.js addon (.node module)
 *
 * This replaced the Koffi loader with a much simpler architecture:
 * - No dynamic symbol binding (all symbols bound at compile time)
 * - No FFI overhead per operation
 * - Automatic memory management via Napi finalizers
 *
 * The .node module is built by cmake-js and contains all 131 C functions
 * directly callable from JavaScript with minimal overhead.
 *
 * IMPORTANT - Napi TypedArray vs ArrayBuffer convention:
 * - Koffi accepted raw ArrayBuffer objects (.buffer property)
 * - Napi expects TypedArray objects directly (Float32Array, BigInt64Array, etc.)
 * - The underlying ArrayBuffer and ByteOffset are extracted automatically by Napi
 * - When creating Napi wrappers, always accept TypedArray, NOT ArrayBuffer.buffer
 * - This convention applies to all buffer parameters (shape, data, weights, etc.)
 */

import { createRequire } from 'module'
import { existsSync, readFileSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { Logger } from '../logger.js'

/**
 * TypedArray type for shape buffers
 * Accept any typed array or buffer-like type
 */
type ShapeBuffer = any

/**
 * Type definition for the loaded Napi module
 * Maps to C function names - all exported from napi_bindings.cpp
 *
 * Note: We pass TypedArrays directly to Napi functions.
 * Napi extracts the underlying ArrayBuffer and ByteOffset automatically.
 */
export type KoffiLibrary = {
  // Utility functions
  ts_version: () => string

  // Tensor factories - take (shapeBuffer, shapeLength, dtype, device, deviceIndex, errBuffer)
  ts_tensor_zeros: (
    shapeBuffer: ShapeBuffer,
    shapeLength: number,
    dtype: number,
    device: number,
    deviceIndex: number,
    errBuffer: ArrayBufferLike
  ) => unknown
  ts_tensor_ones: (
    shapeBuffer: ShapeBuffer,
    shapeLength: number,
    dtype: number,
    device: number,
    deviceIndex: number,
    errBuffer: ArrayBufferLike
  ) => unknown
  ts_tensor_randn: (
    shapeBuffer: ShapeBuffer,
    shapeLength: number,
    dtype: number,
    device: number,
    deviceIndex: number,
    errBuffer: ArrayBufferLike
  ) => unknown
  ts_tensor_rand: (
    shapeBuffer: ShapeBuffer,
    shapeLength: number,
    dtype: number,
    device: number,
    deviceIndex: number,
    errBuffer: ArrayBufferLike
  ) => unknown
  ts_tensor_empty: (
    shapeBuffer: ShapeBuffer,
    shapeLength: number,
    dtype: number,
    device: number,
    deviceIndex: number,
    errBuffer: ArrayBufferLike
  ) => unknown
  ts_tensor_from_buffer: (
    dataBuffer: TypedArray,
    shapeBuffer: ShapeBuffer,
    dtype: number,
    device: number,
    deviceIndex: number
  ) => unknown

  // Tensor properties and manipulation
  ts_tensor_shape: (tensor: unknown, outShape: Int32Array) => number
  ts_tensor_requires_grad: (tensor: unknown) => boolean
  ts_tensor_set_requires_grad: (tensor: unknown, value: boolean) => void
  ts_tensor_dtype: (tensor: unknown) => number
  ts_tensor_clone: (tensor: unknown) => unknown
  ts_tensor_detach: (tensor: unknown) => unknown
  ts_tensor_delete: (tensor: unknown) => void

  // Binary operations
  ts_tensor_add: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_sub: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_mul: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_div: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_matmul: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_minimum: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_maximum: (a: unknown, b: unknown, errBuffer: ArrayBufferLike) => unknown

  // Unary operations
  ts_tensor_relu: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_sigmoid: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_tanh: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_exp: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_log: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_sqrt: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_neg: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_softmax: (tensor: unknown, dim: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_log_softmax: (tensor: unknown, dim: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_transpose: (tensor: unknown, dim0: number, dim1: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_reshape: (tensor: unknown, shapeBuffer: ArrayBufferLike, shapeLength: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_clamp: (tensor: unknown, min: number, max: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_clamp_min: (tensor: unknown, min: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_clamp_max: (tensor: unknown, max: number, errBuffer: ArrayBufferLike) => unknown

  // Reduction operations
  ts_tensor_sum: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_mean: (tensor: unknown, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_sum_dim: (tensor: unknown, dim: bigint, keepdim: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_mean_dim: (tensor: unknown, dim: bigint, keepdim: number, errBuffer: ArrayBufferLike) => unknown

  // Scalar operations
  ts_tensor_add_scalar: (tensor: unknown, scalar: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_sub_scalar: (tensor: unknown, scalar: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_mul_scalar: (tensor: unknown, scalar: number, errBuffer: ArrayBufferLike) => unknown
  ts_tensor_div_scalar: (tensor: unknown, scalar: number, errBuffer: ArrayBufferLike) => unknown

  // Out variants (write result to existing tensor)
  ts_tensor_add_out: (a: unknown, b: unknown, out: unknown, errBuffer: ArrayBufferLike) => void
  ts_tensor_sub_out: (a: unknown, b: unknown, out: unknown, errBuffer: ArrayBufferLike) => void
  ts_tensor_mul_out: (a: unknown, b: unknown, out: unknown, errBuffer: ArrayBufferLike) => void
  ts_tensor_div_out: (a: unknown, b: unknown, out: unknown, errBuffer: ArrayBufferLike) => void
  ts_tensor_matmul_out: (a: unknown, b: unknown, out: unknown, errBuffer: ArrayBufferLike) => void

  // Buffer operations (Napi wrapper extracts buffer size automatically)
  ts_tensor_copy_to_buffer: (tensor: unknown, buffer: ArrayBufferLike) => void

  // Autograd operations (Napi wrapper handles errors internally)
  ts_tensor_backward: (tensor: unknown) => void

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
let libInstance: KoffiLibrary | null = null

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
        case 'arm64':
          return {
            packageName: '@ts-torch-platform/linux-arm64',
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
        case 'arm64':
          return {
            packageName: '@ts-torch-platform/win32-arm64',
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
        Logger.debug(
          `Found workspace root via directory pattern (${hasPackagesDir ? 'packages/' : 'apps/'}): ${dir}`
        )
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
  const cudaInstructions = isCudaSupported
    ? `\nFor GPU/CUDA support:\n` + `  bun run setup:cuda\n\n`
    : ''

  throw new Error(
    `Could not find ts-torch native module for ${process.platform}-${process.arch}.\n\n` +
      `Quick Setup:\n` +
      `  bun run setup          # Download LibTorch and build (CPU)\n` +
      cudaInstructions +
      `Manual Installation:\n` +
      `  bun add ${packageName}\n\n` +
      `Or set TS_TORCH_LIB environment variable to point to your .node file.\n\n` +
      `For debugging, set TS_TORCH_DEBUG=1 to see detailed resolution info.\n\n` +
      `Searched paths:\n${possiblePaths.map((p) => `  - ${p}`).join('\n')}`
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
export function getLib(): KoffiLibrary {
  if (libInstance !== null) {
    return libInstance
  }

  // Setup DLL search path before loading (Windows only)
  setupDllSearchPath()

  const modulePath = getLibraryPath()

  try {
    const require_ = createRequire(import.meta.url)
    libInstance = require_(modulePath) as KoffiLibrary
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
        `Or set LIBTORCH or LIBTORCH_PATH environment variable.`
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
