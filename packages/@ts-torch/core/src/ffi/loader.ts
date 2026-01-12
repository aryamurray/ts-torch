/**
 * Native library loader for ts-torch
 * Handles platform detection, library resolution, and FFI initialization
 */

import koffi from 'koffi'
import { existsSync, readFileSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { FFI_SYMBOLS, type FFISymbols } from './symbols.js'

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
 * Type for a koffi function
 */
type KoffiFunction = (...args: unknown[]) => unknown

/**
 * Library interface with all FFI functions bound
 */
export type KoffiLibrary = {
  [K in keyof FFISymbols]: KoffiFunction
}

/**
 * Cached library instance
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
 * Get platform-specific library suffix
 * Replaces bun:ffi's suffix constant
 */
function getLibrarySuffix(): string {
  switch (process.platform) {
    case 'win32':
      return 'dll'
    case 'darwin':
      return 'dylib'
    default:
      return 'so'
  }
}

/**
 * Get platform-specific package name and library name
 * Maps Node.js process.platform and process.arch to native package names
 */
export function getPlatformPackage(): PlatformInfo {
  const platform = process.platform
  const arch = process.arch

  // Map platform/arch to package names
  // Package naming convention: @ts-torch-platform/{platform}-{arch}
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
 * Check if a CUDA build is valid by verifying .build-meta.json
 */
function isValidCudaBuild(pkgDir: string, libFileName: string, expectedCuda: string): boolean {
  const libPath = join(pkgDir, 'lib', libFileName)
  const metaPath = join(pkgDir, 'lib', '.build-meta.json')

  // Both lib and meta must exist
  if (!existsSync(libPath) || !existsSync(metaPath)) {
    return false
  }

  // Verify meta matches expected CUDA version
  try {
    const meta: BuildMeta = JSON.parse(readFileSync(metaPath, 'utf-8'))
    if (meta.cuda !== expectedCuda) {
      console.warn(`Build meta mismatch: expected ${expectedCuda}, got ${meta.cuda}`)
      return false
    }
    const expectedPlatform = `${process.platform}-${process.arch}`
    if (meta.platform !== expectedPlatform) {
      console.warn(`Platform mismatch: expected ${expectedPlatform}, got ${meta.platform}`)
      return false
    }
    return true
  } catch {
    console.warn(`Invalid .build-meta.json in ${pkgDir}`)
    return false
  }
}

/**
 * Try to find a valid CUDA library
 * @returns Path to CUDA library if found, null otherwise
 */
function findCudaLibrary(): string | null {
  // Only Linux and Windows support CUDA
  if (process.platform !== 'linux' && process.platform !== 'win32') {
    return null
  }

  const platform = process.platform === 'win32' ? 'win32' : 'linux'
  const { libraryName } = getPlatformPackage()
  const suffix = getLibrarySuffix()
  const libFileName = `${libraryName}.${suffix}`

  // Check each CUDA version (latest first)
  for (const cudaVer of CUDA_VERSIONS) {
    const pkg = `@ts-torch-cuda/${platform}-x64-${cudaVer}`
    try {
      const pkgPath = require.resolve(`${pkg}/package.json`)
      const pkgDir = dirname(pkgPath)

      if (isValidCudaBuild(pkgDir, libFileName, cudaVer)) {
        console.log(`Using CUDA ${cudaVer} library`)
        cudaLibPath = join(pkgDir, 'lib', libFileName)
        return cudaLibPath
      }
    } catch {
      // Package not installed, continue to next version
    }
  }

  // Also check development paths
  const cwd = process.cwd()
  for (const cudaVer of CUDA_VERSIONS) {
    const devPath = join(cwd, 'packages', '@ts-torch-cuda', `${platform}-x64-${cudaVer}`)
    if (existsSync(devPath) && isValidCudaBuild(devPath, libFileName, cudaVer)) {
      console.log(`Using CUDA ${cudaVer} library (development)`)
      cudaLibPath = join(devPath, 'lib', libFileName)
      return cudaLibPath
    }
  }

  return null
}

/**
 * Get the full path to the native library
 * Resolution order:
 * 1. TS_TORCH_LIB environment variable (for custom builds)
 * 2. CUDA packages (if installed and valid)
 * 3. CPU platform-specific package (production)
 * 4. Local development paths (workspace monorepo)
 *
 * @throws Error if library cannot be found
 */
export function getLibraryPath(): string {
  // 1. Check environment variable override
  const envPath = process.env.TS_TORCH_LIB
  if (envPath) {
    if (existsSync(envPath)) {
      return resolve(envPath)
    }
    console.warn(`TS_TORCH_LIB set but file not found: ${envPath}`)
  }

  // 2. Check for CUDA packages (prefer GPU over CPU)
  const cudaPath = findCudaLibrary()
  if (cudaPath) {
    return cudaPath
  }

  const { packageName, libraryName } = getPlatformPackage()
  const suffix = getLibrarySuffix()
  const libFileName = `${libraryName}.${suffix}`

  // 3. Try to resolve from installed CPU platform package
  try {
    // Resolve the platform package's package.json
    const packagePath = require.resolve(`${packageName}/package.json`)
    // Handle both forward and backward slashes (Windows uses backslashes)
    const packageDir = packagePath.replace(/[/\\]package\.json$/, '')
    const libPath = join(packageDir, libFileName)

    if (existsSync(libPath)) {
      return libPath
    }
  } catch {
    // Package not found, try development paths
  }

  // 3. Try local development paths (workspace monorepo structure)
  const cwd = process.cwd()
  const platformDir = packageName.replace('@ts-torch-platform/', '')

  const possiblePaths = [
    // Platform package lib directory (from workspace root) - PRIMARY DEV PATH
    join(cwd, 'packages', '@ts-torch-platform', platformDir, 'lib', libFileName),

    // CMake build output (from workspace root)
    join(cwd, 'packages', '@ts-torch', 'core', 'native', 'build', 'Release', libFileName),
    join(cwd, 'packages', '@ts-torch', 'core', 'native', 'build', libFileName),

    // From examples/ or benchmark/ directory (one level up from workspace root)
    join(cwd, '..', 'packages', '@ts-torch-platform', platformDir, 'lib', libFileName),
    join(cwd, '..', 'packages', '@ts-torch', 'core', 'native', 'build', 'Release', libFileName),
    join(cwd, '..', 'packages', '@ts-torch', 'core', 'native', 'build', libFileName),

    // From within a package (e.g., @ts-torch/core)
    join(cwd, '..', '..', '@ts-torch-platform', platformDir, 'lib', libFileName),
    join(cwd, 'native', 'build', 'Release', libFileName),
    join(cwd, 'native', 'build', libFileName),
  ]

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return resolve(path)
    }
  }

  // Library not found
  throw new Error(
    `Could not find ts-torch native library for ${process.platform}-${process.arch}.\n` +
      `Searched paths:\n${possiblePaths.map((p) => `  - ${p}`).join('\n')}\n\n` +
      `Please ensure the platform-specific package is installed:\n` +
      `  bun add ${packageName}\n\n` +
      `Or build from source:\n` +
      `  cd native && cargo build --release\n\n` +
      `Or set TS_TORCH_LIB environment variable to the library path.`,
  )
}

/**
 * Find the libtorch library directory
 * Used to add DLL dependencies to the search path on Windows
 */
function findLibtorchPath(): string | null {
  const cwd = process.cwd()

  // Possible locations for libtorch
  const possiblePaths = [
    // Environment variable (highest priority)
    process.env.LIBTORCH ? join(process.env.LIBTORCH, 'lib') : '',
    process.env.LIBTORCH_PATH ? join(process.env.LIBTORCH_PATH, 'lib') : '',
    // Project root /libtorch/lib
    join(cwd, 'libtorch', 'lib'),
    // Relative to packages/@ts-torch/core
    join(cwd, '..', '..', '..', '..', 'libtorch', 'lib'),
  ].filter(Boolean)

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return resolve(path)
    }
  }

  return null
}

/**
 * Find libtorch path from CUDA package (if using CUDA)
 * CUDA packages store libtorch in their lib/libtorch directory
 */
function findCudaLibtorchPath(): string | null {
  if (!cudaLibPath) {
    return null
  }

  // CUDA lib path is like: .../lib/ts_torch.dll
  // LibTorch is at: .../lib/libtorch/lib
  const libDir = dirname(cudaLibPath)
  const libtorchLib = join(libDir, 'libtorch', 'lib')

  if (existsSync(libtorchLib)) {
    return libtorchLib
  }

  return null
}

/**
 * Setup DLL search path for Windows
 * On Windows, DLL dependencies need to be in PATH or same directory
 */
function setupDllSearchPath(): void {
  if (process.platform !== 'win32') {
    return
  }

  // Check CUDA libtorch path first (if using CUDA)
  const cudaLibtorchLib = findCudaLibtorchPath()
  if (cudaLibtorchLib) {
    const currentPath = process.env.PATH || ''
    if (!currentPath.includes(cudaLibtorchLib)) {
      process.env.PATH = `${cudaLibtorchLib};${currentPath}`
    }
    return
  }

  // Fall back to CPU libtorch path
  const libtorchLib = findLibtorchPath()
  if (libtorchLib) {
    // Add libtorch/lib to PATH so Windows can find dependent DLLs
    const currentPath = process.env.PATH || ''
    if (!currentPath.includes(libtorchLib)) {
      process.env.PATH = `${libtorchLib};${currentPath}`
    }
  }
}

/**
 * Load the native library and return FFI bindings
 * Uses lazy loading and caching for performance
 *
 * @returns Library instance with typed FFI symbols
 * @throws Error if library cannot be loaded
 */
export function getLib(): KoffiLibrary {
  if (libInstance !== null) {
    return libInstance
  }

  // Setup DLL search path before loading (Windows only)
  setupDllSearchPath()

  const libraryPath = getLibraryPath()

  try {
    // Load the native library with koffi
    const lib = koffi.load(libraryPath)

    // Bind all functions from FFI_SYMBOLS
    const symbols: Record<string, KoffiFunction> = {}

    for (const [name, def] of Object.entries(FFI_SYMBOLS)) {
      symbols[name] = lib.func(name, def.returns, def.args as unknown as string[])
    }

    libInstance = symbols as KoffiLibrary
    return libInstance
  } catch (err) {
    const libtorchPath = findLibtorchPath()
    throw new Error(
      `Failed to load ts-torch native library from: ${libraryPath}\n` +
        `Error: ${err instanceof Error ? err.message : String(err)}\n\n` +
        `LibTorch path: ${libtorchPath || 'NOT FOUND'}\n\n` +
        `This may indicate:\n` +
        `  - Library architecture mismatch\n` +
        `  - Missing system dependencies (libtorch, CUDA, etc.)\n` +
        `  - Corrupted library file\n\n` +
        `Ensure libtorch is at project root: ts-torch/libtorch/\n` +
        `Or run "bun run setup" to download and build automatically.\n` +
        `Or set LIBTORCH or LIBTORCH_PATH environment variable.`,
    )
  }
}

/**
 * Close the library and release resources
 * Note: koffi handles cleanup automatically, but we clear the cache
 */
export function closeLib(): void {
  libInstance = null
}

/**
 * Auto-cleanup on process exit
 */
if (typeof process !== 'undefined') {
  process.on('exit', () => {
    closeLib()
  })
}
