/**
 * Native library loader for ts-torch
 * Handles platform detection, library resolution, and FFI initialization
 */

import { dlopen, type Library, suffix } from 'bun:ffi'
import { existsSync } from 'node:fs'
import { resolve, join } from 'node:path'
import { FFI_SYMBOLS, type FFISymbols } from './symbols.js'

/**
 * Cached library instance
 */
let libInstance: Library<FFISymbols> | null = null

/**
 * Platform-specific library information
 */
interface PlatformInfo {
  packageName: string
  libraryName: string
}

/**
 * Get platform-specific package name and library name
 * Maps Node.js process.platform and process.arch to native package names
 */
export function getPlatformPackage(): PlatformInfo {
  const platform = process.platform
  const arch = process.arch

  // Map platform/arch to package names
  switch (platform) {
    case 'darwin':
      switch (arch) {
        case 'arm64':
          return {
            packageName: '@ts-torch/darwin-arm64',
            libraryName: 'libts_torch',
          }
        case 'x64':
          return {
            packageName: '@ts-torch/darwin-x64',
            libraryName: 'libts_torch',
          }
        default:
          throw new Error(`Unsupported macOS architecture: ${arch}`)
      }

    case 'linux':
      switch (arch) {
        case 'x64':
          return {
            packageName: '@ts-torch/linux-x64',
            libraryName: 'libts_torch',
          }
        case 'arm64':
          return {
            packageName: '@ts-torch/linux-arm64',
            libraryName: 'libts_torch',
          }
        default:
          throw new Error(`Unsupported Linux architecture: ${arch}`)
      }

    case 'win32':
      switch (arch) {
        case 'x64':
          return {
            packageName: '@ts-torch/win32-x64',
            libraryName: 'ts_torch',
          }
        case 'arm64':
          return {
            packageName: '@ts-torch/win32-arm64',
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
 * Get the full path to the native library
 * Resolution order:
 * 1. TS_TORCH_LIB environment variable (for custom builds)
 * 2. Platform-specific package (production)
 * 3. Local development paths (workspace monorepo)
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

  const { packageName, libraryName } = getPlatformPackage()
  const libFileName = `${libraryName}.${suffix}`

  // 2. Try to resolve from installed platform package
  try {
    // Resolve the platform package's package.json
    const packagePath = require.resolve(`${packageName}/package.json`)
    const packageDir = packagePath.replace(/\/package\.json$/, '')
    const libPath = join(packageDir, libFileName)

    if (existsSync(libPath)) {
      return libPath
    }
  } catch (err) {
    // Package not found, try development paths
  }

  // 3. Try local development paths (workspace monorepo structure)
  const cwd = process.cwd()
  const possiblePaths = [
    // CMake build output (Windows)
    join(cwd, 'packages', '@ts-torch', 'core', 'native', 'build', 'Release', libFileName),
    join(cwd, 'packages', '@ts-torch', 'core', 'native', 'build', 'Debug', libFileName),

    // Relative from @ts-torch/core package
    join(cwd, '..', '..', 'core', 'native', 'build', 'Release', libFileName),
    join(cwd, '..', '..', 'core', 'native', 'build', 'Debug', libFileName),

    // Current package's native directory (Cargo/Rust style)
    join(cwd, 'native', 'target', 'release', libFileName),
    join(cwd, 'native', 'target', 'debug', libFileName),

    // CMake build from native directory
    join(cwd, 'native', 'build', 'Release', libFileName),
    join(cwd, 'native', 'build', 'Debug', libFileName),

    // Workspace root native directory
    join(cwd, '..', '..', '..', 'native', 'target', 'release', libFileName),
    join(cwd, '..', '..', '..', 'native', 'target', 'debug', libFileName),

    // Platform package in workspace
    join(cwd, '..', packageName, libFileName),
    join(cwd, '..', '..', packageName, libFileName),

    // Platform package lib directory
    join(cwd, 'packages', '@ts-torch-platform', 'win32-x64', 'lib', libFileName),
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
    // Project root /libtorch
    join(cwd, 'libtorch', 'lib'),
    // Relative to packages/@ts-torch/core
    join(cwd, '..', '..', '..', '..', 'libtorch', 'lib'),
    // Environment variable
    process.env.LIBTORCH_PATH ? join(process.env.LIBTORCH_PATH, 'lib') : '',
  ].filter(Boolean)

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return resolve(path)
    }
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
export function getLib(): Library<FFISymbols> {
  if (libInstance !== null) {
    return libInstance
  }

  // Setup DLL search path before loading (Windows only)
  setupDllSearchPath()

  const libraryPath = getLibraryPath()

  try {
    libInstance = dlopen(libraryPath, FFI_SYMBOLS)
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
        `Ensure libtorch is at project root: ts-tools/libtorch\n` +
        `Or set LIBTORCH_PATH environment variable.`,
    )
  }
}

/**
 * Close the library and release resources
 * Should be called on process exit or when library is no longer needed
 */
export function closeLib(): void {
  if (libInstance !== null) {
    libInstance.close()
    libInstance = null
  }
}

/**
 * Auto-cleanup on process exit
 */
if (typeof process !== 'undefined') {
  process.on('exit', () => {
    closeLib()
  })
}
