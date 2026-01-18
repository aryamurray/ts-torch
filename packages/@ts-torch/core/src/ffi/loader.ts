/**
 * Native library loader for ts-torch
 * Handles platform detection, library resolution, and FFI initialization
 */

import koffi from 'koffi'
import { existsSync, readFileSync } from 'node:fs'
import { resolve, join, dirname } from 'node:path'
import { FFI_SYMBOLS, type FFISymbols } from './symbols.js'
import { Logger } from '../logger.js'

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

// Logger is configured via TS_TORCH_DEBUG and TS_TORCH_QUIET env vars automatically

/**
 * Cached workspace root path
 */
let workspaceRoot: string | null = null
let workspaceRootSearched = false

/**
 * Find the workspace root directory using multiple detection strategies
 * Caches the result for performance
 *
 * Detection order:
 * 1. package.json with workspaces field (npm/bun/yarn workspaces)
 * 2. pnpm-workspace.yaml (pnpm workspaces)
 * 3. lerna.json (lerna monorepos)
 * 4. Directory with packages/ or apps/ subdirectory and package.json
 * 5. Git repository root with package.json
 */
function findWorkspaceRoot(): string | null {
  // Return cached result (including null if we already searched)
  if (workspaceRootSearched) {
    return workspaceRoot
  }
  workspaceRootSearched = true

  // Start from loader's directory and traverse up
  const loaderDir = import.meta.dirname
  if (!loaderDir) {
    Logger.debug('Cannot determine loader directory, workspace detection skipped')
    return null
  }

  const root = process.platform === 'win32' ? loaderDir.split(':')[0] + ':\\' : '/'

  // First pass: look for explicit workspace configurations
  let dir = loaderDir
  while (dir !== root) {
    const pkgPath = join(dir, 'package.json')

    // Check for npm/bun/yarn workspaces
    if (existsSync(pkgPath)) {
      try {
        const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'))
        if (pkg.workspaces) {
          // Handle both array and object format (yarn uses objects)
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

    // Check for pnpm workspaces
    if (existsSync(join(dir, 'pnpm-workspace.yaml'))) {
      workspaceRoot = dir
      Logger.debug(`Found workspace root via pnpm-workspace.yaml: ${dir}`)
      return workspaceRoot
    }

    // Check for lerna
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
      // Check for packages/ or apps/ directories (common monorepo patterns)
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

  // Third pass: use git root as fallback (many projects have package root at git root)
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
function getDevPaths(libFileName: string, platformDir: string): string[] {
  const paths: string[] = []
  const root = findWorkspaceRoot()

  if (root) {
    // Primary development paths from workspace root
    paths.push(
      join(root, 'packages', '@ts-torch-platform', platformDir, 'lib', libFileName),
      join(root, 'packages', '@ts-torch', 'core', 'native', 'build', 'Release', libFileName),
      join(root, 'packages', '@ts-torch', 'core', 'native', 'build', libFileName),
    )
  }

  // Fallback: check from cwd for non-workspace scenarios
  const cwd = process.cwd()
  if (cwd !== root) {
    paths.push(
      join(cwd, 'packages', '@ts-torch-platform', platformDir, 'lib', libFileName),
      join(cwd, 'native', 'build', 'Release', libFileName),
      join(cwd, 'native', 'build', libFileName),
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

  // Fallback from cwd
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

  // Both lib and meta must exist
  if (!existsSync(libPath) || !existsSync(metaPath)) {
    return false
  }

  // Verify meta matches expected CUDA version
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
        Logger.info(`Using CUDA ${cudaVer} library`)
        cudaLibPath = join(pkgDir, 'lib', libFileName)
        return cudaLibPath
      }
    } catch {
      // Package not installed, continue to next version
    }
  }

  // Check development paths using workspace root
  const root = findWorkspaceRoot()
  if (root) {
    for (const cudaVer of CUDA_VERSIONS) {
      const devPath = join(root, 'packages', '@ts-torch-cuda', `${platform}-x64-${cudaVer}`)
      if (existsSync(devPath) && isValidCudaBuild(devPath, libFileName, cudaVer)) {
        Logger.info(`Using CUDA ${cudaVer} library (dev)`)
        cudaLibPath = join(devPath, 'lib', libFileName)
        return cudaLibPath
      }
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
    Logger.debug(`TS_TORCH_LIB set but file not found: ${envPath}`)
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
    const libPath = join(packageDir, 'lib', libFileName)

    if (existsSync(libPath)) {
      return libPath
    }
  } catch {
    // Package not found, try development paths
  }

  // 4. Try local development paths (workspace monorepo structure)
  const platformDir = packageName.replace('@ts-torch-platform/', '')
  const possiblePaths = getDevPaths(libFileName, platformDir)

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return resolve(path)
    }
  }

  // Library not found - provide helpful setup instructions
  const isCudaSupported = process.platform === 'linux' || process.platform === 'win32'
  const cudaInstructions = isCudaSupported
    ? `\nFor GPU/CUDA support:\n` +
      `  bun run setup:cuda\n\n`
    : ''

  throw new Error(
    `Could not find ts-torch native library for ${process.platform}-${process.arch}.\n\n` +
      `Quick Setup:\n` +
      `  bun run setup          # Download LibTorch and build (CPU)\n` +
      cudaInstructions +
      `Manual Installation:\n` +
      `  bun add ${packageName}\n\n` +
      `Or set TS_TORCH_LIB environment variable to point to your library.\n\n` +
      `For debugging, set TS_TORCH_DEBUG=1 to see detailed resolution info.\n\n` +
      `Searched paths:\n${possiblePaths.map((p) => `  - ${p}`).join('\n')}`,
  )
}

/**
 * Find the libtorch library directory
 * Used to add DLL dependencies to the search path on Windows
 */
function findLibtorchPath(): string | null {
  // Check environment variables first (highest priority)
  const envPaths = [
    process.env.LIBTORCH ? join(process.env.LIBTORCH, 'lib') : '',
    process.env.LIBTORCH_PATH ? join(process.env.LIBTORCH_PATH, 'lib') : '',
  ].filter(Boolean)

  for (const p of envPaths) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  // Check development paths
  for (const p of getLibtorchDevPaths('libtorch')) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  return null
}

/**
 * Find libtorch path from CUDA package (if using CUDA)
 * CUDA packages store libtorch in their lib/libtorch directory
 * For dev, check root libtorch-cuda directory
 */
function findCudaLibtorchPath(): string | null {
  // Check env vars first (dev/custom builds)
  const envPaths = [
    process.env.LIBTORCH_CUDA ? join(process.env.LIBTORCH_CUDA, 'lib') : '',
    process.env.LIBTORCH_CUDA_PATH ? join(process.env.LIBTORCH_CUDA_PATH, 'lib') : '',
  ].filter(Boolean)

  for (const p of envPaths) {
    if (existsSync(p)) {
      return resolve(p)
    }
  }

  // If we found a CUDA library, check its package's libtorch
  if (cudaLibPath) {
    // CUDA lib path is like: .../lib/ts_torch.dll
    // LibTorch is at: .../lib/libtorch/lib
    const libDir = dirname(cudaLibPath)
    const libtorchLib = join(libDir, 'libtorch', 'lib')

    if (existsSync(libtorchLib)) {
      return libtorchLib
    }
  }

  // Check development paths
  for (const p of getLibtorchDevPaths('libtorch-cuda')) {
    if (existsSync(p)) {
      return resolve(p)
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
  cudaLibPath = null
  // Reset workspace detection for testing
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
