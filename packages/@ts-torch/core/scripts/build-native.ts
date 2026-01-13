#!/usr/bin/env bun
/**
 * Cross-platform native library build script
 * Builds the ts_torch native library and copies it to the appropriate platform package
 */

import { spawnSync } from 'node:child_process'
import { existsSync, copyFileSync, mkdirSync, rmSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const ROOT_DIR = resolve(import.meta.dirname, '..', '..', '..', '..')
const NATIVE_DIR = resolve(import.meta.dirname, '..', 'native')
const BUILD_DIR = join(NATIVE_DIR, 'build')
const _CMAKE_CACHE_FILE = join(BUILD_DIR, 'CMakeCache.txt')
const LIBTORCH_CACHE_FILE = join(BUILD_DIR, '.libtorch-path')

// Platform-specific configuration - now arch-aware for all platforms
interface PlatformConfig {
  packageDir: string
  libName: string
  buildSubdir: string
}

function getPlatformConfig(): PlatformConfig {
  const platform = process.platform
  const arch = process.arch

  switch (platform) {
    case 'win32':
      return {
        packageDir: `@ts-torch-platform/win32-${arch}`,
        libName: 'ts_torch.dll',
        buildSubdir: 'Release',
      }
    case 'darwin':
      return {
        packageDir: `@ts-torch-platform/darwin-${arch}`,
        libName: 'libts_torch.dylib',
        buildSubdir: '',
      }
    case 'linux':
      return {
        packageDir: `@ts-torch-platform/linux-${arch}`,
        libName: 'libts_torch.so',
        buildSubdir: '',
      }
    default:
      throw new Error(`Unsupported platform: ${platform}`)
  }
}

function findLibtorchPath(): string {
  // Check common locations (environment variables first)
  const possiblePaths = [
    process.env.LIBTORCH || '',
    process.env.LIBTORCH_PATH || '',
    join(ROOT_DIR, 'libtorch'),
    '/usr/local/lib/libtorch',
    '/opt/libtorch',
    'C:\\libtorch',
  ]

  for (const p of possiblePaths) {
    if (p && existsSync(p) && existsSync(join(p, 'lib'))) {
      return resolve(p) // Return absolute path
    }
  }

  throw new Error(
    'LibTorch not found. Please either:\n' +
      '  1. Run "bun run setup" to download and build automatically\n' +
      '  2. Place libtorch at the project root: ts-torch/libtorch/\n' +
      '  3. Set LIBTORCH or LIBTORCH_PATH environment variable\n' +
      '  4. Download from https://pytorch.org/get-started/locally/',
  )
}

/**
 * Check if CMake cache needs to be invalidated
 * This happens when LibTorch path changes
 */
function needsCacheInvalidation(libtorchPath: string): boolean {
  if (!existsSync(BUILD_DIR)) {
    return false // No cache to invalidate
  }

  if (!existsSync(LIBTORCH_CACHE_FILE)) {
    return true // No record of previous path, invalidate to be safe
  }

  try {
    const cachedPath = readFileSync(LIBTORCH_CACHE_FILE, 'utf-8').trim()
    if (cachedPath !== libtorchPath) {
      console.log(`LibTorch path changed: ${cachedPath} -> ${libtorchPath}`)
      return true
    }
  } catch {
    return true // Can't read cache file, invalidate
  }

  return false
}

/**
 * Save the LibTorch path for cache invalidation check
 */
function saveLibtorchPath(libtorchPath: string): void {
  try {
    writeFileSync(LIBTORCH_CACHE_FILE, libtorchPath)
  } catch {
    // Non-fatal
  }
}

function runCommand(command: string, args: string[], cwd: string): void {
  console.log(`Running: ${command} ${args.join(' ')}`)
  const result = spawnSync(command, args, {
    cwd,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  })

  if (result.status !== 0) {
    throw new Error(`Command failed with exit code ${result.status}`)
  }
}

async function main(): Promise<void> {
  const config = getPlatformConfig()

  console.log('=== Building ts-torch Native Library ===\n')
  console.log(`Platform: ${process.platform} (${process.arch})`)
  console.log(`Target package: ${config.packageDir}`)

  // Find LibTorch
  const libtorchPath = findLibtorchPath()
  console.log(`LibTorch found at: ${libtorchPath}`)

  // Check if we need to invalidate CMake cache
  if (needsCacheInvalidation(libtorchPath)) {
    console.log('\nCMake cache invalidated due to LibTorch path change')
    if (existsSync(BUILD_DIR)) {
      rmSync(BUILD_DIR, { recursive: true, force: true })
    }
  }

  // Create build directory
  if (!existsSync(BUILD_DIR)) {
    mkdirSync(BUILD_DIR, { recursive: true })
  }

  // Save LibTorch path for future cache checks
  saveLibtorchPath(libtorchPath)

  // Configure CMake - quote the path for Windows paths with spaces
  console.log('\n--- Configuring CMake ---')
  const cmakePrefixPath = process.platform === 'win32'
    ? `"${libtorchPath}"` // Quote for Windows
    : libtorchPath

  runCommand(
    'cmake',
    ['-B', 'build', `-DCMAKE_PREFIX_PATH=${cmakePrefixPath}`, '-DCMAKE_BUILD_TYPE=Release'],
    NATIVE_DIR
  )

  // Build
  console.log('\n--- Building ---')
  runCommand('cmake', ['--build', 'build', '--config', 'Release'], NATIVE_DIR)

  // Copy to platform package
  console.log('\n--- Copying to platform package ---')
  const srcLib = join(BUILD_DIR, config.buildSubdir, config.libName)
  const destDir = join(ROOT_DIR, 'packages', config.packageDir, 'lib')
  const destLib = join(destDir, config.libName)

  if (!existsSync(srcLib)) {
    throw new Error(`Built library not found at: ${srcLib}`)
  }

  if (!existsSync(destDir)) {
    mkdirSync(destDir, { recursive: true })
  }

  copyFileSync(srcLib, destLib)
  console.log(`Copied ${config.libName} to ${destDir}`)

  console.log('\n=== Build complete! ===')
}

main().catch((err) => {
  console.error('Build failed:', err.message)
  process.exit(1)
})
