#!/usr/bin/env bun
/**
 * Cross-platform native library build script for CUDA
 * Builds the ts_torch native library against CUDA-enabled LibTorch
 */

import { spawnSync } from 'node:child_process'
import { existsSync, copyFileSync, mkdirSync, rmSync, readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { LIBTORCH_VERSION, writeBuildMeta } from '@ts-torch-cuda/shared'

const ROOT_DIR = resolve(import.meta.dirname, '..')
const NATIVE_DIR = join(ROOT_DIR, 'packages', '@ts-torch', 'core', 'native')
const BUILD_DIR = join(NATIVE_DIR, 'build-cuda')
const LIBTORCH_CACHE_FILE = join(BUILD_DIR, '.libtorch-cuda-path')

// Default CUDA version for dev builds
const DEFAULT_CUDA_VERSION = 'cu124'

interface PlatformConfig {
  cudaPackageDir: string
  libName: string
  buildSubdir: string
}

function getPlatformConfig(cudaVersion: string): PlatformConfig {
  const platform = process.platform
  const arch = process.arch

  switch (platform) {
    case 'win32':
      if (arch !== 'x64') {
        throw new Error(`CUDA is only supported on x64, got ${arch}`)
      }
      return {
        cudaPackageDir: `@ts-torch-cuda/win32-x64-${cudaVersion}`,
        libName: 'ts_torch.dll',
        buildSubdir: 'Release',
      }
    case 'linux':
      if (arch !== 'x64') {
        throw new Error(`CUDA is only supported on x64, got ${arch}`)
      }
      return {
        cudaPackageDir: `@ts-torch-cuda/linux-x64-${cudaVersion}`,
        libName: 'libts_torch.so',
        buildSubdir: '',
      }
    default:
      throw new Error(`CUDA is not supported on ${platform}. Only Linux and Windows are supported.`)
  }
}

function findCudaLibtorchPath(): string {
  // Check environment variables first
  const envPaths = [
    process.env.LIBTORCH_CUDA,
    process.env.LIBTORCH_CUDA_PATH,
  ]

  for (const p of envPaths) {
    if (p && existsSync(p) && existsSync(join(p, 'lib'))) {
      return resolve(p)
    }
  }

  // Check common locations
  const possiblePaths = [
    join(ROOT_DIR, 'libtorch-cuda'),
    join(ROOT_DIR, 'libtorch_cuda'),
    '/usr/local/lib/libtorch-cuda',
    '/opt/libtorch-cuda',
    'C:\\libtorch-cuda',
  ]

  for (const p of possiblePaths) {
    if (existsSync(p) && existsSync(join(p, 'lib'))) {
      return resolve(p)
    }
  }

  throw new Error(
    'CUDA LibTorch not found. Please either:\n' +
      '  1. Place CUDA libtorch at: ts-torch/libtorch-cuda/\n' +
      '  2. Set LIBTORCH_CUDA or LIBTORCH_CUDA_PATH environment variable\n' +
      '  3. Download from https://pytorch.org/get-started/locally/ (select CUDA version)',
  )
}

function needsCacheInvalidation(libtorchPath: string): boolean {
  if (!existsSync(BUILD_DIR)) {
    return false
  }

  if (!existsSync(LIBTORCH_CACHE_FILE)) {
    return true
  }

  try {
    const cachedPath = readFileSync(LIBTORCH_CACHE_FILE, 'utf-8').trim()
    if (cachedPath !== libtorchPath) {
      console.log(`CUDA LibTorch path changed: ${cachedPath} -> ${libtorchPath}`)
      return true
    }
  } catch {
    return true
  }

  return false
}

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
  // Parse CUDA version from args (default: cu124)
  const cudaVersion = process.argv[2] || DEFAULT_CUDA_VERSION
  if (!['cu118', 'cu121', 'cu124'].includes(cudaVersion)) {
    console.error(`Invalid CUDA version: ${cudaVersion}`)
    console.error('Valid options: cu118, cu121, cu124')
    process.exit(1)
  }

  const config = getPlatformConfig(cudaVersion)

  console.log('=== Building ts-torch Native Library (CUDA) ===\n')
  console.log(`Platform: ${process.platform} (${process.arch})`)
  console.log(`CUDA version: ${cudaVersion}`)
  console.log(`Target package: ${config.cudaPackageDir}`)

  // Find CUDA LibTorch
  const libtorchPath = findCudaLibtorchPath()
  console.log(`CUDA LibTorch found at: ${libtorchPath}`)

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

  // Configure CMake
  console.log('\n--- Configuring CMake ---')
  const cmakePrefixPath = process.platform === 'win32' ? `"${libtorchPath}"` : libtorchPath

  runCommand(
    'cmake',
    ['-B', 'build-cuda', `-DCMAKE_PREFIX_PATH=${cmakePrefixPath}`, '-DCMAKE_BUILD_TYPE=Release'],
    NATIVE_DIR,
  )

  // Build
  console.log('\n--- Building ---')
  runCommand('cmake', ['--build', 'build-cuda', '--config', 'Release'], NATIVE_DIR)

  // Copy to CUDA package
  console.log('\n--- Copying to CUDA package ---')
  const srcLib = join(BUILD_DIR, config.buildSubdir, config.libName)
  const destDir = join(ROOT_DIR, 'packages', config.cudaPackageDir, 'lib')
  const destLib = join(destDir, config.libName)

  if (!existsSync(srcLib)) {
    throw new Error(`Built library not found at: ${srcLib}`)
  }

  if (!existsSync(destDir)) {
    mkdirSync(destDir, { recursive: true })
  }

  copyFileSync(srcLib, destLib)
  console.log(`Copied ${config.libName} to ${destDir}`)

  // Write build metadata
  const metaPath = join(destDir, '.build-meta.json')
  writeBuildMeta(metaPath, cudaVersion)
  console.log(`Wrote build metadata to ${metaPath}`)

  console.log('\n=== CUDA Build complete! ===')
  console.log(`\nTo use: The loader will automatically detect the CUDA library.`)
  console.log(`Make sure libtorch-cuda/lib is in PATH for DLL loading on Windows.`)
}

main().catch((err) => {
  console.error('CUDA Build failed:', err.message)
  process.exit(1)
})
