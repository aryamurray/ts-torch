#!/usr/bin/env bun
/**
 * Cross-platform native library build script
 * Builds the ts_torch native library and copies it to the appropriate platform package
 */

import { execSync, spawnSync } from 'node:child_process'
import { existsSync, copyFileSync, mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'

const ROOT_DIR = resolve(import.meta.dirname, '..', '..', '..', '..')
const NATIVE_DIR = resolve(import.meta.dirname, '..', 'native')
const BUILD_DIR = join(NATIVE_DIR, 'build')

// Platform-specific configuration
const PLATFORM_CONFIG: Record<string, { packageDir: string; libName: string; buildSubdir: string }> = {
  win32: {
    packageDir: '@ts-torch-platform/win32-x64',
    libName: 'ts_torch.dll',
    buildSubdir: 'Release',
  },
  darwin: {
    packageDir: `@ts-torch-platform/darwin-${process.arch}`,
    libName: 'libts_torch.dylib',
    buildSubdir: '',
  },
  linux: {
    packageDir: `@ts-torch-platform/linux-${process.arch}`,
    libName: 'libts_torch.so',
    buildSubdir: '',
  },
}

function findLibtorchPath(): string {
  // Check common locations
  const possiblePaths = [
    join(ROOT_DIR, 'libtorch', 'libtorch'), // Nested (from zip extraction)
    join(ROOT_DIR, 'libtorch'),
    process.env.LIBTORCH || '',
    process.env.LIBTORCH_PATH || '',
    '/usr/local/lib/libtorch',
    '/opt/libtorch',
    'C:\\libtorch',
  ]

  for (const p of possiblePaths) {
    if (p && existsSync(p) && existsSync(join(p, 'lib'))) {
      return p
    }
  }

  throw new Error(
    'LibTorch not found. Please either:\n' +
      '  1. Place libtorch at the project root: ts-torch/libtorch/\n' +
      '  2. Set LIBTORCH or LIBTORCH_PATH environment variable\n' +
      '  3. Download from https://pytorch.org/get-started/locally/',
  )
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
  const platform = process.platform
  const config = PLATFORM_CONFIG[platform]

  if (!config) {
    throw new Error(`Unsupported platform: ${platform}`)
  }

  console.log('=== Building ts-torch Native Library ===\n')

  // Find LibTorch
  const libtorchPath = findLibtorchPath()
  console.log(`LibTorch found at: ${libtorchPath}`)

  // Create build directory
  if (!existsSync(BUILD_DIR)) {
    mkdirSync(BUILD_DIR, { recursive: true })
  }

  // Configure CMake
  console.log('\n--- Configuring CMake ---')
  runCommand('cmake', ['-B', 'build', `-DCMAKE_PREFIX_PATH=${libtorchPath}`, '-DCMAKE_BUILD_TYPE=Release'], NATIVE_DIR)

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
