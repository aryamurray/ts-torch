/**
 * Native library build utilities for CUDA packages
 * Builds ts_torch native library against CUDA-enabled LibTorch
 */

import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, copyFileSync, writeFileSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { LIBTORCH_VERSION } from './download.js'

/**
 * Build metadata stored in .build-meta.json
 */
export interface BuildMeta {
  cuda: string
  libtorch: string
  platform: string
  builtAt: string
}

/**
 * Platform-specific build configuration
 */
interface BuildConfig {
  libName: string
  buildSubdir: string
}

function getBuildConfig(): BuildConfig {
  switch (process.platform) {
    case 'win32':
      return {
        libName: 'ts_torch.dll',
        buildSubdir: 'Release',
      }
    case 'linux':
      return {
        libName: 'libts_torch.so',
        buildSubdir: '',
      }
    default:
      throw new Error(`CUDA is not supported on ${process.platform}`)
  }
}

/**
 * Run a shell command and throw on failure
 */
function runCommand(command: string, args: string[], cwd: string, env?: NodeJS.ProcessEnv): void {
  console.log(`Running: ${command} ${args.join(' ')}`)
  const result = spawnSync(command, args, {
    cwd,
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: { ...process.env, ...env },
  })

  if (result.status !== 0) {
    throw new Error(`Command failed with exit code ${result.status}`)
  }
}

/**
 * Find the @ts-torch/core native source directory
 */
function findNativeSourceDir(): string {
  // Try to resolve @ts-torch/core package
  try {
    const corePkgPath = require.resolve('@ts-torch/core/package.json')
    const coreDir = corePkgPath.replace(/[/\\]package\.json$/, '')
    const nativeDir = join(coreDir, 'native')

    if (existsSync(join(nativeDir, 'CMakeLists.txt'))) {
      return nativeDir
    }
  } catch {
    // Package not resolved, try relative paths
  }

  // Try relative paths for development
  const possiblePaths = [
    // From CUDA package in workspace
    resolve(__dirname, '..', '..', '..', '@ts-torch', 'core', 'native'),
    // From node_modules
    resolve(process.cwd(), 'node_modules', '@ts-torch', 'core', 'native'),
  ]

  for (const p of possiblePaths) {
    if (existsSync(join(p, 'CMakeLists.txt'))) {
      return p
    }
  }

  throw new Error(
    'Could not find @ts-torch/core native source directory.\n' +
      'Ensure @ts-torch/core is installed as a peer dependency.',
  )
}

/**
 * Read existing build metadata
 */
export function readBuildMeta(metaPath: string): BuildMeta | null {
  if (!existsSync(metaPath)) {
    return null
  }

  try {
    return JSON.parse(readFileSync(metaPath, 'utf-8'))
  } catch {
    return null
  }
}

/**
 * Check if rebuild is needed based on build metadata
 */
export function needsRebuild(metaPath: string, cudaVersion: string): boolean {
  const meta = readBuildMeta(metaPath)

  if (!meta) {
    return true
  }

  if (meta.cuda !== cudaVersion || meta.libtorch !== LIBTORCH_VERSION) {
    console.log(
      `Build mismatch: have ${meta.cuda}/${meta.libtorch}, need ${cudaVersion}/${LIBTORCH_VERSION}`,
    )
    return true
  }

  const expectedPlatform = `${process.platform}-${process.arch}`
  if (meta.platform !== expectedPlatform) {
    console.log(`Platform mismatch: have ${meta.platform}, need ${expectedPlatform}`)
    return true
  }

  return false
}

/**
 * Write build metadata
 */
export function writeBuildMeta(metaPath: string, cudaVersion: string): void {
  const meta: BuildMeta = {
    cuda: cudaVersion,
    libtorch: LIBTORCH_VERSION,
    platform: `${process.platform}-${process.arch}`,
    builtAt: new Date().toISOString(),
  }
  writeFileSync(metaPath, JSON.stringify(meta, null, 2))
}

/**
 * Build the native library against CUDA LibTorch
 *
 * @param libtorchPath - Path to LibTorch directory
 * @param outputDir - Directory to copy built library to
 * @param cudaVersion - CUDA version for metadata
 */
export async function buildNative(
  libtorchPath: string,
  outputDir: string,
  cudaVersion: string,
): Promise<void> {
  const config = getBuildConfig()
  const nativeDir = findNativeSourceDir()
  const buildDir = join(nativeDir, 'build-cuda')

  console.log('\n=== Building ts-torch Native Library (CUDA) ===')
  console.log(`Platform: ${process.platform} (${process.arch})`)
  console.log(`LibTorch: ${libtorchPath}`)
  console.log(`Native source: ${nativeDir}`)

  // Create build directory
  if (!existsSync(buildDir)) {
    mkdirSync(buildDir, { recursive: true })
  }

  // Configure CMake
  console.log('\n--- Configuring CMake ---')
  const cmakePrefixPath = process.platform === 'win32' ? `"${libtorchPath}"` : libtorchPath

  runCommand(
    'cmake',
    ['-B', 'build-cuda', `-DCMAKE_PREFIX_PATH=${cmakePrefixPath}`, '-DCMAKE_BUILD_TYPE=Release'],
    nativeDir,
  )

  // Build
  console.log('\n--- Building ---')
  runCommand('cmake', ['--build', 'build-cuda', '--config', 'Release'], nativeDir)

  // Copy to output directory
  console.log('\n--- Copying to output directory ---')
  const srcLib = join(buildDir, config.buildSubdir, config.libName)
  const destLib = join(outputDir, config.libName)

  if (!existsSync(srcLib)) {
    throw new Error(`Built library not found at: ${srcLib}`)
  }

  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true })
  }

  copyFileSync(srcLib, destLib)
  console.log(`Copied ${config.libName} to ${outputDir}`)

  // Write build metadata
  const metaPath = join(outputDir, '.build-meta.json')
  writeBuildMeta(metaPath, cudaVersion)
  console.log(`Wrote build metadata to ${metaPath}`)

  console.log('\n=== Build complete! ===')
}
