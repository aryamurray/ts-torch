#!/usr/bin/env bun
/**
 * Postinstall script for @ts-torch-cuda/win32-x64-cu121
 * Downloads CUDA 12.1 LibTorch and builds the native library
 */

import { existsSync, readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import {
  checkPrerequisites,
  downloadLibTorch,
  buildNative,
  needsRebuild,
} from '@ts-torch-cuda/shared'

const CUDA_VERSION = 'cu121'
const PACKAGE_DIR = resolve(import.meta.dirname, '..')
const LIB_DIR = join(PACKAGE_DIR, 'lib')
const META_FILE = join(LIB_DIR, '.build-meta.json')

/**
 * Check if we're in a development/workspace environment
 */
function isDevEnvironment(): boolean {
  const parentPkg = join(PACKAGE_DIR, '..', '..', '..', 'package.json')
  if (existsSync(parentPkg)) {
    try {
      const content = readFileSync(parentPkg, 'utf-8')
      const pkg = JSON.parse(content)
      if (pkg.workspaces) return true
    } catch {}
  }
  return false
}

async function postinstall(): Promise<void> {
  if (isDevEnvironment()) {
    console.log(`Skipping CUDA ${CUDA_VERSION} postinstall in development environment.`)
    return
  }

  console.log(`\n=== Setting up ts-torch CUDA ${CUDA_VERSION} ===\n`)

  // 1. Check prerequisites first (fail fast)
  checkPrerequisites()

  // 2. Check if rebuild is needed
  const libPath = join(LIB_DIR, 'ts_torch.dll')
  if (existsSync(libPath) && !needsRebuild(META_FILE, CUDA_VERSION)) {
    console.log(`\nCUDA ${CUDA_VERSION} library already built, skipping.`)
    return
  }

  // 3. Download CUDA LibTorch (~2.5GB)
  console.log('\nDownloading CUDA LibTorch...')
  const libtorchPath = await downloadLibTorch(CUDA_VERSION, LIB_DIR)

  // 4. Build native library against CUDA LibTorch
  console.log('\nBuilding native library...')
  await buildNative(libtorchPath, LIB_DIR, CUDA_VERSION)

  console.log(`\n=== CUDA ${CUDA_VERSION} setup complete! ===\n`)
}

postinstall().catch((err) => {
  console.error('\nCUDA setup failed:', err.message)
  process.exit(1)
})
