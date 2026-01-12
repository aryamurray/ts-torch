#!/usr/bin/env bun
/**
 * ts-torch-cuda CLI
 * Detects CUDA driver and helps install the appropriate CUDA package
 */

import { detectCuda } from './detect-cuda.js'
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join, dirname } from 'node:path'

const command = process.argv[2]

/**
 * Detect package manager from lockfiles
 * Walks up the directory tree to find lockfiles
 */
function detectPackageManager(): string {
  let current = process.cwd()
  const root = dirname(current)

  // Walk up directory tree until we find a lockfile or hit root
  while (current !== root) {
    if (existsSync(join(current, 'bun.lockb'))) return 'bun'
    if (existsSync(join(current, 'pnpm-lock.yaml'))) return 'pnpm'
    if (existsSync(join(current, 'yarn.lock'))) return 'yarn'
    if (existsSync(join(current, 'package-lock.json'))) return 'npm'

    const parent = dirname(current)
    if (parent === current) break // Reached filesystem root
    current = parent
  }

  return 'npm' // Default to npm if no lockfile found
}

/**
 * Print CUDA detection results
 */
function printDetection() {
  const cuda = detectCuda()

  if (!cuda.available) {
    console.error('CUDA detection failed.')
    if (cuda.error) {
      console.error(cuda.error)
    }
    console.log('\nFor CPU-only usage, just use @ts-torch/core without CUDA packages.')
    process.exit(1)
  }

  const pm = detectPackageManager()
  const installCmd = pm === 'npm' ? 'install' : 'add'

  console.log(`Detected NVIDIA driver: ${cuda.driverVersion}`)
  console.log(`CUDA compatibility: ${cuda.cudaVersion}`)
  console.log(`Recommended package: ${cuda.recommendedPackage}`)
  console.log(`\nTo install:\n  ${pm} ${installCmd} ${cuda.recommendedPackage}`)

  return cuda
}

/**
 * Print help message
 */
function printHelp() {
  console.log('ts-torch-cuda - CUDA detection and setup')
  console.log('')
  console.log('Commands:')
  console.log('  detect   Show detected CUDA version and recommended package')
  console.log('  install  Detect and install the appropriate CUDA package')
  console.log('')
  console.log('Examples:')
  console.log('  npx ts-torch-cuda detect')
  console.log('  npx ts-torch-cuda install')
  console.log('')
  console.log('Supported CUDA versions:')
  console.log('  cu118 - CUDA 11.8 (NVIDIA driver 520+)')
  console.log('  cu121 - CUDA 12.1 (NVIDIA driver 535+)')
  console.log('  cu124 - CUDA 12.4 (NVIDIA driver 550+)')
}

// Main command handler
switch (command) {
  case 'detect':
    printDetection()
    break

  case 'install': {
    const cuda = printDetection()
    if (!cuda.available) process.exit(1)

    console.log('\nInstalling...')
    const pm = detectPackageManager()
    const installCmd = pm === 'npm' ? 'install' : 'add'

    const result = spawnSync(pm, [installCmd, cuda.recommendedPackage!], {
      stdio: 'inherit',
      shell: true,
    })

    if (result.status !== 0) {
      console.error(`\nFailed to install ${cuda.recommendedPackage}`)
      process.exit(1)
    }

    console.log(`\n✓ CUDA support installed!`)
    break
  }

  case 'help':
  case '--help':
  case '-h':
    printHelp()
    break

  default:
    if (command) {
      console.error(`Unknown command: ${command}\n`)
    }
    printHelp()
    process.exit(command ? 1 : 0)
}
