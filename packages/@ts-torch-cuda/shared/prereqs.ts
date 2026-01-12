/**
 * Prerequisite checks for building ts-torch CUDA packages
 * Verifies CMake and C++ compiler are available before attempting build
 */

import { spawnSync } from 'node:child_process'

// Timeout for command execution (10 seconds should be plenty for version checks)
const COMMAND_TIMEOUT_MS = 10000

/**
 * Check if CMake is available
 */
function checkCMake(): void {
  const result = spawnSync('cmake', ['--version'], {
    shell: true,
    encoding: 'utf-8',
    timeout: COMMAND_TIMEOUT_MS,
  })

  if (result.error) {
    if (result.error.message.includes('ETIMEDOUT') || result.error.message.includes('SIGTERM')) {
      throw new Error('CMake check timed out. Your system may be under heavy load.')
    }
  }

  if (result.status !== 0) {
    throw new Error(
      'CMake not found. Please install CMake:\n' +
        '  - Windows: winget install Kitware.CMake\n' +
        '  - Ubuntu: sudo apt install cmake\n' +
        '  - Fedora: sudo dnf install cmake\n' +
        '  - Or download from https://cmake.org/download/',
    )
  }
}

/**
 * Check if a C++ compiler is available
 */
function checkCompiler(): void {
  if (process.platform === 'win32') {
    // Check for MSVC using 'where cl' which is more reliable
    const whereResult = spawnSync('where', ['cl'], {
      shell: true,
      encoding: 'utf-8',
      timeout: COMMAND_TIMEOUT_MS,
    })

    if (whereResult.status === 0 && whereResult.stdout.trim()) {
      return // MSVC is available (cl.exe found in PATH)
    }

    // Fallback: try running cl directly
    const clResult = spawnSync('cl', [], {
      shell: true,
      encoding: 'utf-8',
      timeout: COMMAND_TIMEOUT_MS,
    })

    // cl.exe outputs version info to stderr when run with no args
    if (clResult.stderr && clResult.stderr.includes('Microsoft')) {
      return // MSVC is available
    }

    throw new Error(
      'MSVC not found. Please install Visual Studio Build Tools:\n' +
        '  winget install Microsoft.VisualStudio.2022.BuildTools\n' +
        '  Or download from https://visualstudio.microsoft.com/downloads/\n\n' +
        'After installation, run this command from a "Developer Command Prompt for VS".',
    )
  } else {
    // Check for g++ on Linux
    const result = spawnSync('g++', ['--version'], {
      shell: true,
      encoding: 'utf-8',
      timeout: COMMAND_TIMEOUT_MS,
    })

    if (result.error) {
      if (result.error.message.includes('ETIMEDOUT') || result.error.message.includes('SIGTERM')) {
        throw new Error('Compiler check timed out. Your system may be under heavy load.')
      }
    }

    if (result.status !== 0) {
      throw new Error(
        'g++ not found. Please install build tools:\n' +
          '  Ubuntu: sudo apt install build-essential\n' +
          '  Fedora: sudo dnf install gcc-c++\n' +
          '  Arch: sudo pacman -S base-devel',
      )
    }
  }
}

/**
 * Check all prerequisites for building CUDA packages
 * Throws an error with helpful message if any prerequisite is missing
 */
export function checkPrerequisites(): void {
  console.log('Checking build prerequisites...')

  checkCMake()
  console.log('  CMake found')

  checkCompiler()
  console.log('  C++ compiler found')

  console.log('All prerequisites satisfied.')
}
