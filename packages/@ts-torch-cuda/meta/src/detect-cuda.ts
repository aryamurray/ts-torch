/**
 * CUDA detection utilities
 * Detects NVIDIA driver version and recommends appropriate CUDA package
 */

import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

export interface CudaInfo {
  available: boolean
  driverVersion?: string // e.g., "550.54.15"
  cudaVersion?: string // e.g., "cu124"
  recommendedPackage?: string
  error?: string
}

/**
 * Find nvidia-smi executable, with Windows PATH fallbacks
 */
function findNvidiaSmi(): string | null {
  // Try nvidia-smi directly (works if in PATH)
  const direct = spawnSync('nvidia-smi', ['--version'], {
    shell: true,
    encoding: 'utf-8',
    timeout: 5000,
  })
  if (direct.status === 0) {
    return 'nvidia-smi'
  }

  if (process.platform === 'win32') {
    // Windows: try known install paths
    const programFiles = process.env.PROGRAMFILES || 'C:\\Program Files'
    const systemRoot = process.env.SYSTEMROOT || 'C:\\Windows'

    const windowsPaths = [
      join(programFiles, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'),
      join(systemRoot, 'System32', 'nvidia-smi.exe'),
      'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
    ]

    for (const p of windowsPaths) {
      if (existsSync(p)) {
        return p
      }
    }

    // Try 'where' command
    const where = spawnSync('where', ['nvidia-smi'], {
      shell: true,
      encoding: 'utf-8',
      timeout: 5000,
    })
    if (where.status === 0 && where.stdout.trim()) {
      return where.stdout.trim().split('\n')[0].trim()
    }
  }

  return null
}

/**
 * Map NVIDIA driver version to CUDA toolkit compatibility
 *
 * Based on NVIDIA's CUDA compatibility table:
 * https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
 */
function mapDriverToCuda(driverMajor: number): string {
  // NVIDIA driver 550.x → CUDA 12.4
  // NVIDIA driver 535.x → CUDA 12.1
  // NVIDIA driver 520.x → CUDA 11.8
  if (driverMajor >= 550) {
    return 'cu124'
  } else if (driverMajor >= 535) {
    return 'cu121'
  } else if (driverMajor >= 520) {
    return 'cu118'
  } else {
    // Older drivers - try cu118 as fallback
    return 'cu118'
  }
}

/**
 * Detect CUDA availability and recommended package
 */
export function detectCuda(): CudaInfo {
  // macOS doesn't support CUDA
  if (process.platform === 'darwin') {
    return {
      available: false,
      error: 'CUDA is not supported on macOS. Only Linux and Windows are supported.',
    }
  }

  const nvidiaSmi = findNvidiaSmi()

  if (!nvidiaSmi) {
    return {
      available: false,
      error:
        process.platform === 'win32'
          ? 'nvidia-smi not found. Ensure NVIDIA drivers are installed.\n' +
            'Download from: https://www.nvidia.com/drivers'
          : 'nvidia-smi not found. Install NVIDIA drivers:\n' +
            '  Ubuntu: sudo apt install nvidia-driver-xxx\n' +
            '  Fedora: sudo dnf install akmod-nvidia',
    }
  }

  // Query driver version
  const result = spawnSync(
    nvidiaSmi,
    ['--query-gpu=driver_version', '--format=csv,noheader'],
    {
      shell: process.platform === 'win32',
      encoding: 'utf-8',
      timeout: 10000,
    },
  )

  if (result.status !== 0 || !result.stdout) {
    return {
      available: false,
      error: 'nvidia-smi found but failed to query GPU. Is an NVIDIA GPU installed?',
    }
  }

  const driverVersion = result.stdout.trim() // e.g., "550.54.15"
  const majorVersion = parseInt(driverVersion.split('.')[0])

  if (isNaN(majorVersion)) {
    return {
      available: false,
      error: `Could not parse driver version: ${driverVersion}`,
    }
  }

  const cudaVersion = mapDriverToCuda(majorVersion)
  const platform = process.platform === 'win32' ? 'win32' : 'linux'
  const recommendedPackage = `@ts-torch-cuda/${platform}-x64-${cudaVersion}`

  return {
    available: true,
    driverVersion,
    cudaVersion,
    recommendedPackage,
  }
}
