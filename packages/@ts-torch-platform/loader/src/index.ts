/**
 * @ts-torch-platform/loader - Platform detection and native binary loading
 *
 * This package detects the current platform and loads the appropriate
 * native binaries for ts-torch operations.
 */

import { platform, arch } from 'node:os'
import { join } from 'node:path'

/**
 * Supported platform identifiers
 */
export type PlatformIdentifier =
  | 'win32-x64'
  | 'win32-arm64'
  | 'darwin-x64'
  | 'darwin-arm64'
  | 'linux-x64'
  | 'linux-arm64'

/**
 * Platform information
 */
export interface PlatformInfo {
  platform: string
  arch: string
  identifier: PlatformIdentifier
  packageName: string
}

/**
 * Get the current platform identifier
 */
export function getPlatformIdentifier(): PlatformIdentifier {
  const plat = platform()
  const architecture = arch()

  const identifier = `${plat}-${architecture}` as PlatformIdentifier

  // Validate supported platforms
  const supported: PlatformIdentifier[] = [
    'win32-x64',
    'win32-arm64',
    'darwin-x64',
    'darwin-arm64',
    'linux-x64',
    'linux-arm64',
  ]

  if (!supported.includes(identifier)) {
    throw new Error(`Unsupported platform: ${identifier}. Supported platforms: ${supported.join(', ')}`)
  }

  return identifier
}

/**
 * Get platform information
 */
export function getPlatformInfo(): PlatformInfo {
  const plat = platform()
  const architecture = arch()
  const identifier = getPlatformIdentifier()
  const packageName = `@ts-torch-platform/${identifier}`

  return {
    platform: plat,
    arch: architecture,
    identifier,
    packageName,
  }
}

/**
 * Get the package name for the current platform
 */
export function getPlatformPackageName(): string {
  const identifier = getPlatformIdentifier()
  return `@ts-torch-platform/${identifier}`
}

/**
 * Try to load the native binary for the current platform
 */
export function loadNativeBinary(): string | null {
  try {
    const packageName = getPlatformPackageName()

    // Try to resolve the platform package
    try {
      const resolved = require.resolve(`${packageName}/package.json`)
      const packageDir = join(resolved, '..')
      const binaryPath = join(packageDir, 'lib', 'ts_torch.node')

      return binaryPath
    } catch (resolveError) {
      console.warn(`Platform package ${packageName} not found: ${resolveError}. Native operations will not be available.`)
      return null
    }
  } catch (error) {
    console.error('Error loading native binary:', error)
    return null
  }
}

/**
 * Check if native binaries are available
 */
export function isNativeAvailable(): boolean {
  return loadNativeBinary() !== null
}

/**
 * Get detailed error information for missing native binaries
 */
export function getMissingBinaryInfo(): string {
  const info = getPlatformInfo()

  return `
Native binaries not found for your platform.

Platform: ${info.platform}
Architecture: ${info.arch}
Required package: ${info.packageName}

To install the required binaries, run:
  bun add ${info.packageName}

If binaries are not available for your platform, ts-torch will fall back
to JavaScript implementations (slower performance).
`.trim()
}

/**
 * Load native binary or throw error with helpful message
 */
export function loadNativeBinaryOrThrow(): string {
  const binaryPath = loadNativeBinary()

  if (!binaryPath) {
    throw new Error(getMissingBinaryInfo())
  }

  return binaryPath
}

/**
 * Validate platform support at runtime
 */
export function validatePlatform(): void {
  try {
    getPlatformIdentifier()
  } catch (error) {
    console.error('Platform validation failed:', error)
    throw error
  }
}
