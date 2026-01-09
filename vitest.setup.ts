import { join } from 'path';
import { platform, arch } from 'os';

/**
 * Global setup for Vitest test environment
 * Configures the TS_TORCH_LIB environment variable for FFI compatibility
 */

// Detect platform and architecture
const detectPlatform = (): string => {
  const platformName = platform();
  switch (platformName) {
    case 'win32':
      return 'win32';
    case 'darwin':
      return 'darwin';
    case 'linux':
      return 'linux';
    default:
      throw new Error(`Unsupported platform: ${platformName}`);
  }
};

const detectArch = (): string => {
  const archName = arch();
  switch (archName) {
    case 'x64':
      return 'x64';
    case 'arm64':
      return 'arm64';
    default:
      throw new Error(`Unsupported architecture: ${archName}`);
  }
};

const getLibraryExtension = (platformName: string): string => {
  switch (platformName) {
    case 'win32':
      return 'dll';
    case 'darwin':
      return 'dylib';
    case 'linux':
      return 'so';
    default:
      throw new Error(`Unsupported platform: ${platformName}`);
  }
};

const getLibraryName = (platformName: string): string => {
  return platformName === 'win32' ? 'ts_torch' : 'libts_torch';
};

// Set up TS_TORCH_LIB environment variable if not already set
if (!process.env.TS_TORCH_LIB) {
  const platformName = detectPlatform();
  const archName = detectArch();
  const extension = getLibraryExtension(platformName);
  const libraryName = getLibraryName(platformName);

  // Construct library path relative to project root
  const libraryPath = join(
    process.cwd(),
    'packages',
    '@ts-torch-platform',
    `${platformName}-${archName}`,
    'lib',
    `${libraryName}.${extension}`
  );

  process.env.TS_TORCH_LIB = libraryPath;

  console.log(`[Vitest Setup] Set TS_TORCH_LIB to: ${libraryPath}`);
} else {
  console.log(`[Vitest Setup] Using existing TS_TORCH_LIB: ${process.env.TS_TORCH_LIB}`);
}
