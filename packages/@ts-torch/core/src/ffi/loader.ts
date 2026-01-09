/**
 * Native library loader for ts-torch
 * Handles platform detection, library resolution, and FFI initialization
 */

import { dlopen, type Library, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { FFI_SYMBOLS, type FFISymbols } from "./symbols.js";

/**
 * Cached library instance
 */
let libInstance: Library<FFISymbols> | null = null;

/**
 * Platform-specific library information
 */
interface PlatformInfo {
  packageName: string;
  libraryName: string;
}

/**
 * Get platform-specific package name and library name
 * Maps Node.js process.platform and process.arch to native package names
 */
export function getPlatformPackage(): PlatformInfo {
  const platform = process.platform;
  const arch = process.arch;

  // Map platform/arch to package names
  switch (platform) {
    case "darwin":
      switch (arch) {
        case "arm64":
          return {
            packageName: "@ts-torch/darwin-arm64",
            libraryName: "libts_torch",
          };
        case "x64":
          return {
            packageName: "@ts-torch/darwin-x64",
            libraryName: "libts_torch",
          };
        default:
          throw new Error(`Unsupported macOS architecture: ${arch}`);
      }

    case "linux":
      switch (arch) {
        case "x64":
          return {
            packageName: "@ts-torch/linux-x64",
            libraryName: "libts_torch",
          };
        case "arm64":
          return {
            packageName: "@ts-torch/linux-arm64",
            libraryName: "libts_torch",
          };
        default:
          throw new Error(`Unsupported Linux architecture: ${arch}`);
      }

    case "win32":
      switch (arch) {
        case "x64":
          return {
            packageName: "@ts-torch/win32-x64",
            libraryName: "ts_torch",
          };
        case "arm64":
          return {
            packageName: "@ts-torch/win32-arm64",
            libraryName: "ts_torch",
          };
        default:
          throw new Error(`Unsupported Windows architecture: ${arch}`);
      }

    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

/**
 * Get the full path to the native library
 * Resolution order:
 * 1. TS_TORCH_LIB environment variable (for custom builds)
 * 2. Platform-specific package (production)
 * 3. Local development paths (workspace monorepo)
 *
 * @throws Error if library cannot be found
 */
export function getLibraryPath(): string {
  // 1. Check environment variable override
  const envPath = process.env.TS_TORCH_LIB;
  if (envPath) {
    if (existsSync(envPath)) {
      return resolve(envPath);
    }
    console.warn(`TS_TORCH_LIB set but file not found: ${envPath}`);
  }

  const { packageName, libraryName } = getPlatformPackage();
  const libFileName = `${libraryName}.${suffix}`;

  // 2. Try to resolve from installed platform package
  try {
    // Resolve the platform package's package.json
    const packagePath = require.resolve(`${packageName}/package.json`);
    const packageDir = packagePath.replace(/\/package\.json$/, "");
    const libPath = join(packageDir, libFileName);

    if (existsSync(libPath)) {
      return libPath;
    }
  } catch (err) {
    // Package not found, try development paths
  }

  // 3. Try local development paths (workspace monorepo structure)
  const cwd = process.cwd();
  const possiblePaths = [
    // Current package's native directory
    join(cwd, "native", "target", "release", libFileName),
    join(cwd, "native", "target", "debug", libFileName),

    // Workspace root native directory
    join(cwd, "..", "..", "..", "native", "target", "release", libFileName),
    join(cwd, "..", "..", "..", "native", "target", "debug", libFileName),

    // Platform package in workspace
    join(cwd, "..", packageName, libFileName),
    join(cwd, "..", "..", packageName, libFileName),
  ];

  for (const path of possiblePaths) {
    if (existsSync(path)) {
      return resolve(path);
    }
  }

  // Library not found
  throw new Error(
    `Could not find ts-torch native library for ${process.platform}-${process.arch}.\n` +
      `Searched paths:\n${possiblePaths.map((p) => `  - ${p}`).join("\n")}\n\n` +
      `Please ensure the platform-specific package is installed:\n` +
      `  bun add ${packageName}\n\n` +
      `Or build from source:\n` +
      `  cd native && cargo build --release\n\n` +
      `Or set TS_TORCH_LIB environment variable to the library path.`
  );
}

/**
 * Load the native library and return FFI bindings
 * Uses lazy loading and caching for performance
 *
 * @returns Library instance with typed FFI symbols
 * @throws Error if library cannot be loaded
 */
export function getLib(): Library<FFISymbols> {
  if (libInstance !== null) {
    return libInstance;
  }

  const libraryPath = getLibraryPath();

  try {
    libInstance = dlopen(libraryPath, FFI_SYMBOLS);
    return libInstance;
  } catch (err) {
    throw new Error(
      `Failed to load ts-torch native library from: ${libraryPath}\n` +
        `Error: ${err instanceof Error ? err.message : String(err)}\n\n` +
        `This may indicate:\n` +
        `  - Library architecture mismatch\n` +
        `  - Missing system dependencies (libtorch, CUDA, etc.)\n` +
        `  - Corrupted library file\n\n` +
        `Try rebuilding the native library:\n` +
        `  cd native && cargo clean && cargo build --release`
    );
  }
}

/**
 * Close the library and release resources
 * Should be called on process exit or when library is no longer needed
 */
export function closeLib(): void {
  if (libInstance !== null) {
    libInstance.close();
    libInstance = null;
  }
}

/**
 * Auto-cleanup on process exit
 */
if (typeof process !== "undefined") {
  process.on("exit", () => {
    closeLib();
  });
}
