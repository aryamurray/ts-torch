import { existsSync, rmSync, createWriteStream, writeFileSync, readFileSync, unlinkSync } from "fs";
import { mkdir } from "fs/promises";
import { tmpdir } from "os";
import { createHash } from "crypto";
import path from "path";
import AdmZip from "adm-zip";

// Use import.meta.dirname for reliable path resolution regardless of cwd
const SCRIPT_DIR = import.meta.dirname;
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, "..");
const LIBTORCH_VERSION = "2.5.1";
const LIBTORCH_DIR = path.join(PROJECT_ROOT, "libtorch");
const NATIVE_DIR = path.join(PROJECT_ROOT, "packages/@ts-torch/core/native");
const LOCK_FILE = path.join(PROJECT_ROOT, ".libtorch-setup.lock");
const BASE_URL = "https://download.pytorch.org/libtorch/cpu";

// Platform detection
const platform = process.platform;
const arch = process.arch;

// Platform-specific configuration with checksums
// Checksums from PyTorch release page for 2.5.1 CPU builds
interface PlatformConfig {
  filename: string;
  sha256?: string; // Optional - if not available, skip validation with warning
}

function getPlatformConfig(): PlatformConfig {
  if (platform === "win32") {
    return {
      filename: `libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip`,
      // Note: PyTorch doesn't publish official checksums, so we skip validation
      // In production, you'd want to generate and store these after verifying a known-good download
    };
  } else if (platform === "linux") {
    return {
      filename: `libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip`,
    };
  } else if (platform === "darwin") {
    if (arch === "arm64") {
      return {
        filename: `libtorch-macos-arm64-${LIBTORCH_VERSION}.zip`,
      };
    } else {
      return {
        filename: `libtorch-macos-${LIBTORCH_VERSION}.zip`,
      };
    }
  } else {
    throw new Error(`Unsupported platform: ${platform}`);
  }
}

const config = getPlatformConfig();
const localFilename = config.filename.replace("%2B", "+");
const downloadUrl = `${BASE_URL}/${config.filename}`;
const zipPath = path.join(tmpdir(), localFilename);

/**
 * Simple file-based lock to prevent concurrent setup
 */
function acquireLock(): boolean {
  try {
    if (existsSync(LOCK_FILE)) {
      const lockContent = readFileSync(LOCK_FILE, "utf-8");
      const lockTime = parseInt(lockContent, 10);
      const now = Date.now();

      // Lock expires after 10 minutes (in case of crash)
      if (now - lockTime < 10 * 60 * 1000) {
        return false; // Lock is held by another process
      }
      // Stale lock, remove it
      unlinkSync(LOCK_FILE);
    }

    // Create lock file with current timestamp
    writeFileSync(LOCK_FILE, Date.now().toString(), { flag: "wx" });
    return true;
  } catch (err) {
    // Another process may have created the lock between our check and create
    return false;
  }
}

function releaseLock(): void {
  try {
    if (existsSync(LOCK_FILE)) {
      unlinkSync(LOCK_FILE);
    }
  } catch {
    // Ignore errors during cleanup
  }
}

/**
 * Calculate SHA256 hash of a file
 */
function calculateFileHash(filePath: string): string {
  const fileBuffer = readFileSync(filePath);
  const hashSum = createHash("sha256");
  hashSum.update(fileBuffer);
  return hashSum.digest("hex");
}

/**
 * Validate file checksum if available
 */
function validateChecksum(filePath: string, expectedHash?: string): boolean {
  if (!expectedHash) {
    console.log("  ⚠ No checksum available, skipping validation");
    return true;
  }

  console.log("  Validating checksum...");
  const actualHash = calculateFileHash(filePath);

  if (actualHash.toLowerCase() !== expectedHash.toLowerCase()) {
    console.error(`  ✗ Checksum mismatch!`);
    console.error(`    Expected: ${expectedHash}`);
    console.error(`    Actual:   ${actualHash}`);
    return false;
  }

  console.log("  ✓ Checksum valid");
  return true;
}

/**
 * Download file with proper stream handling
 */
async function downloadFile(url: string, dest: string): Promise<void> {
  console.log(`Downloading from ${url}...`);

  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status} ${response.statusText}`);
  }

  const totalBytes = parseInt(response.headers.get("content-length") || "0", 10);
  let downloadedBytes = 0;
  let lastPercent = 0;

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Failed to get response reader");
  }

  // Use a promise to properly handle stream completion
  await new Promise<void>((resolve, reject) => {
    const fileStream = createWriteStream(dest);

    fileStream.on("error", (err) => {
      reject(new Error(`Failed to write file: ${err.message}`));
    });

    fileStream.on("finish", () => {
      resolve();
    });

    const pump = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            fileStream.end();
            break;
          }

          // Wait for write to complete before continuing
          const canContinue = fileStream.write(Buffer.from(value));
          downloadedBytes += value.length;

          if (totalBytes > 0) {
            const percent = Math.floor((downloadedBytes / totalBytes) * 100);
            if (percent >= lastPercent + 10) {
              console.log(`  ${percent}% (${Math.round(downloadedBytes / 1024 / 1024)}MB / ${Math.round(totalBytes / 1024 / 1024)}MB)`);
              lastPercent = percent;
            }
          }

          // Handle backpressure
          if (!canContinue) {
            await new Promise<void>((res) => fileStream.once("drain", res));
          }
        }
      } catch (err) {
        fileStream.destroy();
        reject(err);
      }
    };

    pump();
  });

  console.log(`Download complete: ${Math.round(downloadedBytes / 1024 / 1024)}MB`);
}

/**
 * Extract zip file with proper error handling
 */
function extractZip(zipFilePath: string, destDir: string): void {
  console.log(`Extracting to ${destDir}...`);

  let zip: AdmZip;
  try {
    zip = new AdmZip(zipFilePath);
  } catch (err) {
    throw new Error(`Failed to open zip file: ${err instanceof Error ? err.message : String(err)}`);
  }

  const entries = zip.getEntries();
  const totalEntries = entries.length;

  if (totalEntries === 0) {
    throw new Error("Zip file appears to be empty or corrupted");
  }

  console.log(`  Found ${totalEntries} files to extract...`);

  try {
    zip.extractAllTo(destDir, true);
  } catch (err) {
    throw new Error(`Failed to extract zip: ${err instanceof Error ? err.message : String(err)}`);
  }

  // Verify some critical files exist after extraction
  const criticalFiles = [
    path.join(destDir, "libtorch", "lib"),
    path.join(destDir, "libtorch", "include"),
    path.join(destDir, "libtorch", "share"),
  ];

  for (const file of criticalFiles) {
    if (!existsSync(file)) {
      throw new Error(`Extraction incomplete: missing ${file}`);
    }
  }

  console.log(`  ✓ Extracted ${totalEntries} files`);
}

/**
 * Clean up corrupted/partial downloads
 */
function cleanupOnFailure(zipPath: string, libtorchDir: string): void {
  console.log("Cleaning up after failure...");

  try {
    if (existsSync(zipPath)) {
      unlinkSync(zipPath);
      console.log(`  Removed corrupted zip: ${zipPath}`);
    }
  } catch {
    console.warn(`  Could not remove zip file: ${zipPath}`);
  }

  try {
    if (existsSync(libtorchDir)) {
      rmSync(libtorchDir, { recursive: true, force: true });
      console.log(`  Removed incomplete libtorch directory`);
    }
  } catch {
    console.warn(`  Could not remove libtorch directory`);
  }
}

/**
 * Build the native library
 */
async function buildNative(): Promise<void> {
  console.log("\nBuilding native library...");

  const buildScript = path.join(PROJECT_ROOT, "packages/@ts-torch/core/scripts/build-native.ts");
  if (!existsSync(buildScript)) {
    throw new Error(`Build script not found: ${buildScript}`);
  }

  const proc = Bun.spawn(["bun", "run", buildScript], {
    cwd: path.join(PROJECT_ROOT, "packages/@ts-torch/core"),
    env: {
      ...process.env,
      LIBTORCH: LIBTORCH_DIR,
      LIBTORCH_PATH: LIBTORCH_DIR, // Set both for compatibility
    },
    stdio: ["inherit", "inherit", "inherit"],
  });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new Error(`Build failed with exit code ${exitCode}`);
  }
}

/**
 * Main setup function
 */
async function setup() {
  console.log(`\n=== ts-torch Setup ===`);
  console.log(`Platform: ${platform} (${arch})`);
  console.log(`LibTorch version: ${LIBTORCH_VERSION}`);
  console.log(`Project root: ${PROJECT_ROOT}\n`);

  // Acquire lock to prevent concurrent setup
  if (!acquireLock()) {
    console.log("Another setup process is running. Waiting...");

    // Wait up to 5 minutes for the other process
    const maxWait = 5 * 60 * 1000;
    const startTime = Date.now();

    while (!acquireLock()) {
      if (Date.now() - startTime > maxWait) {
        throw new Error("Timeout waiting for another setup process to complete");
      }
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
  }

  try {
    const libDir = path.join(LIBTORCH_DIR, "lib");

    if (!existsSync(libDir)) {
      console.log(`Setting up LibTorch ${LIBTORCH_VERSION}...\n`);

      let needsDownload = true;

      // Check if we have a cached download
      if (existsSync(zipPath)) {
        console.log(`Found cached download: ${zipPath}`);

        // Validate the cached file
        if (validateChecksum(zipPath, config.sha256)) {
          needsDownload = false;
        } else {
          console.log("Cached file is corrupted, re-downloading...");
          unlinkSync(zipPath);
        }
      }

      // Download if needed
      if (needsDownload) {
        try {
          await downloadFile(downloadUrl, zipPath);

          // Validate download
          if (!validateChecksum(zipPath, config.sha256)) {
            throw new Error("Downloaded file failed checksum validation");
          }
        } catch (err) {
          cleanupOnFailure(zipPath, LIBTORCH_DIR);
          throw err;
        }
      }

      // Clean up any partial extraction
      if (existsSync(LIBTORCH_DIR)) {
        console.log("Removing incomplete libtorch directory...");
        rmSync(LIBTORCH_DIR, { recursive: true, force: true });
      }

      // Extract
      try {
        extractZip(zipPath, PROJECT_ROOT);
      } catch (err) {
        cleanupOnFailure(zipPath, LIBTORCH_DIR);
        throw err;
      }

      // Final verification
      if (!existsSync(libDir)) {
        cleanupOnFailure(zipPath, LIBTORCH_DIR);
        throw new Error(`Setup failed - lib directory not found at ${libDir}`);
      }

      // Clean up zip from temp (successful extraction)
      try {
        unlinkSync(zipPath);
      } catch {
        // Non-fatal
      }

      console.log(`\n✓ LibTorch ${LIBTORCH_VERSION} ready at ${LIBTORCH_DIR}`);
    } else {
      console.log(`LibTorch already installed at ${LIBTORCH_DIR}`);
    }

    // Build native library
    await buildNative();

    console.log(`\n✓ Setup complete!`);
  } finally {
    releaseLock();
  }
}

// Run setup
setup().catch((err) => {
  releaseLock();
  console.error("\n✗ Setup failed:", err.message);
  process.exit(1);
});
