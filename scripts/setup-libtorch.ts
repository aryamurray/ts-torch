import { existsSync, rmSync, createWriteStream } from "fs";
import { mkdir } from "fs/promises";
import { tmpdir } from "os";
import path from "path";
import AdmZip from "adm-zip";

const LIBTORCH_VERSION = "2.5.1";
const PROJECT_ROOT = path.resolve(".");
const LIBTORCH_DIR = path.join(PROJECT_ROOT, "libtorch");
const NATIVE_DIR = path.resolve("packages/@ts-torch/core/native");
const BASE_URL = "https://download.pytorch.org/libtorch/cpu";

// Determine platform and architecture
const platform = process.platform;
const arch = process.arch;

let filename: string;

if (platform === "win32") {
  filename = `libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip`;
} else if (platform === "linux") {
  filename = `libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip`;
} else if (platform === "darwin") {
  if (arch === "arm64") {
    filename = `libtorch-macos-arm64-${LIBTORCH_VERSION}.zip`;
  } else {
    filename = `libtorch-macos-${LIBTORCH_VERSION}.zip`;
  }
} else {
  console.error(`Unsupported platform: ${platform}`);
  process.exit(1);
}

const localFilename = filename.replace("%2B", "+");
const downloadUrl = `${BASE_URL}/${filename}`;
const zipPath = path.join(tmpdir(), localFilename);

async function downloadFile(url: string, dest: string): Promise<void> {
  console.log(`Downloading from ${url}...`);

  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status} ${response.statusText}`);
  }

  const totalBytes = parseInt(response.headers.get("content-length") || "0", 10);
  let downloadedBytes = 0;
  let lastPercent = 0;

  const fileStream = createWriteStream(dest);
  const reader = response.body?.getReader();

  if (!reader) {
    throw new Error("Failed to get response reader");
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    fileStream.write(Buffer.from(value));
    downloadedBytes += value.length;

    if (totalBytes > 0) {
      const percent = Math.floor((downloadedBytes / totalBytes) * 100);
      if (percent >= lastPercent + 10) {
        console.log(`  ${percent}% (${Math.round(downloadedBytes / 1024 / 1024)}MB / ${Math.round(totalBytes / 1024 / 1024)}MB)`);
        lastPercent = percent;
      }
    }
  }

  fileStream.close();
  console.log(`Download complete: ${Math.round(downloadedBytes / 1024 / 1024)}MB`);
}

function extractZip(zipFilePath: string, destDir: string): void {
  console.log(`Extracting to ${destDir}...`);

  const zip = new AdmZip(zipFilePath);
  const entries = zip.getEntries();
  const totalEntries = entries.length;
  let extracted = 0;
  let lastPercent = 0;

  zip.extractAllTo(destDir, true);

  console.log(`Extracted ${totalEntries} files`);
}

async function buildNative(): Promise<void> {
  console.log("\nBuilding native library...");

  const buildScript = path.join(NATIVE_DIR, "..", "scripts", "build-native.ts");
  if (!existsSync(buildScript)) {
    throw new Error(`Build script not found: ${buildScript}`);
  }

  const proc = Bun.spawn(["bun", "run", buildScript], {
    cwd: path.resolve("packages/@ts-torch/core"),
    env: { ...process.env, LIBTORCH: LIBTORCH_DIR },
    stdio: ["inherit", "inherit", "inherit"],
  });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    throw new Error(`Build failed with exit code ${exitCode}`);
  }
}

async function setup() {
  try {
    const libDir = path.join(LIBTORCH_DIR, "lib");

    if (!existsSync(libDir)) {
      console.log(`\n=== Setting up LibTorch ${LIBTORCH_VERSION} for ${platform} (${arch}) ===\n`);

      // Download if not already in temp
      if (!existsSync(zipPath)) {
        await downloadFile(downloadUrl, zipPath);
      } else {
        console.log(`Using cached download: ${zipPath}`);
      }

      // Clean up any partial extraction
      if (existsSync(LIBTORCH_DIR)) {
        console.log("Removing incomplete libtorch directory...");
        rmSync(LIBTORCH_DIR, { recursive: true, force: true });
      }

      // Extract using adm-zip (pure JS, cross-platform)
      extractZip(zipPath, PROJECT_ROOT);

      // Verify extraction
      if (!existsSync(libDir)) {
        throw new Error(`Extraction failed - lib directory not found at ${libDir}`);
      }

      // Clean up zip from temp
      rmSync(zipPath, { force: true });

      console.log(`\n✓ LibTorch ${LIBTORCH_VERSION} ready at ${LIBTORCH_DIR}`);
    } else {
      console.log(`LibTorch already installed at ${LIBTORCH_DIR}`);
    }

    // Build native library
    await buildNative();

    console.log(`\n✓ Setup complete!`);
  } catch (error) {
    console.error("\nSetup failed:", error);
    process.exit(1);
  }
}

setup();
