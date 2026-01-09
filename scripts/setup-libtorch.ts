import { existsSync } from "fs";
import { mkdir } from "fs/promises";
import path from "path";
import { execSync } from "child_process";

const LIBTORCH_VERSION = "2.5.1";
const LIBTORCH_DIR = path.resolve("libtorch");
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
  // macOS - check for ARM64 (Apple Silicon)
  if (arch === "arm64") {
    filename = `libtorch-macos-arm64-${LIBTORCH_VERSION}.zip`;
  } else {
    filename = `libtorch-macos-${LIBTORCH_VERSION}.zip`;
  }
} else {
  console.error(`Unsupported platform: ${platform}`);
  process.exit(1);
}

// For local file path, use the non-encoded version (replace %2B with +)
const localFilename = filename.replace("%2B", "+");

const downloadUrl = `${BASE_URL}/${filename}`;
const zipPath = path.join(LIBTORCH_DIR, localFilename);

async function setup() {
  try {
    // Create libtorch directory if it doesn't exist
    if (!existsSync(LIBTORCH_DIR)) {
      await mkdir(LIBTORCH_DIR, { recursive: true });
    }

    // Check if already extracted
    const libDir = path.join(LIBTORCH_DIR, "lib");
    if (!existsSync(libDir)) {
      console.log(`Downloading LibTorch ${LIBTORCH_VERSION} for ${platform} (${arch})...`);
      console.log(`URL: ${downloadUrl}`);

      // Download the file
      execSync(`curl -L -o "${zipPath}" "${downloadUrl}"`, { stdio: "inherit" });

      console.log("Extracting LibTorch...");

      // Extract based on platform
      if (platform === "win32") {
        // Use PowerShell on Windows
        execSync(
          `powershell -Command "Expand-Archive -Path '${zipPath}' -DestinationPath '${LIBTORCH_DIR}' -Force"`,
          { stdio: "inherit" }
        );
      } else {
        // Use unzip on Unix-like systems
        execSync(`unzip -q -o "${zipPath}" -d "${LIBTORCH_DIR}"`, { stdio: "inherit" });
      }

      // Clean up zip file
      execSync(`rm -f "${zipPath}"`, { shell: true });
    } else {
      console.log("LibTorch already extracted, skipping download...");
    }

    console.log(`✓ LibTorch ${LIBTORCH_VERSION} ready at ${LIBTORCH_DIR}`);

    // Build native library using build.sh
    console.log("\nBuilding native library...");
    if (!existsSync(NATIVE_DIR)) {
      console.error(`Native directory not found: ${NATIVE_DIR}`);
      process.exit(1);
    }

    const buildScript = path.join(NATIVE_DIR, "build.sh");
    if (!existsSync(buildScript)) {
      console.error(`Build script not found: ${buildScript}`);
      console.error("Make sure build.sh exists in packages/@ts-torch/core/native/");
      process.exit(1);
    }

    // Set LIBTORCH environment variable and run build script
    const env = { ...process.env, LIBTORCH: LIBTORCH_DIR };

    if (platform === "win32") {
      // On Windows, we need to use the batch or PowerShell version
      // For now, assume build.sh works with bash or we have build.bat
      execSync(`bash "${buildScript}" --libtorch "${LIBTORCH_DIR}" --release`, {
        stdio: "inherit",
        env,
      });
    } else {
      // Linux and macOS
      execSync(`bash "${buildScript}" --libtorch "${LIBTORCH_DIR}" --release`, {
        stdio: "inherit",
        env,
      });
    }

    console.log(`✓ Native library built successfully!`);
  } catch (error) {
    console.error("Failed to setup LibTorch:", error);
    process.exit(1);
  }
}

setup();
