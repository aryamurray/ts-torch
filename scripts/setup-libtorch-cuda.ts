import {
  existsSync,
  rmSync,
  createWriteStream,
  writeFileSync,
  readFileSync,
  unlinkSync,
} from 'fs'
import { mkdir } from 'fs/promises'
import { tmpdir } from 'os'
import { createHash } from 'crypto'
import path from 'path'
import AdmZip from 'adm-zip'
import { LIBTORCH_VERSION, checkPrerequisites } from '@ts-torch-cuda/shared'
import { detectCuda } from '../packages/@ts-torch-cuda/meta/src/detect-cuda.js'

// Use import.meta.dirname for reliable path resolution regardless of cwd
const SCRIPT_DIR = import.meta.dirname
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, '..')
const LIBTORCH_CUDA_DIR = path.join(PROJECT_ROOT, 'libtorch-cuda')
const LOCK_FILE = path.join(PROJECT_ROOT, '.libtorch-cuda-setup.lock')

// Supported CUDA versions (latest first)
const SUPPORTED_CUDA_VERSIONS = ['cu124', 'cu121', 'cu118'] as const
type CudaVersion = (typeof SUPPORTED_CUDA_VERSIONS)[number]

// Default CUDA version
const DEFAULT_CUDA_VERSION: CudaVersion = 'cu124'

// Platform detection
const platform = process.platform
const arch = process.arch

interface PlatformConfig {
  baseUrl: string
  filename: string
  sha256?: string
}

function getCudaBaseUrl(cudaVersion: CudaVersion): string {
  // PyTorch download URLs for CUDA builds
  // Format: https://download.pytorch.org/libtorch/{cuda_version}/...
  return `https://download.pytorch.org/libtorch/${cudaVersion}`
}

function getPlatformConfig(cudaVersion: CudaVersion): PlatformConfig {
  const baseUrl = getCudaBaseUrl(cudaVersion)

  if (platform === 'win32') {
    if (arch !== 'x64') {
      throw new Error(`CUDA is only supported on Windows x64, got ${arch}`)
    }
    return {
      baseUrl,
      filename: `libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2B${cudaVersion}.zip`,
    }
  } else if (platform === 'linux') {
    if (arch !== 'x64') {
      throw new Error(`CUDA is only supported on Linux x64, got ${arch}`)
    }
    return {
      baseUrl,
      filename: `libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${cudaVersion}.zip`,
    }
  } else {
    throw new Error(`CUDA is not supported on ${platform}. Only Linux and Windows are supported.`)
  }
}

function acquireLock(): boolean {
  try {
    if (existsSync(LOCK_FILE)) {
      const lockContent = readFileSync(LOCK_FILE, 'utf-8')
      const lockTime = parseInt(lockContent, 10)
      const now = Date.now()

      // Lock expires after 10 minutes (in case of crash)
      if (now - lockTime < 10 * 60 * 1000) {
        return false
      }
      unlinkSync(LOCK_FILE)
    }

    writeFileSync(LOCK_FILE, Date.now().toString(), { flag: 'wx' })
    return true
  } catch {
    return false
  }
}

function releaseLock(): void {
  try {
    if (existsSync(LOCK_FILE)) {
      unlinkSync(LOCK_FILE)
    }
  } catch {
    // Ignore errors during cleanup
  }
}

function calculateFileHash(filePath: string): string {
  const fileBuffer = readFileSync(filePath)
  const hashSum = createHash('sha256')
  hashSum.update(fileBuffer)
  return hashSum.digest('hex')
}

function validateChecksum(filePath: string, expectedHash?: string): boolean {
  if (!expectedHash) {
    console.log('  No checksum available, skipping validation')
    return true
  }

  console.log('  Validating checksum...')
  const actualHash = calculateFileHash(filePath)

  if (actualHash.toLowerCase() !== expectedHash.toLowerCase()) {
    console.error(`  Checksum mismatch!`)
    console.error(`    Expected: ${expectedHash}`)
    console.error(`    Actual:   ${actualHash}`)
    return false
  }

  console.log('  Checksum valid')
  return true
}

async function downloadFile(url: string, dest: string): Promise<void> {
  console.log(`Downloading from ${url}...`)

  const response = await fetch(url, { redirect: 'follow' })
  if (!response.ok) {
    throw new Error(`Failed to download: ${response.status} ${response.statusText}`)
  }

  const totalBytes = parseInt(response.headers.get('content-length') || '0', 10)
  let downloadedBytes = 0
  let lastPercent = 0

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('Failed to get response reader')
  }

  await new Promise<void>((resolve, reject) => {
    const fileStream = createWriteStream(dest)

    fileStream.on('error', (err) => {
      reject(new Error(`Failed to write file: ${err.message}`))
    })

    fileStream.on('finish', () => {
      resolve()
    })

    const pump = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) {
            fileStream.end()
            break
          }

          const canContinue = fileStream.write(Buffer.from(value))
          downloadedBytes += value.length

          if (totalBytes > 0) {
            const percent = Math.floor((downloadedBytes / totalBytes) * 100)
            if (percent >= lastPercent + 10) {
              console.log(
                `  ${percent}% (${Math.round(downloadedBytes / 1024 / 1024)}MB / ${Math.round(totalBytes / 1024 / 1024)}MB)`,
              )
              lastPercent = percent
            }
          }

          if (!canContinue) {
            await new Promise<void>((res) => fileStream.once('drain', res))
          }
        }
      } catch (err) {
        fileStream.destroy()
        reject(err)
      }
    }

    pump()
  })

  console.log(`Download complete: ${Math.round(downloadedBytes / 1024 / 1024)}MB`)
}

function extractZip(zipFilePath: string, destDir: string): void {
  console.log(`Extracting to ${destDir}...`)

  let zip: AdmZip
  try {
    zip = new AdmZip(zipFilePath)
  } catch (err) {
    throw new Error(`Failed to open zip file: ${err instanceof Error ? err.message : String(err)}`)
  }

  const entries = zip.getEntries()
  const totalEntries = entries.length

  if (totalEntries === 0) {
    throw new Error('Zip file appears to be empty or corrupted')
  }

  console.log(`  Found ${totalEntries} files to extract...`)

  try {
    zip.extractAllTo(destDir, true)
  } catch (err) {
    throw new Error(`Failed to extract zip: ${err instanceof Error ? err.message : String(err)}`)
  }

  // Verify critical directories exist
  const criticalDirs = [
    path.join(destDir, 'libtorch', 'lib'),
    path.join(destDir, 'libtorch', 'include'),
    path.join(destDir, 'libtorch', 'share'),
  ]

  for (const dir of criticalDirs) {
    if (!existsSync(dir)) {
      throw new Error(`Extraction incomplete: missing ${dir}`)
    }
  }

  console.log(`  Extracted ${totalEntries} files`)
}

function cleanupOnFailure(zipPath: string, libtorchDir: string): void {
  console.log('Cleaning up after failure...')

  try {
    if (existsSync(zipPath)) {
      unlinkSync(zipPath)
      console.log(`  Removed corrupted zip: ${zipPath}`)
    }
  } catch {
    console.warn(`  Could not remove zip file: ${zipPath}`)
  }

  try {
    if (existsSync(libtorchDir)) {
      rmSync(libtorchDir, { recursive: true, force: true })
      console.log(`  Removed incomplete libtorch-cuda directory`)
    }
  } catch {
    console.warn(`  Could not remove libtorch-cuda directory`)
  }
}

async function buildNativeCuda(cudaVersion: CudaVersion): Promise<void> {
  console.log('\nBuilding native library with CUDA support...')

  const buildScript = path.join(PROJECT_ROOT, 'scripts/build-native-cuda.ts')
  if (!existsSync(buildScript)) {
    throw new Error(`Build script not found: ${buildScript}`)
  }

  const proc = Bun.spawn(['bun', 'run', buildScript, cudaVersion], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      LIBTORCH_CUDA: LIBTORCH_CUDA_DIR,
      LIBTORCH_CUDA_PATH: LIBTORCH_CUDA_DIR,
    },
    stdio: ['inherit', 'inherit', 'inherit'],
  })

  const exitCode = await proc.exited
  if (exitCode !== 0) {
    throw new Error(`Build failed with exit code ${exitCode}`)
  }
}

async function verifySetup(): Promise<boolean> {
  console.log('\nVerifying CUDA setup...')

  try {
    // Try to load the library and check CUDA
    const verifyCode = `
      import { torch } from '@ts-torch/core';
      const cudaAvailable = torch.cuda.isAvailable();
      const deviceCount = torch.cuda.deviceCount();
      console.log('CUDA available:', cudaAvailable);
      console.log('CUDA devices:', deviceCount);
      if (!cudaAvailable) {
        console.log('\\nNote: CUDA not detected. This could mean:');
        console.log('  - No NVIDIA GPU present');
        console.log('  - NVIDIA driver not installed');
        console.log('  - CUDA version mismatch');
      }
      process.exit(cudaAvailable ? 0 : 1);
    `

    const proc = Bun.spawn(['bun', '-e', verifyCode], {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        LIBTORCH_CUDA: LIBTORCH_CUDA_DIR,
        LIBTORCH_CUDA_PATH: LIBTORCH_CUDA_DIR,
        TS_TORCH_QUIET: '1', // Suppress loader info during verify
      },
      stdio: ['inherit', 'inherit', 'inherit'],
    })

    const exitCode = await proc.exited
    return exitCode === 0
  } catch (err) {
    console.error('Verification failed:', err instanceof Error ? err.message : String(err))
    return false
  }
}

function parseCudaVersion(arg: string | undefined): CudaVersion {
  if (!arg) {
    return DEFAULT_CUDA_VERSION
  }

  // Normalize input (accept "124", "cu124", "12.4", etc.)
  let normalized = arg.toLowerCase().replace(/\./g, '')
  if (!normalized.startsWith('cu')) {
    normalized = `cu${normalized}`
  }

  if (!SUPPORTED_CUDA_VERSIONS.includes(normalized as CudaVersion)) {
    console.error(`Unsupported CUDA version: ${arg}`)
    console.error(`Supported versions: ${SUPPORTED_CUDA_VERSIONS.join(', ')}`)
    process.exit(1)
  }

  return normalized as CudaVersion
}

function checkCudaCompatibility(requestedVersion: CudaVersion): void {
  console.log('Checking CUDA compatibility...')

  const cudaInfo = detectCuda()

  if (!cudaInfo.available) {
    console.log(`  Warning: ${cudaInfo.error}`)
    console.log(`  Continuing with ${requestedVersion} anyway (may not work without GPU)`)
    console.log('')
    return
  }

  console.log(`  NVIDIA driver version: ${cudaInfo.driverVersion}`)
  console.log(`  Recommended CUDA version: ${cudaInfo.cudaVersion}`)

  if (cudaInfo.cudaVersion !== requestedVersion) {
    console.log('')
    console.log(`  Warning: You requested ${requestedVersion} but your driver supports ${cudaInfo.cudaVersion}`)
    console.log(`  This may cause runtime errors. Consider using:`)
    console.log(`    bun run setup:cuda ${cudaInfo.cudaVersion}`)
    console.log('')
  } else {
    console.log(`  Requested version matches driver capabilities`)
  }
  console.log('')
}

async function setup() {
  const cudaVersion = parseCudaVersion(process.argv[2])
  const config = getPlatformConfig(cudaVersion)
  const localFilename = config.filename.replace('%2B', '+')
  const downloadUrl = `${config.baseUrl}/${config.filename}`
  const zipPath = path.join(tmpdir(), localFilename)

  console.log(`\n=== ts-torch CUDA Setup ===`)
  console.log(`Platform: ${platform} (${arch})`)
  console.log(`LibTorch version: ${LIBTORCH_VERSION}`)
  console.log(`CUDA version: ${cudaVersion}`)
  console.log(`Project root: ${PROJECT_ROOT}\n`)

  // Check build prerequisites before proceeding
  checkPrerequisites()
  console.log('')

  // Check CUDA driver compatibility
  checkCudaCompatibility(cudaVersion)

  // Acquire lock
  if (!acquireLock()) {
    console.log('Another setup process is running. Waiting...')

    const maxWait = 5 * 60 * 1000
    const startTime = Date.now()

    while (!acquireLock()) {
      if (Date.now() - startTime > maxWait) {
        throw new Error('Timeout waiting for another setup process to complete')
      }
      await new Promise((resolve) => setTimeout(resolve, 5000))
    }
  }

  try {
    const libDir = path.join(LIBTORCH_CUDA_DIR, 'lib')

    if (!existsSync(libDir)) {
      console.log(`Setting up LibTorch ${LIBTORCH_VERSION} with CUDA ${cudaVersion}...\n`)

      let needsDownload = true

      // Check for cached download
      if (existsSync(zipPath)) {
        console.log(`Found cached download: ${zipPath}`)

        if (validateChecksum(zipPath, config.sha256)) {
          needsDownload = false
        } else {
          console.log('Cached file is corrupted, re-downloading...')
          unlinkSync(zipPath)
        }
      }

      // Download if needed
      if (needsDownload) {
        try {
          await downloadFile(downloadUrl, zipPath)

          if (!validateChecksum(zipPath, config.sha256)) {
            throw new Error('Downloaded file failed checksum validation')
          }
        } catch (err) {
          cleanupOnFailure(zipPath, LIBTORCH_CUDA_DIR)
          throw err
        }
      }

      // Clean up partial extraction
      if (existsSync(LIBTORCH_CUDA_DIR)) {
        console.log('Removing incomplete libtorch-cuda directory...')
        rmSync(LIBTORCH_CUDA_DIR, { recursive: true, force: true })
      }

      // Extract to a temp location first, then rename
      const extractDir = path.join(PROJECT_ROOT, '.libtorch-cuda-extract-temp')
      if (existsSync(extractDir)) {
        rmSync(extractDir, { recursive: true, force: true })
      }

      try {
        extractZip(zipPath, extractDir)

        // The zip extracts to a "libtorch" subfolder, rename to libtorch-cuda
        const extractedLibtorch = path.join(extractDir, 'libtorch')
        if (!existsSync(extractedLibtorch)) {
          throw new Error('Extracted archive does not contain libtorch folder')
        }

        // Move to final location
        await mkdir(path.dirname(LIBTORCH_CUDA_DIR), { recursive: true })

        // On Windows, use rename. On Unix, we could also use rename
        const { renameSync } = await import('fs')
        renameSync(extractedLibtorch, LIBTORCH_CUDA_DIR)

        // Clean up temp dir
        rmSync(extractDir, { recursive: true, force: true })
      } catch (err) {
        cleanupOnFailure(zipPath, LIBTORCH_CUDA_DIR)
        if (existsSync(extractDir)) {
          rmSync(extractDir, { recursive: true, force: true })
        }
        throw err
      }

      // Final verification
      if (!existsSync(libDir)) {
        cleanupOnFailure(zipPath, LIBTORCH_CUDA_DIR)
        throw new Error(`Setup failed - lib directory not found at ${libDir}`)
      }

      // Clean up zip
      try {
        unlinkSync(zipPath)
      } catch {
        // Non-fatal
      }

      console.log(`\nLibTorch ${LIBTORCH_VERSION} (CUDA ${cudaVersion}) ready at ${LIBTORCH_CUDA_DIR}`)
    } else {
      console.log(`LibTorch CUDA already installed at ${LIBTORCH_CUDA_DIR}`)
    }

    // Build native library
    await buildNativeCuda(cudaVersion)

    // Verify the setup works
    const cudaWorks = await verifySetup()

    console.log(`\n=== Setup Complete ===`)
    if (cudaWorks) {
      console.log(`CUDA is working! You can now use GPU acceleration.`)
    } else {
      console.log(`Native library built, but CUDA not detected.`)
      console.log(`The library will still work for CPU operations.`)
      console.log(`\nTo debug: TS_TORCH_DEBUG=1 bun run your-script.ts`)
    }
  } finally {
    releaseLock()
  }
}

// Run setup
setup().catch((err) => {
  releaseLock()
  console.error('\nSetup failed:', err.message)
  process.exit(1)
})
