/**
 * CUDA LibTorch download utilities
 * Downloads and extracts CUDA-enabled LibTorch from PyTorch's servers
 */

import { createWriteStream, existsSync, rmSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import AdmZip from 'adm-zip'

const LIBTORCH_VERSION = '2.5.1'

/**
 * CUDA version configuration
 */
interface CudaConfig {
  cudaVersion: string // e.g., 'cu124'
  downloadUrl: string
  filename: string
}

/**
 * Get download configuration for a CUDA version
 */
export function getCudaConfig(cudaVersion: string): CudaConfig {
  const platform = process.platform

  let filename: string
  if (platform === 'win32') {
    filename = `libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2B${cudaVersion}.zip`
  } else if (platform === 'linux') {
    filename = `libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${cudaVersion}.zip`
  } else {
    throw new Error(`CUDA is not supported on ${platform}. Only Linux and Windows are supported.`)
  }

  const downloadUrl = `https://download.pytorch.org/libtorch/${cudaVersion}/${filename}`

  return {
    cudaVersion,
    downloadUrl,
    filename: filename.replace('%2B', '+'),
  }
}

// Timeout for stalled downloads (60 seconds without data)
const DOWNLOAD_STALL_TIMEOUT_MS = 60000

/**
 * Download a file with progress reporting and stall detection
 */
async function downloadFile(url: string, dest: string): Promise<void> {
  console.log(`Downloading from ${url}...`)

  // Use AbortController for timeout on initial connection
  const controller = new AbortController()
  const connectionTimeout = setTimeout(() => controller.abort(), 30000) // 30s connection timeout

  let response: Response
  try {
    response = await fetch(url, { redirect: 'follow', signal: controller.signal })
  } finally {
    clearTimeout(connectionTimeout)
  }

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
    let stallTimeout: ReturnType<typeof setTimeout> | null = null

    const resetStallTimeout = () => {
      if (stallTimeout) clearTimeout(stallTimeout)
      stallTimeout = setTimeout(() => {
        reader.cancel('Download stalled - no data received for 60 seconds')
        fileStream.destroy()
        reject(new Error('Download stalled - no data received for 60 seconds. Check your network connection.'))
      }, DOWNLOAD_STALL_TIMEOUT_MS)
    }

    const cleanup = () => {
      if (stallTimeout) clearTimeout(stallTimeout)
    }

    fileStream.on('error', (err) => {
      cleanup()
      reject(new Error(`Failed to write file: ${err.message}`))
    })

    fileStream.on('finish', () => {
      cleanup()
      resolve()
    })

    const pump = async () => {
      resetStallTimeout() // Start initial timeout

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) {
            cleanup()
            fileStream.end()
            break
          }

          resetStallTimeout() // Reset timeout on data received

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
        cleanup()
        fileStream.destroy()
        reject(err)
      }
    }

    pump()
  })

  console.log(`Download complete: ${Math.round(downloadedBytes / 1024 / 1024)}MB`)
}

/**
 * Extract zip file with error handling
 */
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

  // Verify critical files exist after extraction
  const criticalFiles = [
    join(destDir, 'libtorch', 'lib'),
    join(destDir, 'libtorch', 'include'),
    join(destDir, 'libtorch', 'share'),
  ]

  for (const file of criticalFiles) {
    if (!existsSync(file)) {
      throw new Error(`Extraction incomplete: missing ${file}`)
    }
  }

  console.log(`  Extracted ${totalEntries} files`)
}

/**
 * Download and extract CUDA LibTorch to the specified directory
 *
 * @param cudaVersion - CUDA version (e.g., 'cu118', 'cu121', 'cu124')
 * @param destDir - Destination directory (LibTorch will be extracted to destDir/libtorch)
 */
export async function downloadLibTorch(cudaVersion: string, destDir: string): Promise<string> {
  const config = getCudaConfig(cudaVersion)

  // Check if already downloaded
  const libtorchDir = join(destDir, 'libtorch')
  if (existsSync(join(libtorchDir, 'lib'))) {
    console.log(`LibTorch ${cudaVersion} already exists at ${libtorchDir}`)
    return libtorchDir
  }

  // Ensure destination exists
  await mkdir(destDir, { recursive: true })

  // Download to temp directory
  const zipPath = join(tmpdir(), config.filename)

  try {
    // Download if not cached
    if (!existsSync(zipPath)) {
      await downloadFile(config.downloadUrl, zipPath)
    } else {
      console.log(`Using cached download: ${zipPath}`)
    }

    // Clean up any partial extraction
    if (existsSync(libtorchDir)) {
      console.log('Removing incomplete libtorch directory...')
      rmSync(libtorchDir, { recursive: true, force: true })
    }

    // Extract
    extractZip(zipPath, destDir)

    // Verify
    if (!existsSync(join(libtorchDir, 'lib'))) {
      throw new Error(`Extraction failed - lib directory not found at ${libtorchDir}`)
    }

    console.log(`LibTorch ${cudaVersion} ready at ${libtorchDir}`)
    return libtorchDir
  } catch (err) {
    // Clean up on failure
    try {
      if (existsSync(zipPath)) {
        rmSync(zipPath)
      }
      if (existsSync(libtorchDir)) {
        rmSync(libtorchDir, { recursive: true, force: true })
      }
    } catch {
      // Ignore cleanup errors
    }
    throw err
  }
}

export { LIBTORCH_VERSION }
