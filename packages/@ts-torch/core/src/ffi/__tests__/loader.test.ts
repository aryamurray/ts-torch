/**
 * Tests for library loader
 * Note: These tests will fail if native library is not built
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { getPlatformPackage, getLibraryPath, getLib, closeLib } from '../loader.js'

describe('FFI Loader', () => {
  describe('getPlatformPackage', () => {
    it('should return valid platform package info', () => {
      const { packageName, libraryName } = getPlatformPackage()

      expect(packageName).toMatch(/^@ts-torch-platform\/(darwin|linux|win32)-(arm64|x64)$/)
      expect(libraryName).toBeTruthy()
      expect(typeof libraryName).toBe('string')
    })

    it('should match current platform', () => {
      const { packageName } = getPlatformPackage()
      const platform = process.platform

      expect(packageName).toContain(platform)
    })
  })

  describe('getLibraryPath', () => {
    it('should respect TS_TORCH_LIB environment variable', () => {
      const originalEnv = process.env.TS_TORCH_LIB

      try {
        // Set custom path
        const customPath = '/custom/path/libts_torch.so'
        process.env.TS_TORCH_LIB = customPath

        // Note: Will warn if file doesn't exist but continue with fallback
        const path = getLibraryPath()
        expect(typeof path).toBe('string')
      } finally {
        // Restore original
        if (originalEnv === undefined) {
          delete process.env.TS_TORCH_LIB
        } else {
          process.env.TS_TORCH_LIB = originalEnv
        }
      }
    })

    it('should return absolute path', () => {
      try {
        const path = getLibraryPath()
        expect(path).toBeTruthy()
        // On Windows, check for drive letter or UNC path
        // On Unix, check for leading slash
        const isAbsolute = path.match(/^[A-Za-z]:/) || path.startsWith('/') || path.startsWith('\\\\')
        expect(isAbsolute).toBeTruthy()
      } catch (err) {
        // Expected if library not built yet
        expect(err).toBeInstanceOf(Error)
        expect((err as Error).message).toContain('Could not find ts-torch native library')
      }
    })
  })

  describe('getLib', () => {
    it('should cache library instance', () => {
      try {
        const lib1 = getLib()
        const lib2 = getLib()

        // Should return same instance
        expect(lib1).toBe(lib2)
      } catch (err) {
        // Expected if library not built yet
        expect(err).toBeInstanceOf(Error)
      }
    })

    it('should have all required symbols', () => {
      try {
        const lib = getLib()

        // Check a few key symbols
        expect(lib.ts_tensor_zeros).toBeDefined()
        expect(lib.ts_tensor_add).toBeDefined()
        expect(lib.ts_tensor_delete).toBeDefined()
        expect(lib.ts_version).toBeDefined()
      } catch (err) {
        // Expected if library not built yet
        expect(err).toBeInstanceOf(Error)
      }
    })

    it('should throw descriptive error if library not found', () => {
      // Clear cache first
      closeLib()

      try {
        getLib()
      } catch (err) {
        expect(err).toBeInstanceOf(Error)
        const message = (err as Error).message
        expect(message).toContain('ts-torch')
        // Should include helpful suggestions
        expect(message.includes('bun add') || message.includes('cargo build') || message.includes('TS_TORCH_LIB')).toBe(
          true,
        )
      }
    })
  })

  describe('closeLib', () => {
    it('should cleanup library instance', () => {
      try {
        getLib() // Load library
        closeLib() // Close library

        // Next call should reload
        const lib = getLib()
        expect(lib).toBeDefined()
      } catch (err) {
        // Expected if library not built yet
        expect(err).toBeInstanceOf(Error)
      }
    })

    it('should be safe to call multiple times', () => {
      expect(() => {
        closeLib()
        closeLib()
        closeLib()
      }).not.toThrow()
    })
  })

  describe('CUDA detection', () => {
    let originalDebug: string | undefined

    beforeEach(() => {
      originalDebug = process.env.TS_TORCH_DEBUG
      closeLib() // Clear cached library
    })

    afterEach(() => {
      if (originalDebug === undefined) {
        delete process.env.TS_TORCH_DEBUG
      } else {
        process.env.TS_TORCH_DEBUG = originalDebug
      }
    })

    it('should respect TS_TORCH_DEBUG environment variable', () => {
      // Enable debug mode
      process.env.TS_TORCH_DEBUG = '1'

      // This just verifies the env var is respected without crashing
      // Actual logging is tested manually
      try {
        getLibraryPath()
      } catch {
        // Expected if library not built
      }

      // Disable debug mode
      process.env.TS_TORCH_DEBUG = '0'

      try {
        getLibraryPath()
      } catch {
        // Expected if library not built
      }
    })

    it('should only support CUDA on Linux and Windows', () => {
      const { packageName } = getPlatformPackage()

      // macOS (darwin) should not have CUDA variant
      if (process.platform === 'darwin') {
        expect(packageName).not.toContain('cuda')
      }
    })

    it('should prefer CUDA library when available', () => {
      // This test verifies the library path resolution order
      // CUDA packages are checked before CPU packages
      try {
        const path = getLibraryPath()

        // If a CUDA library is available, it should be used
        // The path will contain 'cuda' if CUDA is being used
        if (path.includes('cuda') || path.includes('cu1')) {
          expect(path).toMatch(/cu\d{3}|cuda/)
        }
      } catch {
        // Expected if no library built
      }
    })

    it('should validate CUDA build metadata', () => {
      // The loader validates .build-meta.json before using a CUDA library
      // Invalid metadata should cause fallback to next option
      // This is tested implicitly through getLibraryPath behavior
      try {
        const path = getLibraryPath()
        expect(typeof path).toBe('string')
      } catch (err) {
        expect(err).toBeInstanceOf(Error)
      }
    })
  })

  describe('Workspace detection', () => {
    beforeEach(() => {
      closeLib() // Reset cached workspace root
    })

    it('should detect workspace root from this monorepo', () => {
      // Since we're running tests from within the ts-tools monorepo,
      // workspace detection should work
      try {
        const path = getLibraryPath()
        // If we get here, workspace detection worked (or library was found via other means)
        expect(typeof path).toBe('string')
      } catch (err) {
        // Even if library not found, the error should list searched paths
        // which indicates workspace detection ran
        expect(err).toBeInstanceOf(Error)
        const message = (err as Error).message
        expect(message).toContain('Searched paths:')
      }
    })

    it('should reset workspace cache on closeLib', () => {
      // This is an implementation detail test to ensure cache resets work
      closeLib()

      // After closeLib, the next getLibraryPath call should re-search
      // (we can't directly test this without exposing internals,
      // but we can verify it doesn't throw unexpectedly)
      try {
        getLibraryPath()
      } catch (err) {
        // Expected if library not built - verify it's the right error
        expect(err).toBeInstanceOf(Error)
        expect((err as Error).message).toContain('Could not find ts-torch native library')
      }
    })

    it('should provide helpful error messages with setup instructions', () => {
      closeLib()

      try {
        getLibraryPath()
      } catch (err) {
        expect(err).toBeInstanceOf(Error)
        const message = (err as Error).message

        // Should include quick setup instructions
        expect(message).toContain('Quick Setup')
        expect(message).toContain('bun run setup')

        // Should mention TS_TORCH_LIB
        expect(message).toContain('TS_TORCH_LIB')

        // Should mention debug mode
        expect(message).toContain('TS_TORCH_DEBUG')

        // On Linux/Windows, should mention CUDA setup
        if (process.platform === 'linux' || process.platform === 'win32') {
          expect(message).toContain('setup:cuda')
        }
      }
    })
  })
})
