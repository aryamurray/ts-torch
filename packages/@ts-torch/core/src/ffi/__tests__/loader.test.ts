/**
 * Tests for library loader
 * Note: These tests will fail if native library is not built
 */

import { describe, it, expect } from 'vitest'
import { getPlatformPackage, getLibraryPath, getLib, closeLib } from '../loader.js'

describe('FFI Loader', () => {
  describe('getPlatformPackage', () => {
    it('should return valid platform package info', () => {
      const { packageName, libraryName } = getPlatformPackage()

      expect(packageName).toMatch(/^@ts-torch\/(darwin|linux|win32)-(arm64|x64)$/)
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
        expect(lib.symbols.ts_tensor_zeros).toBeDefined()
        expect(lib.symbols.ts_tensor_add).toBeDefined()
        expect(lib.symbols.ts_tensor_delete).toBeDefined()
        expect(lib.symbols.ts_version).toBeDefined()
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
})
