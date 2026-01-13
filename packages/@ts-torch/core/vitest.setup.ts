/**
 * Vitest setup file for @ts-torch/core
 *
 * This file runs before all tests and sets up custom matchers
 */

import { beforeAll, afterAll } from 'vitest';
import { setupTensorMatchers } from './src/test/utils.js';
import { closeLib } from './src/ffi/loader.js';

beforeAll(() => {
  setupTensorMatchers();
});

afterAll(async () => {
  // Explicitly close the native library to prevent worker crash during teardown
  closeLib();
  // Small delay to allow cleanup
  await new Promise(resolve => setTimeout(resolve, 100));
});
