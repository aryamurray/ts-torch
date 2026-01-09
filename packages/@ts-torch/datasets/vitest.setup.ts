/**
 * Vitest setup file for @ts-torch/datasets
 *
 * This file runs before all tests and sets up custom matchers
 */

import { beforeAll } from 'vitest';

// Import from source instead of built dist (to avoid build issues)
const setupPath = new URL('../../core/src/test/utils.ts', import.meta.url);

beforeAll(async () => {
  const { setupTensorMatchers } = await import(setupPath.href);
  setupTensorMatchers();
});
