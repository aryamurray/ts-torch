/**
 * Vitest setup file for @ts-torch/core
 *
 * This file runs before all tests and sets up custom matchers
 */

import { beforeAll } from 'vitest';
import { setupTensorMatchers } from './src/test/utils.js';

beforeAll(() => {
  setupTensorMatchers();
});
