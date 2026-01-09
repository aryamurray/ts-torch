/**
 * @ts-torch/test-utils
 *
 * Shared test utilities for the ts-torch monorepo
 */

// Matchers
export {
  setupTensorMatchers,
  tensorMatchers,
} from './matchers/tensor-matchers.js';

// Mocks
export {
  MockTensor,
  mockTensorFactories,
  createMockFFI,
  type MockDtype,
  type MockFFI,
  type MockFFISymbols,
} from './mocks/index.js';

// Fixtures
export {
  TensorFixtures,
  createTestTensor,
  type TensorFixture,
  type TensorPattern,
} from './fixtures/index.js';

// Helpers
export {
  ScopeTestHelper,
  scopedTest,
  scopedTestAsync,
  createTestScope,
} from './helpers/index.js';
