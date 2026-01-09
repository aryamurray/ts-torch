/**
 * Quick test to verify native library loads correctly
 */

import { torch } from '@ts-torch/core'

console.log('Testing native library connection...\n')

try {
  // Test version
  const version = torch.version()
  console.log('LibTorch version:', version)

  // Test CUDA availability
  console.log('CUDA available:', torch.cuda.isAvailable())
  console.log('CUDA device count:', torch.cuda.deviceCount())

  // Test basic tensor creation
  console.log('\nTesting tensor creation...')

  torch.run(() => {
    const zeros = torch.zeros([2, 3] as const)
    console.log('Created zeros tensor:', zeros.shape)

    const ones = torch.ones([3, 4] as const)
    console.log('Created ones tensor:', ones.shape)

    const randn = torch.randn([2, 2] as const)
    console.log('Created randn tensor:', randn.shape)
  })

  console.log('\n✓ Native library connected successfully!')
} catch (error) {
  console.error('✗ Error loading native library:')
  console.error(error)
  process.exit(1)
}
