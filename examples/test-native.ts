/**
 * Quick test to verify native library loads correctly
 */

import { device, run, cuda } from '@ts-torch/core'

const cpu = device.cpu()

console.log('Testing native library connection...\n')

try {
  // Test CUDA availability
  console.log('CUDA available:', cuda.isAvailable())
  console.log('CUDA device count:', cuda.deviceCount())

  // Test basic tensor creation
  console.log('\nTesting tensor creation...')

  run(() => {
    const zeros = cpu.zeros([2, 3] as const)
    console.log('Created zeros tensor:', zeros.shape)

    const ones = cpu.ones([3, 4] as const)
    console.log('Created ones tensor:', ones.shape)

    const randn = cpu.randn([2, 2] as const)
    console.log('Created randn tensor:', randn.shape)
  })

  console.log('\n✓ Native library connected successfully!')
} catch (error) {
  console.error('✗ Error loading native library:')
  console.error(error)
  process.exit(1)
}
