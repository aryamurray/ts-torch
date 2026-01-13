import { device, int64 } from '@ts-torch/core'
import { crossEntropyLoss } from '@ts-torch/optim'

const cpu = device.cpu()

console.log('Testing basic tensor...')
const t = cpu.zeros([2, 3] as const)
console.log('Created tensor:', t.shape)

console.log('Testing randn...')
const r = cpu.randn([2, 3] as const)
console.log('Created randn:', r.shape)

console.log('Testing requires_grad...')
r.requiresGrad = true
console.log('requiresGrad:', r.requiresGrad)

console.log('Testing forward pass...')
const a = cpu.randn([2, 3] as const)
const b = cpu.randn([2, 3] as const)
a.requiresGrad = true
const c = a.add(b)
console.log('Add result shape:', c.shape)

console.log('Testing loss...')
const logits = cpu.randn([4, 10] as const)
logits.requiresGrad = true
const targets = cpu.tensor([0, 1, 2, 3], [4] as const, int64)
console.log('Targets shape:', targets.shape)

const loss = crossEntropyLoss(logits as any, targets as any)
console.log('Loss computed')

console.log('Testing backward...')
loss.backward()
console.log('Backward completed!')

console.log('All tests passed!')
