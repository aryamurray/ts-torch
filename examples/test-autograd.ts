import { torch } from '@ts-torch/core'

console.log('Testing basic tensor...')
const t = torch.zeros([2, 3] as const)
console.log('Created tensor:', t.shape)

console.log('Testing randn...')
const r = torch.randn([2, 3] as const)
console.log('Created randn:', r.shape)

console.log('Testing requires_grad...')
r.requiresGrad = true
console.log('requiresGrad:', r.requiresGrad)

console.log('Testing forward pass...')
const a = torch.randn([2, 3] as const)
const b = torch.randn([2, 3] as const)
a.requiresGrad = true
const c = a.add(b)
console.log('Add result shape:', c.shape)

console.log('Testing loss...')
const logits = torch.randn([4, 10] as const)
logits.requiresGrad = true
const targets = torch.tensor([0, 1, 2, 3], [4] as const, torch.int64)
console.log('Targets shape:', targets.shape)

const loss = torch.nn.crossEntropyLoss(logits as any, targets as any)
console.log('Loss computed')

console.log('Testing backward...')
loss.backward()
console.log('Backward completed!')

console.log('All tests passed!')
