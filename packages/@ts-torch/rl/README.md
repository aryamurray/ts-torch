# @ts-torch/rl

Declarative reinforcement learning for ts-torch.

## Overview

This package provides a high-level, Stable Baselines 3-style API for reinforcement learning. It includes policy gradient agents (PPO, A2C), off-policy agents (DQN, SAC), replay buffers, and environment utilities.

## Features

- **On-Policy Agents**: PPO, A2C with actor-critic policies
- **Off-Policy Agents**: DQN, SAC for discrete and continuous control
- **Vectorized Environments**: Run multiple environments in parallel
- **Replay Buffers**: Uniform and prioritized experience replay
- **Flexible Policies**: MLP policies with configurable architectures
- **Checkpoint Support**: Save and resume training
- **Built-in Environments**: CartPole and more

## Installation

```bash
bun add @ts-torch/rl
```

## Quick Start

```typescript
import { RL } from '@ts-torch/rl'
import { device } from '@ts-torch/core'

// 1. Create vectorized environment
const vecEnv = RL.vecEnv.dummy(() => RL.envs.CartPole(), { nEnvs: 8 })

// 2. Create PPO agent
const ppo = RL.ppo({
  policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
  learningRate: 3e-4,
  nSteps: 2048,
  batchSize: 64,
  nEpochs: 10,
}).init(device.cuda(0), vecEnv)

// 3. Train
await ppo.learn({ totalTimesteps: 1_000_000 })

// 4. Inference
const action = ppo.predict(observation, true)  // deterministic=true
```

## API Reference

### Agents

#### PPO (Proximal Policy Optimization)

```typescript
import { RL, ppo } from '@ts-torch/rl'

const agent = RL.ppo({
  policy: {
    netArch: { pi: [64, 64], vf: [64, 64] },
    activation: 'tanh',
  },
  learningRate: 3e-4,
  nSteps: 2048,
  batchSize: 64,
  nEpochs: 10,
  clipRange: 0.2,
  entCoef: 0.01,
  vfCoef: 0.5,
  maxGradNorm: 0.5,
}).init(device, vecEnv)

await agent.learn({ totalTimesteps: 1_000_000 })
```

#### A2C (Advantage Actor-Critic)

```typescript
import { RL, a2c } from '@ts-torch/rl'

const agent = RL.a2c({
  policy: { netArch: { pi: [64, 64], vf: [64, 64] } },
  learningRate: 7e-4,
  nSteps: 5,
  entCoef: 0.01,
  vfCoef: 0.5,
}).init(device, vecEnv)

await agent.learn({ totalTimesteps: 1_000_000 })
```

#### SAC (Soft Actor-Critic)

For continuous control tasks:

```typescript
import { RL, sac } from '@ts-torch/rl'

const agent = RL.sac({
  policy: { netArch: { pi: [256, 256], qf: [256, 256] } },
  learningRate: 3e-4,
  bufferSize: 1_000_000,
  batchSize: 256,
  tau: 0.005,
  gamma: 0.99,
  autoEntropyTuning: true,
}).init(device, vecEnv)

await agent.learn({ totalTimesteps: 1_000_000 })
```

#### DQN (Deep Q-Network)

```typescript
import { RL, dqn } from '@ts-torch/rl'

const agent = RL.dqn({
  netArch: [64, 64],
  learningRate: 1e-4,
  bufferSize: 100_000,
  batchSize: 32,
  explorationFraction: 0.1,
  explorationFinalEps: 0.05,
})
```

### Environments

#### Built-in Environments

```typescript
import { RL, CartPole } from '@ts-torch/rl'

// Using RL namespace
const env = RL.envs.CartPole()

// Direct import
const env = CartPole()
```

#### Vectorized Environments

```typescript
import { RL, dummyVecEnv } from '@ts-torch/rl'

// Run 8 environments in parallel
const vecEnv = RL.vecEnv.dummy(() => RL.envs.CartPole(), { nEnvs: 8 })

// Or using the direct function
const vecEnv = dummyVecEnv(() => CartPole(), { nEnvs: 8 })
```

#### Custom Environments

```typescript
import { env, type EnvConfig } from '@ts-torch/rl'

const myEnv = env({
  observationSpace: box({ low: -1, high: 1, shape: [4] }),
  actionSpace: discrete(2),
  reset: () => ({ observation: [0, 0, 0, 0], info: {} }),
  step: (action) => ({
    observation: [0, 0, 0, 0],
    reward: 1,
    terminated: false,
    truncated: false,
    info: {},
  }),
})
```

### Spaces

```typescript
import { discrete, box } from '@ts-torch/rl'

// Discrete action space with 4 actions
const actionSpace = discrete(4)

// Continuous observation space
const obsSpace = box({
  low: [-1, -1, -1, -1],
  high: [1, 1, 1, 1],
  shape: [4],
})
```

### Replay Buffers

```typescript
import { ReplayBuffer, RolloutBuffer } from '@ts-torch/rl'

// Standard replay buffer
const buffer = new ReplayBuffer({
  size: 100_000,
  observationShape: [4],
  actionShape: [1],
})

// With Prioritized Experience Replay (PER)
const perBuffer = new ReplayBuffer({
  size: 100_000,
  observationShape: [4],
  actionShape: [1],
  per: {
    alpha: 0.6,
    beta: 0.4,
    betaIncrement: 0.001,
  },
})
```

### Policies

```typescript
import { actorCriticPolicy, mlpPolicy, sacPolicy } from '@ts-torch/rl'

// Actor-Critic policy for PPO/A2C
const policy = actorCriticPolicy({
  netArch: { pi: [64, 64], vf: [64, 64] },
  activation: 'tanh',
  orthoInit: true,
})

// SAC policy for continuous control
const sacPol = sacPolicy({
  netArch: { pi: [256, 256], qf: [256, 256] },
  activation: 'relu',
})
```

### Callbacks

```typescript
await agent.learn({
  totalTimesteps: 1_000_000,
  callbacks: {
    onTrainingStart: () => console.log('Training started'),
    onStep: ({ step, reward }) => {
      if (step % 1000 === 0) console.log(`Step ${step}`)
    },
    onEpisodeEnd: ({ episode, reward }) => {
      console.log(`Episode ${episode}: reward=${reward}`)
    },
    onTrainingEnd: () => console.log('Training complete'),
  },
})
```

### Multi-Objective RL

```typescript
import { MORL, RL } from '@ts-torch/rl'

// Sample weights from simplex
const weights = MORL.sampleSimplex(3)  // 3 objectives

// Scalarize multi-objective reward
const scalarReward = RL.scalarize(weights, [r1, r2, r3])

// Generate weight grid for evaluation
const grid = MORL.weightGrid(3, 10)
```

### Checkpointing

```typescript
import { saveCheckpoint, loadCheckpoint } from '@ts-torch/rl'

// Save agent state
await saveCheckpoint(agent, 'checkpoints/ppo_1m.pt')

// Load and resume
const restored = await loadCheckpoint('checkpoints/ppo_1m.pt', agent)
```

## License

MIT
