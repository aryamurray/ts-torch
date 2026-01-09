/**
 * Learning rate schedulers for optimizers
 *
 * Provides various learning rate scheduling strategies to adjust the learning rate
 * during training, which can improve convergence and final model performance.
 */

import { Optimizer } from "./optimizer.js";

/**
 * Base class for learning rate schedulers
 *
 * All schedulers should inherit from this class and implement the getLr method.
 */
export abstract class LRScheduler {
  protected optimizer: Optimizer;
  protected lastEpoch: number;
  protected baseLrs: number[];

  constructor(optimizer: Optimizer, lastEpoch = -1) {
    this.optimizer = optimizer;
    this.lastEpoch = lastEpoch;

    // Store initial learning rates for each parameter group
    this.baseLrs = optimizer["paramGroups"].map((group) => group.lr ?? optimizer["defaults"].lr);

    if (lastEpoch === -1) {
      // Initialize learning rates
      this.step();
    }
  }

  /**
   * Compute learning rate for the current epoch
   * @returns Array of learning rates for each parameter group
   */
  protected abstract getLr(): number[];

  /**
   * Perform a single step of the scheduler
   * Updates the learning rate for all parameter groups
   */
  step(): void {
    this.lastEpoch += 1;
    const lrs = this.getLr();

    const paramGroups = this.optimizer["paramGroups"];
    for (let i = 0; i < paramGroups.length; i++) {
      const group = paramGroups[i];
      const lr = lrs[i];
      if (group && lr !== undefined) {
        group.lr = lr;
      }
    }
  }

  /**
   * Get the current learning rates
   */
  getCurrentLr(): number[] {
    return this.optimizer["paramGroups"].map((group) => group.lr ?? this.optimizer["defaults"].lr);
  }

  /**
   * Get the last epoch number
   */
  getLastEpoch(): number {
    return this.lastEpoch;
  }
}

/**
 * Step learning rate scheduler
 *
 * Decays the learning rate by gamma every stepSize epochs.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new StepLR(optimizer, 30, 0.1);
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train();
 *   validate();
 *   scheduler.step();
 * }
 * // Learning rate will be 0.1 for epochs [0, 29]
 * //                      0.01 for epochs [30, 59]
 * //                      0.001 for epochs [60, 89]
 * //                      0.0001 for epochs [90, 99]
 * ```
 */
export class StepLR extends LRScheduler {
  private stepSize: number;
  private gamma: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param stepSize - Period of learning rate decay
   * @param gamma - Multiplicative factor of learning rate decay (default: 0.1)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, stepSize: number, gamma = 0.1, lastEpoch = -1) {
    if (stepSize <= 0) {
      throw new Error("Step size must be positive");
    }
    if (gamma <= 0 || gamma > 1) {
      throw new Error("Gamma must be in (0, 1]");
    }

    super(optimizer, lastEpoch);
    this.stepSize = stepSize;
    this.gamma = gamma;
  }

  protected getLr(): number[] {
    const multiplier = Math.pow(this.gamma, Math.floor(this.lastEpoch / this.stepSize));
    return this.baseLrs.map((baseLr) => baseLr * multiplier);
  }
}

/**
 * Multi-step learning rate scheduler
 *
 * Decays the learning rate by gamma once the number of epochs reaches one of the milestones.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new MultiStepLR(optimizer, [30, 80], 0.1);
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train();
 *   validate();
 *   scheduler.step();
 * }
 * // Learning rate will be 0.1 for epochs [0, 29]
 * //                      0.01 for epochs [30, 79]
 * //                      0.001 for epochs [80, 99]
 * ```
 */
export class MultiStepLR extends LRScheduler {
  private milestones: Set<number>;
  private gamma: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param milestones - List of epoch indices at which to decay the learning rate
   * @param gamma - Multiplicative factor of learning rate decay (default: 0.1)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, milestones: number[], gamma = 0.1, lastEpoch = -1) {
    if (gamma <= 0 || gamma > 1) {
      throw new Error("Gamma must be in (0, 1]");
    }

    super(optimizer, lastEpoch);
    this.milestones = new Set(milestones.sort((a, b) => a - b));
    this.gamma = gamma;
  }

  protected getLr(): number[] {
    let multiplier = 1.0;
    for (let i = 1; i <= this.lastEpoch; i++) {
      if (this.milestones.has(i)) {
        multiplier *= this.gamma;
      }
    }
    return this.baseLrs.map((baseLr) => baseLr * multiplier);
  }
}

/**
 * Exponential learning rate scheduler
 *
 * Decays the learning rate by gamma every epoch.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new ExponentialLR(optimizer, 0.95);
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train();
 *   validate();
 *   scheduler.step();
 * }
 * // Learning rate = 0.1 * 0.95^epoch
 * ```
 */
export class ExponentialLR extends LRScheduler {
  private gamma: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param gamma - Multiplicative factor of learning rate decay
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, gamma: number, lastEpoch = -1) {
    if (gamma <= 0 || gamma > 1) {
      throw new Error("Gamma must be in (0, 1]");
    }

    super(optimizer, lastEpoch);
    this.gamma = gamma;
  }

  protected getLr(): number[] {
    const multiplier = Math.pow(this.gamma, this.lastEpoch);
    return this.baseLrs.map((baseLr) => baseLr * multiplier);
  }
}

/**
 * Cosine annealing learning rate scheduler
 *
 * Sets the learning rate using a cosine annealing schedule. The learning rate
 * is annealed from the initial value to etaMin over tMax epochs.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new CosineAnnealingLR(optimizer, 50, 0.001);
 *
 * for (let epoch = 0; epoch < 50; epoch++) {
 *   train();
 *   validate();
 *   scheduler.step();
 * }
 * // Learning rate follows a cosine curve from 0.1 to 0.001
 * ```
 */
export class CosineAnnealingLR extends LRScheduler {
  private tMax: number;
  private etaMin: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param tMax - Maximum number of iterations (epochs)
   * @param etaMin - Minimum learning rate (default: 0)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, tMax: number, etaMin = 0, lastEpoch = -1) {
    if (tMax <= 0) {
      throw new Error("T_max must be positive");
    }
    if (etaMin < 0) {
      throw new Error("eta_min must be non-negative");
    }

    super(optimizer, lastEpoch);
    this.tMax = tMax;
    this.etaMin = etaMin;
  }

  protected getLr(): number[] {
    if (this.lastEpoch === 0) {
      return this.baseLrs;
    }

    const cosineAnnealing = (baseLr: number): number => {
      return (
        this.etaMin +
        ((baseLr - this.etaMin) * (1 + Math.cos((Math.PI * this.lastEpoch) / this.tMax))) / 2
      );
    };

    return this.baseLrs.map(cosineAnnealing);
  }
}

/**
 * Cosine annealing with warm restarts
 *
 * Sets the learning rate using a cosine annealing schedule with periodic restarts.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new CosineAnnealingWarmRestarts(optimizer, 10, 2);
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train();
 *   validate();
 *   scheduler.step();
 * }
 * // Learning rate restarts every T_0 * T_mult epochs with cosine annealing
 * ```
 */
export class CosineAnnealingWarmRestarts extends LRScheduler {
  private tMult: number;
  private etaMin: number;
  private tCur: number;
  private tI: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param t0 - Number of iterations for the first restart
   * @param tMult - A factor increases t_i after a restart (default: 1)
   * @param etaMin - Minimum learning rate (default: 0)
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, t0: number, tMult = 1, etaMin = 0, lastEpoch = -1) {
    if (t0 <= 0) {
      throw new Error("T_0 must be positive");
    }
    if (tMult < 1) {
      throw new Error("T_mult must be >= 1");
    }
    if (etaMin < 0) {
      throw new Error("eta_min must be non-negative");
    }

    super(optimizer, lastEpoch);
    this.tMult = tMult;
    this.etaMin = etaMin;
    this.tCur = 0;
    this.tI = t0;
  }

  protected getLr(): number[] {
    const cosineAnnealing = (baseLr: number): number => {
      return (
        this.etaMin + ((baseLr - this.etaMin) * (1 + Math.cos((Math.PI * this.tCur) / this.tI))) / 2
      );
    };

    return this.baseLrs.map(cosineAnnealing);
  }

  override step(): void {
    this.lastEpoch += 1;
    this.tCur += 1;

    if (this.tCur >= this.tI) {
      this.tCur = 0;
      this.tI = this.tI * this.tMult;
    }

    const lrs = this.getLr();
    const paramGroups = this.optimizer["paramGroups"];
    for (let i = 0; i < paramGroups.length; i++) {
      const group = paramGroups[i];
      const lr = lrs[i];
      if (group && lr !== undefined) {
        group.lr = lr;
      }
    }
  }
}

/**
 * Reduce learning rate on plateau
 *
 * Reduces learning rate when a metric has stopped improving.
 * Models often benefit from reducing the learning rate by a factor of 2-10
 * once learning stagnates.
 *
 * @example
 * ```typescript
 * const optimizer = new SGD(model.parameters(), { lr: 0.1 });
 * const scheduler = new ReduceLROnPlateau(optimizer, 'min', {
 *   factor: 0.1,
 *   patience: 10,
 *   threshold: 0.0001,
 * });
 *
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train();
 *   const valLoss = validate();
 *   scheduler.step(valLoss);
 * }
 * ```
 */
export class ReduceLROnPlateau extends LRScheduler {
  private mode: "min" | "max";
  private factor: number;
  private patience: number;
  private threshold: number;
  private thresholdMode: "rel" | "abs";
  private cooldown: number;
  private minLr: number | number[];
  private eps: number;

  private numBadEpochs: number;
  private cooldownCounter: number;
  private best: number | null;

  /**
   * @param optimizer - Wrapped optimizer
   * @param mode - One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing
   * @param options - Scheduler options
   */
  constructor(
    optimizer: Optimizer,
    mode: "min" | "max" = "min",
    options?: {
      factor?: number;
      patience?: number;
      threshold?: number;
      thresholdMode?: "rel" | "abs";
      cooldown?: number;
      minLr?: number | number[];
      eps?: number;
    },
  ) {
    super(optimizer, -1);

    this.mode = mode;
    this.factor = options?.factor ?? 0.1;
    this.patience = options?.patience ?? 10;
    this.threshold = options?.threshold ?? 1e-4;
    this.thresholdMode = options?.thresholdMode ?? "rel";
    this.cooldown = options?.cooldown ?? 0;
    this.minLr = options?.minLr ?? 0;
    this.eps = options?.eps ?? 1e-8;

    if (this.factor >= 1.0) {
      throw new Error("Factor should be < 1.0");
    }
    if (this.patience < 0) {
      throw new Error("Patience should be non-negative");
    }

    this.numBadEpochs = 0;
    this.cooldownCounter = 0;
    this.best = null;
  }

  protected getLr(): number[] {
    // This is never called for ReduceLROnPlateau since we override step()
    return this.getCurrentLr();
  }

  /**
   * Update the learning rate based on the metric value
   * @param metrics - The metric value to monitor
   */
  override step(metrics?: number): void {
    if (metrics === undefined) {
      throw new Error("ReduceLROnPlateau requires a metric value");
    }

    const current = metrics;

    if (this.best === null) {
      this.best = current;
    } else if (this.isImproved(current)) {
      this.best = current;
      this.numBadEpochs = 0;
    } else {
      this.numBadEpochs += 1;
    }

    if (this.cooldownCounter > 0) {
      this.cooldownCounter -= 1;
      this.numBadEpochs = 0;
    }

    if (this.numBadEpochs > this.patience) {
      this.reduceLr();
      this.cooldownCounter = this.cooldown;
      this.numBadEpochs = 0;
    }

    this.lastEpoch += 1;
  }

  private isImproved(current: number): boolean {
    if (this.best === null) return true;

    if (this.mode === "min") {
      const threshold =
        this.thresholdMode === "rel"
          ? this.best * (1 - this.threshold)
          : this.best - this.threshold;
      return current < threshold;
    } else {
      const threshold =
        this.thresholdMode === "rel"
          ? this.best * (1 + this.threshold)
          : this.best + this.threshold;
      return current > threshold;
    }
  }

  private reduceLr(): void {
    const paramGroups = this.optimizer["paramGroups"];
    for (let i = 0; i < paramGroups.length; i++) {
      const group = paramGroups[i];
      if (!group) continue;

      const oldLr = group.lr ?? this.optimizer["defaults"].lr;
      const newLr = Math.max(oldLr * this.factor, this.getMinLr(i));

      if (oldLr - newLr > this.eps) {
        group.lr = newLr;
      }
    }
  }

  private getMinLr(index: number): number {
    if (typeof this.minLr === "number") {
      return this.minLr;
    }
    return this.minLr[index] ?? 0;
  }
}

/**
 * Linear learning rate warmup
 *
 * Linearly increases the learning rate from 0 to the base learning rate
 * over a specified number of warmup steps.
 *
 * @example
 * ```typescript
 * const optimizer = new Adam(model.parameters(), { lr: 0.001 });
 * const scheduler = new LinearWarmup(optimizer, 1000);
 *
 * for (let step = 0; step < totalSteps; step++) {
 *   train();
 *   scheduler.step();
 * }
 * ```
 */
export class LinearWarmup extends LRScheduler {
  private warmupSteps: number;

  /**
   * @param optimizer - Wrapped optimizer
   * @param warmupSteps - Number of warmup steps
   * @param lastEpoch - The index of last epoch (default: -1)
   */
  constructor(optimizer: Optimizer, warmupSteps: number, lastEpoch = -1) {
    if (warmupSteps <= 0) {
      throw new Error("Warmup steps must be positive");
    }

    super(optimizer, lastEpoch);
    this.warmupSteps = warmupSteps;
  }

  protected getLr(): number[] {
    if (this.lastEpoch >= this.warmupSteps) {
      return this.baseLrs;
    }

    const multiplier = this.lastEpoch / this.warmupSteps;
    return this.baseLrs.map((baseLr) => baseLr * multiplier);
  }
}
