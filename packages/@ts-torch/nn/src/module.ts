/**
 * Base Module class for all neural network layers with advanced type safety
 *
 * Provides type-safe shape inference, module composition via .pipe(), and
 * parameter management for training.
 */

import type { Shape, DType, DeviceType } from '@ts-torch/core'

/**
 * Tensor interface matching core implementation
 *
 * This is a minimal interface for type checking in the nn package.
 * The actual Tensor implementation is in @ts-torch/core.
 *
 * @template S - Tensor shape (e.g., readonly [32, 784] for a batch of 32 784-dimensional vectors)
 * @template D - Data type (e.g., DType<'float32'>)
 * @template Dev - Device type ('cpu' | 'cuda' | 'mps')
 *
 * @remarks
 * Binary operations like matmul and add enforce device consistency through
 * the Dev type parameter - both operands must be on the same device.
 */
export interface Tensor<S extends Shape = Shape, D extends DType<string> = DType<any>, Dev extends DeviceType = DeviceType> {
  readonly shape: S
  readonly dtype: D
  readonly device: Dev

  // Core operations that modules might need
  relu(): Tensor<S, D, Dev>
  sigmoid(): Tensor<S, D, Dev>
  tanh(): Tensor<S, D, Dev>
  softmax(dim: number): Tensor<S, D, Dev>
  matmul<S2 extends Shape>(other: Tensor<S2, D, Dev>): Tensor<any, D, Dev>
  add<S2 extends Shape>(other: Tensor<S2, D, Dev> | number): Tensor<any, D, Dev>
  transpose(dim0: number, dim1: number): Tensor<any, D, Dev>
  escape(): this
  free(): void

  // Device operations (copy semantics - source remains valid)
  to<TargetDev extends DeviceType>(device: TargetDev): Tensor<S, D, TargetDev>
  cpu(): Tensor<S, D, 'cpu'>
  cuda(index?: number): Tensor<S, D, 'cuda'>

  // Move operations (move semantics - source becomes invalid after call)
  move<TargetDev extends DeviceType>(device: TargetDev): Tensor<S, D, TargetDev>
  moveCpu(): Tensor<S, D, 'cpu'>
  moveCuda(index?: number): Tensor<S, D, 'cuda'>

  // Gradient operations
  detach(): Tensor<S, D, Dev>
  requiresGrad: boolean
}

/**
 * Default float32 dtype type
 */
export type float32 = DType<'float32'>

/**
 * Parameter wrapper for trainable tensors
 *
 * Wraps a tensor and enables gradient tracking for training.
 *
 * @template S - Tensor shape
 * @template D - Data type
 * @template Dev - Device type
 */
export class Parameter<S extends Shape = Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType> {
  private _requiresGrad: boolean

  constructor(
    public data: Tensor<S, D, Dev>,
    requiresGrad: boolean = true,
  ) {
    this._requiresGrad = requiresGrad

    // Set requires_grad on the underlying tensor
    if (requiresGrad && 'requiresGrad' in data) {
      ;(data as any).requiresGrad = true
    }
  }

  /**
   * Check if this parameter requires gradient
   */
  get requiresGrad(): boolean {
    return this._requiresGrad
  }

  /**
   * Set whether this parameter requires gradient
   */
  set requiresGrad(value: boolean) {
    this._requiresGrad = value
    if ('requiresGrad' in this.data) {
      ;(this.data as any).requiresGrad = value
    }
  }

  /**
   * Get the gradient of this parameter
   */
  get grad(): Tensor<S, D, Dev> | null {
    if ('grad' in this.data) {
      return (this.data as any).grad
    }
    return null
  }

  /**
   * Zero out the gradient
   */
  zeroGrad(): void {
    if ('zeroGrad' in this.data && typeof (this.data as any).zeroGrad === 'function') {
      ;(this.data as any).zeroGrad()
    }
  }
}

/**
 * Base class for all neural network modules with advanced type safety
 *
 * Type parameters:
 * @template InShape - Input tensor shape
 * @template OutShape - Output tensor shape
 * @template D - Data type (defaults to float32)
 * @template Dev - Device type (defaults to any device)
 *
 * @example
 * ```ts
 * // Define a typed layer
 * class MyLayer extends Module<
 *   readonly [number, 128], // Input: [Batch, 128]
 *   readonly [number, 64],  // Output: [Batch, 64]
 * > {
 *   forward(input: Tensor<readonly [number, 128]>) {
 *     return this.process(input); // Type-checked!
 *   }
 * }
 *
 * // Chain modules with .pipe()
 * const model = new Linear(128, 64)
 *   .pipe(new ReLU())
 *   .pipe(new Linear(64, 10));
 * ```
 */
export abstract class Module<
  InShape extends Shape = Shape,
  OutShape extends Shape = Shape,
  D extends DType<string> = float32,
  Dev extends DeviceType = DeviceType,
> {
  protected _training = true
  protected _parameters: Map<string, Parameter<any, D, Dev>> = new Map()
  protected _modules: Map<string, Module<any, any, D, Dev>> = new Map()

  /**
   * Forward pass - must be implemented by subclasses
   *
   * @param input - Input tensor with shape InShape
   * @returns Output tensor with shape OutShape
   */
  abstract forward(input: Tensor<InShape, D, Dev>): Tensor<OutShape, D, Dev>

  /**
   * Callable syntax for forward pass
   * Allows: model(input) instead of model.forward(input)
   *
   * @param input - Input tensor
   * @returns Output tensor
   */
  __call__(input: Tensor<InShape, D, Dev>): Tensor<OutShape, D, Dev> {
    return this.forward(input)
  }

  /**
   * Compose this module with another module via type-safe piping
   *
   * The output shape of this module must match the input shape of the next module.
   * This is enforced at compile time!
   *
   * @template NextOut - Output shape of the next module
   * @param next - Module to pipe into (its input shape must match our output shape)
   * @returns A composed PipedModule
   *
   * @example
   * ```ts
   * const layer1 = new Linear(784, 128);
   * const layer2 = new Linear(128, 10);
   * const composed = layer1.pipe(layer2); // Type-safe!
   *
   * // This would be a compile error:
   * const layer3 = new Linear(64, 10);
   * const invalid = layer1.pipe(layer3); // ERROR: 128 !== 64
   * ```
   */
  pipe<NextOut extends Shape>(next: Module<OutShape, NextOut, D, Dev>): PipedModule<InShape, OutShape, NextOut, D, Dev> {
    return new PipedModule<InShape, OutShape, NextOut, D, Dev>(this, next)
  }

  /**
   * Set training mode
   *
   * @param mode - True for training, false for evaluation
   * @returns this for chaining
   */
  train(mode: boolean = true): this {
    this._training = mode

    // Propagate to submodules
    for (const module of this._modules.values()) {
      module.train(mode)
    }

    return this
  }

  /**
   * Set evaluation mode (disables dropout, etc.)
   *
   * @returns this for chaining
   */
  eval(): this {
    return this.train(false)
  }

  /**
   * Check if module is in training mode
   */
  get training(): boolean {
    return this._training
  }

  /**
   * Get all parameters in this module (including nested modules)
   *
   * @returns Array of all parameters
   */
  parameters(): Parameter<any, D, Dev>[] {
    const params: Parameter<any, D, Dev>[] = Array.from(this._parameters.values())

    for (const module of this._modules.values()) {
      params.push(...module.parameters())
    }

    return params
  }

  /**
   * Get all parameters with their names (including nested modules)
   * Names use dot notation for nested parameters: "layer1.weight"
   *
   * @returns Map of parameter name to Parameter
   */
  namedParameters(): Map<string, Parameter<any, D, Dev>> {
    const namedParams = new Map(this._parameters)

    for (const [name, module] of this._modules.entries()) {
      for (const [paramName, param] of module.namedParameters().entries()) {
        namedParams.set(`${name}.${paramName}`, param)
      }
    }

    return namedParams
  }

  /**
   * Move module to specified device (cpu, cuda, mps)
   *
   * **IMPORTANT: Mutation semantics**
   *
   * This method mutates the module in-place AND returns `this` with an updated type.
   * This matches PyTorch's behavior. After calling `.to()`, only use the returned
   * reference - the original reference's type will be stale.
   *
   * @template TargetDev - Target device type
   * @param device - Target device
   * @returns this (mutated) with updated device type
   *
   * @example
   * ```ts
   * // CORRECT: Use the returned reference
   * const model = new Linear(10, 5).to('cuda')  // model: Linear<..., 'cuda'>
   *
   * // CORRECT: Chain calls
   * const cudaModel = nn.sequence(cuda, [
   *   nn.linear(784, 128),
   *   nn.relu()
   * ])  // sequence() calls .to() internally
   *
   * // INCORRECT: Don't use original reference after .to()
   * const cpuModel = new Linear(10, 5)  // cpuModel: Linear<..., 'cpu'>
   * const cudaModel = cpuModel.to('cuda')
   * // cpuModel's type is now stale - data is actually on 'cuda'!
   * ```
   */
  to<TargetDev extends DeviceType>(device: TargetDev): Module<InShape, OutShape, D, TargetDev> {
    // Move all parameters to the device
    // We detach first to create a fresh leaf tensor, then move and re-enable gradients
    for (const param of this.parameters()) {
      const wasRequiresGrad = param.data.requiresGrad
      const oldData = param.data // Keep reference for cleanup

      // Detach to break gradient history (creates intermediate tensor)
      const detached = oldData.detach()

      // Move to target device (creates final tensor)
      const moved = detached.to(device)

      // FREE old data and intermediate tensors to prevent memory leak
      oldData.free()
      detached.free()

      if (wasRequiresGrad) {
        moved.requiresGrad = true
      }

      // Mutation: param.data is replaced with moved tensor
      // Cast is intentional - we're mutating the parameter's data in place
      ;(param as unknown as Parameter<any, D, TargetDev>).data = moved
    }

    // Recursively move submodules
    for (const child of this._modules.values()) {
      child.to(device)
    }

    // Return this with updated type - caller should use returned reference
    return this as unknown as Module<InShape, OutShape, D, TargetDev>
  }

  /**
   * Zero all gradients
   */
  zeroGrad(): void {
    for (const param of this.parameters()) {
      param.zeroGrad()
    }
  }

  /**
   * Register a parameter
   *
   * @param name - Parameter name
   * @param param - Parameter to register
   */
  protected registerParameter(name: string, param: Parameter<any, D, Dev>): void {
    this._parameters.set(name, param)
  }

  /**
   * Register a submodule
   *
   * @param name - Module name
   * @param module - Module to register
   */
  protected registerModule(name: string, module: Module<any, any, D, Dev>): void {
    this._modules.set(name, module)
  }

  /**
   * String representation of the module
   */
  toString(): string {
    return `${this.constructor.name}()`
  }
}

/**
 * Piped module created by .pipe() composition
 *
 * Chains two modules together: output of first becomes input of second.
 * Type-safe: ensures intermediate shapes match at compile time.
 *
 * @template In - Input shape to first module
 * @template Mid - Intermediate shape (output of first, input of second)
 * @template Out - Final output shape
 * @template D - Data type
 * @template Dev - Device type
 *
 * @internal
 */
export class PipedModule<
  In extends Shape,
  Mid extends Shape,
  Out extends Shape,
  D extends DType<string> = float32,
  Dev extends DeviceType = DeviceType,
> extends Module<In, Out, D, Dev> {
  constructor(
    private first: Module<In, Mid, D, Dev>,
    private second: Module<Mid, Out, D, Dev>,
  ) {
    super()
    this.registerModule('0', this.first)
    this.registerModule('1', this.second)
  }

  /**
   * Forward pass through both modules in sequence
   *
   * @param input - Input tensor
   * @returns Output tensor after both transformations
   */
  forward(input: Tensor<In, D, Dev>): Tensor<Out, D, Dev> {
    const intermediate = this.first.forward(input)
    return this.second.forward(intermediate)
  }

  /**
   * Override pipe to support further chaining
   * This allows: a.pipe(b).pipe(c).pipe(d)...
   */
  override pipe<NextOut extends Shape>(next: Module<Out, NextOut, D, Dev>): PipedModule<In, Out, NextOut, D, Dev> {
    return new PipedModule<In, Out, NextOut, D, Dev>(this, next)
  }

  override toString(): string {
    return `PipedModule(\n  ${this.first.toString()}\n  -> ${this.second.toString()}\n)`
  }
}
