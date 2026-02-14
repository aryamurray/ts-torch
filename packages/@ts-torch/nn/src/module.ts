/**
 * Base Module class for all neural network layers with advanced type safety
 *
 * Provides type-safe shape inference, module composition via .pipe(), and
 * parameter management for training.
 */

import type { Shape, DType, DeviceType } from '@ts-torch/core'
import { fromArray, DType as DTypeNs } from '@ts-torch/core'
import type { StateDict } from './safetensors.js'
import { validateStateDict } from './validation.js'

/**
 * Convert a simple glob pattern to a RegExp.
 * Supports `*` as wildcard (matches any characters including dots).
 */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&')
  return new RegExp('^' + escaped.replace(/\*/g, '.*') + '$')
}

/**
 * Options for loadWeights()
 */
export interface LoadWeightsOptions {
  strict?: boolean
  include?: string[]
  exclude?: string[]
}

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
  logSoftmax(dim: number): Tensor<S, D, Dev>
  exp(): Tensor<S, D, Dev>
  log(): Tensor<S, D, Dev>
  sqrt(): Tensor<S, D, Dev>
  clamp(min: number, max: number): Tensor<S, D, Dev>
  clampMin(min: number): Tensor<S, D, Dev>
  clampMax(max: number): Tensor<S, D, Dev>
  dropout(p?: number, training?: boolean): Tensor<S, D, Dev>
  sumDim(dim: number, keepdim?: boolean): Tensor<any, D, Dev>
  matmul<S2 extends Shape>(other: Tensor<S2, D, Dev>): Tensor<any, D, Dev>
  add<S2 extends Shape>(other: Tensor<S2, D, Dev> | number): Tensor<any, D, Dev>
  addScalar(scalar: number): Tensor<S, D, Dev>
  sub<S2 extends Shape>(other: Tensor<S2, D, Dev>): Tensor<any, D, Dev>
  mul<S2 extends Shape>(other: Tensor<S2, D, Dev>): Tensor<any, D, Dev>
  mulScalar(scalar: number): Tensor<S, D, Dev>
  div<S2 extends Shape>(other: Tensor<S2, D, Dev>): Tensor<any, D, Dev>
  maximum<S2 extends Shape>(other: Tensor<S2, D, Dev>): Tensor<any, D, Dev>
  transpose(dim0: number, dim1: number): Tensor<any, D, Dev>

  // In-place operations (for optimizers)
  addScaledInplace(other: Tensor<S, D, Dev>, scalar: number): void
  addInplace(other: Tensor<S, D, Dev>): void
  subInplace(other: Tensor<S, D, Dev>): void
  mulInplace(other: Tensor<S, D, Dev>): void
  mulScalarInplace(scalar: number): void

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

  /**
   * In-place scaled add: data += scalar * other
   * Used by optimizers for efficient parameter updates
   */
  addScaledInplace(other: Tensor<S, D, Dev>, scalar: number): void {
    if ('addScaledInplace' in this.data && typeof (this.data as any).addScaledInplace === 'function') {
      ;(this.data as any).addScaledInplace(other, scalar)
    }
  }

  /**
   * In-place add: data += other
   */
  addInplace(other: Tensor<S, D, Dev>): void {
    if ('addInplace' in this.data && typeof (this.data as any).addInplace === 'function') {
      ;(this.data as any).addInplace(other)
    }
  }

  /**
   * In-place subtract: data -= other
   */
  subInplace(other: Tensor<S, D, Dev>): void {
    if ('subInplace' in this.data && typeof (this.data as any).subInplace === 'function') {
      ;(this.data as any).subInplace(other)
    }
  }

  /**
   * In-place multiply: data *= other
   */
  mulInplace(other: Tensor<S, D, Dev>): void {
    if ('mulInplace' in this.data && typeof (this.data as any).mulInplace === 'function') {
      ;(this.data as any).mulInplace(other)
    }
  }

  /**
   * In-place scalar multiply: data *= scalar
   */
  mulScalarInplace(scalar: number): void {
    if ('mulScalarInplace' in this.data && typeof (this.data as any).mulScalarInplace === 'function') {
      ;(this.data as any).mulScalarInplace(scalar)
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
export class Module<
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
   *
   * Note: This is not abstract to allow subclasses (like Transformers) to
   * override with different signatures (e.g., multiple input tensors).
   */
  forward(_input: Tensor<InShape, D, Dev>): Tensor<OutShape, D, Dev> {
    throw new Error(`forward() must be implemented by ${this.constructor.name}`)
  }

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
   * Get all child modules registered with this module
   *
   * @returns Map of module name to Module
   */
  modules(): Map<string, Module<any, any, D, Dev>> {
    return this._modules
  }

  /**
   * Get all modules recursively with dot-notation names
   * Includes this module at '' key (PyTorch convention).
   *
   * @returns Map of module path to Module
   */
  namedModules(): Map<string, Module<any, any, D, Dev>> {
    const result = new Map<string, Module<any, any, D, Dev>>()
    result.set('', this)
    for (const [name, module] of this._modules.entries()) {
      result.set(name, module)
      for (const [childName, childModule] of module.namedModules().entries()) {
        if (childName === '') continue
        result.set(`${name}.${childName}`, childModule)
      }
    }
    return result
  }

  /**
   * Count the total number of parameters in this module
   *
   * @param filter - 'all' (default), 'trainable', or 'frozen'
   * @returns Total element count across all parameters
   */
  parameterCount(filter: 'all' | 'trainable' | 'frozen' = 'all'): number {
    let count = 0
    for (const param of this.parameters()) {
      if (filter === 'trainable' && !param.requiresGrad) continue
      if (filter === 'frozen' && param.requiresGrad) continue
      const shape = param.data.shape as readonly number[]
      count += shape.reduce((a, d) => a * d, 1)
    }
    return count
  }

  /**
   * Count parameters that belong directly to this module (not children)
   */
  protected _directParameterCount(): number {
    let count = 0
    for (const param of this._parameters.values()) {
      const shape = param.data.shape as readonly number[]
      count += shape.reduce((a, d) => a * d, 1)
    }
    return count
  }

  /**
   * Return an output shape hint string for summary display.
   * Override in shape-transforming modules. Returns null by default,
   * which means summary() carries forward the previous module's shape.
   */
  protected _outputShapeHint(): string | null {
    return null
  }

  /**
   * Print a formatted summary table of this model's layers, shapes, and parameter counts.
   *
   * @returns Formatted table string
   */
  summary(): string {
    const modules = this.namedModules()
    const rows: [string, string, string, number][] = []
    let currentShape = '-'

    for (const [name, mod] of modules) {
      if (name === '') continue
      const hint = mod._outputShapeHint()
      if (hint !== null) currentShape = hint
      rows.push([name, mod.constructor.name, currentShape, mod._directParameterCount()])
    }

    // Calculate column widths
    const headers = ['Layer', 'Type', 'Output Shape', 'Params']
    const formattedParams = rows.map(r => r[3].toLocaleString('en-US'))
    const totalParams = this.parameterCount()
    const trainableParams = this.parameterCount('trainable')
    const frozenParams = this.parameterCount('frozen')
    const totalFormatted = totalParams.toLocaleString('en-US')

    const colWidths = [
      Math.max(headers[0]!.length, ...rows.map(r => r[0].length), 'Total'.length),
      Math.max(headers[1]!.length, ...rows.map(r => r[1].length)),
      Math.max(headers[2]!.length, ...rows.map(r => r[2].length)),
      Math.max(headers[3]!.length, ...formattedParams.map(p => p.length), totalFormatted.length),
    ]

    const pad = (s: string, w: number) => s + ' '.repeat(Math.max(0, w - s.length))
    const padRight = (s: string, w: number) => ' '.repeat(Math.max(0, w - s.length)) + s
    const hLine = (left: string, mid: string, right: string) =>
      `${left}${colWidths.map(w => '─'.repeat(w + 2)).join(mid)}${right}`

    const lines: string[] = []
    lines.push(hLine('┌', '┬', '┐'))
    lines.push(`│ ${pad(headers[0]!, colWidths[0]!)} │ ${pad(headers[1]!, colWidths[1]!)} │ ${pad(headers[2]!, colWidths[2]!)} │ ${padRight(headers[3]!, colWidths[3]!)} │`)
    lines.push(hLine('├', '┼', '┤'))

    for (let i = 0; i < rows.length; i++) {
      const [name, type, shape] = rows[i]!
      lines.push(`│ ${pad(name, colWidths[0]!)} │ ${pad(type, colWidths[1]!)} │ ${pad(shape, colWidths[2]!)} │ ${padRight(formattedParams[i]!, colWidths[3]!)} │`)
    }

    lines.push(hLine('├', '┼', '┤'))
    lines.push(`│ ${pad('Total', colWidths[0]!)} │ ${pad('', colWidths[1]!)} │ ${pad('', colWidths[2]!)} │ ${padRight(totalFormatted, colWidths[3]!)} │`)
    lines.push(hLine('└', '┴', '┘'))
    lines.push(`Trainable params: ${trainableParams.toLocaleString('en-US')}`)
    lines.push(`Non-trainable params: ${frozenParams.toLocaleString('en-US')}`)

    return lines.join('\n')
  }

  /**
   * Freeze parameters (set requiresGrad = false).
   * If pattern is provided, only matching parameter names are frozen (glob syntax).
   *
   * @param pattern - Optional glob pattern to match parameter names
   * @returns this for chaining
   */
  freeze(pattern?: string): this {
    const regex = pattern ? globToRegex(pattern) : null
    for (const [name, param] of this.namedParameters()) {
      if (regex && !regex.test(name)) continue
      param.requiresGrad = false
    }
    return this
  }

  /**
   * Unfreeze parameters (set requiresGrad = true).
   * If pattern is provided, only matching parameter names are unfrozen (glob syntax).
   *
   * @param pattern - Optional glob pattern to match parameter names
   * @returns this for chaining
   */
  unfreeze(pattern?: string): this {
    const regex = pattern ? globToRegex(pattern) : null
    for (const [name, param] of this.namedParameters()) {
      if (regex && !regex.test(name)) continue
      param.requiresGrad = true
    }
    return this
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
   * // CORRECT: Use fluent builders with .init()
   * const cudaModel = nn.sequence(
   *   nn.input(784),
   *   nn.fc(128).relu(),
   *   nn.fc(10)
   * ).init(device.cuda(0))
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
   * Get a serializable state dictionary containing all parameters
   *
   * @returns Record mapping parameter names to TensorData
   *
   * @example
   * ```ts
   * const state = model.stateDict()
   * await saveSafetensors('./model.safetensors', state)
   * ```
   */
  stateDict(): StateDict {
    const state: StateDict = {}

    for (const [name, param] of this.namedParameters()) {
      const tensor = param.data as any

      // Extract data using toArray() which returns the correct TypedArray per dtype
      if (typeof tensor.toArray !== 'function') {
        throw new Error(`Cannot serialize parameter "${name}": tensor has no toArray() method`)
      }
      const data = tensor.toArray()

      // Extract shape
      const shape: number[] = Array.isArray(tensor.shape) ? [...tensor.shape] : [data.length]

      // Read dtype from tensor
      if (!tensor.dtype?.name) {
        throw new Error(`Cannot serialize parameter "${name}": tensor has no dtype.name`)
      }
      const dtype: string = tensor.dtype.name

      state[name] = { data, shape, dtype }
    }

    return state
  }

  /**
   * Load parameters from a state dictionary
   *
   * @param state - State dictionary from stateDict() or checkpoint
   * @param strict - If true (default), throws if keys don't match exactly
   *
   * @example
   * ```ts
   * const { tensors } = await loadSafetensors('./model.safetensors')
   * model.loadStateDict(tensors)
   * ```
   */
  loadStateDict(state: StateDict, strict: boolean = true): void {
    // Validate first, mutate second
    validateStateDict(this, state, strict)

    const currentParams = this.namedParameters()

    // Load parameters by creating new tensors and replacing old ones
    for (const [name, param] of currentParams) {
      const tensorData = state[name]
      if (!tensorData) continue

      const oldTensor = param.data as any

      // Resolve dtype object from string name
      const dtypeObj = DTypeNs[tensorData.dtype as keyof typeof DTypeNs]
      if (!dtypeObj) {
        throw new Error(`Unknown dtype: ${tensorData.dtype}`)
      }

      // Create new tensor from state dict data
      const newTensor = fromArray(
        tensorData.data,
        tensorData.shape as readonly number[],
        dtypeObj,
      )

      // Move to same device as old tensor if needed
      const device = oldTensor.device
      let finalTensor = newTensor
      if (device && device !== 'cpu') {
        finalTensor = newTensor.to(device)
        newTensor.free()
      }

      // Preserve requiresGrad
      if (oldTensor.requiresGrad) {
        finalTensor.requiresGrad = true
      }

      // Free old tensor and replace
      oldTensor.free()
      ;(param as any).data = finalTensor
    }
  }

  /**
   * Save model to a directory (config.json + model.safetensors).
   * Requires the model to have been created via config.init() or config.load().
   *
   * @param directory - Output directory path
   * @param metadata - Optional metadata (epoch, loss, etc.) stored in safetensors header
   */
  async save(directory: string, metadata?: Record<string, unknown>): Promise<void> {
    const config = (this as any)._config
    if (!config) {
      throw new Error(
        'Cannot save: model has no _config. Model must be created via config.init() or config.load().',
      )
    }

    const { mkdir, writeFile, rename, rm } = await import('node:fs/promises')
    const { join, dirname, basename } = await import('node:path')
    const { randomUUID } = await import('node:crypto')
    const { saveSafetensors, serializeMetadata } = await import('./safetensors.js')

    const tmpDir = join(dirname(directory), `.${basename(directory)}.tmp-${randomUUID().slice(0, 8)}`)
    try {
      await mkdir(tmpDir, { recursive: true })
      await writeFile(join(tmpDir, 'config.json'), JSON.stringify(config, null, 2))
      await saveSafetensors(
        join(tmpDir, 'model.safetensors'),
        this.stateDict(),
        serializeMetadata(metadata),
      )
      await rm(directory, { recursive: true, force: true }).catch(() => {})
      await rename(tmpDir, directory)
    } catch (err) {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
      throw err
    }
  }

  /**
   * Load weights from a directory containing model.safetensors into this model.
   * Useful for transfer learning with an existing model instance.
   *
   * @param directory - Directory containing model.safetensors
   * @param options - boolean for backward compat (strict mode), or LoadWeightsOptions
   * @returns Deserialized metadata from the safetensors file
   */
  async loadWeights(
    directory: string,
    options?: boolean | LoadWeightsOptions,
  ): Promise<Record<string, unknown>> {
    const { join } = await import('node:path')
    const { loadSafetensors, deserializeMetadata } = await import('./safetensors.js')

    // Normalize options
    let strict: boolean
    let include: string[] | undefined
    let exclude: string[] | undefined

    if (typeof options === 'boolean') {
      strict = options
    } else if (options) {
      include = options.include
      exclude = options.exclude
      // When include/exclude used, strict defaults to false
      strict = options.strict ?? (include || exclude ? false : true)
    } else {
      strict = true
    }

    const { tensors, metadata } = await loadSafetensors(join(directory, 'model.safetensors'))

    // Filter state dict keys if include/exclude specified
    if (include || exclude) {
      const includeRegexes = include?.map(globToRegex)
      const excludeRegexes = exclude?.map(globToRegex)

      for (const key of Object.keys(tensors)) {
        // If include is specified, key must match at least one pattern
        if (includeRegexes && !includeRegexes.some(r => r.test(key))) {
          delete tensors[key]
          continue
        }
        // If exclude is specified, key must not match any pattern
        if (excludeRegexes && excludeRegexes.some(r => r.test(key))) {
          delete tensors[key]
        }
      }
    }

    this.loadStateDict(tensors, strict)
    return deserializeMetadata(metadata)
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
