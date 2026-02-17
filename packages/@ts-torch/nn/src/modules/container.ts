/**
 * Container modules for composing neural network layers
 *
 * Provides Sequential for building linear pipelines with type-safe shape inference.
 */

import { Module, type Tensor, type float32 } from '../module.js'
import type { Shape, DType, DeviceType } from '@ts-torch/core'

/**
 * Sequential container for linear module composition
 *
 * Chains multiple modules together in sequence. Type inference ensures
 * that output shapes match input shapes throughout the pipeline.
 *
 * @template In - Input shape to first module
 * @template Out - Output shape from last module
 * @template D - Data type (default: float32)
 * @template Dev - Device type (default: any device)
 *
 * @example
 * ```ts
 * // Type-safe sequential construction
 * const model = new Sequential<
 *   readonly [number, 784],  // Input shape
 *   readonly [number, 10],   // Output shape
 * >(
 *   new Linear(784, 128),
 *   new ReLU(),
 *   new Linear(128, 10),
 *   new Softmax()
 * );
 *
 * // Type inference works!
 * const input: Tensor<readonly [32, 784]> = ...;
 * const output = model.forward(input); // Type: Tensor<readonly [32, 10]>
 * ```
 */
export class Sequential<
  In extends Shape = Shape,
  Out extends Shape = Shape,
  D extends DType<string> = float32,
  Dev extends DeviceType = DeviceType,
> extends Module<In, Out, D, Dev> {
  private readonly _layers: Module<any, any, D, Dev>[]

  /** Serialized config from SequenceDef.toJSON(), set during init()/load() */
  _config?: object

  /**
   * Create a new Sequential container
   *
   * @param modules - Modules to chain together in order
   *
   * @remarks
   * TypeScript cannot automatically infer the intermediate shapes when
   * modules are passed as rest parameters. For full type safety, use
   * the explicit type parameters or use .pipe() chaining instead.
   */
  constructor(...modules: Module<any, any, D, Dev>[]) {
    super()

    if (modules.length === 0) {
      throw new Error('Sequential requires at least one module')
    }

    this._layers = modules

    // Register all modules with numeric keys
    modules.forEach((module, index) => {
      this.registerModule(String(index), module)
    })
  }

  /**
   * Forward pass through all modules in sequence
   *
   * @param input - Input tensor
   * @returns Output tensor after all transformations
   */
  forward(input: Tensor<In, D, Dev>): Tensor<Out, D, Dev> {
    let output: any = input

    for (const module of this._layers) {
      output = module.forward(output) as Tensor<Shape, D, Dev>
    }

    return output as Tensor<Out, D, Dev>
  }

  /**
   * Append a module to the end of the sequential
   *
   * @param module - Module to append
   * @returns New Sequential with appended module
   */
  append<NextOut extends Shape>(module: Module<Out, NextOut, D, Dev>): Sequential<In, NextOut, D, Dev> {
    return new Sequential<In, NextOut, D, Dev>(...this._layers, module)
  }

  /**
   * Get module at specified index
   *
   * @param index - Module index
   * @returns Module at index
   */
  at(index: number): Module<any, any, D, Dev> | undefined {
    return this._layers[index]
  }

  /**
   * Get number of modules in sequential
   */
  get length(): number {
    return this._layers.length
  }

  /**
   * Iterate over modules
   */
  *[Symbol.iterator](): Iterator<Module<any, any, D, Dev>> {
    yield* this._layers
  }

  override toString(): string {
    const moduleStrs = this._layers.map((m, i) => `  (${i}): ${m.toString()}`)
    return `Sequential(\n${moduleStrs.join('\n')}\n)`
  }
}

/**
 * Type-safe Sequential builder with full shape inference
 *
 * This helper provides better type inference than the Sequential constructor
 * by using method chaining to track shapes at each step.
 *
 * @template In - Current input shape
 * @template D - Data type
 * @template Dev - Device type
 *
 * @example
 * ```ts
 * // Full type inference through the chain!
 * const model = sequential<readonly [number, 784]>()
 *   .add(new Linear(784, 128))
 *   .add(new ReLU())
 *   .add(new Linear(128, 64))
 *   .add(new ReLU())
 *   .add(new Linear(64, 10))
 *   .build();
 *
 * // Type is: Sequential<readonly [number, 784], readonly [number, 10]>
 * ```
 */
export class SequentialBuilder<
  In extends Shape = Shape,
  Out extends Shape = Shape,
  D extends DType<string> = float32,
  Dev extends DeviceType = DeviceType,
> {
  private modules: Module<any, any, D, Dev>[] = []

  private constructor(
    private readonly inputShape?: In,
    modules: Module<any, any, D, Dev>[] = [],
  ) {
    this.modules = modules
  }

  /**
   * Create a new sequential builder
   *
   * @template In - Input shape for the first module
   * @param _inputShape - Phantom parameter for type inference (not used at runtime)
   * @returns New sequential builder
   *
   * @example
   * ```ts
   * const builder = sequential<readonly [number, 784]>();
   * ```
   */
  static create<In extends Shape, D extends DType<string> = float32, Dev extends DeviceType = DeviceType>(
    _inputShape?: In,
  ): SequentialBuilder<In, In, D, Dev> {
    return new SequentialBuilder<In, In, D, Dev>(_inputShape)
  }

  /**
   * Add a module to the sequential
   *
   * @template NextOut - Output shape of the module being added
   * @param module - Module to add (must accept current Out as input)
   * @returns Updated builder with new output shape
   */
  add<NextOut extends Shape>(module: Module<Out, NextOut, D, Dev>): SequentialBuilder<In, NextOut, D, Dev> {
    const newBuilder = new SequentialBuilder<In, NextOut, D, Dev>(this.inputShape, [...this.modules, module])
    return newBuilder
  }

  /**
   * Build the final Sequential module
   *
   * @returns Sequential module with full type information
   */
  build(): Sequential<In, Out, D, Dev> {
    return new Sequential<In, Out, D, Dev>(...this.modules)
  }
}

/**
 * Helper function to create a type-safe sequential builder
 *
 * @template In - Input shape
 * @template D - Data type
 * @template Dev - Device type
 * @returns Sequential builder
 *
 * @example
 * ```ts
 * const model = sequential<readonly [number, 784]>()
 *   .add(new Linear(784, 128))
 *   .add(new ReLU())
 *   .add(new Linear(128, 10))
 *   .build();
 * ```
 */
export function sequential<
  In extends Shape,
  D extends DType<string> = float32,
  Dev extends DeviceType = DeviceType,
>(): SequentialBuilder<In, In, D, Dev> {
  return SequentialBuilder.create<In, D, Dev>()
}

/**
 * Dynamic list of modules where parameters are visible in parameters() and stateDict().
 * Modules are registered with numeric string keys ("0", "1", ...).
 *
 * Users iterate manually in their custom forward — ModuleList does not define forward().
 */
export class ModuleList<D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<
  any,
  any,
  D,
  Dev
> {
  private _list: Module<any, any, D, Dev>[] = []

  constructor(modules?: Module<any, any, D, Dev>[]) {
    super()
    if (modules) {
      for (const m of modules) {
        this.append(m)
      }
    }
  }

  append(module: Module<any, any, D, Dev>): this {
    this.registerModule(String(this._list.length), module)
    this._list.push(module)
    return this
  }

  at(index: number): Module<any, any, D, Dev> | undefined {
    return this._list[index]
  }

  get length(): number {
    return this._list.length
  }

  *[Symbol.iterator](): Iterator<Module<any, any, D, Dev>> {
    yield* this._list
  }
}

/**
 * Dynamic named collection of modules where parameters are visible in parameters() and stateDict().
 * Modules are registered with their string key.
 */
export class ModuleDict<D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<
  any,
  any,
  D,
  Dev
> {
  private _dict: Map<string, Module<any, any, D, Dev>> = new Map()

  constructor(modules?: Record<string, Module<any, any, D, Dev>>) {
    super()
    if (modules) {
      for (const [k, v] of Object.entries(modules)) {
        this.set(k, v)
      }
    }
  }

  set(key: string, module: Module<any, any, D, Dev>): this {
    this.registerModule(key, module)
    this._dict.set(key, module)
    return this
  }

  get(key: string): Module<any, any, D, Dev> | undefined {
    return this._dict.get(key)
  }

  has(key: string): boolean {
    return this._dict.has(key)
  }

  keys(): IterableIterator<string> {
    return this._dict.keys()
  }

  values(): IterableIterator<Module<any, any, D, Dev>> {
    return this._dict.values()
  }

  entries(): IterableIterator<[string, Module<any, any, D, Dev>]> {
    return this._dict.entries()
  }

  get size(): number {
    return this._dict.size
  }
}

/**
 * Multi-head model: shared backbone + named head branches.
 * Created by the builder API when nn.heads() is the terminal block.
 *
 * State dict keys: shared.0.weight, head.pi.0.weight, head.vf.0.weight
 */
export class HeadedSequential<D extends DType<string> = float32, Dev extends DeviceType = DeviceType> extends Module<
  Shape,
  Shape,
  D,
  Dev
> {
  private _shared: Sequential<Shape, Shape, D, Dev>
  private _heads: Map<string, Sequential<Shape, Shape, D, Dev>>
  private _defaultHead: string

  /** Serialized config from SequenceDef.toJSON(), set during init()/load() */
  _config?: object

  constructor(
    shared: Sequential<Shape, Shape, D, Dev>,
    heads: Record<string, Sequential<Shape, Shape, D, Dev>>,
    defaultHead?: string,
  ) {
    super()
    this._shared = shared
    this._heads = new Map(Object.entries(heads))
    this._defaultHead = defaultHead ?? Object.keys(heads)[0]!

    this.registerModule('shared', shared)
    for (const [name, head] of Object.entries(heads)) {
      this.registerModule(`head.${name}`, head)
    }
  }

  forward(input: Tensor<Shape, D, Dev>): Tensor<Shape, D, Dev>
  forward(input: Tensor<Shape, D, Dev>, headName: string): Tensor<Shape, D, Dev>
  forward(input: Tensor<Shape, D, Dev>, headName?: string): Tensor<Shape, D, Dev> {
    const features = this._shared.forward(input)
    const head = this._heads.get(headName ?? this._defaultHead)
    if (!head) {
      throw new Error(`Unknown head: "${headName}". Available heads: ${[...this._heads.keys()].join(', ')}`)
    }
    return head.forward(features)
  }

  forwardAll(input: Tensor<Shape, D, Dev>): Record<string, Tensor<Shape, D, Dev>> {
    const features = this._shared.forward(input)
    const result: Record<string, Tensor<Shape, D, Dev>> = {}
    for (const [name, head] of this._heads) {
      result[name] = head.forward(features)
    }
    return result
  }

  get headNames(): string[] {
    return [...this._heads.keys()]
  }

  get defaultHeadName(): string {
    return this._defaultHead
  }

  getHead(name: string): Sequential<Shape, Shape, D, Dev> | undefined {
    return this._heads.get(name)
  }
}
