/**
 * Container modules for composing neural network layers
 *
 * Provides Sequential for building linear pipelines with type-safe shape inference.
 */

import { Module, type Tensor, type float32 } from '../module.js';
import type { Shape, DType } from '@ts-torch/core';

/**
 * Sequential container for linear module composition
 *
 * Chains multiple modules together in sequence. Type inference ensures
 * that output shapes match input shapes throughout the pipeline.
 *
 * @template In - Input shape to first module
 * @template Out - Output shape from last module
 * @template D - Data type (default: float32)
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
  D extends DType<string> = float32
> extends Module<In, Out, D> {
  private readonly modules: Module<any, any, D>[];

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
  constructor(...modules: Module<any, any, D>[]) {
    super();

    if (modules.length === 0) {
      throw new Error('Sequential requires at least one module');
    }

    this.modules = modules;

    // Register all modules with numeric keys
    modules.forEach((module, index) => {
      this.registerModule(String(index), module);
    });
  }

  /**
   * Forward pass through all modules in sequence
   *
   * @param input - Input tensor
   * @returns Output tensor after all transformations
   */
  forward(input: Tensor<In, D>): Tensor<Out, D> {
    let output: any = input;

    for (const module of this.modules) {
      output = module.forward(output);
    }

    return output as Tensor<Out, D>;
  }

  /**
   * Append a module to the end of the sequential
   *
   * @param module - Module to append
   * @returns New Sequential with appended module
   */
  append<NextOut extends Shape>(
    module: Module<Out, NextOut, D>
  ): Sequential<In, NextOut, D> {
    return new Sequential<In, NextOut, D>(...this.modules, module);
  }

  /**
   * Get module at specified index
   *
   * @param index - Module index
   * @returns Module at index
   */
  at(index: number): Module<any, any, D> | undefined {
    return this.modules[index];
  }

  /**
   * Get number of modules in sequential
   */
  get length(): number {
    return this.modules.length;
  }

  /**
   * Iterate over modules
   */
  *[Symbol.iterator](): Iterator<Module<any, any, D>> {
    yield* this.modules;
  }

  override toString(): string {
    const moduleStrs = this.modules.map((m, i) => `  (${i}): ${m.toString()}`);
    return `Sequential(\n${moduleStrs.join('\n')}\n)`;
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
  D extends DType<string> = float32
> {
  private modules: Module<any, any, D>[] = [];

  private constructor(
    private readonly inputShape?: In,
    modules: Module<any, any, D>[] = []
  ) {
    this.modules = modules;
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
  static create<In extends Shape, D extends DType<string> = float32>(
    _inputShape?: In
  ): SequentialBuilder<In, In, D> {
    return new SequentialBuilder<In, In, D>(_inputShape);
  }

  /**
   * Add a module to the sequential
   *
   * @template NextOut - Output shape of the module being added
   * @param module - Module to add (must accept current Out as input)
   * @returns Updated builder with new output shape
   */
  add<NextOut extends Shape>(
    module: Module<Out, NextOut, D>
  ): SequentialBuilder<In, NextOut, D> {
    const newBuilder = new SequentialBuilder<In, NextOut, D>(
      this.inputShape,
      [...this.modules, module]
    );
    return newBuilder;
  }

  /**
   * Build the final Sequential module
   *
   * @returns Sequential module with full type information
   */
  build(): Sequential<In, Out, D> {
    return new Sequential<In, Out, D>(...this.modules);
  }
}

/**
 * Helper function to create a type-safe sequential builder
 *
 * @template In - Input shape
 * @template D - Data type
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
  D extends DType<string> = float32
>(): SequentialBuilder<In, In, D> {
  return SequentialBuilder.create<In, D>();
}
