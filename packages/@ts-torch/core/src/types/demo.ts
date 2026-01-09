// @ts-nocheck
/**
 * Practical demonstration of ts-torch type system
 *
 * This file shows real-world usage patterns and demonstrates
 * how compile-time shape checking prevents common errors.
 */

import type {
  TensorType,
  MatMulShape,
  BroadcastShape,
  TransposeShape,
  ReshapeValid,
  Dim,
  DTypeName,
  PromoteDType,
} from "./index";

import { DTypeConstants } from "./index";

// ============================================================================
// Example 1: Neural Network Layer Types
// ============================================================================

/**
 * Define a type-safe linear layer
 */
interface LinearLayer<
  InFeatures extends number,
  OutFeatures extends number,
  D extends DTypeName = "float32"
> {
  weight: TensorType<[InFeatures, OutFeatures], D>;
  bias: TensorType<[OutFeatures], D>;
}

/**
 * Forward pass type for linear layer
 * Input: [batch, in_features]
 * Output: [batch, out_features]
 */
type LinearForward<
  Batch extends Dim<"batch">,
  InFeatures extends number,
  OutFeatures extends number
> = MatMulShape<[Batch, InFeatures], [InFeatures, OutFeatures]>;

// Usage example
type BatchDim = Dim<"batch">;
type Input = TensorType<[BatchDim, 512], "float32">;
type Layer = LinearLayer<512, 256, "float32">;
type Output = TensorType<LinearForward<BatchDim, 512, 256>, "float32">;

// ============================================================================
// Example 2: Convolutional Layer Shapes
// ============================================================================

/**
 * Conv2d layer types (simplified)
 */
interface Conv2dLayer<
  InChannels extends number,
  OutChannels extends number,
  KernelSize extends number,
  D extends DTypeName = "float32"
> {
  weight: TensorType<[OutChannels, InChannels, KernelSize, KernelSize], D>;
  bias: TensorType<[OutChannels], D>;
}

// NCHW format: [batch, channels, height, width]
type Conv2dInput = TensorType<[Dim<"batch">, 3, 224, 224], "float32">;
type Conv2dLayer1 = Conv2dLayer<3, 64, 3, "float32">;

// ============================================================================
// Example 3: Multi-Head Attention
// ============================================================================

/**
 * Multi-head attention configuration
 */
interface AttentionConfig {
  hiddenDim: number;
  numHeads: number;
  headDim: number;
}

/**
 * Self-attention types
 */
interface SelfAttention<
  HiddenDim extends number,
  D extends DTypeName = "float32"
> {
  qkvProjection: TensorType<[HiddenDim, number], D>;
  outputProjection: TensorType<[HiddenDim, HiddenDim], D>;
}

// BERT-base configuration
type BertAttention = SelfAttention<768, "float32">;

// ============================================================================
// Example 4: Batch Operations with Broadcasting
// ============================================================================

/**
 * Batch normalization shapes
 */
type BatchNormInput = TensorType<[32, 64, 224, 224], "float32">; // NCHW
type BatchNormScale = TensorType<[1, 64, 1, 1], "float32">; // Broadcastable

// The scale broadcasts to match input
type BatchNormOutput = TensorType<
  BroadcastShape<[32, 64, 224, 224], [1, 64, 1, 1]>,
  "float32"
>;

// ============================================================================
// Example 5: Reshape and View Operations
// ============================================================================

/**
 * Flatten operation
 */
type ImageBatch = [32, 3, 224, 224]; // 32 images, 3 channels, 224x224
type FlattenedBatch = ReshapeValid<ImageBatch, [32, 150528]>; // 3 * 224 * 224 = 150528

/**
 * Unflatten operation
 */
type FlatVector = [32, 2048];
type Unflattened = ReshapeValid<FlatVector, [32, 32, 64]>; // 32 * 64 = 2048

// ============================================================================
// Example 6: Transpose for Different Layouts
// ============================================================================

/**
 * Convert between NCHW and NHWC
 */
type NCHW = [32, 3, 224, 224];
type NHWC = [32, 224, 224, 3];

// This would be the permutation to convert NCHW -> NHWC
// type ToNHWC = PermuteShape<NCHW, [0, 2, 3, 1]>;

/**
 * Matrix transpose
 */
type Matrix = [100, 50];
type MatrixT = TransposeShape<Matrix, 0, 1>; // [50, 100]

// ============================================================================
// Example 7: DType Promotion
// ============================================================================

/**
 * Mixed precision operations
 */
type Float32Tensor = TensorType<[100, 50], "float32">;
type Float16Tensor = TensorType<[100, 50], "float16">;

// When combining different dtypes, promote to higher precision
type PromotedDType = PromoteDType<"float32", "float16">; // "float32"
type ResultTensor = TensorType<[100, 50], PromotedDType>;

/**
 * Integer to float promotion
 */
type IntTensor = TensorType<[10, 10], "int32">;
type FloatTensor = TensorType<[10, 10], "float32">;
type MixedResult = PromoteDType<"int32", "float32">; // "float32"

// ============================================================================
// Example 8: Dynamic Batch Dimensions
// ============================================================================

/**
 * Transformer encoder layer with dynamic batch and sequence length
 */
interface TransformerEncoderLayer<
  HiddenDim extends number,
  FFNDim extends number,
  D extends DTypeName = "float32"
> {
  selfAttn: SelfAttention<HiddenDim, D>;
  feedForward: {
    linear1: LinearLayer<HiddenDim, FFNDim, D>;
    linear2: LinearLayer<FFNDim, HiddenDim, D>;
  };
}

// Input has dynamic batch and sequence length
type TransformerInput = TensorType<
  [Dim<"batch">, Dim<"seq_len">, 768],
  "float32"
>;

// Output has same shape
type TransformerOutput = TensorType<
  [Dim<"batch">, Dim<"seq_len">, 768],
  "float32"
>;

// ============================================================================
// Example 9: Runtime DType Constants
// ============================================================================

/**
 * Accessing runtime dtype information
 */
const float32Info = DTypeConstants.float32;
console.log(`DType: ${float32Info.name}`); // "float32"
console.log(`Value: ${float32Info.value}`); // 0
console.log(`Bytes: ${float32Info.bytes}`); // 4

const int64Info = DTypeConstants.int64;
console.log(`DType: ${int64Info.name}`); // "int64"
console.log(`Value: ${int64Info.value}`); // 3
console.log(`Bytes: ${int64Info.bytes}`); // 8

// ============================================================================
// Example 10: Type-Safe Tensor Creation (Conceptual)
// ============================================================================

/**
 * Conceptual API for type-safe tensor creation
 * (This would be implemented in the runtime Tensor class)
 */
interface TensorFactory {
  /**
   * Create a tensor with explicit shape and dtype types
   */
  create<S extends readonly number[], D extends DTypeName>(
    shape: S,
    dtype: D,
    data?: ArrayLike<number>
  ): TensorType<S, D>;

  /**
   * Zeros with inferred shape
   */
  zeros<S extends readonly number[], D extends DTypeName = "float32">(
    shape: S,
    dtype?: D
  ): TensorType<S, D>;

  /**
   * Ones with inferred shape
   */
  ones<S extends readonly number[], D extends DTypeName = "float32">(
    shape: S,
    dtype?: D
  ): TensorType<S, D>;
}

// Usage would look like:
// const tensor = factory.zeros([3, 224, 224], "float32");
// Type would be: TensorType<[3, 224, 224], "float32">

// ============================================================================
// Example 11: Common Computer Vision Architectures
// ============================================================================

/**
 * ResNet block shapes
 */
namespace ResNet {
  // Input: [batch, 64, 56, 56]
  export type _Input = TensorType<[Dim<"batch">, 64, 56, 56], "float32">;

  // After 3x3 conv: [batch, 64, 56, 56]
  export type AfterConv1 = TensorType<[Dim<"batch">, 64, 56, 56], "float32">;

  // After another 3x3 conv: [batch, 64, 56, 56]
  export type AfterConv2 = TensorType<[Dim<"batch">, 64, 56, 56], "float32">;

  // After residual connection (element-wise add, uses broadcasting)
  export type _Output = TensorType<
    BroadcastShape<
      [Dim<"batch">, 64, 56, 56],
      [Dim<"batch">, 64, 56, 56]
    >,
    "float32"
  >;
}

/**
 * Vision Transformer (ViT) shapes
 */
namespace ViT {
  // Input image: [batch, 3, 224, 224]
  export type ImageInput = TensorType<[Dim<"batch">, 3, 224, 224], "float32">;

  // Patch embedding: 16x16 patches -> 196 patches
  export type NumPatches = 196; // (224/16)^2
  export type EmbedDim = 768;
  export type PatchEmbedding = TensorType<[Dim<"batch">, NumPatches, EmbedDim], "float32">;

  // After transformer encoder (same shape)
  export type EncoderOutput = TensorType<[Dim<"batch">, NumPatches, EmbedDim], "float32">;

  // Classification head (take first token)
  export type ClassToken = TensorType<[Dim<"batch">, EmbedDim], "float32">;
  export type Logits = TensorType<[Dim<"batch">, 1000], "float32">; // ImageNet classes
}

// ============================================================================
// Example 12: NLP Model Shapes
// ============================================================================

/**
 * GPT-2 style transformer
 */
namespace GPT2 {
  export type VocabSize = 50257;
  export type EmbedDim = 768;
  export type ContextLength = 1024;

  // Input token IDs: [batch, seq_len]
  export type TokenIDs = TensorType<[Dim<"batch">, Dim<"seq_len">], "int32">;

  // Token embeddings: [batch, seq_len, embed_dim]
  export type Embeddings = TensorType<[Dim<"batch">, Dim<"seq_len">, EmbedDim], "float32">;

  // After transformer layers (same shape)
  export type Hidden = TensorType<[Dim<"batch">, Dim<"seq_len">, EmbedDim], "float32">;

  // Language modeling head: [batch, seq_len, vocab_size]
  export type Logits = TensorType<[Dim<"batch">, Dim<"seq_len">, VocabSize], "float32">;
}

/**
 * Sequence-to-sequence model
 */
namespace Seq2Seq {
  // Encoder input: [batch, src_len, embed_dim]
  export type EncoderInput = TensorType<[Dim<"batch">, Dim<"src_len">, 512], "float32">;

  // Encoder output: [batch, src_len, hidden_dim]
  export type EncoderOutput = TensorType<[Dim<"batch">, Dim<"src_len">, 512], "float32">;

  // Decoder input: [batch, tgt_len, embed_dim]
  export type DecoderInput = TensorType<[Dim<"batch">, Dim<"tgt_len">, 512], "float32">;

  // Decoder output: [batch, tgt_len, hidden_dim]
  export type DecoderOutput = TensorType<[Dim<"batch">, Dim<"tgt_len">, 512], "float32">;
}

// ============================================================================
// Export types for documentation
// ============================================================================

export type {
  LinearLayer,
  LinearForward,
  Conv2dLayer,
  AttentionConfig,
  SelfAttention,
  TransformerEncoderLayer,
  TensorFactory,
};

export {
  ResNet,
  ViT,
  GPT2,
  Seq2Seq,
};
