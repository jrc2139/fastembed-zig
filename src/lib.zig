//! fastembed-zig - Fast text embeddings in Zig
//!
//! A high-performance text embedding library built on ONNX Runtime
//! and HuggingFace tokenizers.
//!
//! ## Embedder Types
//!
//! This library provides two embedder implementations:
//!
//! - **Embedder**: Standard allocating embedder. Each `embed()` call allocates
//!   and returns owned memory. Simple and flexible.
//!
//! - **FastEmbedder**: Zero-allocation embedder. Pre-allocates all buffers at init.
//!   After init, `embed()` performs zero heap allocations. Ideal for high-throughput
//!   pipelines and real-time applications.
//!
//! ## Quick Start (Standard)
//!
//! ```zig
//! const fe = @import("fastembed");
//!
//! var embedder = try fe.Embedder.init(allocator, .{
//!     .model_path = "models/bge-small-en-v1.5",
//! });
//! defer embedder.deinit();
//!
//! const embeddings = try embedder.embed(&.{ "Hello", "World" });
//! defer allocator.free(embeddings);
//! ```
//!
//! ## Quick Start (Zero-Allocation)
//!
//! ```zig
//! const fe = @import("fastembed");
//!
//! var embedder = try fe.FastEmbedder.init(allocator, .{
//!     .model_path = "models/bge-small-en-v1.5",
//!     .max_batch_size = 64,  // Pre-allocate for up to 64 texts
//! });
//! defer embedder.deinit();
//!
//! // Zero-allocation hot path
//! for (batches) |batch| {
//!     const embeddings = try embedder.embed(batch);
//!     // embeddings valid until next embed() call
//! }
//! ```

const std = @import("std");

// Core embedding functionality (allocating)
pub const embedding = @import("embedding.zig");
pub const Embedder = embedding.Embedder;
pub const EmbedderOptions = embedding.EmbedderOptions;
pub const EmbedderError = embedding.EmbedderError;

// Zero-allocation embedding (FastEmbedder)
pub const fast_embedding = @import("fast_embedding.zig");
pub const FastEmbedder = fast_embedding.FastEmbedder;
pub const FastEmbedderOptions = fast_embedding.FastEmbedderOptions;
pub const FastEmbedderError = fast_embedding.FastEmbedderError;

// Execution providers (for GPU/Neural Engine acceleration)
pub const ExecutionProvider = embedding.ExecutionProvider;
pub const CoreMLOptions = embedding.CoreMLOptions;
pub const CoreMLComputeUnits = embedding.CoreMLComputeUnits;

// Model registry
pub const models = @import("models.zig");
pub const Model = models.Model;
pub const ModelConfig = models.ModelConfig;
pub const MemoryProfile = models.MemoryProfile;
pub const getAvailableMemoryMB = models.getAvailableMemoryMB;
pub const CpuFeatures = models.CpuFeatures;
pub const getOptimalOnnxFile = models.getOptimalOnnxFile;

// Pooling strategies
pub const pooling = @import("pooling.zig");
pub const PoolingStrategy = pooling.PoolingStrategy;

// Normalization utilities
pub const normalize = @import("normalize.zig");

// Reranker (cross-encoder for semantic reranking)
pub const reranker = @import("reranker.zig");
pub const Reranker = reranker.Reranker;
pub const RerankerOptions = reranker.RerankerOptions;
pub const RerankerError = reranker.RerankerError;
pub const RerankerModel = reranker.RerankerModel;

// Tokenizer (pure Zig implementation via tokenizer-zig)
pub const tokenizer = @import("tokenizer/tokenizer.zig");
pub const Tokenizer = tokenizer.Tokenizer;
pub const TokenizerError = tokenizer.TokenizerError;

// Zero-allocation tokenizer
pub const FastTokenizer = tokenizer.FastTokenizer;
pub const FastTokenizerOptions = tokenizer.FastTokenizerOptions;

// Low-level access (ONNX)
pub const onnx = @import("onnx/session.zig");
pub const fast_onnx = @import("onnx/fast_session.zig");

// Zero-allocation session types
pub const FastSession = fast_onnx.FastSession;
pub const FastSessionConfig = fast_onnx.FastSessionConfig;
pub const IoBinding = fast_onnx.IoBinding;
const c_api = @import("onnx/c_api.zig");

// Build configuration - indicates what this binary was compiled with
pub const cuda_enabled = c_api.cuda_enabled;
pub const coreml_enabled = c_api.coreml_enabled;
pub const dynamic_ort = c_api.dynamic_ort;

// Dynamic loading utilities
pub const isCudaRuntimeAvailable = c_api.isCudaRuntimeAvailable;
pub const isDynamicCudaLoaded = c_api.isDynamicCudaLoaded;

// Utility functions

/// Compute cosine similarity between two embedding vectors
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return normalize.cosineSimilarity(a, b);
}

/// Compute dot product between two vectors (for normalized embeddings)
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    return normalize.dotProduct(a, b);
}

test {
    // Run all module tests
    std.testing.refAllDecls(@This());
}

test "library compiles" {
    try std.testing.expect(true);
}
