//! fastembed-zig - Fast text embeddings in Zig
//!
//! A high-performance text embedding library built on ONNX Runtime
//! and HuggingFace tokenizers.
//!
//! ## Quick Start
//!
//! ```zig
//! const fe = @import("fastembed");
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     // Initialize embedder with local model
//!     var embedder = try fe.Embedder.init(allocator, .{
//!         .model_path = "models/bge-small-en-v1.5",
//!     });
//!     defer embedder.deinit();
//!
//!     // Generate embeddings
//!     const texts = &[_][]const u8{ "Hello world", "How are you?" };
//!     const embeddings = try embedder.embed(texts);
//!     defer allocator.free(embeddings);
//!
//!     // embeddings is []f32 — [num_texts * hidden_dim] flattened
//!     const dim = embedder.getDimension(); // 384 for BGE-small
//!     for (0..texts.len) |i| {
//!         const vec = embeddings[i * dim .. (i + 1) * dim];
//!         std.debug.print("Embedding {d}: [{d:.4}, ...]\n", .{ i, vec[0] });
//!     }
//! }
//! ```

const std = @import("std");

// Core embedding functionality
pub const embedding = @import("embedding.zig");
pub const Embedder = embedding.Embedder;
pub const EmbedderOptions = embedding.EmbedderOptions;
pub const EmbedderError = embedding.EmbedderError;

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

// Low-level access (ONNX)
pub const onnx = @import("onnx/session.zig");
const c_api = @import("onnx/c_api.zig");

// Build configuration - indicates what this binary was compiled with
pub const cuda_enabled = c_api.cuda_enabled;
pub const coreml_enabled = c_api.coreml_enabled;

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
