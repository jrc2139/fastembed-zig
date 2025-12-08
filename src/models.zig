//! Model registry for supported embedding models
//!
//! Defines metadata and configuration for supported models.

const pooling = @import("pooling.zig");

/// Supported embedding models
pub const Model = enum {
    /// BAAI BGE small English model (384 dimensions)
    bge_small_en_v1_5,
    /// Sentence Transformers all-MiniLM-L6-v2 (384 dimensions)
    all_minilm_l6_v2,
    /// Multilingual E5 large (1024 dimensions)
    multilingual_e5_large,
    /// BAAI BGE small Chinese model (512 dimensions)
    bge_small_zh_v1_5,
    /// Google EmbeddingGemma 300M (768 dimensions) - Matryoshka support
    embedding_gemma_300m,
    /// Google EmbeddingGemma 300M Q4 quantized (768 dimensions)
    embedding_gemma_300m_q4,
    /// Google EmbeddingGemma 300M Q4F16 quantized (768 dimensions) - better for CoreML
    embedding_gemma_300m_q4f16,
    /// Google EmbeddingGemma 300M FP16 (768 dimensions) - full precision for CoreML
    embedding_gemma_300m_fp16,
    /// IBM Granite Embedding English R2 Q4F16 (768 dimensions) - ModernBERT architecture
    granite_embedding_english_r2,
    /// IBM Granite Embedding English R2 FP16 embedded (768 dimensions) - for CoreML
    granite_embedding_english_r2_fp16,

    /// Get model configuration
    pub fn getConfig(self: Model) ModelConfig {
        return switch (self) {
            .bge_small_en_v1_5 => .{
                .name = "bge-small-en-v1.5",
                .hf_repo = "BAAI/bge-small-en-v1.5",
                .hidden_dim = 384,
                .max_seq_len = 512,
                .pooling = .cls,
                .normalize = true,
                .query_prefix = "query: ",
                .passage_prefix = null,
                .use_token_type_ids = true,
                .model_file = "model.onnx",
                .output_is_pooled = false,
            },
            .all_minilm_l6_v2 => .{
                .name = "all-MiniLM-L6-v2",
                .hf_repo = "sentence-transformers/all-MiniLM-L6-v2",
                .hidden_dim = 384,
                .max_seq_len = 256,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = true,
                .model_file = "model.onnx",
                .output_is_pooled = false,
            },
            .multilingual_e5_large => .{
                .name = "multilingual-e5-large",
                .hf_repo = "intfloat/multilingual-e5-large",
                .hidden_dim = 1024,
                .max_seq_len = 512,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = "query: ",
                .passage_prefix = "passage: ",
                .use_token_type_ids = true,
                .model_file = "model.onnx",
                .output_is_pooled = false,
            },
            .bge_small_zh_v1_5 => .{
                .name = "bge-small-zh-v1.5",
                .hf_repo = "BAAI/bge-small-zh-v1.5",
                .hidden_dim = 512,
                .max_seq_len = 512,
                .pooling = .cls,
                .normalize = true,
                .query_prefix = "为这个句子生成表示以用于检索相关文章：",
                .passage_prefix = null,
                .use_token_type_ids = true,
                .model_file = "model.onnx",
                .output_is_pooled = false,
            },
            .embedding_gemma_300m => .{
                .name = "embeddinggemma-300m",
                .hf_repo = "onnx-community/embeddinggemma-300m-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 2048,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model.onnx",
                .output_is_pooled = true,
                .output_name = "sentence_embedding",
            },
            .embedding_gemma_300m_q4 => .{
                .name = "embeddinggemma-300m-q4",
                .hf_repo = "onnx-community/embeddinggemma-300m-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 2048,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model_q4.onnx",
                .output_is_pooled = true,
                .output_name = "sentence_embedding",
            },
            .embedding_gemma_300m_q4f16 => .{
                .name = "embeddinggemma-300m-q4f16",
                .hf_repo = "onnx-community/embeddinggemma-300m-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 2048,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model_q4f16.onnx",
                .output_is_pooled = true,
                .output_name = "sentence_embedding",
            },
            .embedding_gemma_300m_fp16 => .{
                .name = "embeddinggemma-300m-fp16",
                .hf_repo = "onnx-community/embeddinggemma-300m-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 2048,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model_fp16.onnx",
                .output_is_pooled = true,
                .output_name = "sentence_embedding",
            },
            .granite_embedding_english_r2 => .{
                .name = "granite-embedding-english-r2",
                .hf_repo = "onnx-community/granite-embedding-english-r2-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false, // ModernBERT doesn't use token_type_ids
                .model_file = "onnx/model_quantized.onnx",
                .output_is_pooled = false, // BERT-style output needs pooling
                .output_name = null, // Use default output (last_hidden_state)
            },
            .granite_embedding_english_r2_fp16 => .{
                .name = "granite-embedding-english-r2-fp16",
                .hf_repo = "onnx-community/granite-embedding-english-r2-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model_fp16_embedded.onnx", // Single file, no external data
                .output_is_pooled = false,
                .output_name = null,
            },
        };
    }
};

/// Model configuration
pub const ModelConfig = struct {
    /// Display name
    name: []const u8,
    /// HuggingFace repository ID
    hf_repo: []const u8,
    /// Output embedding dimension
    hidden_dim: usize,
    /// Maximum input sequence length
    max_seq_len: usize,
    /// Pooling strategy
    pooling: pooling.PoolingStrategy,
    /// Whether to L2-normalize embeddings
    normalize: bool,
    /// Prefix to add for query embeddings (for asymmetric models)
    query_prefix: ?[]const u8,
    /// Prefix to add for passage/document embeddings
    passage_prefix: ?[]const u8,
    /// Whether model uses token_type_ids input (false for Gemma)
    use_token_type_ids: bool = true,
    /// Path to model file within model directory
    model_file: []const u8 = "model.onnx",
    /// Whether model output is already pooled (true for Gemma's sentence_embedding)
    output_is_pooled: bool = false,
    /// Name of output tensor to use (null = use first output)
    output_name: ?[]const u8 = null,
};

/// Files needed for a model
pub const ModelFiles = struct {
    pub const model_onnx = "model.onnx";
    pub const tokenizer_json = "tokenizer.json";
    pub const config_json = "config.json";
    pub const special_tokens_map = "special_tokens_map.json";
};

/// Get HuggingFace download URL for a model file
pub fn getHfUrl(repo: []const u8, filename: []const u8) []const u8 {
    _ = repo;
    _ = filename;
    // Format: https://huggingface.co/{repo}/resolve/main/{filename}
    // For now, return empty - actual implementation would build URL
    return "";
}

test "Model config" {
    const std = @import("std");
    const config = Model.bge_small_en_v1_5.getConfig();

    try std.testing.expectEqual(@as(usize, 384), config.hidden_dim);
    try std.testing.expectEqual(pooling.PoolingStrategy.cls, config.pooling);
    try std.testing.expect(config.normalize);
}
