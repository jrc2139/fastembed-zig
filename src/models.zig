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
                // EmbeddingGemma task prompts per Google docs
                .query_prefix = "task: code retrieval | query: ",
                .passage_prefix = "title: none | text: ",
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
                // EmbeddingGemma task prompts per Google docs
                .query_prefix = "task: code retrieval | query: ",
                .passage_prefix = "title: none | text: ",
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
                // EmbeddingGemma task prompts per Google docs
                .query_prefix = "task: code retrieval | query: ",
                .passage_prefix = "title: none | text: ",
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
                // EmbeddingGemma task prompts per Google docs
                .query_prefix = "task: code retrieval | query: ",
                .passage_prefix = "title: none | text: ",
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

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "BGE small English config" {
    const std = @import("std");
    const config = Model.bge_small_en_v1_5.getConfig();

    try std.testing.expectEqualStrings("bge-small-en-v1.5", config.name);
    try std.testing.expectEqualStrings("BAAI/bge-small-en-v1.5", config.hf_repo);
    try std.testing.expectEqual(@as(usize, 384), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 512), config.max_seq_len);
    try std.testing.expectEqual(pooling.PoolingStrategy.cls, config.pooling);
    try std.testing.expect(config.normalize);
    try std.testing.expectEqualStrings("query: ", config.query_prefix.?);
    try std.testing.expect(config.passage_prefix == null);
    try std.testing.expect(config.use_token_type_ids);
    try std.testing.expectEqualStrings("model.onnx", config.model_file);
    try std.testing.expect(!config.output_is_pooled);
    try std.testing.expect(config.output_name == null);
}

test "all-MiniLM-L6-v2 config" {
    const std = @import("std");
    const config = Model.all_minilm_l6_v2.getConfig();

    try std.testing.expectEqualStrings("all-MiniLM-L6-v2", config.name);
    try std.testing.expectEqualStrings("sentence-transformers/all-MiniLM-L6-v2", config.hf_repo);
    try std.testing.expectEqual(@as(usize, 384), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 256), config.max_seq_len);
    try std.testing.expectEqual(pooling.PoolingStrategy.mean, config.pooling);
    try std.testing.expect(config.normalize);
    try std.testing.expect(config.query_prefix == null);
    try std.testing.expect(config.passage_prefix == null);
    try std.testing.expect(config.use_token_type_ids);
    try std.testing.expectEqualStrings("model.onnx", config.model_file);
    try std.testing.expect(!config.output_is_pooled);
}

test "multilingual E5 large config" {
    const std = @import("std");
    const config = Model.multilingual_e5_large.getConfig();

    try std.testing.expectEqualStrings("multilingual-e5-large", config.name);
    try std.testing.expectEqualStrings("intfloat/multilingual-e5-large", config.hf_repo);
    try std.testing.expectEqual(@as(usize, 1024), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 512), config.max_seq_len);
    try std.testing.expectEqual(pooling.PoolingStrategy.mean, config.pooling);
    try std.testing.expect(config.normalize);
    try std.testing.expectEqualStrings("query: ", config.query_prefix.?);
    try std.testing.expectEqualStrings("passage: ", config.passage_prefix.?);
    try std.testing.expect(config.use_token_type_ids);
}

test "BGE small Chinese config" {
    const std = @import("std");
    const config = Model.bge_small_zh_v1_5.getConfig();

    try std.testing.expectEqualStrings("bge-small-zh-v1.5", config.name);
    try std.testing.expectEqualStrings("BAAI/bge-small-zh-v1.5", config.hf_repo);
    try std.testing.expectEqual(@as(usize, 512), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 512), config.max_seq_len);
    try std.testing.expectEqual(pooling.PoolingStrategy.cls, config.pooling);
    try std.testing.expect(config.normalize);
    // Chinese query prefix
    try std.testing.expectEqualStrings("为这个句子生成表示以用于检索相关文章：", config.query_prefix.?);
    try std.testing.expect(config.passage_prefix == null);
    try std.testing.expect(config.use_token_type_ids);
}

test "EmbeddingGemma 300M base config" {
    const std = @import("std");
    const config = Model.embedding_gemma_300m.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m", config.name);
    try std.testing.expectEqualStrings("onnx-community/embeddinggemma-300m-ONNX", config.hf_repo);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 2048), config.max_seq_len);
    try std.testing.expectEqual(pooling.PoolingStrategy.mean, config.pooling);
    try std.testing.expect(config.normalize);
    // EmbeddingGemma uses task prompts per Google docs
    try std.testing.expectEqualStrings("task: code retrieval | query: ", config.query_prefix.?);
    try std.testing.expectEqualStrings("title: none | text: ", config.passage_prefix.?);
    // Gemma doesn't use token_type_ids
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expectEqualStrings("onnx/model.onnx", config.model_file);
    // Gemma output is already pooled
    try std.testing.expect(config.output_is_pooled);
    try std.testing.expectEqualStrings("sentence_embedding", config.output_name.?);
}

test "EmbeddingGemma 300M Q4 config" {
    const std = @import("std");
    const config = Model.embedding_gemma_300m_q4.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m-q4", config.name);
    try std.testing.expectEqualStrings("onnx/model_q4.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(config.output_is_pooled);
    try std.testing.expectEqualStrings("sentence_embedding", config.output_name.?);
}

test "EmbeddingGemma 300M Q4F16 config" {
    const std = @import("std");
    const config = Model.embedding_gemma_300m_q4f16.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m-q4f16", config.name);
    try std.testing.expectEqualStrings("onnx/model_q4f16.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(config.output_is_pooled);
}

test "EmbeddingGemma 300M FP16 config" {
    const std = @import("std");
    const config = Model.embedding_gemma_300m_fp16.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m-fp16", config.name);
    try std.testing.expectEqualStrings("onnx/model_fp16.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(config.output_is_pooled);
}

test "Granite Embedding English R2 config" {
    const std = @import("std");
    const config = Model.granite_embedding_english_r2.getConfig();

    try std.testing.expectEqualStrings("granite-embedding-english-r2", config.name);
    try std.testing.expectEqualStrings("onnx-community/granite-embedding-english-r2-ONNX", config.hf_repo);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    // Granite supports long sequences
    try std.testing.expectEqual(@as(usize, 8192), config.max_seq_len);
    try std.testing.expectEqual(pooling.PoolingStrategy.mean, config.pooling);
    try std.testing.expect(config.normalize);
    // ModernBERT architecture - no token_type_ids
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expectEqualStrings("onnx/model_quantized.onnx", config.model_file);
    // BERT-style output needs pooling
    try std.testing.expect(!config.output_is_pooled);
    try std.testing.expect(config.output_name == null);
}

test "Granite Embedding English R2 FP16 config" {
    const std = @import("std");
    const config = Model.granite_embedding_english_r2_fp16.getConfig();

    try std.testing.expectEqualStrings("granite-embedding-english-r2-fp16", config.name);
    try std.testing.expectEqualStrings("onnx/model_fp16_embedded.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 8192), config.max_seq_len);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(!config.output_is_pooled);
}

test "All models have valid configs" {
    const std = @import("std");
    const models = [_]Model{
        .bge_small_en_v1_5,
        .all_minilm_l6_v2,
        .multilingual_e5_large,
        .bge_small_zh_v1_5,
        .embedding_gemma_300m,
        .embedding_gemma_300m_q4,
        .embedding_gemma_300m_q4f16,
        .embedding_gemma_300m_fp16,
        .granite_embedding_english_r2,
        .granite_embedding_english_r2_fp16,
    };

    for (models) |model| {
        const config = model.getConfig();
        // All models should have basic properties
        try std.testing.expect(config.name.len > 0);
        try std.testing.expect(config.hf_repo.len > 0);
        try std.testing.expect(config.hidden_dim > 0);
        try std.testing.expect(config.max_seq_len > 0);
        try std.testing.expect(config.model_file.len > 0);
        // All current models normalize
        try std.testing.expect(config.normalize);
    }
}

test "Model enum iteration" {
    const std = @import("std");
    // Verify we have exactly 10 models
    const model_count = @typeInfo(Model).@"enum".fields.len;
    try std.testing.expectEqual(@as(usize, 10), model_count);
}

test "Pooling strategy distribution" {
    const std = @import("std");
    var cls_count: usize = 0;
    var mean_count: usize = 0;

    const models = [_]Model{
        .bge_small_en_v1_5,
        .all_minilm_l6_v2,
        .multilingual_e5_large,
        .bge_small_zh_v1_5,
        .embedding_gemma_300m,
        .embedding_gemma_300m_q4,
        .embedding_gemma_300m_q4f16,
        .embedding_gemma_300m_fp16,
        .granite_embedding_english_r2,
        .granite_embedding_english_r2_fp16,
    };

    for (models) |model| {
        const config = model.getConfig();
        switch (config.pooling) {
            .cls => cls_count += 1,
            .mean => mean_count += 1,
        }
    }

    // BGE models use CLS pooling, others use mean
    try std.testing.expectEqual(@as(usize, 2), cls_count);
    try std.testing.expectEqual(@as(usize, 8), mean_count);
}

test "Hidden dimensions" {
    const std = @import("std");

    // 384 dim models (small)
    try std.testing.expectEqual(@as(usize, 384), Model.bge_small_en_v1_5.getConfig().hidden_dim);
    try std.testing.expectEqual(@as(usize, 384), Model.all_minilm_l6_v2.getConfig().hidden_dim);

    // 512 dim models (Chinese)
    try std.testing.expectEqual(@as(usize, 512), Model.bge_small_zh_v1_5.getConfig().hidden_dim);

    // 768 dim models (base)
    try std.testing.expectEqual(@as(usize, 768), Model.embedding_gemma_300m.getConfig().hidden_dim);
    try std.testing.expectEqual(@as(usize, 768), Model.granite_embedding_english_r2.getConfig().hidden_dim);

    // 1024 dim models (large)
    try std.testing.expectEqual(@as(usize, 1024), Model.multilingual_e5_large.getConfig().hidden_dim);
}

test "Max sequence lengths" {
    const std = @import("std");

    // 256 for MiniLM
    try std.testing.expectEqual(@as(usize, 256), Model.all_minilm_l6_v2.getConfig().max_seq_len);

    // 512 for most BERT-based models
    try std.testing.expectEqual(@as(usize, 512), Model.bge_small_en_v1_5.getConfig().max_seq_len);
    try std.testing.expectEqual(@as(usize, 512), Model.multilingual_e5_large.getConfig().max_seq_len);

    // 2048 for Gemma models
    try std.testing.expectEqual(@as(usize, 2048), Model.embedding_gemma_300m.getConfig().max_seq_len);
    try std.testing.expectEqual(@as(usize, 2048), Model.embedding_gemma_300m_q4.getConfig().max_seq_len);

    // 8192 for Granite (long context)
    try std.testing.expectEqual(@as(usize, 8192), Model.granite_embedding_english_r2.getConfig().max_seq_len);
}

test "Token type IDs requirement" {
    const std = @import("std");

    // BERT-based models use token_type_ids
    try std.testing.expect(Model.bge_small_en_v1_5.getConfig().use_token_type_ids);
    try std.testing.expect(Model.all_minilm_l6_v2.getConfig().use_token_type_ids);
    try std.testing.expect(Model.multilingual_e5_large.getConfig().use_token_type_ids);
    try std.testing.expect(Model.bge_small_zh_v1_5.getConfig().use_token_type_ids);

    // Gemma models don't use token_type_ids
    try std.testing.expect(!Model.embedding_gemma_300m.getConfig().use_token_type_ids);
    try std.testing.expect(!Model.embedding_gemma_300m_q4.getConfig().use_token_type_ids);
    try std.testing.expect(!Model.embedding_gemma_300m_q4f16.getConfig().use_token_type_ids);
    try std.testing.expect(!Model.embedding_gemma_300m_fp16.getConfig().use_token_type_ids);

    // Granite (ModernBERT) doesn't use token_type_ids
    try std.testing.expect(!Model.granite_embedding_english_r2.getConfig().use_token_type_ids);
    try std.testing.expect(!Model.granite_embedding_english_r2_fp16.getConfig().use_token_type_ids);
}

test "Pre-pooled vs requires pooling" {
    const std = @import("std");

    // Gemma models output pre-pooled sentence_embedding
    try std.testing.expect(Model.embedding_gemma_300m.getConfig().output_is_pooled);
    try std.testing.expect(Model.embedding_gemma_300m_q4.getConfig().output_is_pooled);
    try std.testing.expect(Model.embedding_gemma_300m_q4f16.getConfig().output_is_pooled);
    try std.testing.expect(Model.embedding_gemma_300m_fp16.getConfig().output_is_pooled);

    // All other models need pooling applied
    try std.testing.expect(!Model.bge_small_en_v1_5.getConfig().output_is_pooled);
    try std.testing.expect(!Model.all_minilm_l6_v2.getConfig().output_is_pooled);
    try std.testing.expect(!Model.multilingual_e5_large.getConfig().output_is_pooled);
    try std.testing.expect(!Model.bge_small_zh_v1_5.getConfig().output_is_pooled);
    try std.testing.expect(!Model.granite_embedding_english_r2.getConfig().output_is_pooled);
    try std.testing.expect(!Model.granite_embedding_english_r2_fp16.getConfig().output_is_pooled);
}

test "Query prefix behavior" {
    const std = @import("std");

    // Asymmetric models with query prefix
    try std.testing.expect(Model.bge_small_en_v1_5.getConfig().query_prefix != null);
    try std.testing.expect(Model.multilingual_e5_large.getConfig().query_prefix != null);
    try std.testing.expect(Model.bge_small_zh_v1_5.getConfig().query_prefix != null);
    // EmbeddingGemma uses task prompts
    try std.testing.expect(Model.embedding_gemma_300m.getConfig().query_prefix != null);

    // Symmetric models without query prefix
    try std.testing.expect(Model.all_minilm_l6_v2.getConfig().query_prefix == null);
    try std.testing.expect(Model.granite_embedding_english_r2.getConfig().query_prefix == null);
}

test "Passage prefix behavior" {
    const std = @import("std");

    // Models with passage prefix
    try std.testing.expect(Model.multilingual_e5_large.getConfig().passage_prefix != null);
    try std.testing.expectEqualStrings("passage: ", Model.multilingual_e5_large.getConfig().passage_prefix.?);
    // EmbeddingGemma uses task prompts
    try std.testing.expect(Model.embedding_gemma_300m.getConfig().passage_prefix != null);
    try std.testing.expectEqualStrings("title: none | text: ", Model.embedding_gemma_300m.getConfig().passage_prefix.?);

    // Models without passage prefix
    try std.testing.expect(Model.bge_small_en_v1_5.getConfig().passage_prefix == null);
    try std.testing.expect(Model.all_minilm_l6_v2.getConfig().passage_prefix == null);
}

test "ModelFiles constants" {
    const std = @import("std");

    try std.testing.expectEqualStrings("model.onnx", ModelFiles.model_onnx);
    try std.testing.expectEqualStrings("tokenizer.json", ModelFiles.tokenizer_json);
    try std.testing.expectEqualStrings("config.json", ModelFiles.config_json);
    try std.testing.expectEqualStrings("special_tokens_map.json", ModelFiles.special_tokens_map);
}

test "getHfUrl returns empty for now" {
    const std = @import("std");
    const url = getHfUrl("BAAI/bge-small-en-v1.5", "model.onnx");
    try std.testing.expectEqual(@as(usize, 0), url.len);
}

test "Gemma variants share same HF repo" {
    const std = @import("std");
    const base = Model.embedding_gemma_300m.getConfig().hf_repo;
    const q4 = Model.embedding_gemma_300m_q4.getConfig().hf_repo;
    const q4f16 = Model.embedding_gemma_300m_q4f16.getConfig().hf_repo;
    const fp16 = Model.embedding_gemma_300m_fp16.getConfig().hf_repo;

    try std.testing.expectEqualStrings(base, q4);
    try std.testing.expectEqualStrings(base, q4f16);
    try std.testing.expectEqualStrings(base, fp16);
}

test "Granite variants share same HF repo" {
    const std = @import("std");
    const quantized = Model.granite_embedding_english_r2.getConfig().hf_repo;
    const fp16 = Model.granite_embedding_english_r2_fp16.getConfig().hf_repo;

    try std.testing.expectEqualStrings(quantized, fp16);
}

test "Output names for pooled models" {
    const std = @import("std");

    // Gemma models use "sentence_embedding" output
    try std.testing.expectEqualStrings("sentence_embedding", Model.embedding_gemma_300m.getConfig().output_name.?);
    try std.testing.expectEqualStrings("sentence_embedding", Model.embedding_gemma_300m_q4.getConfig().output_name.?);

    // Non-pooled models use default (null)
    try std.testing.expect(Model.bge_small_en_v1_5.getConfig().output_name == null);
    try std.testing.expect(Model.granite_embedding_english_r2.getConfig().output_name == null);
}
