//! Model registry for supported embedding models
//!
//! Defines metadata and configuration for supported models.

const std = @import("std");
const builtin = @import("builtin");
const pooling = @import("pooling.zig");

/// CPU feature detection for optimal model file selection
/// Detects AVX2, AVX-512, and AVX-512 VNNI at runtime for x86_64
pub const CpuFeatures = struct {
    has_avx2: bool = false,
    has_avx512f: bool = false,
    has_avx512vnni: bool = false,

    /// Detect CPU features at runtime using CPUID (x86_64 only)
    pub fn detect() CpuFeatures {
        if (builtin.cpu.arch != .x86_64) {
            return .{};
        }

        var features = CpuFeatures{};

        // CPUID leaf 7, subleaf 0: Extended features
        // EBX bit 5 = AVX2
        // EBX bit 16 = AVX-512F (foundation)
        // ECX bit 11 = AVX-512 VNNI
        var eax: u32 = undefined;
        var ebx: u32 = undefined;
        var ecx: u32 = undefined;
        var edx: u32 = undefined;
        asm volatile ("cpuid"
            : [_] "={eax}" (eax),
              [_] "={ebx}" (ebx),
              [_] "={ecx}" (ecx),
              [_] "={edx}" (edx),
            : [_] "{eax}" (@as(u32, 7)),
              [_] "{ecx}" (@as(u32, 0)),
        );
        _ = .{ eax, edx }; // Suppress unused warnings
        features.has_avx2 = (ebx & (1 << 5)) != 0;
        features.has_avx512f = (ebx & (1 << 16)) != 0;
        features.has_avx512vnni = (ecx & (1 << 11)) != 0;

        return features;
    }
};

/// Get the optimal ONNX model file for the current CPU
/// Returns the best quantized variant based on detected features:
/// - ARM64: model_qint8_arm64.onnx (QINT8 for ARM NEON)
/// - x86_64 + AVX-512 VNNI: model_qint8_avx512_vnni.onnx (best for quantized)
/// - x86_64 + AVX-512: model_qint8_avx512.onnx (if available)
/// - x86_64 + AVX2: model_quint8_avx2.onnx (widely compatible)
/// - Fallback: model.onnx (FP32)
pub fn getOptimalOnnxFile() []const u8 {
    if (builtin.cpu.arch == .aarch64) {
        return "onnx/model_qint8_arm64.onnx";
    }

    if (builtin.cpu.arch == .x86_64) {
        // macOS Intel - use AVX2 (all Intel Macs since ~2013 have AVX2)
        if (builtin.os.tag == .macos) {
            return "onnx/model_quint8_avx2.onnx";
        }

        // Linux/Windows x86_64 - detect best available
        const features = CpuFeatures.detect();
        if (features.has_avx512vnni) {
            return "onnx/model_qint8_avx512_vnni.onnx";
        } else if (features.has_avx512f) {
            // AVX-512 without VNNI - fall back to AVX2 variant
            // (most AVX-512 models need VNNI for quantized ops)
            return "onnx/model_quint8_avx2.onnx";
        } else if (features.has_avx2) {
            return "onnx/model_quint8_avx2.onnx";
        } else {
            return "onnx/model.onnx"; // FP32 fallback for old CPUs
        }
    }

    return "onnx/model.onnx"; // Generic fallback
}

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
    /// IBM Granite Embedding Small English R2 (384 dimensions) - 47M params, fast
    granite_embedding_small_english_r2,
    /// IBM Granite Embedding Small English R2 Q4 quantized (384 dimensions) - fastest
    granite_embedding_small_english_r2_q4,
    /// IBM Granite Embedding Small English R2 QINT8 ARM64 (384 dimensions) - optimized for Apple Silicon
    granite_embedding_small_english_r2_qint8,
    /// IBM Granite Embedding English R2 O4 (768 dimensions) - CUDA optimized, opset 18
    granite_embedding_english_r2_o4,

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
                // Small model - low memory, fast
                .memory_profile = .{ .base_memory_mb = 200, .per_batch_item_mb = 8.0, .optimal_cpu_batch = 64, .optimal_gpu_batch = 128 },
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
                // Very small model - lowest memory
                .memory_profile = .{ .base_memory_mb = 150, .per_batch_item_mb = 6.0, .optimal_cpu_batch = 64, .optimal_gpu_batch = 128 },
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
                // Large model - high memory usage
                .memory_profile = .{ .base_memory_mb = 1500, .per_batch_item_mb = 40.0, .optimal_cpu_batch = 16, .optimal_gpu_batch = 32 },
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
                // Small model similar to bge-small-en
                .memory_profile = .{ .base_memory_mb = 200, .per_batch_item_mb = 8.0, .optimal_cpu_batch = 64, .optimal_gpu_batch = 128 },
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
                // Gemma 300M is memory hungry due to long context
                .memory_profile = .{ .base_memory_mb = 2000, .per_batch_item_mb = 150.0, .optimal_cpu_batch = 8, .optimal_gpu_batch = 16 },
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
                // Q4 quantized uses less memory but still memory hungry
                .memory_profile = .{ .base_memory_mb = 1000, .per_batch_item_mb = 100.0, .optimal_cpu_batch = 16, .optimal_gpu_batch = 32 },
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
                // Q4F16 - GPU batch 64 for better throughput
                .memory_profile = .{ .base_memory_mb = 1000, .per_batch_item_mb = 50.0, .optimal_cpu_batch = 8, .optimal_gpu_batch = 64 },
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
                // FP16 - GPU batch 64, no dequantization overhead
                .memory_profile = .{ .base_memory_mb = 1500, .per_batch_item_mb = 60.0, .optimal_cpu_batch = 8, .optimal_gpu_batch = 64 },
            },
            .granite_embedding_english_r2 => .{
                .name = "granite-embedding-english-r2",
                .hf_repo = "jrc2139/granite-embedding-english-r2-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false, // ModernBERT doesn't use token_type_ids
                // Architecture-specific model files for CPU:
                .model_file = if (builtin.cpu.arch == .aarch64)
                    "onnx/model_qint8_arm64.onnx"
                else
                    "onnx/model_quint8_avx2.onnx",
                .output_is_pooled = false, // BERT-style output needs pooling
                .output_name = null, // Use default output (last_hidden_state)
                // Granite has very long context (8192) - memory scales with sequence length
                .memory_profile = .{ .base_memory_mb = 1500, .per_batch_item_mb = 80.0, .optimal_cpu_batch = 8, .optimal_gpu_batch = 16 },
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
                // FP16 uses more memory than quantized
                .memory_profile = .{ .base_memory_mb = 2000, .per_batch_item_mb = 100.0, .optimal_cpu_batch = 8, .optimal_gpu_batch = 16 },
            },
            .granite_embedding_small_english_r2 => .{
                .name = "granite-embedding-small-english-r2",
                .hf_repo = "onnx-community/granite-embedding-small-english-r2-ONNX",
                .hidden_dim = 384,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false, // ModernBERT doesn't use token_type_ids
                .model_file = "onnx/model.onnx",
                .output_is_pooled = false,
                .output_name = null,
                // Small model (47M params) - fast and memory efficient
                .memory_profile = .{ .base_memory_mb = 200, .per_batch_item_mb = 10.0, .optimal_cpu_batch = 32, .optimal_gpu_batch = 64 },
            },
            .granite_embedding_small_english_r2_q4 => .{
                .name = "granite-embedding-small-english-r2-q4",
                .hf_repo = "onnx-community/granite-embedding-small-english-r2-ONNX",
                .hidden_dim = 384,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model_quantized.onnx",
                .output_is_pooled = false,
                .output_name = null,
                // Q4 quantized - smallest and fastest
                .memory_profile = .{ .base_memory_mb = 100, .per_batch_item_mb = 5.0, .optimal_cpu_batch = 64, .optimal_gpu_batch = 128 },
            },
            .granite_embedding_small_english_r2_qint8 => .{
                .name = "granite-embedding-small-english-r2-qint8",
                .hf_repo = "jrc2139/granite-embedding-small-english-r2-ONNX",
                .hidden_dim = 384,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                // Architecture-specific model files:
                // - ARM64 (Apple Silicon): model_qint8_arm64.onnx (QINT8 optimized for ARM NEON)
                // - x86_64: model_quint8_avx2.onnx (QUINT8 optimized for AVX2, widely compatible)
                .model_file = if (builtin.cpu.arch == .aarch64)
                    "onnx/model_qint8_arm64.onnx"
                else if (builtin.cpu.arch == .x86_64)
                    "onnx/model_quint8_avx2.onnx"
                else
                    "onnx/model_qint8_arm64.onnx", // fallback
                .output_is_pooled = false,
                .output_name = null,
                // Quantized model - optimized for inference, single file (no external data)
                .memory_profile = .{ .base_memory_mb = 100, .per_batch_item_mb = 5.0, .optimal_cpu_batch = 64, .optimal_gpu_batch = 128 },
            },
            .granite_embedding_english_r2_o4 => .{
                // Same model as granite_embedding_english_r2, but uses FP16 CUDA ONNX file
                .name = "granite-embedding-english-r2", // Same directory as CPU variant
                .hf_repo = "jrc2139/granite-embedding-english-r2-ONNX",
                .hidden_dim = 768,
                .max_seq_len = 8192,
                .pooling = .mean,
                .normalize = true,
                .query_prefix = null,
                .passage_prefix = null,
                .use_token_type_ids = false,
                .model_file = "onnx/model_f16_cuda.onnx", // FP16 optimized for CUDA
                .output_is_pooled = false,
                .output_name = null,
                // FP16 model optimized for GPU inference
                .memory_profile = .{ .base_memory_mb = 1500, .per_batch_item_mb = 80.0, .optimal_cpu_batch = 8, .optimal_gpu_batch = 64 },
            },
        };
    }

    /// Get the optimal model file for this model type based on runtime CPU detection
    /// For models with multiple quantized variants (granite_embedding_english_r2, etc.),
    /// this uses runtime CPU feature detection to select the best variant.
    /// For other models, returns the compile-time default from getConfig().
    pub fn getModelFile(self: Model) []const u8 {
        return switch (self) {
            // Models with runtime CPU-specific variants
            .granite_embedding_english_r2,
            .granite_embedding_small_english_r2_qint8,
            => getOptimalOnnxFile(),

            // All other models use compile-time default
            else => self.getConfig().model_file,
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
    /// Memory profile for batch size optimization
    memory_profile: MemoryProfile = .{},
};

/// Memory profile for calculating optimal batch sizes
/// Based on empirical measurements from fastembed-go benchmarks
pub const MemoryProfile = struct {
    /// Base memory used by model weights (MB)
    base_memory_mb: u32 = 500,
    /// Memory per batch item at max sequence length (MB)
    per_batch_item_mb: f32 = 20.0,
    /// Optimal batch size for CPU (found via benchmarking)
    /// Smaller batches with parallelization often outperform large batches
    optimal_cpu_batch: u32 = 32,
    /// Optimal batch size for GPU
    optimal_gpu_batch: u32 = 64,

    /// Calculate optimal batch size given available memory (MB)
    /// Returns a batch size that fits in memory while being efficient
    pub fn calculateOptimalBatch(self: MemoryProfile, available_memory_mb: u32, is_gpu: bool) u32 {
        // Use 70% of available memory for safety margin
        const usable_mb = @as(f32, @floatFromInt(available_memory_mb)) * 0.7;
        const available_for_batches = usable_mb - @as(f32, @floatFromInt(self.base_memory_mb));

        if (available_for_batches < self.per_batch_item_mb) {
            return 1; // Minimum batch size
        }

        const memory_based_batch: u32 = @intFromFloat(available_for_batches / self.per_batch_item_mb);

        // Use the smaller of memory-based and optimal batch size
        const optimal = if (is_gpu) self.optimal_gpu_batch else self.optimal_cpu_batch;
        return @min(memory_based_batch, optimal);
    }

    /// Get default batch size (when memory is unknown)
    pub fn getDefaultBatch(self: MemoryProfile, is_gpu: bool) u32 {
        return if (is_gpu) self.optimal_gpu_batch else self.optimal_cpu_batch;
    }

    /// Get optimal batch size with automatic memory detection
    pub fn getAutoBatch(self: MemoryProfile, is_gpu: bool) u32 {
        const available_mb = getAvailableMemoryMB();
        if (available_mb > 0) {
            return self.calculateOptimalBatch(available_mb, is_gpu);
        }
        return self.getDefaultBatch(is_gpu);
    }
};

/// Get available system memory in MB
/// Returns 0 if detection fails
pub fn getAvailableMemoryMB() u32 {
    if (builtin.os.tag == .macos) {
        return getMacOSMemoryMB();
    } else if (builtin.os.tag == .linux) {
        return getLinuxMemoryMB();
    }
    return 0; // Unknown OS
}

fn getMacOSMemoryMB() u32 {
    // Use sysctl to get physical memory
    const c = @cImport({
        @cInclude("sys/sysctl.h");
    });

    var mem_size: u64 = 0;
    var size: usize = @sizeOf(u64);
    var mib = [_]c_int{ c.CTL_HW, c.HW_MEMSIZE };

    if (c.sysctl(&mib, 2, &mem_size, &size, null, 0) == 0) {
        return @intCast(mem_size / (1024 * 1024));
    }
    return 0;
}

fn getLinuxMemoryMB() u32 {
    // Read from /proc/meminfo
    const file = std.fs.openFileAbsolute("/proc/meminfo", .{}) catch return 0;
    defer file.close();

    var buf: [256]u8 = undefined;
    const bytes_read = file.read(&buf) catch return 0;
    const content = buf[0..bytes_read];

    // Parse "MemAvailable: XXXX kB" or fall back to "MemFree: XXXX kB"
    if (parseMemInfoLine(content, "MemAvailable:")) |kb| {
        return @intCast(kb / 1024);
    }
    if (parseMemInfoLine(content, "MemFree:")) |kb| {
        return @intCast(kb / 1024);
    }
    return 0;
}

fn parseMemInfoLine(content: []const u8, prefix: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, prefix)) {
            // Skip prefix and whitespace
            var rest = std.mem.trimLeft(u8, line[prefix.len..], " \t");
            // Parse number
            var num_end: usize = 0;
            while (num_end < rest.len and rest[num_end] >= '0' and rest[num_end] <= '9') {
                num_end += 1;
            }
            if (num_end > 0) {
                return std.fmt.parseInt(u64, rest[0..num_end], 10) catch null;
            }
        }
    }
    return null;
}

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
    const config = Model.bge_small_en_v1_5.getConfig();

    try std.testing.expectEqual(@as(usize, 384), config.hidden_dim);
    try std.testing.expectEqual(pooling.PoolingStrategy.cls, config.pooling);
    try std.testing.expect(config.normalize);
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "BGE small English config" {
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
    const config = Model.embedding_gemma_300m_q4.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m-q4", config.name);
    try std.testing.expectEqualStrings("onnx/model_q4.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(config.output_is_pooled);
    try std.testing.expectEqualStrings("sentence_embedding", config.output_name.?);
}

test "EmbeddingGemma 300M Q4F16 config" {
    const config = Model.embedding_gemma_300m_q4f16.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m-q4f16", config.name);
    try std.testing.expectEqualStrings("onnx/model_q4f16.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(config.output_is_pooled);
}

test "EmbeddingGemma 300M FP16 config" {
    const config = Model.embedding_gemma_300m_fp16.getConfig();

    try std.testing.expectEqualStrings("embeddinggemma-300m-fp16", config.name);
    try std.testing.expectEqualStrings("onnx/model_fp16.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(config.output_is_pooled);
}

test "Granite Embedding English R2 config" {
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
    const config = Model.granite_embedding_english_r2_fp16.getConfig();

    try std.testing.expectEqualStrings("granite-embedding-english-r2-fp16", config.name);
    try std.testing.expectEqualStrings("onnx/model_fp16_embedded.onnx", config.model_file);
    try std.testing.expectEqual(@as(usize, 768), config.hidden_dim);
    try std.testing.expectEqual(@as(usize, 8192), config.max_seq_len);
    try std.testing.expect(!config.use_token_type_ids);
    try std.testing.expect(!config.output_is_pooled);
}

test "All models have valid configs" {
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

    // Verify we have exactly 13 models
    const model_count = @typeInfo(Model).@"enum".fields.len;
    try std.testing.expectEqual(@as(usize, 13), model_count);
}

test "Pooling strategy distribution" {
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
    try std.testing.expectEqualStrings("model.onnx", ModelFiles.model_onnx);
    try std.testing.expectEqualStrings("tokenizer.json", ModelFiles.tokenizer_json);
    try std.testing.expectEqualStrings("config.json", ModelFiles.config_json);
    try std.testing.expectEqualStrings("special_tokens_map.json", ModelFiles.special_tokens_map);
}

test "getHfUrl returns empty for now" {
    const url = getHfUrl("BAAI/bge-small-en-v1.5", "model.onnx");
    try std.testing.expectEqual(@as(usize, 0), url.len);
}

test "Gemma variants share same HF repo" {
    const base = Model.embedding_gemma_300m.getConfig().hf_repo;
    const q4 = Model.embedding_gemma_300m_q4.getConfig().hf_repo;
    const q4f16 = Model.embedding_gemma_300m_q4f16.getConfig().hf_repo;
    const fp16 = Model.embedding_gemma_300m_fp16.getConfig().hf_repo;

    try std.testing.expectEqualStrings(base, q4);
    try std.testing.expectEqualStrings(base, q4f16);
    try std.testing.expectEqualStrings(base, fp16);
}

test "Granite variants share same HF repo" {
    const quantized = Model.granite_embedding_english_r2.getConfig().hf_repo;
    const fp16 = Model.granite_embedding_english_r2_fp16.getConfig().hf_repo;

    try std.testing.expectEqualStrings(quantized, fp16);
}

test "Output names for pooled models" {

    // Gemma models use "sentence_embedding" output
    try std.testing.expectEqualStrings("sentence_embedding", Model.embedding_gemma_300m.getConfig().output_name.?);
    try std.testing.expectEqualStrings("sentence_embedding", Model.embedding_gemma_300m_q4.getConfig().output_name.?);

    // Non-pooled models use default (null)
    try std.testing.expect(Model.bge_small_en_v1_5.getConfig().output_name == null);
    try std.testing.expect(Model.granite_embedding_english_r2.getConfig().output_name == null);
}
