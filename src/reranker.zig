//! Cross-encoder reranker for semantic search
//!
//! Takes query-document pairs and produces relevance scores.
//! Uses ModernBERT-based models optimized for reranking.
//!
//! ## Quick Start
//!
//! ```zig
//! const fe = @import("fastembed");
//!
//! var reranker = try fe.Reranker.init(allocator, .{
//!     .model_path = "models/granite-reranker",
//! });
//! defer reranker.deinit();
//!
//! const scores = try reranker.rerank("search query", &[_][]const u8{
//!     "relevant document",
//!     "less relevant document",
//! });
//! defer allocator.free(scores);
//! // scores[0] > scores[1] if doc 1 is more relevant
//! ```

const std = @import("std");
const tokenizer_mod = @import("tokenizer/tokenizer.zig");
const onnx = @import("onnx/session.zig");
const models_mod = @import("models.zig");

pub const RerankerError = error{
    TokenizerError,
    ModelError,
    InferenceError,
    InvalidInput,
    OutOfMemory,
};

const builtin = @import("builtin");

// Re-use CPU feature detection from models.zig
const CpuFeatures = models_mod.CpuFeatures;

/// Get the optimal reranker ONNX model file for the current CPU
/// Uses runtime detection for x86_64 to select the best quantized variant
fn getOptimalRerankerOnnxFile() []const u8 {
    // Reranker uses same file naming as embedding model
    return models_mod.getOptimalOnnxFile();
}

/// Reranker model types
pub const RerankerModel = enum {
    /// Granite Embedding Reranker English R2 (CPU - auto-selects arch-specific quantized file)
    granite_reranker_english_r2,
    /// Granite Embedding Reranker English R2 (O4 optimized for CUDA)
    granite_reranker_english_r2_o4,

    pub fn getConfig(self: RerankerModel) RerankerConfig {
        return switch (self) {
            .granite_reranker_english_r2 => .{
                // Compile-time fallback - use getModelFile() for runtime detection
                .model_file = if (builtin.cpu.arch == .aarch64)
                    "onnx/model_qint8_arm64.onnx"
                else
                    "onnx/model_quint8_avx2.onnx",
                .hidden_dim = 768,
                .max_seq_len = 8192,
                .num_labels = 1,
            },
            .granite_reranker_english_r2_o4 => .{
                .model_file = "onnx/model_f16_cuda.onnx", // FP16 optimized for CUDA
                .hidden_dim = 768,
                .max_seq_len = 8192,
                .num_labels = 1,
            },
        };
    }

    /// Get the optimal model file for this reranker type based on runtime CPU detection
    /// For CPU variants, uses runtime CPU feature detection to select the best variant.
    /// For CUDA, returns the FP16 CUDA-optimized file.
    pub fn getModelFile(self: RerankerModel) []const u8 {
        return switch (self) {
            .granite_reranker_english_r2 => getOptimalRerankerOnnxFile(),
            .granite_reranker_english_r2_o4 => "onnx/model_f16_cuda.onnx",
        };
    }
};

/// Configuration for a reranker model
pub const RerankerConfig = struct {
    model_file: []const u8,
    hidden_dim: usize,
    max_seq_len: usize,
    num_labels: usize,
};

/// Execution provider types (re-exported from onnx module)
pub const ExecutionProvider = onnx.ExecutionProvider;

/// Options for creating a Reranker
pub const RerankerOptions = struct {
    /// Model to use (default: CPU with arch-specific quantization)
    model: RerankerModel = .granite_reranker_english_r2,
    /// Path to model directory (required)
    model_path: ?[]const u8 = null,
    /// Maximum sequence length (0 = use model default of 8192)
    max_seq_len: usize = 0,
    /// Execution provider (default: CPU)
    execution_provider: ExecutionProvider = .{ .cpu = {} },
    /// Number of threads for intra-op parallelism (0 = ONNX default)
    intra_op_num_threads: u32 = 0,
    /// Number of threads for inter-op parallelism (0 = ONNX default)
    inter_op_num_threads: u32 = 0,
};

/// Cross-encoder reranker for semantic search
pub const Reranker = struct {
    allocator: std.mem.Allocator,
    tokenizer: tokenizer_mod.Tokenizer,
    session: onnx.Session,
    env: onnx.Environment,
    config: RerankerConfig,
    max_seq_len: usize,

    const Self = @This();

    /// Initialize reranker from a local model directory
    ///
    /// The directory should contain:
    /// - onnx/model_*.onnx (one of the quantized variants)
    /// - tokenizer.json
    pub fn init(allocator: std.mem.Allocator, options: RerankerOptions) RerankerError!Self {
        const config = options.model.getConfig();

        const model_path = options.model_path orelse {
            return RerankerError.ModelError;
        };

        // Build paths - use runtime CPU detection for optimal model file
        const model_file = options.model.getModelFile();
        const model_onnx_path = std.fs.path.join(allocator, &.{ model_path, model_file }) catch return RerankerError.OutOfMemory;
        defer allocator.free(model_onnx_path);

        const tokenizer_path = std.fs.path.join(allocator, &.{ model_path, "tokenizer.json" }) catch return RerankerError.OutOfMemory;
        defer allocator.free(tokenizer_path);

        // Load tokenizer
        var tok = tokenizer_mod.Tokenizer.fromFile(allocator, tokenizer_path) catch {
            return RerankerError.TokenizerError;
        };
        errdefer tok.deinit();

        // Initialize ONNX environment
        var env = onnx.Environment.init() catch {
            return RerankerError.ModelError;
        };
        errdefer env.deinit();

        // Convert path to null-terminated
        const model_onnx_path_z = allocator.allocSentinel(u8, model_onnx_path.len, 0) catch return RerankerError.OutOfMemory;
        defer allocator.free(model_onnx_path_z);
        @memcpy(model_onnx_path_z, model_onnx_path);

        // Load ONNX model
        var session = onnx.Session.initWithOptions(env, model_onnx_path_z, allocator, .{
            .execution_provider = options.execution_provider,
            .intra_op_num_threads = options.intra_op_num_threads,
            .inter_op_num_threads = options.inter_op_num_threads,
        }) catch {
            return RerankerError.ModelError;
        };
        errdefer session.deinit();

        const max_seq_len = if (options.max_seq_len > 0) options.max_seq_len else config.max_seq_len;

        return Self{
            .allocator = allocator,
            .tokenizer = tok,
            .session = session,
            .env = env,
            .config = config,
            .max_seq_len = max_seq_len,
        };
    }

    /// Release all resources
    pub fn deinit(self: *Self) void {
        self.session.deinit();
        self.env.deinit();
        var tok = self.tokenizer;
        tok.deinit();
    }

    /// Rerank documents against a query
    ///
    /// Returns relevance scores for each document. Higher scores = more relevant.
    /// The scores are logits from the classification head, not normalized probabilities.
    pub fn rerank(self: *Self, query: []const u8, documents: []const []const u8) RerankerError![]f32 {
        if (documents.len == 0) {
            return &[_]f32{};
        }

        const scores = self.allocator.alloc(f32, documents.len) catch return RerankerError.OutOfMemory;
        errdefer self.allocator.free(scores);

        // Process each query-document pair
        // CrossEncoder models expect: [CLS] query [SEP] document [SEP]
        for (documents, 0..) |doc, i| {
            scores[i] = try self.scorePair(query, doc);
        }

        return scores;
    }

    /// Rerank and return indices sorted by relevance (highest first)
    pub fn rerankWithIndices(self: *Self, query: []const u8, documents: []const []const u8) RerankerError!struct {
        scores: []f32,
        indices: []usize,
    } {
        const scores = try self.rerank(query, documents);
        errdefer self.allocator.free(scores);

        // Create index array
        var indices = self.allocator.alloc(usize, documents.len) catch return RerankerError.OutOfMemory;
        errdefer self.allocator.free(indices);

        for (0..indices.len) |i| {
            indices[i] = i;
        }

        // Sort indices by scores (descending)
        std.mem.sort(usize, indices, scores, struct {
            fn lessThan(s: []f32, a: usize, b: usize) bool {
                return s[b] < s[a]; // Descending order
            }
        }.lessThan);

        return .{ .scores = scores, .indices = indices };
    }

    /// Score a single query-document pair
    fn scorePair(self: *Self, query: []const u8, document: []const u8) RerankerError!f32 {
        // Encode query and document with [SEP] separator
        // CrossEncoder format: [CLS] query [SEP] document [SEP]
        const query_tokens = self.tokenizer.encodeAlloc(self.allocator, query, false) catch {
            return RerankerError.TokenizerError;
        };
        defer self.allocator.free(query_tokens);

        const doc_tokens = self.tokenizer.encodeAlloc(self.allocator, document, false) catch {
            return RerankerError.TokenizerError;
        };
        defer self.allocator.free(doc_tokens);

        // Get special token IDs (ModernBERT uses standard BERT tokenizer)
        const cls_id: i32 = self.tokenizer.tokenToId("[CLS]") orelse 101; // Default BERT CLS
        const sep_id: i32 = self.tokenizer.tokenToId("[SEP]") orelse 102; // Default BERT SEP

        // Build combined sequence: [CLS] query [SEP] document [SEP]
        // Total length = 1 + query_len + 1 + doc_len + 1 = query_len + doc_len + 3
        const total_tokens = query_tokens.len + doc_tokens.len + 3;
        const seq_len = @min(total_tokens, self.max_seq_len);

        // Allocate input tensors
        var input_ids = self.allocator.alloc(i64, seq_len) catch return RerankerError.OutOfMemory;
        defer self.allocator.free(input_ids);
        var attention_mask = self.allocator.alloc(i64, seq_len) catch return RerankerError.OutOfMemory;
        defer self.allocator.free(attention_mask);

        // Build sequence
        var pos: usize = 0;
        input_ids[pos] = @as(i64, cls_id);
        pos += 1;

        // Query tokens (truncate if needed to leave room for doc)
        const max_query_len = (self.max_seq_len - 3) / 2; // Half for query, half for doc
        const actual_query_len = @min(query_tokens.len, max_query_len);
        for (query_tokens[0..actual_query_len]) |tok| {
            if (pos >= seq_len - 2) break; // Leave room for SEP and at least one doc token
            input_ids[pos] = @as(i64, tok);
            pos += 1;
        }

        input_ids[pos] = @as(i64, sep_id);
        pos += 1;

        // Document tokens (use remaining space)
        const remaining = seq_len - pos - 1; // -1 for final SEP
        const actual_doc_len = @min(doc_tokens.len, remaining);
        for (doc_tokens[0..actual_doc_len]) |tok| {
            input_ids[pos] = @as(i64, tok);
            pos += 1;
        }

        input_ids[pos] = @as(i64, sep_id);
        pos += 1;

        // Set attention mask (all 1s for real tokens)
        @memset(attention_mask[0..pos], 1);
        if (pos < seq_len) {
            @memset(attention_mask[pos..], 0);
            @memset(input_ids[pos..], 0);
        }

        // Run inference
        // ModernBERT reranker outputs logits directly (not hidden states)
        const logits = self.session.run(
            input_ids[0..pos],
            attention_mask[0..pos],
            null, // ModernBERT doesn't use token_type_ids
            1, // batch_size = 1
            pos,
            "logits", // Output name for classification models
        ) catch {
            return RerankerError.InferenceError;
        };
        defer self.allocator.free(logits);

        // Return the logit (single score for binary classification)
        return logits[0];
    }
};

test "Reranker struct compiles" {
    try std.testing.expect(@sizeOf(Reranker) > 0);
}

test "RerankerModel enum" {
    const models = [_]RerankerModel{
        .granite_reranker_english_r2,
        .granite_reranker_english_r2_o4,
    };

    for (models) |m| {
        const config = m.getConfig();
        try std.testing.expect(config.hidden_dim == 768);
        try std.testing.expect(config.max_seq_len == 8192);
        try std.testing.expect(config.num_labels == 1);
    }
}

test "RerankerOptions default values" {
    const opts = RerankerOptions{};

    try std.testing.expectEqual(RerankerModel.granite_reranker_english_r2, opts.model);
    try std.testing.expect(opts.model_path == null);
    try std.testing.expectEqual(@as(usize, 0), opts.max_seq_len);
}
