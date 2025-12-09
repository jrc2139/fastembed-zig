//! Text embedding using transformer models
//!
//! The main Embedder struct that combines tokenization and ONNX inference
//! to generate text embeddings.

const std = @import("std");
const tokenizer_mod = @import("tokenizer/tokenizer.zig");
const onnx = @import("onnx/session.zig");
const pooling = @import("pooling.zig");
const normalize = @import("normalize.zig");
const models = @import("models.zig");

pub const EmbedderError = error{
    TokenizerError,
    ModelError,
    InferenceError,
    InvalidInput,
    OutOfMemory,
};

/// Execution provider types (re-exported from onnx module)
pub const ExecutionProvider = onnx.ExecutionProvider;
pub const CoreMLOptions = onnx.CoreMLOptions;
pub const CoreMLComputeUnits = onnx.CoreMLComputeUnits;

/// Options for creating an Embedder
pub const EmbedderOptions = struct {
    /// Model to use (default: Granite Embedding English R2 Q4)
    model: models.Model = .granite_embedding_english_r2,
    /// Path to model directory (if using local model)
    model_path: ?[]const u8 = null,
    /// Maximum sequence length (0 = use model default)
    max_seq_len: usize = 0,
    /// Whether to normalize embeddings (default: use model setting)
    normalize: ?bool = null,
    /// Execution provider (default: CPU)
    execution_provider: ExecutionProvider = .{ .cpu = {} },
};

/// Text embedder using transformer models
pub const Embedder = struct {
    allocator: std.mem.Allocator,
    tokenizer: tokenizer_mod.Tokenizer,
    session: onnx.Session,
    env: onnx.Environment,
    config: models.ModelConfig,
    max_seq_len: usize,
    do_normalize: bool,

    const Self = @This();

    /// Initialize embedder from a local model directory
    ///
    /// The directory should contain:
    /// - model.onnx
    /// - tokenizer.json
    pub fn init(allocator: std.mem.Allocator, options: EmbedderOptions) EmbedderError!Self {
        const config = options.model.getConfig();

        // Determine model path
        const model_path = options.model_path orelse {
            // TODO: Download model if not present
            return EmbedderError.ModelError;
        };

        // Build paths using model-specific file path
        const model_onnx_path = std.fs.path.join(allocator, &.{ model_path, config.model_file }) catch return EmbedderError.OutOfMemory;
        defer allocator.free(model_onnx_path);

        const tokenizer_path = std.fs.path.join(allocator, &.{ model_path, "tokenizer.json" }) catch return EmbedderError.OutOfMemory;
        defer allocator.free(tokenizer_path);

        // Load tokenizer
        var tok = tokenizer_mod.Tokenizer.fromFile(allocator, tokenizer_path) catch {
            return EmbedderError.TokenizerError;
        };
        errdefer tok.deinit();

        // Initialize ONNX environment
        var env = onnx.Environment.init() catch {
            return EmbedderError.ModelError;
        };
        errdefer env.deinit();

        // Convert path to null-terminated
        const model_onnx_path_z = allocator.allocSentinel(u8, model_onnx_path.len, 0) catch return EmbedderError.OutOfMemory;
        defer allocator.free(model_onnx_path_z);
        @memcpy(model_onnx_path_z, model_onnx_path);

        // Load ONNX model with configured execution provider
        var session = onnx.Session.initWithOptions(env, model_onnx_path_z, allocator, .{
            .execution_provider = options.execution_provider,
        }) catch {
            return EmbedderError.ModelError;
        };
        errdefer session.deinit();

        const max_seq_len = if (options.max_seq_len > 0) options.max_seq_len else config.max_seq_len;
        const do_normalize = options.normalize orelse config.normalize;

        return Self{
            .allocator = allocator,
            .tokenizer = tok,
            .session = session,
            .env = env,
            .config = config,
            .max_seq_len = max_seq_len,
            .do_normalize = do_normalize,
        };
    }

    /// Release all resources
    pub fn deinit(self: *Self) void {
        self.session.deinit();
        self.env.deinit();
        var tok = self.tokenizer;
        tok.deinit();
    }

    /// Generate embeddings for a list of texts
    ///
    /// Automatically splits into optimal batch sizes based on model memory profile.
    /// Returns: [num_texts][hidden_dim] as flattened array
    pub fn embed(self: *Self, texts: []const []const u8) EmbedderError![]f32 {
        if (texts.len == 0) {
            return &[_]f32{};
        }

        // Get optimal batch size from model memory profile
        // Use CPU batch size (TODO: detect GPU and use optimal_gpu_batch)
        const optimal_batch = self.config.memory_profile.getDefaultBatch(false);
        const batch_size: usize = @min(texts.len, optimal_batch);

        // If input fits in a single batch, process directly
        if (texts.len <= batch_size) {
            return self.embedBatch(texts);
        }

        // Split into multiple batches
        const dim = self.config.hidden_dim;
        const total_embeddings = self.allocator.alloc(f32, texts.len * dim) catch return EmbedderError.OutOfMemory;
        errdefer self.allocator.free(total_embeddings);

        var processed: usize = 0;
        while (processed < texts.len) {
            const end = @min(processed + batch_size, texts.len);
            const batch_texts = texts[processed..end];

            const batch_embeddings = try self.embedBatch(batch_texts);
            defer self.allocator.free(batch_embeddings);

            // Copy batch results into the combined output
            @memcpy(total_embeddings[processed * dim .. end * dim], batch_embeddings);

            processed = end;
        }

        return total_embeddings;
    }

    /// Internal: Generate embeddings for a single batch (no chunking)
    fn embedBatch(self: *Self, texts: []const []const u8) EmbedderError![]f32 {
        const batch_size = texts.len;

        // Tokenize all texts and track per-sequence lengths
        var seq_lengths = self.allocator.alloc(usize, batch_size) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(seq_lengths);

        // Phase 1.3: Pre-allocate token buffer with estimated capacity
        const estimated_tokens = batch_size * self.max_seq_len;
        var all_tokens = std.ArrayListUnmanaged(i64){};
        all_tokens.ensureTotalCapacity(self.allocator, estimated_tokens) catch return EmbedderError.OutOfMemory;
        defer all_tokens.deinit(self.allocator);

        var actual_seq_len: usize = 0;

        for (texts, 0..) |text, i| {
            const tokens = self.tokenizer.encodeAlloc(self.allocator, text, true) catch {
                return EmbedderError.TokenizerError;
            };
            defer self.allocator.free(tokens);

            // Truncate if needed
            const seq_len = @min(tokens.len, self.max_seq_len);
            seq_lengths[i] = seq_len;
            actual_seq_len = @max(actual_seq_len, seq_len);

            // Store tokens - capacity already ensured, append with i32->i64 conversion
            for (tokens[0..seq_len]) |tok| {
                all_tokens.appendAssumeCapacity(@as(i64, tok));
            }
        }

        // Pad all sequences to same length
        const padded_len = actual_seq_len;

        var input_ids = self.allocator.alloc(i64, batch_size * padded_len) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(input_ids);
        var attention_mask = self.allocator.alloc(i64, batch_size * padded_len) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(attention_mask);

        // Phase 1.4: Only allocate token_type_ids if model uses them
        const token_type_ids: ?[]i64 = if (self.config.use_token_type_ids) blk: {
            const ids = self.allocator.alloc(i64, batch_size * padded_len) catch return EmbedderError.OutOfMemory;
            @memset(ids, 0); // token_type_ids are always 0 for single-sequence
            break :blk ids;
        } else null;
        defer if (token_type_ids) |ids| self.allocator.free(ids);

        // Phase 1.2: Only initialize attention_mask with zeros (for padding)
        // input_ids will be fully written, token_type_ids initialized above if needed
        @memset(attention_mask, 0);

        // Phase 1.1: Use memcpy for token copying instead of element-by-element
        var src_offset: usize = 0;
        for (0..batch_size) |b| {
            const seq_len = seq_lengths[b];
            const dst_offset = b * padded_len;

            // Bulk copy tokens using memcpy
            @memcpy(input_ids[dst_offset..][0..seq_len], all_tokens.items[src_offset..][0..seq_len]);
            // Bulk set attention mask to 1 for valid tokens
            @memset(attention_mask[dst_offset..][0..seq_len], 1);
            // Pad input_ids with zeros for remaining positions
            if (seq_len < padded_len) {
                @memset(input_ids[dst_offset + seq_len ..][0 .. padded_len - seq_len], 0);
            }

            src_offset += seq_len;
        }

        // Run inference (pass token_type_ids only if model uses them)
        const hidden_states = self.session.run(
            input_ids,
            attention_mask,
            token_type_ids,
            batch_size,
            padded_len,
            self.config.output_name,
        ) catch {
            return EmbedderError.InferenceError;
        };

        // Handle pooling based on model output type
        const embeddings = if (self.config.output_is_pooled) blk: {
            // Model already outputs pooled embeddings (e.g., Gemma's sentence_embedding)
            // hidden_states is already [batch_size, hidden_dim]
            break :blk hidden_states;
        } else blk: {
            defer self.allocator.free(hidden_states);
            // Need to pool the token-level outputs
            break :blk switch (self.config.pooling) {
                .cls => pooling.poolCls(hidden_states, batch_size, padded_len, self.config.hidden_dim, self.allocator) catch return EmbedderError.OutOfMemory,
                .mean => pooling.poolMean(hidden_states, attention_mask, batch_size, padded_len, self.config.hidden_dim, self.allocator) catch return EmbedderError.OutOfMemory,
            };
        };

        // Normalize if needed
        if (self.do_normalize) {
            normalize.l2NormalizeBatch(embeddings, batch_size, self.config.hidden_dim);
        }

        return embeddings;
    }

    /// Generate embedding for a single text
    pub fn embedOne(self: *Self, text: []const u8) EmbedderError![]f32 {
        const texts = [_][]const u8{text};
        return self.embed(&texts);
    }

    /// Generate query embedding (adds query prefix for asymmetric models)
    pub fn queryEmbed(self: *Self, query: []const u8) EmbedderError![]f32 {
        if (self.config.query_prefix) |prefix| {
            const prefixed = std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, query }) catch {
                return EmbedderError.OutOfMemory;
            };
            defer self.allocator.free(prefixed);
            return self.embedOne(prefixed);
        }
        return self.embedOne(query);
    }

    /// Generate passage embedding (adds passage prefix for asymmetric models)
    pub fn passageEmbed(self: *Self, passage: []const u8) EmbedderError![]f32 {
        if (self.config.passage_prefix) |prefix| {
            const prefixed = std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, passage }) catch {
                return EmbedderError.OutOfMemory;
            };
            defer self.allocator.free(prefixed);
            return self.embedOne(prefixed);
        }
        return self.embedOne(passage);
    }

    /// Generate passage embeddings for multiple texts (adds passage prefix for asymmetric models)
    /// Returns: [num_texts][hidden_dim] as flattened array
    pub fn passageEmbedBatch(self: *Self, passages: []const []const u8) EmbedderError![]f32 {
        if (self.config.passage_prefix) |prefix| {
            // Allocate prefixed texts
            var prefixed_texts = self.allocator.alloc([]u8, passages.len) catch return EmbedderError.OutOfMemory;
            defer {
                for (prefixed_texts) |text| {
                    self.allocator.free(text);
                }
                self.allocator.free(prefixed_texts);
            }

            // Prefix each passage
            for (passages, 0..) |passage, i| {
                prefixed_texts[i] = std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, passage }) catch {
                    // Clean up already allocated texts on error
                    for (prefixed_texts[0..i]) |text| {
                        self.allocator.free(text);
                    }
                    return EmbedderError.OutOfMemory;
                };
            }

            // Cast [][]u8 to [][]const u8 for embed()
            const const_texts: []const []const u8 = @ptrCast(prefixed_texts);
            return self.embed(const_texts);
        }
        return self.embed(passages);
    }

    /// Get embedding dimension
    pub fn getDimension(self: Self) usize {
        return self.config.hidden_dim;
    }
};

test "Embedder struct compiles" {
    try std.testing.expect(@sizeOf(Embedder) > 0);
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "EmbedderError enum" {
    // Test that all error types exist
    const errors = [_]EmbedderError{
        EmbedderError.TokenizerError,
        EmbedderError.ModelError,
        EmbedderError.InferenceError,
        EmbedderError.InvalidInput,
        EmbedderError.OutOfMemory,
    };

    try std.testing.expectEqual(@as(usize, 5), errors.len);
}

test "EmbedderOptions default values" {
    const opts = EmbedderOptions{};

    try std.testing.expectEqual(models.Model.granite_embedding_english_r2, opts.model);
    try std.testing.expect(opts.model_path == null);
    try std.testing.expectEqual(@as(usize, 0), opts.max_seq_len);
    try std.testing.expect(opts.normalize == null);
    try std.testing.expectEqual(ExecutionProvider{ .cpu = {} }, opts.execution_provider);
}

test "EmbedderOptions with custom values" {
    const opts = EmbedderOptions{
        .model = .bge_small_en_v1_5,
        .model_path = "/some/path",
        .max_seq_len = 128,
        .normalize = false,
        .execution_provider = .{ .cpu = {} },
    };

    try std.testing.expectEqual(models.Model.bge_small_en_v1_5, opts.model);
    try std.testing.expectEqualStrings("/some/path", opts.model_path.?);
    try std.testing.expectEqual(@as(usize, 128), opts.max_seq_len);
    try std.testing.expectEqual(false, opts.normalize.?);
}

test "EmbedderOptions with all model types" {
    const model_types = [_]models.Model{
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

    for (model_types) |model| {
        const opts = EmbedderOptions{
            .model = model,
        };
        try std.testing.expectEqual(model, opts.model);
    }
}

test "ExecutionProvider CPU type" {
    const provider = ExecutionProvider{ .cpu = {} };
    _ = provider;
    // Just ensure it compiles and is valid
    try std.testing.expect(true);
}

test "ExecutionProvider CoreML type with default options" {
    const provider = ExecutionProvider{ .coreml = .{} };
    _ = provider;
    try std.testing.expect(true);
}

test "ExecutionProvider CoreML type with custom options" {
    const provider = ExecutionProvider{
        .coreml = .{
            .compute_units = .cpu_and_neural_engine,
            .require_static_input_shapes = true,
        },
    };
    _ = provider;
    try std.testing.expect(true);
}

test "CoreMLComputeUnits variants" {
    const cpu_only = CoreMLComputeUnits.cpu_only;
    const cpu_and_gpu = CoreMLComputeUnits.cpu_and_gpu;
    const all = CoreMLComputeUnits.all;
    const cpu_and_neural_engine = CoreMLComputeUnits.cpu_and_neural_engine;

    try std.testing.expect(cpu_only != cpu_and_gpu);
    try std.testing.expect(cpu_and_gpu != all);
    try std.testing.expect(all != cpu_and_neural_engine);
}

test "CoreMLOptions default values" {
    const opts = CoreMLOptions{};

    try std.testing.expectEqual(CoreMLComputeUnits.cpu_and_neural_engine, opts.compute_units);
    try std.testing.expectEqual(false, opts.require_static_input_shapes);
}

test "CoreMLOptions custom values" {
    const opts = CoreMLOptions{
        .compute_units = .cpu_and_neural_engine,
        .require_static_input_shapes = true,
    };

    try std.testing.expectEqual(CoreMLComputeUnits.cpu_and_neural_engine, opts.compute_units);
    try std.testing.expectEqual(true, opts.require_static_input_shapes);
}

test "EmbedderOptions max_seq_len override" {
    // Test that max_seq_len = 0 means "use model default"
    const opts_default = EmbedderOptions{
        .model = .bge_small_en_v1_5, // default is 512
        .max_seq_len = 0,
    };
    try std.testing.expectEqual(@as(usize, 0), opts_default.max_seq_len);

    // Custom override
    const opts_custom = EmbedderOptions{
        .model = .bge_small_en_v1_5,
        .max_seq_len = 256,
    };
    try std.testing.expectEqual(@as(usize, 256), opts_custom.max_seq_len);
}

test "EmbedderOptions normalize override" {
    // null means use model default
    const opts_default = EmbedderOptions{
        .normalize = null,
    };
    try std.testing.expect(opts_default.normalize == null);

    // Explicit true
    const opts_true = EmbedderOptions{
        .normalize = true,
    };
    try std.testing.expectEqual(true, opts_true.normalize.?);

    // Explicit false (override model's normalize=true)
    const opts_false = EmbedderOptions{
        .normalize = false,
    };
    try std.testing.expectEqual(false, opts_false.normalize.?);
}

test "Embedder struct size is reasonable" {
    // Embedder should be a reasonable size (not bloated)
    const size = @sizeOf(Embedder);
    try std.testing.expect(size > 0);
    try std.testing.expect(size < 4096); // Should not be massive
}

test "Embedder has expected fields" {
    // Use @hasField to verify structure
    try std.testing.expect(@hasField(Embedder, "allocator"));
    try std.testing.expect(@hasField(Embedder, "tokenizer"));
    try std.testing.expect(@hasField(Embedder, "session"));
    try std.testing.expect(@hasField(Embedder, "env"));
    try std.testing.expect(@hasField(Embedder, "config"));
    try std.testing.expect(@hasField(Embedder, "max_seq_len"));
    try std.testing.expect(@hasField(Embedder, "do_normalize"));
}
