//! Zero-Allocation Text Embedding
//!
//! FastEmbedder provides zero-allocation text embedding by combining:
//! - FastTokenizer: Arena-based tokenization with buffer reuse
//! - FastSession: IoBinding-based inference with pre-allocated tensors
//!
//! After initialization, the embed() method performs zero heap allocations,
//! making it ideal for high-throughput embedding pipelines.
//!
//! Usage:
//! ```zig
//! var embedder = try FastEmbedder.init(allocator, .{
//!     .model_path = "models/bge-small-en-v1.5",
//! });
//! defer embedder.deinit();
//!
//! // Zero-allocation embedding loop
//! for (texts) |text| {
//!     const embedding = try embedder.embedOne(text);
//!     // embedding is valid until next embed call
//! }
//! ```

const std = @import("std");
const tokenizer_mod = @import("tokenizer/tokenizer.zig");
const fast_session = @import("onnx/fast_session.zig");
const session_mod = @import("onnx/session.zig");
const pooling = @import("pooling.zig");
const normalize = @import("normalize.zig");
const models = @import("models.zig");

pub const FastTokenizer = tokenizer_mod.FastTokenizer;
pub const FastTokenizerOptions = tokenizer_mod.FastTokenizerOptions;
pub const FastSession = fast_session.FastSession;
pub const FastSessionConfig = fast_session.FastSessionConfig;
pub const ExecutionProvider = session_mod.ExecutionProvider;
pub const CoreMLOptions = session_mod.CoreMLOptions;
pub const CoreMLComputeUnits = session_mod.CoreMLComputeUnits;

/// Error type for fast embedder operations
pub const FastEmbedderError = error{
    TokenizerError,
    ModelError,
    InferenceError,
    InvalidInput,
    OutOfMemory,
    ConfigurationError,
    BatchTooLarge,
    SequenceTooLong,
};

/// Options for FastEmbedder initialization
pub const FastEmbedderOptions = struct {
    /// Model to use
    model: models.Model = .granite_embedding_english_r2,
    /// Path to model directory (required)
    model_path: ?[]const u8 = null,
    /// Maximum batch size for pre-allocation
    max_batch_size: usize = 32,
    /// Maximum sequence length (0 = use 512 default, capped for reasonable memory)
    max_seq_len: usize = 0,
    /// Whether to normalize embeddings (null = use model default)
    do_normalize: ?bool = null,
    /// Execution provider
    execution_provider: ExecutionProvider = .{ .cpu = {} },
    /// Number of intra-op threads (0 = use all cores)
    intra_op_num_threads: u32 = 0,
    /// Number of inter-op threads (0 = use all cores)
    inter_op_num_threads: u32 = 0,
};

/// Zero-allocation text embedder.
///
/// After initialization, all embed methods perform zero heap allocations
/// by reusing pre-allocated buffers. The returned embeddings are valid
/// until the next embed call.
///
/// Thread Safety: NOT thread-safe. Use separate instances per thread.
pub const FastEmbedder = struct {
    allocator: std.mem.Allocator,
    tokenizer: FastTokenizer,
    session: FastSession,
    config: models.ModelConfig,
    max_batch_size: usize,
    max_seq_len: usize,
    do_normalize: bool,

    // Per-sequence length tracking for attention mask
    seq_lengths: []usize,

    // Pooled embeddings buffer (for non-pooled models)
    pooled_buffer: []f32,

    const Self = @This();

    /// Initialize FastEmbedder with pre-allocated buffers.
    ///
    /// This allocates all required memory upfront. After init,
    /// embed operations are zero-allocation.
    pub fn init(allocator: std.mem.Allocator, options: FastEmbedderOptions) FastEmbedderError!Self {
        const model_config = options.model.getConfig();

        const model_path = options.model_path orelse {
            return FastEmbedderError.ModelError;
        };

        // Determine limits - use 512 as default to avoid huge memory allocations
        // for models like granite which support up to 8192 tokens
        const max_seq_len = if (options.max_seq_len > 0)
            options.max_seq_len
        else
            @min(model_config.max_seq_len, 512);

        const max_batch_size = options.max_batch_size;
        const do_normalize = options.do_normalize orelse model_config.normalize;

        // Build paths
        const model_file = options.model.getModelFile();
        const model_onnx_path = std.fs.path.join(allocator, &.{ model_path, model_file }) catch {
            return FastEmbedderError.OutOfMemory;
        };
        defer allocator.free(model_onnx_path);

        const tokenizer_path = std.fs.path.join(allocator, &.{ model_path, "tokenizer.json" }) catch {
            return FastEmbedderError.OutOfMemory;
        };
        defer allocator.free(tokenizer_path);

        // Create zero-sentinel path for ONNX
        const model_onnx_path_z = allocator.allocSentinel(u8, model_onnx_path.len, 0) catch {
            return FastEmbedderError.OutOfMemory;
        };
        defer allocator.free(model_onnx_path_z);
        @memcpy(model_onnx_path_z, model_onnx_path);

        // Initialize FastTokenizer
        var tokenizer = FastTokenizer.fromFile(allocator, tokenizer_path, .{
            .max_sequence_length = 16384, // Max input bytes
            .max_tokens = @intCast(max_seq_len),
        }) catch {
            return FastEmbedderError.TokenizerError;
        };
        errdefer tokenizer.deinit();

        // Initialize ONNX environment
        var env = session_mod.Environment.init() catch {
            return FastEmbedderError.ModelError;
        };
        errdefer env.deinit();

        // Initialize FastSession with pre-allocated buffers
        var session = FastSession.init(env, model_onnx_path_z, allocator, .{
            .max_batch_size = max_batch_size,
            .max_seq_len = max_seq_len,
            .hidden_dim = model_config.hidden_dim,
            .output_is_pooled = model_config.output_is_pooled,
            .use_token_type_ids = model_config.use_token_type_ids,
        }, .{
            .execution_provider = options.execution_provider,
            .intra_op_num_threads = options.intra_op_num_threads,
            .inter_op_num_threads = options.inter_op_num_threads,
        }) catch {
            return FastEmbedderError.ModelError;
        };
        errdefer session.deinit();

        // Allocate sequence length tracking
        const seq_lengths = allocator.alloc(usize, max_batch_size) catch {
            return FastEmbedderError.OutOfMemory;
        };
        errdefer allocator.free(seq_lengths);

        // Allocate pooled buffer for non-pooled models
        const pooled_buffer = allocator.alloc(f32, max_batch_size * model_config.hidden_dim) catch {
            return FastEmbedderError.OutOfMemory;
        };
        errdefer allocator.free(pooled_buffer);

        return Self{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .session = session,
            .config = model_config,
            .max_batch_size = max_batch_size,
            .max_seq_len = max_seq_len,
            .do_normalize = do_normalize,
            .seq_lengths = seq_lengths,
            .pooled_buffer = pooled_buffer,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.pooled_buffer);
        self.allocator.free(self.seq_lengths);
        self.session.deinit();
        self.tokenizer.deinit();
    }

    /// Embed a single text. Zero allocations after init.
    ///
    /// Returns a slice of the internal embeddings buffer.
    /// Valid until the next embed call.
    pub fn embedOne(self: *Self, text: []const u8) FastEmbedderError![]f32 {
        const texts = [_][]const u8{text};
        return self.embed(&texts);
    }

    /// Embed multiple texts. Zero allocations after init.
    ///
    /// Returns [batch_size * hidden_dim] floats as a flat array.
    /// Valid until the next embed call.
    pub fn embed(self: *Self, texts: []const []const u8) FastEmbedderError![]f32 {
        if (texts.len == 0) {
            return &[_]f32{};
        }

        if (texts.len > self.max_batch_size) {
            return FastEmbedderError.BatchTooLarge;
        }

        const batch_size = texts.len;

        // Get input buffers
        var actual_max_seq_len: usize = 0;

        // Phase 1: Tokenize all texts and find max sequence length
        for (texts, 0..) |text, i| {
            const encoding = self.tokenizer.encode(text) catch {
                return FastEmbedderError.TokenizerError;
            };

            // Truncate to max_seq_len
            const seq_len = @min(@as(usize, encoding.len), self.max_seq_len);
            self.seq_lengths[i] = seq_len;
            actual_max_seq_len = @max(actual_max_seq_len, seq_len);
        }

        // Phase 2: Fill input buffers with padding
        const buffers = self.session.getInputBuffers(batch_size, actual_max_seq_len);

        // Clear buffers (set to padding)
        @memset(buffers.input_ids, 0);
        @memset(buffers.attention_mask, 0);

        // Phase 3: Re-tokenize and fill buffers
        for (texts, 0..) |text, batch_idx| {
            const encoding = self.tokenizer.encode(text) catch {
                return FastEmbedderError.TokenizerError;
            };

            const seq_len = self.seq_lengths[batch_idx];
            const ids = encoding.getIds();
            const offset = batch_idx * actual_max_seq_len;

            // Copy token IDs (with i32->i64 conversion)
            for (0..seq_len) |i| {
                buffers.input_ids[offset + i] = @intCast(ids[i]);
                buffers.attention_mask[offset + i] = 1;
            }
        }

        // Phase 4: Run inference (zero allocations via IoBinding)
        const hidden_states = self.session.run(batch_size, actual_max_seq_len) catch {
            return FastEmbedderError.InferenceError;
        };

        // Phase 5: Pool if needed
        var embeddings: []f32 = undefined;
        if (self.config.output_is_pooled) {
            embeddings = hidden_states[0 .. batch_size * self.config.hidden_dim];
        } else {
            // Pool token outputs to get sentence embeddings
            const output_seq_len = hidden_states.len / (batch_size * self.config.hidden_dim);
            embeddings = switch (self.config.pooling) {
                .cls => self.poolCls(hidden_states, batch_size, output_seq_len),
                .mean => self.poolMean(hidden_states, buffers.attention_mask, batch_size, output_seq_len, actual_max_seq_len),
            };
        }

        // Phase 6: Normalize if needed
        if (self.do_normalize) {
            normalize.l2NormalizeBatch(embeddings, batch_size, self.config.hidden_dim);
        }

        return embeddings;
    }

    /// CLS pooling: take first token's embedding
    fn poolCls(self: *Self, hidden_states: []f32, batch_size: usize, seq_len: usize) []f32 {
        const hidden_dim = self.config.hidden_dim;

        for (0..batch_size) |b| {
            const src_offset = b * seq_len * hidden_dim;
            const dst_offset = b * hidden_dim;
            @memcpy(self.pooled_buffer[dst_offset..][0..hidden_dim], hidden_states[src_offset..][0..hidden_dim]);
        }

        return self.pooled_buffer[0 .. batch_size * hidden_dim];
    }

    /// Mean pooling: average over non-padding tokens
    fn poolMean(
        self: *Self,
        hidden_states: []f32,
        attention_mask: []i64,
        batch_size: usize,
        output_seq_len: usize,
        input_seq_len: usize,
    ) []f32 {
        const hidden_dim = self.config.hidden_dim;
        const effective_seq_len = @min(output_seq_len, input_seq_len);

        for (0..batch_size) |b| {
            const mask_offset = b * input_seq_len;

            // Count valid tokens
            var valid_count: f32 = 0;
            for (0..effective_seq_len) |s| {
                if (attention_mask[mask_offset + s] != 0) {
                    valid_count += 1;
                }
            }
            if (valid_count == 0) valid_count = 1; // Avoid division by zero

            // Compute mean
            const dst_offset = b * hidden_dim;
            @memset(self.pooled_buffer[dst_offset..][0..hidden_dim], 0);

            for (0..effective_seq_len) |s| {
                if (attention_mask[mask_offset + s] != 0) {
                    const src_offset = b * output_seq_len * hidden_dim + s * hidden_dim;
                    for (0..hidden_dim) |d| {
                        self.pooled_buffer[dst_offset + d] += hidden_states[src_offset + d];
                    }
                }
            }

            // Divide by count
            for (0..hidden_dim) |d| {
                self.pooled_buffer[dst_offset + d] /= valid_count;
            }
        }

        return self.pooled_buffer[0 .. batch_size * hidden_dim];
    }

    /// Get embedding dimension
    pub fn getDimension(self: Self) usize {
        return self.config.hidden_dim;
    }

    /// Get the token count for a text (uses tokenizer arena, no alloc)
    pub fn getTokenCount(self: *Self, text: []const u8) usize {
        return self.tokenizer.tokenCount(text);
    }

    /// Get memory usage of pre-allocated buffers
    pub fn memoryUsage(self: *const Self) usize {
        const input_size = self.max_batch_size * self.max_seq_len * @sizeOf(i64) * 2; // input_ids + mask
        const output_size = self.session.output_buffer.len * @sizeOf(f32);
        const pooled_size = self.pooled_buffer.len * @sizeOf(f32);
        const seq_lengths_size = self.seq_lengths.len * @sizeOf(usize);
        const tokenizer_arena = self.tokenizer.arenaMemoryUsage();

        return input_size + output_size + pooled_size + seq_lengths_size + tokenizer_arena;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "FastEmbedderOptions defaults" {
    const opts = FastEmbedderOptions{};
    try std.testing.expectEqual(models.Model.granite_embedding_english_r2, opts.model);
    try std.testing.expect(opts.model_path == null);
    try std.testing.expectEqual(@as(usize, 32), opts.max_batch_size);
    try std.testing.expectEqual(@as(usize, 0), opts.max_seq_len);
    try std.testing.expect(opts.do_normalize == null);
}

test "FastEmbedder struct has expected fields" {
    try std.testing.expect(@hasField(FastEmbedder, "tokenizer"));
    try std.testing.expect(@hasField(FastEmbedder, "session"));
    try std.testing.expect(@hasField(FastEmbedder, "config"));
    try std.testing.expect(@hasField(FastEmbedder, "pooled_buffer"));
    try std.testing.expect(@hasField(FastEmbedder, "seq_lengths"));
}

test "FastEmbedder.init fails without model_path" {
    const allocator = std.testing.allocator;

    const result = FastEmbedder.init(allocator, .{
        .model = .bge_small_en_v1_5,
        .model_path = null,
    });

    try std.testing.expectError(FastEmbedderError.ModelError, result);
}

test "FastEmbedderError enum" {
    const errors = [_]FastEmbedderError{
        FastEmbedderError.TokenizerError,
        FastEmbedderError.ModelError,
        FastEmbedderError.InferenceError,
        FastEmbedderError.InvalidInput,
        FastEmbedderError.OutOfMemory,
        FastEmbedderError.ConfigurationError,
        FastEmbedderError.BatchTooLarge,
        FastEmbedderError.SequenceTooLong,
    };

    try std.testing.expectEqual(@as(usize, 8), errors.len);
}
