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

/// Options for creating an Embedder
pub const EmbedderOptions = struct {
    /// Model to use (default: BGE small English)
    model: models.Model = .bge_small_en_v1_5,
    /// Path to model directory (if using local model)
    model_path: ?[]const u8 = null,
    /// Maximum sequence length (0 = use model default)
    max_seq_len: usize = 0,
    /// Whether to normalize embeddings (default: use model setting)
    normalize: ?bool = null,
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

        // Load ONNX model
        var session = onnx.Session.init(env, model_onnx_path_z, allocator) catch {
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
    /// Returns: [num_texts][hidden_dim] as flattened array
    pub fn embed(self: *Self, texts: []const []const u8) EmbedderError![]f32 {
        if (texts.len == 0) {
            return &[_]f32{};
        }

        const batch_size = texts.len;

        // Tokenize all texts and track per-sequence lengths
        var seq_lengths = self.allocator.alloc(usize, batch_size) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(seq_lengths);

        var all_tokens = std.ArrayListUnmanaged(i64){};
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

            // Store tokens
            for (tokens[0..seq_len]) |tok| {
                all_tokens.append(self.allocator, tok) catch return EmbedderError.OutOfMemory;
            }
        }

        // Pad all sequences to same length
        const padded_len = actual_seq_len;

        var input_ids = self.allocator.alloc(i64, batch_size * padded_len) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(input_ids);
        var attention_mask = self.allocator.alloc(i64, batch_size * padded_len) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(attention_mask);
        var token_type_ids = self.allocator.alloc(i64, batch_size * padded_len) catch return EmbedderError.OutOfMemory;
        defer self.allocator.free(token_type_ids);

        // Initialize with padding
        @memset(input_ids, 0);
        @memset(attention_mask, 0);
        @memset(token_type_ids, 0);

        // Copy tokenized data using tracked sequence lengths
        var src_offset: usize = 0;
        for (0..batch_size) |b| {
            const seq_len = seq_lengths[b];
            const dst_offset = b * padded_len;

            // Copy tokens for this sequence
            for (0..seq_len) |i| {
                input_ids[dst_offset + i] = all_tokens.items[src_offset + i];
                attention_mask[dst_offset + i] = 1;
                token_type_ids[dst_offset + i] = 0;
            }
            src_offset += seq_len;
        }

        // Run inference (pass token_type_ids only if model uses them)
        const hidden_states = self.session.run(
            input_ids,
            attention_mask,
            if (self.config.use_token_type_ids) token_type_ids else null,
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

    /// Get embedding dimension
    pub fn getDimension(self: Self) usize {
        return self.config.hidden_dim;
    }
};

test "Embedder struct compiles" {
    try std.testing.expect(@sizeOf(Embedder) > 0);
}
