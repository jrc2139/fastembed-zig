//! High-level Tokenizer wrapper
//!
//! Provides an idiomatic Zig interface to HuggingFace tokenizers.
//! Now powered by pure Zig tokenizer-zig library for maximum performance.
//!
//! This module provides two tokenizer types:
//! - `Tokenizer`: Standard allocating tokenizer for flexibility
//! - `FastTokenizer`: Zero-allocation tokenizer for maximum throughput

const std = @import("std");
const tokenizer_lib = @import("tokenizer");

/// Error type for tokenizer operations
pub const TokenizerError = error{
    /// Failed to create tokenizer from JSON
    InvalidJson,
    /// Tokenizer handle is null/invalid
    InvalidHandle,
    /// Memory allocation failed
    OutOfMemory,
    /// Failed to read tokenizer file
    FileError,
    /// Tokenization failed
    TokenizationError,
    /// Decoding failed
    DecodingError,
};

/// A HuggingFace-compatible tokenizer (pure Zig implementation)
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    inner: tokenizer_lib.Tokenizer,

    const Self = @This();

    /// Create a tokenizer from a JSON configuration string
    pub fn fromJson(allocator: std.mem.Allocator, json: []const u8) TokenizerError!Self {
        const inner = tokenizer_lib.Tokenizer.fromJson(allocator, json) catch {
            return TokenizerError.InvalidJson;
        };
        return Self{
            .allocator = allocator,
            .inner = inner,
        };
    }

    /// Create a tokenizer from a tokenizer.json file
    pub fn fromFile(allocator: std.mem.Allocator, path: []const u8) TokenizerError!Self {
        const inner = tokenizer_lib.Tokenizer.fromFile(allocator, path) catch |err| {
            return switch (err) {
                error.FileNotFound, error.AccessDenied, error.IsDir => TokenizerError.FileError,
                error.OutOfMemory => TokenizerError.OutOfMemory,
                else => TokenizerError.InvalidJson,
            };
        };
        return Self{
            .allocator = allocator,
            .inner = inner,
        };
    }

    /// Release the tokenizer
    pub fn deinit(self: *Self) void {
        self.inner.deinit();
    }

    /// Encode text and copy results to caller-owned memory
    /// Returns token IDs as i32 for compatibility with existing ONNX code
    pub fn encodeAlloc(self: *Self, allocator: std.mem.Allocator, text: []const u8, add_special_tokens: bool) TokenizerError![]i32 {
        var encoding = self.inner.encode(text, add_special_tokens) catch {
            return TokenizerError.TokenizationError;
        };
        defer encoding.deinit();

        // Convert u32 IDs to i32 for ONNX compatibility
        const result = allocator.alloc(i32, encoding.ids.len) catch {
            return TokenizerError.OutOfMemory;
        };

        for (encoding.ids, 0..) |id, i| {
            result[i] = @intCast(id);
        }

        return result;
    }

    /// Encode text to token IDs (legacy API)
    ///
    /// Note: For better performance, prefer encodeAlloc() which doesn't require
    /// an extra copy. This method exists for API compatibility.
    pub fn encode(self: *Self, text: []const u8, add_special_tokens: bool) []const i32 {
        // Use internal allocator for legacy API
        return self.encodeAlloc(self.allocator, text, add_special_tokens) catch &[_]i32{};
    }

    /// Free tokens returned by encode() (no-op for pure Zig implementation)
    pub fn freeTokens(self: *Self, tokens: []const i32) void {
        if (tokens.len == 0) return;
        // The encode() method allocates with self.allocator
        self.allocator.free(@constCast(tokens));
    }

    /// Decode token IDs back to text
    pub fn decode(self: *Self, token_ids: []const i32, skip_special_tokens: bool) ?[]const u8 {
        // Convert i32 to u32
        const u32_ids = self.allocator.alloc(u32, token_ids.len) catch return null;
        defer self.allocator.free(u32_ids);

        for (token_ids, 0..) |id, i| {
            u32_ids[i] = @intCast(id);
        }

        const result = self.inner.decode(u32_ids, skip_special_tokens) catch return null;
        // Store in a persistent location - caller should copy if needed
        // Note: This returns allocator-owned memory
        return result;
    }

    /// Decode token IDs and copy result to caller-owned memory
    pub fn decodeAlloc(self: *Self, allocator: std.mem.Allocator, token_ids: []const i32, skip_special_tokens: bool) TokenizerError![]u8 {
        // Convert i32 to u32
        const u32_ids = self.allocator.alloc(u32, token_ids.len) catch {
            return TokenizerError.OutOfMemory;
        };
        defer self.allocator.free(u32_ids);

        for (token_ids, 0..) |id, i| {
            u32_ids[i] = @intCast(id);
        }

        const result = self.inner.decode(u32_ids, skip_special_tokens) catch {
            return TokenizerError.DecodingError;
        };

        // If caller is using a different allocator, copy the result
        if (allocator.ptr != self.allocator.ptr) {
            const copy = allocator.alloc(u8, result.len) catch {
                self.allocator.free(result);
                return TokenizerError.OutOfMemory;
            };
            @memcpy(copy, result);
            self.allocator.free(result);
            return copy;
        }

        return result;
    }

    /// Get the number of tokens in the given text without allocating
    /// Returns 0 on tokenization error
    pub fn tokenCount(self: *Self, text: []const u8, add_special_tokens: bool) usize {
        var encoding = self.inner.encode(text, add_special_tokens) catch {
            return 0;
        };
        defer encoding.deinit();
        return encoding.ids.len;
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) usize {
        return @constCast(&self.inner).getVocabSize();
    }

    /// Convert token ID to token string
    pub fn idToToken(self: *const Self, id: u32) ?[]const u8 {
        return @constCast(&self.inner).idToToken(id);
    }

    /// Convert token string to ID
    /// Returns i32 for API compatibility (use null for not found)
    pub fn tokenToId(self: *const Self, token: []const u8) ?i32 {
        if (@constCast(&self.inner).tokenToId(token)) |id| {
            return @intCast(id);
        }
        return null;
    }
};

test "Tokenizer struct is defined" {
    try std.testing.expect(@sizeOf(Tokenizer) > 0);
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "TokenizerError enum" {
    const errors = [_]TokenizerError{
        TokenizerError.InvalidJson,
        TokenizerError.InvalidHandle,
        TokenizerError.OutOfMemory,
        TokenizerError.FileError,
        TokenizerError.TokenizationError,
        TokenizerError.DecodingError,
    };

    try std.testing.expectEqual(@as(usize, 6), errors.len);
}

test "Tokenizer.fromFile - file not found" {
    const allocator = std.testing.allocator;

    const result = Tokenizer.fromFile(allocator, "/nonexistent/path/tokenizer.json");
    try std.testing.expectError(TokenizerError.FileError, result);
}

test "Tokenizer.fromFile - directory instead of file" {
    const allocator = std.testing.allocator;

    // Try to open a directory as if it were a file
    const result = Tokenizer.fromFile(allocator, "/tmp");
    // Should fail with FileError (IsDir)
    try std.testing.expectError(TokenizerError.FileError, result);
}

test "Tokenizer.fromJson - invalid JSON" {
    const allocator = std.testing.allocator;

    const result = Tokenizer.fromJson(allocator, "not valid json at all");
    try std.testing.expectError(TokenizerError.InvalidJson, result);
}

test "Tokenizer.fromJson - empty JSON" {
    const allocator = std.testing.allocator;

    const result = Tokenizer.fromJson(allocator, "");
    try std.testing.expectError(TokenizerError.InvalidJson, result);
}

test "Tokenizer.fromJson - valid JSON but wrong format" {
    const allocator = std.testing.allocator;

    // Valid JSON but not a tokenizer config
    const result = Tokenizer.fromJson(allocator, "{}");
    try std.testing.expectError(TokenizerError.InvalidJson, result);
}

test "Tokenizer struct has expected fields" {
    try std.testing.expect(@hasField(Tokenizer, "allocator"));
    try std.testing.expect(@hasField(Tokenizer, "inner"));
}

test "Tokenizer struct size is reasonable" {
    const size = @sizeOf(Tokenizer);
    try std.testing.expect(size > 0);
    try std.testing.expect(size < 4096);
}

// =============================================================================
// FastTokenizer - Zero-Allocation Tokenizer
// =============================================================================

/// Options for FastTokenizer initialization
pub const FastTokenizerOptions = struct {
    /// Maximum input sequence length in bytes
    max_sequence_length: u32 = 8192,
    /// Maximum output tokens per sequence
    max_tokens: u32 = 512,
};

/// A zero-allocation tokenizer for maximum throughput.
///
/// After initialization, `encode()` performs zero allocations by reusing
/// pre-allocated arena buffers. The returned encoding is valid until
/// the next `encode()` call.
///
/// Use this when:
/// - Processing many texts in a tight loop
/// - Building real-time embedding pipelines
/// - Memory allocation overhead is a bottleneck
///
/// Limitations:
/// - Encoding is invalidated on next encode() call
/// - Fixed maximum sequence/token limits set at init
/// - Not suitable if you need to keep multiple encodings alive
pub const FastTokenizer = struct {
    allocator: std.mem.Allocator,
    inner: tokenizer_lib.FastTokenizer,

    const Self = @This();

    /// Create a fast tokenizer from a JSON configuration string
    pub fn fromJson(allocator: std.mem.Allocator, json: []const u8, opts: FastTokenizerOptions) TokenizerError!Self {
        const inner = tokenizer_lib.FastTokenizer.fromJson(allocator, json, .{
            .max_sequence_length = opts.max_sequence_length,
            .max_tokens = opts.max_tokens,
        }) catch {
            return TokenizerError.InvalidJson;
        };
        return Self{
            .allocator = allocator,
            .inner = inner,
        };
    }

    /// Create a fast tokenizer from a tokenizer.json file
    pub fn fromFile(allocator: std.mem.Allocator, path: []const u8, opts: FastTokenizerOptions) TokenizerError!Self {
        const inner = tokenizer_lib.FastTokenizer.fromFile(allocator, path, .{
            .max_sequence_length = opts.max_sequence_length,
            .max_tokens = opts.max_tokens,
        }) catch |err| {
            return switch (err) {
                error.FileNotFound, error.AccessDenied, error.IsDir => TokenizerError.FileError,
                error.OutOfMemory => TokenizerError.OutOfMemory,
                else => TokenizerError.InvalidJson,
            };
        };
        return Self{
            .allocator = allocator,
            .inner = inner,
        };
    }

    /// Release the tokenizer and all pre-allocated buffers
    pub fn deinit(self: *Self) void {
        self.inner.deinit();
    }

    /// Zero-allocation encode - returns pointer to arena's encoding.
    ///
    /// IMPORTANT: The returned encoding is only valid until the next encode() call.
    /// If you need to keep the result, copy the IDs to your own buffer.
    ///
    /// Returns a SpanEncoding with:
    /// - `ids`: Token IDs as u32
    /// - `len`: Number of tokens
    /// - `getIds()`: Get slice of active IDs
    /// - `getAttentionMask()`: Get attention mask slice
    pub fn encode(self: *Self, text: []const u8) TokenizerError!*tokenizer_lib.SpanEncoding {
        return self.inner.encode(text) catch {
            return TokenizerError.TokenizationError;
        };
    }

    /// Encode and copy IDs to caller's buffer (still zero-alloc on tokenizer side)
    ///
    /// Returns the number of tokens written. If buffer is too small, returns
    /// as many tokens as fit.
    pub fn encodeInto(self: *Self, text: []const u8, out_ids: []i64) TokenizerError!usize {
        const encoding = try self.encode(text);
        const ids = encoding.getIds();
        const count = @min(ids.len, out_ids.len);

        for (0..count) |i| {
            out_ids[i] = @intCast(ids[i]);
        }

        return count;
    }

    /// Encode and copy IDs + attention mask to caller's buffers
    ///
    /// Returns the number of tokens written.
    pub fn encodeWithMaskInto(
        self: *Self,
        text: []const u8,
        out_ids: []i64,
        out_mask: []i64,
    ) TokenizerError!usize {
        const encoding = try self.encode(text);
        const ids = encoding.getIds();
        const mask = encoding.getAttentionMask();
        const count = @min(ids.len, @min(out_ids.len, out_mask.len));

        for (0..count) |i| {
            out_ids[i] = @intCast(ids[i]);
            out_mask[i] = @intCast(mask[i]);
        }

        return count;
    }

    /// Get the number of tokens for a text without copying
    ///
    /// Note: This still performs tokenization (using the arena), just doesn't
    /// copy the result. Use this for pre-flight token counting.
    pub fn tokenCount(self: *Self, text: []const u8) usize {
        const encoding = self.encode(text) catch return 0;
        return encoding.len;
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: *const Self) usize {
        return @constCast(&self.inner).getVocabSize();
    }

    /// Get memory usage of pre-allocated arena buffers
    pub fn arenaMemoryUsage(self: *const Self) usize {
        return self.inner.arenaMemoryUsage();
    }
};

// =============================================================================
// FastTokenizer Tests
// =============================================================================

test "FastTokenizer struct is defined" {
    try std.testing.expect(@sizeOf(FastTokenizer) > 0);
}

test "FastTokenizerOptions defaults" {
    const opts = FastTokenizerOptions{};
    try std.testing.expectEqual(@as(u32, 8192), opts.max_sequence_length);
    try std.testing.expectEqual(@as(u32, 512), opts.max_tokens);
}

test "FastTokenizer.fromFile - file not found" {
    const allocator = std.testing.allocator;

    const result = FastTokenizer.fromFile(allocator, "/nonexistent/path/tokenizer.json", .{});
    try std.testing.expectError(TokenizerError.FileError, result);
}

test "FastTokenizer.fromJson - invalid JSON" {
    const allocator = std.testing.allocator;

    const result = FastTokenizer.fromJson(allocator, "not valid json", .{});
    try std.testing.expectError(TokenizerError.InvalidJson, result);
}
