//! High-level Tokenizer wrapper
//!
//! Provides an idiomatic Zig interface to HuggingFace tokenizers.
//! Now powered by pure Zig tokenizer-zig library for maximum performance.

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
