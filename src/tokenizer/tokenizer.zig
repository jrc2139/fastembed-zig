//! High-level Tokenizer wrapper
//!
//! Provides an idiomatic Zig interface to HuggingFace tokenizers.

const std = @import("std");
const c_api = @import("c_api.zig");

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
};

/// A HuggingFace-compatible tokenizer
pub const Tokenizer = struct {
    handle: c_api.TokenizerHandle,

    const Self = @This();

    /// Create a tokenizer from a JSON configuration string
    pub fn fromJson(json: []const u8) TokenizerError!Self {
        const handle = c_api.newFromStr(json);
        if (handle == null) {
            return TokenizerError.InvalidJson;
        }
        return Self{ .handle = handle.? };
    }

    /// Create a tokenizer from a tokenizer.json file
    pub fn fromFile(allocator: std.mem.Allocator, path: []const u8) TokenizerError!Self {
        const file = std.fs.cwd().openFile(path, .{}) catch {
            return TokenizerError.FileError;
        };
        defer file.close();

        const json = file.readToEndAlloc(allocator, 100 * 1024 * 1024) catch {
            return TokenizerError.OutOfMemory;
        };
        defer allocator.free(json);

        return fromJson(json);
    }

    /// Release the tokenizer
    pub fn deinit(self: *Self) void {
        c_api.free(self.handle);
        self.handle = undefined;
    }

    /// Encode text to token IDs
    ///
    /// Returns a slice of token IDs. Caller must call `freeTokens` when done.
    pub fn encode(self: Self, text: []const u8, add_special_tokens: bool) []const i32 {
        var result: c_api.TokenizerEncodeResult = undefined;
        c_api.encode(self.handle, text, add_special_tokens, &result);

        if (result.token_ids == null or result.len == 0) {
            return &[_]i32{};
        }

        // Cast c_int* to i32 slice
        const ptr: [*]const i32 = @ptrCast(result.token_ids);
        return ptr[0..result.len];
    }

    /// Free tokens returned by encode()
    pub fn freeTokens(self: Self, tokens: []const i32) void {
        _ = self;
        if (tokens.len == 0) return;

        var result = c_api.TokenizerEncodeResult{
            .token_ids = @ptrCast(@constCast(tokens.ptr)),
            .len = tokens.len,
        };
        c_api.freeEncodeResults(&result, 1);
    }

    /// Encode text and copy results to caller-owned memory
    pub fn encodeAlloc(self: Self, allocator: std.mem.Allocator, text: []const u8, add_special_tokens: bool) TokenizerError![]i32 {
        const tokens = self.encode(text, add_special_tokens);
        defer self.freeTokens(tokens);

        const copy = allocator.alloc(i32, tokens.len) catch {
            return TokenizerError.OutOfMemory;
        };
        @memcpy(copy, tokens);
        return copy;
    }

    /// Decode token IDs back to text
    pub fn decode(self: Self, token_ids: []const i32, skip_special_tokens: bool) ?[]const u8 {
        // Convert i32 slice to u32 slice for C API
        const u32_ptr: [*]const u32 = @ptrCast(token_ids.ptr);
        c_api.decode(self.handle, u32_ptr[0..token_ids.len], skip_special_tokens);
        return c_api.getDecodeStr(self.handle);
    }

    /// Decode token IDs and copy result to caller-owned memory
    pub fn decodeAlloc(self: Self, allocator: std.mem.Allocator, token_ids: []const i32, skip_special_tokens: bool) TokenizerError![]u8 {
        const text = self.decode(token_ids, skip_special_tokens) orelse return TokenizerError.InvalidHandle;

        const copy = allocator.alloc(u8, text.len) catch {
            return TokenizerError.OutOfMemory;
        };
        @memcpy(copy, text);
        return copy;
    }

    /// Get vocabulary size
    pub fn getVocabSize(self: Self) usize {
        return c_api.getVocabSize(self.handle);
    }

    /// Convert token ID to token string
    pub fn idToToken(self: Self, id: u32) ?[]const u8 {
        return c_api.idToToken(self.handle, id);
    }

    /// Convert token string to ID
    pub fn tokenToId(self: Self, token: []const u8) ?i32 {
        return c_api.tokenToId(self.handle, token);
    }
};

test "Tokenizer struct is defined" {
    try std.testing.expect(@sizeOf(Tokenizer) > 0);
}
