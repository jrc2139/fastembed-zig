//! Raw C bindings for HuggingFace Tokenizers
//!
//! This module provides direct access to the tokenizers-cpp C API via @cImport.
//! For a higher-level idiomatic Zig API, use the `tokenizer` module.

pub const c = @cImport({
    @cInclude("tokenizers_c.h");
});

// Re-export types for convenience
pub const TokenizerHandle = c.TokenizerHandle;
pub const TokenizerEncodeResult = c.TokenizerEncodeResult;

/// Create a new tokenizer from a JSON configuration string (tokenizer.json contents)
pub fn newFromStr(json: []const u8) ?TokenizerHandle {
    return c.tokenizers_new_from_str(json.ptr, json.len);
}

/// Create a new BPE tokenizer from vocab, merges, and added tokens
pub fn byteLevelBpeNewFromStr(
    vocab: []const u8,
    merges: []const u8,
    added_tokens: []const u8,
) ?TokenizerHandle {
    return c.byte_level_bpe_tokenizers_new_from_str(
        vocab.ptr,
        vocab.len,
        merges.ptr,
        merges.len,
        added_tokens.ptr,
        added_tokens.len,
    );
}

/// Encode a text string to token IDs
pub fn encode(
    handle: TokenizerHandle,
    text: []const u8,
    add_special_tokens: bool,
    result: *TokenizerEncodeResult,
) void {
    c.tokenizers_encode(
        handle,
        text.ptr,
        text.len,
        if (add_special_tokens) 1 else 0,
        result,
    );
}

/// Encode multiple text strings to token IDs
pub fn encodeBatch(
    handle: TokenizerHandle,
    texts: []const []const u8,
    add_special_tokens: bool,
    results: []TokenizerEncodeResult,
) void {
    // Note: This is a simplified version - full implementation would need allocator
    _ = texts;
    _ = handle;
    _ = add_special_tokens;
    _ = results;
    @panic("encodeBatch not yet implemented - use encode in a loop");
}

/// Free encode results
pub fn freeEncodeResults(results: *TokenizerEncodeResult, num_seqs: usize) void {
    c.tokenizers_free_encode_results(results, num_seqs);
}

/// Decode token IDs back to text (stores result internally)
pub fn decode(handle: TokenizerHandle, token_ids: []const u32, skip_special_tokens: bool) void {
    c.tokenizers_decode(
        handle,
        token_ids.ptr,
        token_ids.len,
        if (skip_special_tokens) 1 else 0,
    );
}

/// Get the decoded string from the last decode() call
pub fn getDecodeStr(handle: TokenizerHandle) ?[]const u8 {
    var data: [*c]const u8 = undefined;
    var len: usize = undefined;
    c.tokenizers_get_decode_str(handle, &data, &len);
    if (data == null or len == 0) {
        return null;
    }
    return data[0..len];
}

/// Get vocabulary size
pub fn getVocabSize(handle: TokenizerHandle) usize {
    var size: usize = undefined;
    c.tokenizers_get_vocab_size(handle, &size);
    return size;
}

/// Get token string from ID
pub fn idToToken(handle: TokenizerHandle, id: u32) ?[]const u8 {
    var data: [*c]const u8 = undefined;
    var len: usize = undefined;
    c.tokenizers_id_to_token(handle, id, &data, &len);
    if (data == null or len == 0) {
        return null;
    }
    return data[0..len];
}

/// Get token ID from string, returns null if not in vocabulary
pub fn tokenToId(handle: TokenizerHandle, token: []const u8) ?i32 {
    var id: i32 = undefined;
    c.tokenizers_token_to_id(handle, token.ptr, token.len, &id);
    if (id < 0) {
        return null;
    }
    return id;
}

/// Free a tokenizer handle
pub fn free(handle: TokenizerHandle) void {
    c.tokenizers_free(handle);
}

test "c_api types are defined" {
    const std = @import("std");
    // Just check that types exist
    try std.testing.expect(@sizeOf(TokenizerHandle) > 0);
    try std.testing.expect(@sizeOf(TokenizerEncodeResult) > 0);
}
