//! Pooling strategies for transformer embeddings
//!
//! Converts sequence of hidden states to a single embedding vector.

const std = @import("std");

/// Pooling strategy
pub const PoolingStrategy = enum {
    /// Use the [CLS] token embedding (first token)
    cls,
    /// Mean pooling over all tokens (weighted by attention mask)
    mean,
};

/// CLS pooling: extract the first token's hidden state
///
/// Parameters:
/// - hidden_states: [batch_size, seq_len, hidden_dim] flattened
/// - batch_size: number of sequences
/// - seq_len: sequence length
/// - hidden_dim: dimension of hidden state
/// - allocator: for result allocation
///
/// Returns: [batch_size, hidden_dim] flattened
pub fn poolCls(
    hidden_states: []const f32,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    allocator: std.mem.Allocator,
) ![]f32 {
    const result = try allocator.alloc(f32, batch_size * hidden_dim);
    errdefer allocator.free(result);

    for (0..batch_size) |b| {
        const batch_offset = b * seq_len * hidden_dim;
        // First token (CLS) is at offset 0 within each sequence
        const src = hidden_states[batch_offset .. batch_offset + hidden_dim];
        const dst = result[b * hidden_dim .. (b + 1) * hidden_dim];
        @memcpy(dst, src);
    }

    return result;
}

/// Mean pooling: average hidden states weighted by attention mask
///
/// Parameters:
/// - hidden_states: [batch_size, seq_len, hidden_dim] flattened
/// - attention_mask: [batch_size, seq_len] flattened (1 for real tokens, 0 for padding)
/// - batch_size: number of sequences
/// - seq_len: sequence length
/// - hidden_dim: dimension of hidden state
/// - allocator: for result allocation
///
/// Returns: [batch_size, hidden_dim] flattened
pub fn poolMean(
    hidden_states: []const f32,
    attention_mask: []const i64,
    batch_size: usize,
    seq_len: usize,
    hidden_dim: usize,
    allocator: std.mem.Allocator,
) ![]f32 {
    const result = try allocator.alloc(f32, batch_size * hidden_dim);
    errdefer allocator.free(result);

    for (0..batch_size) |b| {
        const batch_hs_offset = b * seq_len * hidden_dim;
        const batch_mask_offset = b * seq_len;
        const dst = result[b * hidden_dim .. (b + 1) * hidden_dim];

        // Initialize to zero
        @memset(dst, 0);

        // Sum hidden states weighted by attention mask
        var mask_sum: f32 = 0;
        for (0..seq_len) |s| {
            const mask_val: f32 = @floatFromInt(attention_mask[batch_mask_offset + s]);
            if (mask_val > 0) {
                mask_sum += mask_val;
                const token_offset = batch_hs_offset + s * hidden_dim;
                for (0..hidden_dim) |d| {
                    dst[d] += hidden_states[token_offset + d] * mask_val;
                }
            }
        }

        // Normalize by sum of mask (avoid division by zero)
        if (mask_sum > 0) {
            for (0..hidden_dim) |d| {
                dst[d] /= mask_sum;
            }
        }
    }

    return result;
}

test "CLS pooling" {
    const allocator = std.testing.allocator;

    // [1, 3, 4] - 1 batch, 3 tokens, 4 dims
    const hidden_states = [_]f32{
        // Token 0 (CLS)
        1.0, 2.0,  3.0,  4.0,
        // Token 1
        5.0, 6.0,  7.0,  8.0,
        // Token 2
        9.0, 10.0, 11.0, 12.0,
    };

    const result = try poolCls(&hidden_states, 1, 3, 4, allocator);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, result);
}

test "Mean pooling" {
    const allocator = std.testing.allocator;

    // [1, 3, 2] - 1 batch, 3 tokens, 2 dims
    const hidden_states = [_]f32{
        1.0, 2.0, // Token 0
        3.0, 4.0, // Token 1
        5.0, 6.0, // Token 2 (padding)
    };

    const attention_mask = [_]i64{ 1, 1, 0 }; // Only first 2 tokens are real

    const result = try poolMean(&hidden_states, &attention_mask, 1, 3, 2, allocator);
    defer allocator.free(result);

    // Mean of first 2 tokens: (1+3)/2=2, (2+4)/2=3
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[1], 0.001);
}
