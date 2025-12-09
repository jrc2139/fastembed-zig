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
        const batch_hs_base = b * seq_len * hidden_dim;
        const mask_slice = attention_mask[b * seq_len ..][0..seq_len];
        const dst = result[b * hidden_dim ..][0..hidden_dim];

        // Initialize to zero
        @memset(dst, 0);

        // Count valid tokens first (simpler branch prediction)
        var token_count: f32 = 0;
        for (mask_slice) |m| {
            if (m != 0) token_count += 1;
        }

        // Sum hidden states for valid tokens using contiguous slice access
        for (mask_slice, 0..) |m, s| {
            if (m != 0) {
                // Get contiguous hidden state slice for this token
                const src = hidden_states[batch_hs_base + s * hidden_dim ..][0..hidden_dim];
                // Add src to dst element-wise (cache-friendly contiguous access)
                for (dst, src) |*d, val| {
                    d.* += val;
                }
            }
        }

        // Normalize by token count (avoid division by zero)
        if (token_count > 0) {
            const scale = 1.0 / token_count;
            for (dst) |*d| {
                d.* *= scale;
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

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "CLS pooling - multiple batches" {
    const allocator = std.testing.allocator;

    // [2, 3, 4] - 2 batches, 3 tokens, 4 dims
    const hidden_states = [_]f32{
        // Batch 0
        1.0, 2.0, 3.0, 4.0, // CLS token
        5.0, 6.0, 7.0, 8.0, // Token 1
        9.0, 10.0, 11.0, 12.0, // Token 2
        // Batch 1
        13.0, 14.0, 15.0, 16.0, // CLS token
        17.0, 18.0, 19.0, 20.0, // Token 1
        21.0, 22.0, 23.0, 24.0, // Token 2
    };

    const result = try poolCls(&hidden_states, 2, 3, 4, allocator);
    defer allocator.free(result);

    // Batch 0 CLS
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, result[0..4]);
    // Batch 1 CLS
    try std.testing.expectEqualSlices(f32, &[_]f32{ 13.0, 14.0, 15.0, 16.0 }, result[4..8]);
}

test "CLS pooling - single token sequence" {
    const allocator = std.testing.allocator;

    // [1, 1, 3] - 1 batch, 1 token, 3 dims
    const hidden_states = [_]f32{ 1.0, 2.0, 3.0 };

    const result = try poolCls(&hidden_states, 1, 1, 3, allocator);
    defer allocator.free(result);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0 }, result);
}

test "CLS pooling - high dimensional" {
    const allocator = std.testing.allocator;

    const hidden_dim = 384; // Like BGE
    const seq_len = 128;

    // Create test data
    var hidden_states: [hidden_dim * seq_len]f32 = undefined;
    for (&hidden_states, 0..) |*v, i| {
        v.* = @floatFromInt(i % hidden_dim);
    }

    const result = try poolCls(&hidden_states, 1, seq_len, hidden_dim, allocator);
    defer allocator.free(result);

    // CLS should be first hidden_dim values
    try std.testing.expectEqual(@as(usize, hidden_dim), result.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[1], 0.001);
}

test "Mean pooling - all tokens valid" {
    const allocator = std.testing.allocator;

    // [1, 3, 2] - all tokens valid
    const hidden_states = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    const attention_mask = [_]i64{ 1, 1, 1 };

    const result = try poolMean(&hidden_states, &attention_mask, 1, 3, 2, allocator);
    defer allocator.free(result);

    // Mean: (1+3+5)/3=3, (2+4+6)/3=4
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 0.001);
}

test "Mean pooling - all tokens masked" {
    const allocator = std.testing.allocator;

    // [1, 3, 2] - all tokens masked (edge case)
    const hidden_states = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    const attention_mask = [_]i64{ 0, 0, 0 };

    const result = try poolMean(&hidden_states, &attention_mask, 1, 3, 2, allocator);
    defer allocator.free(result);

    // All zeros when mask is empty
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.001);
}

test "Mean pooling - multiple batches different mask lengths" {
    const allocator = std.testing.allocator;

    // [2, 4, 2] - 2 batches, 4 tokens, 2 dims
    const hidden_states = [_]f32{
        // Batch 0: 3 valid tokens
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0,  8.0, // padding
        // Batch 1: 2 valid tokens
        10.0, 20.0,
        30.0, 40.0,
        50.0, 60.0, // padding
        70.0, 80.0, // padding
    };
    const attention_mask = [_]i64{
        1, 1, 1, 0, // Batch 0
        1, 1, 0, 0, // Batch 1
    };

    const result = try poolMean(&hidden_states, &attention_mask, 2, 4, 2, allocator);
    defer allocator.free(result);

    // Batch 0: mean of (1+3+5)/3=3, (2+4+6)/3=4
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 0.001);

    // Batch 1: mean of (10+30)/2=20, (20+40)/2=30
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), result[3], 0.001);
}

test "Mean pooling - only first token valid (like CLS)" {
    const allocator = std.testing.allocator;

    // [1, 3, 2]
    const hidden_states = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    const attention_mask = [_]i64{ 1, 0, 0 };

    const result = try poolMean(&hidden_states, &attention_mask, 1, 3, 2, allocator);
    defer allocator.free(result);

    // Only first token, so mean equals first token
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 0.001);
}

test "Mean pooling - sparse mask (non-contiguous)" {
    const allocator = std.testing.allocator;

    // [1, 4, 2] - tokens 0 and 2 are valid, 1 and 3 are padding
    const hidden_states = [_]f32{
        1.0, 2.0, // valid
        3.0, 4.0, // padding
        5.0, 6.0, // valid
        7.0, 8.0, // padding
    };
    const attention_mask = [_]i64{ 1, 0, 1, 0 };

    const result = try poolMean(&hidden_states, &attention_mask, 1, 4, 2, allocator);
    defer allocator.free(result);

    // Mean of tokens 0 and 2: (1+5)/2=3, (2+6)/2=4
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 0.001);
}

test "Mean pooling - high dimensional" {
    const allocator = std.testing.allocator;

    const hidden_dim = 768; // Like BERT
    const seq_len = 128;

    // Create test data: all ones
    var hidden_states: [hidden_dim * seq_len]f32 = undefined;
    for (&hidden_states) |*v| {
        v.* = 1.0;
    }

    // Only first 64 tokens are valid
    var attention_mask: [seq_len]i64 = undefined;
    for (&attention_mask, 0..) |*m, i| {
        m.* = if (i < 64) 1 else 0;
    }

    const result = try poolMean(&hidden_states, &attention_mask, 1, seq_len, hidden_dim, allocator);
    defer allocator.free(result);

    // All inputs are 1, so mean should be 1
    for (result) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v, 0.001);
    }
}

test "Mean pooling - negative values" {
    const allocator = std.testing.allocator;

    const hidden_states = [_]f32{
        -1.0, 2.0,
        3.0,  -4.0,
        -5.0, 6.0,
    };
    const attention_mask = [_]i64{ 1, 1, 1 };

    const result = try poolMean(&hidden_states, &attention_mask, 1, 3, 2, allocator);
    defer allocator.free(result);

    // Mean: (-1+3-5)/3=-1, (2-4+6)/3=4/3≈1.333
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.333), result[1], 0.01);
}

test "CLS vs Mean pooling - different results" {
    const allocator = std.testing.allocator;

    const hidden_states = [_]f32{
        1.0, 2.0, // CLS
        3.0, 4.0, // Token 1
        5.0, 6.0, // Token 2
    };
    const attention_mask = [_]i64{ 1, 1, 1 };

    const cls_result = try poolCls(&hidden_states, 1, 3, 2, allocator);
    defer allocator.free(cls_result);

    const mean_result = try poolMean(&hidden_states, &attention_mask, 1, 3, 2, allocator);
    defer allocator.free(mean_result);

    // CLS should be [1, 2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cls_result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), cls_result[1], 0.001);

    // Mean should be [3, 4]
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), mean_result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), mean_result[1], 0.001);
}
