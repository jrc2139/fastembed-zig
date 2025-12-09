//! Vector normalization utilities
//!
//! L2 normalization for embedding vectors.

const std = @import("std");

/// L2 normalize a vector in-place
///
/// Converts vector to unit length: v = v / ||v||
pub fn l2Normalize(vector: []f32) void {
    var norm_sq: f32 = 0;
    for (vector) |v| {
        norm_sq += v * v;
    }

    if (norm_sq > 0) {
        const norm = @sqrt(norm_sq);
        for (vector) |*v| {
            v.* /= norm;
        }
    }
}

/// L2 normalize multiple vectors in-place
///
/// Parameters:
/// - vectors: [num_vectors * dim] flattened array
/// - num_vectors: number of vectors
/// - dim: dimension of each vector
pub fn l2NormalizeBatch(vectors: []f32, num_vectors: usize, dim: usize) void {
    for (0..num_vectors) |i| {
        const start = i * dim;
        const end = start + dim;
        l2Normalize(vectors[start..end]);
    }
}

/// Compute L2 norm of a vector
pub fn l2Norm(vector: []const f32) f32 {
    var norm_sq: f32 = 0;
    for (vector) |v| {
        norm_sq += v * v;
    }
    return @sqrt(norm_sq);
}

/// Compute cosine similarity between two vectors
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var dot: f32 = 0;
    var norm_a_sq: f32 = 0;
    var norm_b_sq: f32 = 0;

    for (a, b) |va, vb| {
        dot += va * vb;
        norm_a_sq += va * va;
        norm_b_sq += vb * vb;
    }

    const norm_a = @sqrt(norm_a_sq);
    const norm_b = @sqrt(norm_b_sq);

    if (norm_a == 0 or norm_b == 0) {
        return 0;
    }

    return dot / (norm_a * norm_b);
}

/// Compute dot product (for pre-normalized vectors, this equals cosine similarity)
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var dot: f32 = 0;
    for (a, b) |va, vb| {
        dot += va * vb;
    }
    return dot;
}

test "L2 normalize" {
    var vec = [_]f32{ 3.0, 4.0 };
    l2Normalize(&vec);

    // 3-4-5 triangle: normalized should be [0.6, 0.8]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), vec[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vec[1], 0.001);

    // Norm should be 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l2Norm(&vec), 0.001);
}

test "L2 normalize batch" {
    var vecs = [_]f32{
        3.0, 4.0, // vec 0
        0.0, 5.0, // vec 1
    };
    l2NormalizeBatch(&vecs, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f32, 0.6), vecs[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vecs[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vecs[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vecs[3], 0.001);
}

test "Cosine similarity" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0 };
    const c = [_]f32{ 1.0, 0.0 };

    // Orthogonal vectors
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &b), 0.001);

    // Identical vectors
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &c), 0.001);
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "L2 normalize - zero vector stays zero" {
    var vec = [_]f32{ 0.0, 0.0, 0.0 };
    l2Normalize(&vec);

    // Zero vector should remain zero (no division by zero)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vec[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vec[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vec[2], 0.0001);
}

test "L2 normalize - negative values" {
    var vec = [_]f32{ -3.0, 4.0 };
    l2Normalize(&vec);

    // Should normalize correctly with negative values
    try std.testing.expectApproxEqAbs(@as(f32, -0.6), vec[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vec[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l2Norm(&vec), 0.001);
}

test "L2 normalize - all negative values" {
    var vec = [_]f32{ -3.0, -4.0 };
    l2Normalize(&vec);

    try std.testing.expectApproxEqAbs(@as(f32, -0.6), vec[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.8), vec[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l2Norm(&vec), 0.001);
}

test "L2 normalize - single element" {
    var vec = [_]f32{5.0};
    l2Normalize(&vec);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vec[0], 0.001);
}

test "L2 normalize - already normalized" {
    var vec = [_]f32{ 0.6, 0.8 };
    l2Normalize(&vec);

    // Should remain unchanged (within tolerance)
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), vec[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vec[1], 0.001);
}

test "L2 normalize - very small values" {
    var vec = [_]f32{ 1e-10, 1e-10 };
    l2Normalize(&vec);

    // Should normalize even very small values
    const norm = l2Norm(&vec);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
}

test "L2 normalize - high dimensional vector" {
    var vec: [128]f32 = undefined;
    for (&vec, 0..) |*v, i| {
        v.* = @floatFromInt(i + 1);
    }
    l2Normalize(&vec);

    // Norm should be 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l2Norm(&vec), 0.001);
}

test "L2 normalize batch - mixed vectors" {
    var vecs = [_]f32{
        3.0, 4.0, // vec 0: norm = 5
        0.0, 0.0, // vec 1: zero vector
        -3.0, 4.0, // vec 2: with negatives
        1.0, 0.0, // vec 3: already unit length
    };
    l2NormalizeBatch(&vecs, 4, 2);

    // vec 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), vecs[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vecs[1], 0.001);

    // vec 1: zero stays zero
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vecs[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vecs[3], 0.001);

    // vec 2
    try std.testing.expectApproxEqAbs(@as(f32, -0.6), vecs[4], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), vecs[5], 0.001);

    // vec 3
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vecs[6], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vecs[7], 0.001);
}

test "L2 norm - pythagorean triples" {
    // 3-4-5 triangle
    const v1 = [_]f32{ 3.0, 4.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), l2Norm(&v1), 0.001);

    // 5-12-13 triangle
    const v2 = [_]f32{ 5.0, 12.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), l2Norm(&v2), 0.001);

    // 8-15-17 triangle
    const v3 = [_]f32{ 8.0, 15.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 17.0), l2Norm(&v3), 0.001);
}

test "L2 norm - 3D vectors" {
    // sqrt(1^2 + 2^2 + 2^2) = sqrt(9) = 3
    const v = [_]f32{ 1.0, 2.0, 2.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), l2Norm(&v), 0.001);
}

test "Cosine similarity - parallel vectors (same direction)" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 2.0, 4.0, 6.0 }; // Same direction, different magnitude

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.001);
}

test "Cosine similarity - anti-parallel vectors (opposite direction)" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ -1.0, -2.0, -3.0 };

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), cosineSimilarity(&a, &b), 0.001);
}

test "Cosine similarity - zero vector returns 0" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const zero = [_]f32{ 0.0, 0.0, 0.0 };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &zero), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&zero, &a), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&zero, &zero), 0.001);
}

test "Cosine similarity - 45 degree angle" {
    const a = [_]f32{ 1.0, 0.0 };
    const b = [_]f32{ 1.0, 1.0 };

    // cos(45°) = 1/sqrt(2) ≈ 0.7071
    try std.testing.expectApproxEqAbs(@as(f32, 0.7071), cosineSimilarity(&a, &b), 0.001);
}

test "Cosine similarity - normalized vs unnormalized gives same result" {
    var a = [_]f32{ 3.0, 4.0 };
    var b = [_]f32{ 1.0, 2.0 };

    const sim_unnorm = cosineSimilarity(&a, &b);

    l2Normalize(&a);
    l2Normalize(&b);
    const sim_norm = cosineSimilarity(&a, &b);

    try std.testing.expectApproxEqAbs(sim_unnorm, sim_norm, 0.001);
}

test "Dot product - basic computation" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dotProduct(&a, &b), 0.001);
}

test "Dot product - orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0 };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dotProduct(&a, &b), 0.001);
}

test "Dot product - equals cosine similarity for normalized vectors" {
    var a = [_]f32{ 3.0, 4.0 };
    var b = [_]f32{ 1.0, 2.0 };

    l2Normalize(&a);
    l2Normalize(&b);

    const dot = dotProduct(&a, &b);
    const cosine = cosineSimilarity(&a, &b);

    try std.testing.expectApproxEqAbs(dot, cosine, 0.001);
}

test "Dot product - negative values" {
    const a = [_]f32{ -1.0, 2.0, -3.0 };
    const b = [_]f32{ 4.0, -5.0, 6.0 };

    // -1*4 + 2*(-5) + (-3)*6 = -4 - 10 - 18 = -32
    try std.testing.expectApproxEqAbs(@as(f32, -32.0), dotProduct(&a, &b), 0.001);
}

test "Dot product - self dot product equals squared norm" {
    const v = [_]f32{ 3.0, 4.0 };

    // v · v = ||v||^2
    const dot = dotProduct(&v, &v);
    const norm = l2Norm(&v);

    try std.testing.expectApproxEqAbs(dot, norm * norm, 0.001);
}
