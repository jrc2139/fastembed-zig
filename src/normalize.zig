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
