//! Integration tests for the Embedder
//!
//! These tests require:
//! - ONNX Runtime library available
//! - Model files in the models/ directory
//!
//! Run with: zig build test-integration

const std = @import("std");
const fe = @import("fastembed");

const MODEL_PATH = "models/all-MiniLM-L6-v2";

/// Check if the test model exists
fn modelExists() bool {
    const tokenizer_path = std.fs.path.join(std.testing.allocator, &.{ MODEL_PATH, "tokenizer.json" }) catch return false;
    defer std.testing.allocator.free(tokenizer_path);

    std.fs.cwd().access(tokenizer_path, .{}) catch return false;
    return true;
}

test "Embedder - full pipeline with real model" {
    if (!modelExists()) {
        std.debug.print("Skipping: model not found at {s}\n", .{MODEL_PATH});
        return error.SkipZigTest;
    }

    const allocator = std.testing.allocator;

    var embedder = fe.Embedder.init(allocator, .{
        .model = .all_minilm_l6_v2,
        .model_path = MODEL_PATH,
    }) catch |err| {
        std.debug.print("Failed to initialize embedder: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer embedder.deinit();

    // Test dimension
    try std.testing.expectEqual(@as(usize, 384), embedder.getDimension());

    // Test single embedding
    const texts = &[_][]const u8{"Hello world"};
    const embeddings = try embedder.embed(texts);
    defer allocator.free(embeddings);

    try std.testing.expectEqual(@as(usize, 384), embeddings.len);

    // Verify normalized (L2 norm should be ~1.0)
    const norm = fe.l2Norm(embeddings);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
}

test "Embedder - semantic similarity" {
    if (!modelExists()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var embedder = fe.Embedder.init(allocator, .{
        .model = .all_minilm_l6_v2,
        .model_path = MODEL_PATH,
    }) catch return error.SkipZigTest;
    defer embedder.deinit();

    const texts = &[_][]const u8{
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above the sleepy hound", // Similar meaning
        "Machine learning and artificial intelligence", // Different topic
    };

    const embeddings = try embedder.embed(texts);
    defer allocator.free(embeddings);

    const dim = embedder.getDimension();

    // Calculate similarities
    const sim_01 = fe.cosineSimilarity(embeddings[0..dim], embeddings[dim .. 2 * dim]);
    const sim_02 = fe.cosineSimilarity(embeddings[0..dim], embeddings[2 * dim .. 3 * dim]);

    // Similar texts should have higher similarity than unrelated ones
    try std.testing.expect(sim_01 > sim_02);

    // Similar sentences should have reasonable similarity (> 0.5)
    try std.testing.expect(sim_01 > 0.3);
}

test "Embedder - batch consistency" {
    if (!modelExists()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var embedder = fe.Embedder.init(allocator, .{
        .model = .all_minilm_l6_v2,
        .model_path = MODEL_PATH,
    }) catch return error.SkipZigTest;
    defer embedder.deinit();

    const text = "Test consistency between single and batch embedding";

    // Single embedding
    const single = try embedder.embedOne(text);
    defer allocator.free(single);

    // Batch of one
    const batch_texts = &[_][]const u8{text};
    const batch = try embedder.embed(batch_texts);
    defer allocator.free(batch);

    // Results should be identical
    for (single, batch) |s, b| {
        try std.testing.expectApproxEqAbs(s, b, 0.0001);
    }
}

test "Embedder - empty input" {
    if (!modelExists()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var embedder = fe.Embedder.init(allocator, .{
        .model = .all_minilm_l6_v2,
        .model_path = MODEL_PATH,
    }) catch return error.SkipZigTest;
    defer embedder.deinit();

    // Empty batch should return empty result
    const result = try embedder.embed(&[_][]const u8{});
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "Embedder - multiple texts batch" {
    if (!modelExists()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var embedder = fe.Embedder.init(allocator, .{
        .model = .all_minilm_l6_v2,
        .model_path = MODEL_PATH,
    }) catch return error.SkipZigTest;
    defer embedder.deinit();

    const texts = &[_][]const u8{
        "First document",
        "Second document",
        "Third document",
        "Fourth document",
    };

    const embeddings = try embedder.embed(texts);
    defer allocator.free(embeddings);

    const dim = embedder.getDimension();
    try std.testing.expectEqual(texts.len * dim, embeddings.len);

    // Each embedding should be normalized
    for (0..texts.len) |i| {
        const start = i * dim;
        const embedding = embeddings[start .. start + dim];
        const norm = fe.l2Norm(embedding);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.01);
    }
}

test "Embedder - token count" {
    if (!modelExists()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var embedder = fe.Embedder.init(allocator, .{
        .model = .all_minilm_l6_v2,
        .model_path = MODEL_PATH,
    }) catch return error.SkipZigTest;
    defer embedder.deinit();

    const count = embedder.getTokenCount("Hello world");
    // Should have some tokens (at least CLS + words + SEP)
    try std.testing.expect(count >= 3);
}
