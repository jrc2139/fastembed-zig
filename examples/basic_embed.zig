//! Basic embedding example
//!
//! This example demonstrates how to:
//! 1. Load an embedding model
//! 2. Generate embeddings for text
//! 3. Compute similarity between embeddings
//!
//! Usage: zig build run -- <model_directory> [model_type]
//!
//! Model types: bge, minilm, e5, gemma, gemma-q4

const std = @import("std");
const fe = @import("fastembed");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get model path from command line
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_directory> [model_type]\n", .{args[0]});
        std.debug.print("\nModel types:\n", .{});
        std.debug.print("  bge       - BGE small English (default)\n", .{});
        std.debug.print("  minilm    - all-MiniLM-L6-v2\n", .{});
        std.debug.print("  e5        - Multilingual E5 large\n", .{});
        std.debug.print("  gemma     - EmbeddingGemma 300M\n", .{});
        std.debug.print("  gemma-q4  - EmbeddingGemma 300M (Q4 quantized)\n", .{});
        std.debug.print("  gemma-fp16 - EmbeddingGemma 300M (FP16)\n", .{});
        std.debug.print("  granite    - Granite Embedding English R2 (Q4F16)\n", .{});
        std.debug.print("  granite-fp16 - Granite Embedding English R2 FP16 (CoreML)\n", .{});
        std.debug.print("\nExamples:\n", .{});
        std.debug.print("  {s} models/bge-small-en-v1.5\n", .{args[0]});
        std.debug.print("  {s} models/embeddinggemma-300m gemma\n", .{args[0]});
        return;
    }

    const model_path = args[1];

    // Parse model type
    const model_type: fe.Model = if (args.len > 2) blk: {
        const type_str = args[2];
        if (std.mem.eql(u8, type_str, "bge")) {
            break :blk .bge_small_en_v1_5;
        } else if (std.mem.eql(u8, type_str, "minilm")) {
            break :blk .all_minilm_l6_v2;
        } else if (std.mem.eql(u8, type_str, "e5")) {
            break :blk .multilingual_e5_large;
        } else if (std.mem.eql(u8, type_str, "gemma")) {
            break :blk .embedding_gemma_300m;
        } else if (std.mem.eql(u8, type_str, "gemma-q4")) {
            break :blk .embedding_gemma_300m_q4;
        } else if (std.mem.eql(u8, type_str, "gemma-fp16")) {
            break :blk .embedding_gemma_300m_fp16;
        } else if (std.mem.eql(u8, type_str, "granite")) {
            break :blk .granite_embedding_english_r2;
        } else if (std.mem.eql(u8, type_str, "granite-fp16")) {
            break :blk .granite_embedding_english_r2_fp16;
        } else {
            std.debug.print("Unknown model type: {s}\n", .{type_str});
            return;
        }
    } else .bge_small_en_v1_5;

    std.debug.print("Loading model from: {s}\n", .{model_path});
    std.debug.print("Model type: {s}\n", .{model_type.getConfig().name});

    // Initialize embedder
    var embedder = fe.Embedder.init(allocator, .{
        .model = model_type,
        .model_path = model_path,
    }) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        std.debug.print("\nMake sure the directory contains the model files and tokenizer.json\n", .{});
        return;
    };
    defer embedder.deinit();

    std.debug.print("Model loaded! Dimension: {d}\n", .{embedder.getDimension()});

    // Sample texts
    const texts = [_][]const u8{
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above a sleepy canine",
        "Machine learning is a subset of artificial intelligence",
        "The weather is nice today",
    };

    std.debug.print("\nGenerating embeddings for {d} texts...\n", .{texts.len});

    // Generate embeddings
    const embeddings = embedder.embed(&texts) catch |err| {
        std.debug.print("Failed to generate embeddings: {}\n", .{err});
        return;
    };
    defer allocator.free(embeddings);

    const dim = embedder.getDimension();

    // Print first few values of each embedding
    std.debug.print("\nEmbeddings (first 5 values):\n", .{});
    for (0..texts.len) |i| {
        const vec = embeddings[i * dim .. (i + 1) * dim];
        std.debug.print("  [{d}] \"{s}\"\n", .{ i, texts[i][0..@min(40, texts[i].len)] });
        std.debug.print("      [{d:.4}, {d:.4}, {d:.4}, {d:.4}, {d:.4}, ...]\n", .{
            vec[0], vec[1], vec[2], vec[3], vec[4],
        });
    }

    // Compute pairwise similarities
    std.debug.print("\nCosine Similarities:\n", .{});
    for (0..texts.len) |i| {
        for (i + 1..texts.len) |j| {
            const vec_i = embeddings[i * dim .. (i + 1) * dim];
            const vec_j = embeddings[j * dim .. (j + 1) * dim];
            const sim = fe.cosineSimilarity(vec_i, vec_j);
            std.debug.print("  [{d}] vs [{d}]: {d:.4}\n", .{ i, j, sim });
        }
    }

    std.debug.print("\nNote: Similar texts should have higher similarity scores.\n", .{});
    std.debug.print("Texts [0] and [1] should be most similar (both about foxes).\n", .{});
}
