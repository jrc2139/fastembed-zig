//! Throughput benchmark for fastembed-zig
//!
//! Compares against fastembed-go benchmark results
//!
//! Usage: zig build benchmark -- <model_directory> [model_type]

const std = @import("std");
const fe = @import("fastembed");

// Long texts for throughput testing (same as Go benchmark)
const longTexts = [_][]const u8{
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
    "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision and natural language processing.",
    "Natural language processing (NLP) is an interdisciplinary subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The result is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.",
    "Transformers are a type of neural network architecture that has become the foundation for many state-of-the-art natural language processing models. Unlike recurrent neural networks, transformers process all input tokens simultaneously using self-attention mechanisms, allowing them to capture long-range dependencies more effectively and enabling parallel computation during training.",
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_directory> [model_type]\n", .{args[0]});
        std.debug.print("\nModel types: granite (default), bge, minilm\n", .{});
        return;
    }

    const model_path = args[1];
    const model_type: fe.Model = if (args.len > 2) blk: {
        const type_str = args[2];
        if (std.mem.eql(u8, type_str, "granite")) {
            break :blk .granite_embedding_english_r2;
        } else if (std.mem.eql(u8, type_str, "bge")) {
            break :blk .bge_small_en_v1_5;
        } else if (std.mem.eql(u8, type_str, "minilm")) {
            break :blk .all_minilm_l6_v2;
        } else {
            break :blk .granite_embedding_english_r2;
        }
    } else .granite_embedding_english_r2;

    std.debug.print("Loading model: {s}\n", .{model_type.getConfig().name});
    std.debug.print("Path: {s}\n", .{model_path});

    var embedder = fe.Embedder.init(allocator, .{
        .model = model_type,
        .model_path = model_path,
    }) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return;
    };
    defer embedder.deinit();

    std.debug.print("Model loaded! Dimension: {d}\n\n", .{embedder.getDimension()});

    // Warmup
    _ = embedder.embedOne("warmup") catch {};

    // Test different batch sizes
    const batchSizes = [_]usize{ 1, 5, 10, 20, 50 };
    const iterations = 5;

    std.debug.print("Throughput Benchmark (averaging {d} iterations):\n", .{iterations});
    std.debug.print("---------------------------------------------------\n", .{});

    for (batchSizes) |batchSize| {
        // Create batch
        var batch = try allocator.alloc([]const u8, batchSize);
        defer allocator.free(batch);

        for (0..batchSize) |i| {
            batch[i] = longTexts[i % longTexts.len];
        }

        // Time the embeddings
        var total_ns: i128 = 0;

        for (0..iterations) |_| {
            const start = std.time.nanoTimestamp();
            const embeddings = embedder.embed(batch) catch |err| {
                std.debug.print("Failed to embed: {}\n", .{err});
                return;
            };
            const elapsed = std.time.nanoTimestamp() - start;
            total_ns += elapsed;
            allocator.free(embeddings);
        }

        const avg_ns = @divTrunc(total_ns, iterations);
        const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const texts_per_sec = @as(f64, @floatFromInt(batchSize)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);

        std.debug.print("Batch size {:>3}: {:>7.0}ms avg, {:>5.1} texts/sec\n", .{
            batchSize,
            avg_ms,
            texts_per_sec,
        });
    }

    std.debug.print("\n", .{});
}
