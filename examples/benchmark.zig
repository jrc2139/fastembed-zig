//! Throughput benchmark for fastembed-zig
//!
//! Usage: zig build benchmark -- <model_directory> [model_type] [provider]

const std = @import("std");
const fe = @import("fastembed");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_directory> [model_type] [provider]\n", .{args[0]});
        std.debug.print("\nModel types: granite (default), bge, minilm, gemma-q4, gemma-fp16\n", .{});
        std.debug.print("Providers: cpu (default), coreml, coreml-all, coreml-cpu, auto\n", .{});
        return;
    }

    const model_path = args[1];
    const model_type: fe.Model = if (args.len > 2) blk: {
        const type_str = args[2];
        if (std.mem.eql(u8, type_str, "granite")) {
            break :blk .granite_embedding_english_r2;
        } else if (std.mem.eql(u8, type_str, "granite-fp16")) {
            break :blk .granite_embedding_english_r2_fp16;
        } else if (std.mem.eql(u8, type_str, "bge")) {
            break :blk .bge_small_en_v1_5;
        } else if (std.mem.eql(u8, type_str, "minilm")) {
            break :blk .all_minilm_l6_v2;
        } else if (std.mem.eql(u8, type_str, "gemma-q4")) {
            break :blk .embedding_gemma_300m_q4;
        } else if (std.mem.eql(u8, type_str, "gemma-fp16")) {
            break :blk .embedding_gemma_300m_fp16;
        } else {
            break :blk .granite_embedding_english_r2;
        }
    } else .granite_embedding_english_r2;

    // Parse execution provider
    const exec_provider: fe.ExecutionProvider = if (args.len > 3) blk: {
        const provider_str = args[3];
        if (std.mem.eql(u8, provider_str, "coreml")) {
            break :blk fe.ExecutionProvider.coremlProvider();
        } else if (std.mem.eql(u8, provider_str, "coreml-all")) {
            break :blk fe.ExecutionProvider.coremlPerformance();
        } else if (std.mem.eql(u8, provider_str, "coreml-cpu")) {
            break :blk fe.ExecutionProvider.coremlSafe();
        } else if (std.mem.eql(u8, provider_str, "auto")) {
            break :blk fe.ExecutionProvider.autoProvider();
        } else {
            break :blk .{ .cpu = {} };
        }
    } else .{ .cpu = {} };

    std.debug.print("Loading model: {s}\n", .{model_type.getConfig().name});
    std.debug.print("Path: {s}\n", .{model_path});
    std.debug.print("Provider: {s}\n", .{exec_provider.getName()});

    var embedder = fe.Embedder.init(allocator, .{
        .model = model_type,
        .model_path = model_path,
        .execution_provider = exec_provider,
    }) catch |err| {
        std.debug.print("Failed to load model: {}\n", .{err});
        return;
    };
    defer embedder.deinit();

    std.debug.print("Model loaded! Dimension: {d}\n\n", .{embedder.getDimension()});

    // Sample texts (5 varied-length sentences)
    const base_texts = [_][]const u8{
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "In the realm of natural language processing, embedding models transform text into dense vector representations that capture semantic meaning and enable similarity comparisons.",
        "The development of large language models has revolutionized how we approach tasks like text classification, semantic search, question answering systems, and document summarization.",
    };

    std.debug.print("Throughput Benchmark (3 iterations each, warmup first):\n", .{});
    std.debug.print("-----------------------------------------------------------\n", .{});

    // Test different batch sizes to compare with fastembed-go
    const batch_sizes = [_]usize{ 10, 100, 500, 1000 };

    for (batch_sizes) |batch_size| {
        // Build batch of texts
        var texts = try allocator.alloc([]const u8, batch_size);
        defer allocator.free(texts);
        for (0..batch_size) |i| {
            texts[i] = base_texts[i % base_texts.len];
        }

        // Warmup run (first run is often slower due to ONNX initialization)
        {
            const emb = embedder.embed(texts) catch |err| {
                std.debug.print("Failed: {}\n", .{err});
                return;
            };
            allocator.free(emb);
        }

        // Timed runs
        var total_ns: i128 = 0;
        const iterations = 3;

        for (0..iterations) |_| {
            const start = std.time.nanoTimestamp();
            const emb = embedder.embed(texts) catch |err| {
                std.debug.print("Failed: {}\n", .{err});
                return;
            };
            const elapsed = std.time.nanoTimestamp() - start;
            total_ns += elapsed;
            allocator.free(emb);
        }

        const avg_ns: i128 = @divTrunc(total_ns, iterations);
        const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
        const tps = @as(f64, @floatFromInt(@as(i128, batch_size))) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        std.debug.print("total={d:>5}: {d:>7.0}ms avg, {d:>6.1} texts/sec\n", .{ batch_size, avg_ms, tps });
    }

    std.debug.print("\n", .{});
}
