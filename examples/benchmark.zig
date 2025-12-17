//! Throughput benchmark for fastembed-zig
//!
//! Compares standard Embedder vs zero-allocation FastEmbedder vs Async inference.
//!
//! Usage: zig build benchmark -- <model_directory> [model_type] [provider] [mode]
//!
//! Modes:
//!   both  - Compare both embedder types (default)
//!   fast  - Only test FastEmbedder
//!   std   - Only test standard Embedder
//!   async - Test async inference (pipelining)
//!   all   - Test all modes including async

const std = @import("std");
const fe = @import("fastembed");
const onnx = fe.onnx;

const BenchMode = enum { both, fast, std_only, async_only, all };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_directory> [model_type] [provider] [mode]\n", .{args[0]});
        std.debug.print("\nModel types: granite (default), granite-small, bge, minilm, gemma-q4, gemma-fp16\n", .{});
        std.debug.print("Providers: cpu (default), coreml, coreml-all, coreml-cpu, auto\n", .{});
        std.debug.print("Modes: both (default), fast, std, async, all\n", .{});
        return;
    }

    const model_path = args[1];
    const model_type: fe.Model = if (args.len > 2) blk: {
        const type_str = args[2];
        if (std.mem.eql(u8, type_str, "granite")) {
            break :blk .granite_embedding_english_r2;
        } else if (std.mem.eql(u8, type_str, "granite-small")) {
            break :blk .granite_embedding_small_english_r2;
        } else if (std.mem.eql(u8, type_str, "granite-small-qint8")) {
            break :blk .granite_embedding_small_english_r2_qint8;
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

    // Parse benchmark mode
    const mode: BenchMode = if (args.len > 4) blk: {
        const mode_str = args[4];
        if (std.mem.eql(u8, mode_str, "fast")) {
            break :blk .fast;
        } else if (std.mem.eql(u8, mode_str, "std")) {
            break :blk .std_only;
        } else if (std.mem.eql(u8, mode_str, "async")) {
            break :blk .async_only;
        } else if (std.mem.eql(u8, mode_str, "all")) {
            break :blk .all;
        } else {
            break :blk .both;
        }
    } else .both;

    std.debug.print("=== fastembed-zig Benchmark ===\n\n", .{});
    std.debug.print("Model: {s}\n", .{model_type.getConfig().name});
    std.debug.print("Path:  {s}\n", .{model_path});
    std.debug.print("Provider: {s}\n", .{exec_provider.getName()});
    std.debug.print("Mode: {s}\n\n", .{@tagName(mode)});

    // Sample texts (5 varied-length sentences)
    const base_texts = [_][]const u8{
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "In the realm of natural language processing, embedding models transform text into dense vector representations that capture semantic meaning and enable similarity comparisons.",
        "The development of large language models has revolutionized how we approach tasks like text classification, semantic search, question answering systems, and document summarization.",
    };

    // Test batch sizes
    const batch_sizes = [_]usize{ 10, 32, 64, 100 };
    const max_batch_for_fast = 64; // FastEmbedder pre-allocates for max batch (keep reasonable for memory)
    const iterations = 5;

    // Run standard Embedder benchmark
    if (mode == .both or mode == .std_only or mode == .all) {
        std.debug.print("--- Standard Embedder (allocates per embed) ---\n", .{});

        var embedder = fe.Embedder.init(allocator, .{
            .model = model_type,
            .model_path = model_path,
            .execution_provider = exec_provider,
        }) catch |err| {
            std.debug.print("Failed to load Embedder: {}\n", .{err});
            return;
        };
        defer embedder.deinit();

        std.debug.print("Dimension: {d}\n\n", .{embedder.getDimension()});

        for (batch_sizes) |batch_size| {
            try benchmarkStd(allocator, &embedder, &base_texts, batch_size, iterations);
        }
        std.debug.print("\n", .{});
    }

    // Run FastEmbedder benchmark
    if (mode == .both or mode == .fast or mode == .all) {
        std.debug.print("--- FastEmbedder (zero-allocation after init) ---\n", .{});

        var fast_embedder = fe.FastEmbedder.init(allocator, .{
            .model = model_type,
            .model_path = model_path,
            .execution_provider = exec_provider,
            .max_batch_size = max_batch_for_fast,
        }) catch |err| {
            std.debug.print("Failed to load FastEmbedder: {}\n", .{err});
            // Fall back to just showing standard results
            if (mode == .fast) return;
            std.debug.print("(Skipping FastEmbedder)\n\n", .{});
            return;
        };
        defer fast_embedder.deinit();

        std.debug.print("Dimension: {d}\n", .{fast_embedder.getDimension()});
        std.debug.print("Pre-allocated memory: {d:.2} MB\n\n", .{@as(f64, @floatFromInt(fast_embedder.memoryUsage())) / (1024.0 * 1024.0)});

        for (batch_sizes) |batch_size| {
            if (batch_size <= max_batch_for_fast) {
                try benchmarkFast(allocator, &fast_embedder, &base_texts, batch_size, iterations);
            }
        }
        std.debug.print("\n", .{});
    }

    // Run Async benchmark (uses Session.runAsync directly)
    if (mode == .async_only or mode == .all) {
        std.debug.print("--- Async Inference (callback-based) ---\n", .{});

        // Build paths
        const model_file = model_type.getModelFile();
        const model_onnx_path = try std.fs.path.join(allocator, &.{ model_path, model_file });
        defer allocator.free(model_onnx_path);

        const tokenizer_path = try std.fs.path.join(allocator, &.{ model_path, "tokenizer.json" });
        defer allocator.free(tokenizer_path);

        // Load tokenizer
        var tok = fe.Tokenizer.fromFile(allocator, tokenizer_path) catch |err| {
            std.debug.print("Failed to load tokenizer: {}\n", .{err});
            return;
        };
        defer tok.deinit();

        // Load ONNX session
        const model_onnx_path_z = try allocator.allocSentinel(u8, model_onnx_path.len, 0);
        defer allocator.free(model_onnx_path_z);
        @memcpy(model_onnx_path_z, model_onnx_path);

        var env = onnx.Environment.init() catch |err| {
            std.debug.print("Failed to create environment: {}\n", .{err});
            return;
        };
        defer env.deinit();

        var session = onnx.Session.initWithOptions(env, model_onnx_path_z, allocator, .{
            .execution_provider = exec_provider,
        }) catch |err| {
            std.debug.print("Failed to load session: {}\n", .{err});
            return;
        };
        defer session.deinit();

        const config = model_type.getConfig();
        std.debug.print("Dimension: {d}\n\n", .{config.hidden_dim});

        // For async, we test smaller batches but with pipelining
        const async_batch_sizes = [_]usize{ 1, 4, 8, 16 };
        for (async_batch_sizes) |batch_size| {
            try benchmarkAsync(allocator, &session, &tok, &base_texts, batch_size, iterations, config);
        }
        std.debug.print("\n", .{});
    }

    // Summary
    if (mode == .both or mode == .all) {
        std.debug.print("=== Summary ===\n", .{});
        std.debug.print("- Embedder: Standard sync inference with per-call allocation\n", .{});
        std.debug.print("- FastEmbed: Zero-allocation sync inference with IoBinding\n", .{});
        if (mode == .all) {
            std.debug.print("- Async: Callback-based async inference (enables pipelining)\n", .{});
        }
        std.debug.print("\nFastEmbedder eliminates per-call allocations, reducing GC pressure\n", .{});
        std.debug.print("and providing more predictable latency for real-time applications.\n", .{});
    }
}

fn benchmarkStd(
    allocator: std.mem.Allocator,
    embedder: *fe.Embedder,
    base_texts: []const []const u8,
    batch_size: usize,
    iterations: usize,
) !void {
    // Build batch of texts
    var texts = try allocator.alloc([]const u8, batch_size);
    defer allocator.free(texts);
    for (0..batch_size) |i| {
        texts[i] = base_texts[i % base_texts.len];
    }

    // Warmup
    {
        const emb = embedder.embed(texts) catch |err| {
            std.debug.print("Embed failed: {}\n", .{err});
            return;
        };
        allocator.free(emb);
    }

    // Timed runs
    var total_ns: i128 = 0;
    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        const emb = embedder.embed(texts) catch |err| {
            std.debug.print("Embed failed: {}\n", .{err});
            return;
        };
        const elapsed = std.time.nanoTimestamp() - start;
        total_ns += elapsed;
        allocator.free(emb);
    }

    printResults("Embedder", batch_size, total_ns, iterations);
}

fn benchmarkFast(
    allocator: std.mem.Allocator,
    embedder: *fe.FastEmbedder,
    base_texts: []const []const u8,
    batch_size: usize,
    iterations: usize,
) !void {
    // Build batch of texts
    var texts = try allocator.alloc([]const u8, batch_size);
    defer allocator.free(texts);
    for (0..batch_size) |i| {
        texts[i] = base_texts[i % base_texts.len];
    }

    // Warmup
    _ = embedder.embed(texts) catch |err| {
        std.debug.print("Embed failed: {}\n", .{err});
        return;
    };

    // Timed runs - NO allocations after warmup!
    var total_ns: i128 = 0;
    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        _ = embedder.embed(texts) catch |err| {
            std.debug.print("Embed failed: {}\n", .{err});
            return;
        };
        const elapsed = std.time.nanoTimestamp() - start;
        total_ns += elapsed;
        // No free needed - result is borrowed from internal buffer
    }

    printResults("FastEmbed", batch_size, total_ns, iterations);
}

fn printResults(name: []const u8, batch_size: usize, total_ns: i128, iterations: usize) void {
    const avg_ns: i128 = @divTrunc(total_ns, @as(i128, @intCast(iterations)));
    const avg_ms = @as(f64, @floatFromInt(avg_ns)) / 1_000_000.0;
    const tps = @as(f64, @floatFromInt(@as(i128, batch_size))) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);

    std.debug.print("{s:<10} batch={d:>4}: {d:>7.1}ms avg, {d:>7.1} texts/sec\n", .{
        name, batch_size, avg_ms, tps,
    });
}

/// Context for async callback
const AsyncBenchContext = struct {
    completed: std.atomic.Value(bool),
    result: ?[]f32,
    err: ?onnx.OnnxError,
    allocator: std.mem.Allocator,
};

/// Callback for async benchmark
fn asyncCallback(user_data: ?*anyopaque, result: onnx.Session.AsyncResult) void {
    const ctx: *AsyncBenchContext = @ptrCast(@alignCast(user_data));
    switch (result) {
        .success => |data| {
            ctx.result = data;
            ctx.err = null;
        },
        .err => |e| {
            ctx.result = null;
            ctx.err = e;
        },
    }
    ctx.completed.store(true, .release);
}

fn benchmarkAsync(
    allocator: std.mem.Allocator,
    session: *onnx.Session,
    tokenizer: *fe.Tokenizer,
    base_texts: []const []const u8,
    batch_size: usize,
    iterations: usize,
    config: fe.models.ModelConfig,
) !void {
    _ = config;

    // Build batch of texts
    var texts = try allocator.alloc([]const u8, batch_size);
    defer allocator.free(texts);
    for (0..batch_size) |i| {
        texts[i] = base_texts[i % base_texts.len];
    }

    // Tokenize all texts and find max sequence length
    const max_seq_len: usize = 512;
    var actual_max_seq: usize = 0;

    var all_encodings = try allocator.alloc([]i32, batch_size);
    defer {
        for (all_encodings) |enc| {
            allocator.free(enc);
        }
        allocator.free(all_encodings);
    }

    for (texts, 0..) |text, i| {
        const enc = tokenizer.encodeAlloc(allocator, text, true) catch {
            std.debug.print("Tokenization failed\n", .{});
            return;
        };
        all_encodings[i] = enc;
        actual_max_seq = @max(actual_max_seq, @min(enc.len, max_seq_len));
    }

    // Prepare input tensors
    const input_size = batch_size * actual_max_seq;
    var input_ids = try allocator.alloc(i64, input_size);
    defer allocator.free(input_ids);
    var attention_mask = try allocator.alloc(i64, input_size);
    defer allocator.free(attention_mask);

    // Fill input buffers
    @memset(input_ids, 0);
    @memset(attention_mask, 0);

    for (0..batch_size) |b| {
        const enc = all_encodings[b];
        const seq_len = @min(enc.len, actual_max_seq);
        const offset = b * actual_max_seq;
        for (0..seq_len) |s| {
            input_ids[offset + s] = @intCast(enc[s]);
            attention_mask[offset + s] = 1;
        }
    }

    // Warmup with async
    {
        var ctx = AsyncBenchContext{
            .completed = std.atomic.Value(bool).init(false),
            .result = null,
            .err = null,
            .allocator = allocator,
        };

        session.runAsync(input_ids, attention_mask, null, batch_size, actual_max_seq, asyncCallback, &ctx) catch |err| {
            std.debug.print("Async warmup failed: {}\n", .{err});
            return;
        };

        // Spin wait for completion
        while (!ctx.completed.load(.acquire)) {
            std.Thread.sleep(1000); // 1us
        }

        if (ctx.result) |r| {
            allocator.free(r);
        }
    }

    // Timed async runs
    var total_ns: i128 = 0;
    for (0..iterations) |_| {
        var ctx = AsyncBenchContext{
            .completed = std.atomic.Value(bool).init(false),
            .result = null,
            .err = null,
            .allocator = allocator,
        };

        const start = std.time.nanoTimestamp();

        session.runAsync(input_ids, attention_mask, null, batch_size, actual_max_seq, asyncCallback, &ctx) catch |err| {
            std.debug.print("Async run failed: {}\n", .{err});
            return;
        };

        // Spin wait for completion
        while (!ctx.completed.load(.acquire)) {
            std.Thread.sleep(1000); // 1us
        }

        const elapsed = std.time.nanoTimestamp() - start;
        total_ns += elapsed;

        if (ctx.result) |r| {
            allocator.free(r);
        } else if (ctx.err) |e| {
            std.debug.print("Async inference error: {}\n", .{e});
            return;
        }
    }

    printResults("Async", batch_size, total_ns, iterations);
}
