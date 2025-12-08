//! Test tokenizer directly (bypassing embedder)
const std = @import("std");
const fe = @import("fastembed");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <model_directory>\n", .{args[0]});
        return;
    }

    const model_path = args[1];
    const tokenizer_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_path});
    defer allocator.free(tokenizer_path);

    std.debug.print("Loading tokenizer from: {s}\n", .{tokenizer_path});

    var tokenizer = fe.Tokenizer.fromFile(allocator, tokenizer_path) catch |err| {
        std.debug.print("Failed to load tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    std.debug.print("Tokenizer loaded!\n\n", .{});

    // Test 1: Compile-time string
    std.debug.print("Test 1: Compile-time string...\n", .{});
    const compile_time = "Hello world this is a test";
    const tokens1 = try tokenizer.encodeAlloc(allocator, compile_time, true);
    allocator.free(tokens1);
    std.debug.print("Test 1 OK! ({d} tokens)\n", .{tokens1.len});

    // Test 2: Another compile-time string
    std.debug.print("Test 2: Another compile-time string...\n", .{});
    const compile_time2 = "Goodbye world";
    const tokens2 = try tokenizer.encodeAlloc(allocator, compile_time2, true);
    allocator.free(tokens2);
    std.debug.print("Test 2 OK! ({d} tokens)\n", .{tokens2.len});

    // Test 3: Runtime-allocated string
    std.debug.print("Test 3: Runtime-allocated string...\n", .{});
    const runtime_buf = try allocator.alloc(u8, 50);
    defer allocator.free(runtime_buf);
    @memset(runtime_buf, 'a');

    const tokens3 = try tokenizer.encodeAlloc(allocator, runtime_buf, true);
    allocator.free(tokens3);
    std.debug.print("Test 3 OK! ({d} tokens)\n", .{tokens3.len});

    std.debug.print("\nAll tokenizer tests passed!\n", .{});
}
