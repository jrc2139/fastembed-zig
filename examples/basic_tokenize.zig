//! Basic tokenization example
//!
//! This example demonstrates how to:
//! 1. Load a HuggingFace tokenizer from a JSON file
//! 2. Encode text to token IDs
//! 3. Decode token IDs back to text
//!
//! Usage: zig build run -- path/to/tokenizer.json "Text to tokenize"

const std = @import("std");
const fe = @import("fastembed");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get command line args
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <tokenizer.json> <text>\n", .{args[0]});
        std.debug.print("\nExample:\n", .{});
        std.debug.print("  {s} models/bge-small-en-v1.5/tokenizer.json \"Hello world\"\n", .{args[0]});
        return;
    }

    const tokenizer_path = args[1];
    const text = args[2];

    std.debug.print("Loading tokenizer from: {s}\n", .{tokenizer_path});

    // Load tokenizer
    var tokenizer = fe.Tokenizer.fromFile(allocator, tokenizer_path) catch |err| {
        std.debug.print("Failed to load tokenizer: {}\n", .{err});
        return;
    };
    defer tokenizer.deinit();

    std.debug.print("Vocabulary size: {d}\n", .{tokenizer.getVocabSize()});

    // Encode text
    std.debug.print("\nEncoding: \"{s}\"\n", .{text});

    const tokens = try tokenizer.encodeAlloc(allocator, text, true);
    defer allocator.free(tokens);

    std.debug.print("Token IDs ({d}): ", .{tokens.len});
    for (tokens) |token_id| {
        std.debug.print("{d} ", .{token_id});
    }
    std.debug.print("\n", .{});

    // Show token strings
    std.debug.print("Tokens: ", .{});
    for (tokens) |token_id| {
        if (tokenizer.idToToken(@intCast(token_id))) |token_str| {
            std.debug.print("\"{s}\" ", .{token_str});
        } else {
            std.debug.print("[UNK] ", .{});
        }
    }
    std.debug.print("\n", .{});

    // Decode back to text
    if (tokenizer.decode(tokens, false)) |decoded| {
        std.debug.print("\nDecoded: \"{s}\"\n", .{decoded});
    }
}
