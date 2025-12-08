const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build option for static linking
    const use_static = b.option(bool, "static", "Link against static ONNX Runtime libraries") orelse false;

    // -------------------------------------------------------------------------
    // Dependencies
    // -------------------------------------------------------------------------

    // Tokenizer dependency (pure Zig - no more C library!)
    const tokenizer_dep = b.dependency("tokenizer", .{
        .target = target,
        .optimize = optimize,
    });
    const tokenizer_mod = tokenizer_dep.module("tokenizer");

    // ONNX Runtime paths
    const ort_include = if (use_static)
        b.path("deps/onnxruntime-static/include")
    else
        b.path("deps/onnxruntime/include");

    const ort_lib = if (use_static)
        b.path("deps/onnxruntime-static/lib")
    else
        b.path("deps/onnxruntime/lib");

    // -------------------------------------------------------------------------
    // Helper to configure a module with ONNX Runtime dependencies
    // -------------------------------------------------------------------------
    const configureOrt = struct {
        fn configure(
            module: *std.Build.Module,
            tgt: std.Build.ResolvedTarget,
            o_include: std.Build.LazyPath,
            o_lib: std.Build.LazyPath,
            static: bool,
        ) void {
            // ONNX Runtime
            module.addIncludePath(o_include);
            module.addLibraryPath(o_lib);

            if (static) {
                // Link the combined static library
                module.linkSystemLibrary("onnxruntime_all", .{ .preferred_link_mode = .static });
                // Link C++ standard library
                if (tgt.result.os.tag == .macos) {
                    module.linkSystemLibrary("c++", .{});
                } else {
                    module.linkSystemLibrary("stdc++", .{ .preferred_link_mode = .static });
                }
            } else {
                module.linkSystemLibrary("onnxruntime", .{});
                module.addRPath(o_lib);
            }

            // System dependencies
            module.link_libc = true;
            if (tgt.result.os.tag == .macos) {
                module.linkFramework("Foundation", .{});
                module.linkFramework("CoreML", .{});
            }
        }
    }.configure;

    // -------------------------------------------------------------------------
    // Library Module
    // -------------------------------------------------------------------------
    const fastembed_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tokenizer", .module = tokenizer_mod },
        },
    });
    configureOrt(fastembed_mod, target, ort_include, ort_lib, use_static);

    // Export the configured module for other packages
    _ = b.addModule("fastembed", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tokenizer", .module = tokenizer_mod },
        },
    });

    // Also add the dependency paths to the exported module
    const exported_mod = b.modules.get("fastembed").?;
    configureOrt(exported_mod, target, ort_include, ort_lib, use_static);

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tokenizer", .module = tokenizer_mod },
            },
        }),
    });
    configureOrt(lib_tests.root_module, target, ort_include, ort_lib, use_static);

    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_lib_tests.step);

    // -------------------------------------------------------------------------
    // Example: Basic Tokenization
    // -------------------------------------------------------------------------
    const tokenize_example = b.addExecutable(.{
        .name = "basic_tokenize",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_tokenize.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fastembed", .module = fastembed_mod },
            },
        }),
    });
    configureOrt(tokenize_example.root_module, target, ort_include, ort_lib, use_static);
    b.installArtifact(tokenize_example);

    // -------------------------------------------------------------------------
    // Example: Basic Embedding
    // -------------------------------------------------------------------------
    const embed_example = b.addExecutable(.{
        .name = "basic_embed",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/basic_embed.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fastembed", .module = fastembed_mod },
            },
        }),
    });
    configureOrt(embed_example.root_module, target, ort_include, ort_lib, use_static);
    b.installArtifact(embed_example);

    const run_embed = b.addRunArtifact(embed_example);
    run_embed.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_embed.addArgs(args);
    }

    const run_step = b.step("run", "Run the basic embedding example");
    run_step.dependOn(&run_embed.step);

    // -------------------------------------------------------------------------
    // Example: Benchmark
    // -------------------------------------------------------------------------
    const benchmark_example = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/benchmark.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fastembed", .module = fastembed_mod },
            },
        }),
    });
    configureOrt(benchmark_example.root_module, target, ort_include, ort_lib, use_static);
    b.installArtifact(benchmark_example);

    const run_benchmark = b.addRunArtifact(benchmark_example);
    run_benchmark.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_benchmark.addArgs(args);
    }

    const benchmark_step = b.step("benchmark", "Run the throughput benchmark");
    benchmark_step.dependOn(&run_benchmark.step);

    // -------------------------------------------------------------------------
    // Example: Test Batch
    // -------------------------------------------------------------------------
    const test_batch_example = b.addExecutable(.{
        .name = "test_batch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/test_batch.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "fastembed", .module = fastembed_mod },
            },
        }),
    });
    configureOrt(test_batch_example.root_module, target, ort_include, ort_lib, use_static);
    b.installArtifact(test_batch_example);

    // -------------------------------------------------------------------------
    // Check (for ZLS)
    // -------------------------------------------------------------------------
    const check = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tokenizer", .module = tokenizer_mod },
            },
        }),
    });
    check.root_module.addIncludePath(ort_include);

    const check_step = b.step("check", "Check for compilation errors");
    check_step.dependOn(&check.step);
}
