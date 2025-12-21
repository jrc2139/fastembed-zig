const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options (can be passed from parent project)
    const dynamic_ort = b.option(bool, "dynamic_ort", "Load ONNX Runtime dynamically at runtime") orelse false;
    const coreml_enabled = b.option(bool, "coreml_enabled", "Enable CoreML execution provider (macOS only)") orelse true;
    const cuda_enabled = b.option(bool, "cuda_enabled", "Enable CUDA execution provider") orelse false;

    // Create build options module
    const build_options = b.addOptions();
    build_options.addOption(bool, "coreml_enabled", coreml_enabled);
    build_options.addOption(bool, "cuda_enabled", cuda_enabled);
    build_options.addOption(bool, "dynamic_ort", dynamic_ort);
    const build_options_mod = build_options.createModule();

    // -------------------------------------------------------------------------
    // Dependencies via build.zig.zon
    // -------------------------------------------------------------------------
    const onnxruntime_dep = b.dependency("onnxruntime_zig", .{
        .target = target,
        .optimize = optimize,
        .cuda_enabled = cuda_enabled,
        .coreml_enabled = coreml_enabled,
        .dynamic_ort = dynamic_ort,
    });
    const onnxruntime_mod = onnxruntime_dep.module("onnxruntime");

    const tokenizer_dep = b.dependency("tokenizer_zig", .{
        .target = target,
        .optimize = optimize,
    });
    const tokenizer_mod = tokenizer_dep.module("tokenizer");

    // -------------------------------------------------------------------------
    // Fastembed module (exported for consumers)
    // -------------------------------------------------------------------------
    const fastembed_mod = b.addModule("fastembed", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tokenizer", .module = tokenizer_mod },
            .{ .name = "build_options", .module = build_options_mod },
            .{ .name = "onnxruntime-zig", .module = onnxruntime_mod },
        },
    });

    // Add ORT include path for @cImport
    fastembed_mod.addIncludePath(b.path("deps/onnxruntime/include"));

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
                .{ .name = "build_options", .module = build_options_mod },
                .{ .name = "onnxruntime-zig", .module = onnxruntime_mod },
            },
        }),
    });
    lib_tests.root_module.addIncludePath(b.path("deps/onnxruntime/include"));
    lib_tests.root_module.addLibraryPath(b.path("deps/onnxruntime/lib"));
    if (!dynamic_ort) {
        lib_tests.root_module.linkSystemLibrary("onnxruntime", .{});
        lib_tests.root_module.addRPath(b.path("deps/onnxruntime/lib"));
    }
    lib_tests.linkLibC();

    if (target.result.os.tag == .macos) {
        lib_tests.root_module.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            lib_tests.root_module.linkFramework("CoreML", .{});
        }
    }

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&b.addRunArtifact(lib_tests).step);

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
    embed_example.root_module.addIncludePath(b.path("deps/onnxruntime/include"));
    embed_example.linkLibC();

    if (!dynamic_ort) {
        embed_example.root_module.addLibraryPath(b.path("deps/onnxruntime/lib"));
        embed_example.root_module.linkSystemLibrary("onnxruntime", .{});
        embed_example.root_module.addRPath(b.path("deps/onnxruntime/lib"));
    }

    if (target.result.os.tag == .macos) {
        embed_example.root_module.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            embed_example.root_module.linkFramework("CoreML", .{});
        }
    }

    b.installArtifact(embed_example);

    const run_embed = b.addRunArtifact(embed_example);
    run_embed.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_embed.addArgs(args);

    const run_step = b.step("run", "Run the basic embedding example");
    run_step.dependOn(&run_embed.step);

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
    b.installArtifact(tokenize_example);

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
    benchmark_example.root_module.addIncludePath(b.path("deps/onnxruntime/include"));
    benchmark_example.linkLibC();
    if (!dynamic_ort) {
        benchmark_example.root_module.addLibraryPath(b.path("deps/onnxruntime/lib"));
        benchmark_example.root_module.linkSystemLibrary("onnxruntime", .{});
        benchmark_example.root_module.addRPath(b.path("deps/onnxruntime/lib"));
    }
    if (target.result.os.tag == .macos) {
        benchmark_example.root_module.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            benchmark_example.root_module.linkFramework("CoreML", .{});
        }
    }
    b.installArtifact(benchmark_example);

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
                .{ .name = "build_options", .module = build_options_mod },
                .{ .name = "onnxruntime-zig", .module = onnxruntime_mod },
            },
        }),
    });
    check.root_module.addIncludePath(b.path("deps/onnxruntime/include"));
    check.linkLibC();

    const check_step = b.step("check", "Check for compilation errors");
    check_step.dependOn(&check.step);
}
