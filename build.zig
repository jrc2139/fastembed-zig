const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build option for static linking
    const use_static = b.option(bool, "static", "Link against static ONNX Runtime libraries") orelse false;

    // Build option for CoreML support (default: true for dynamic, false for static on macOS)
    const default_coreml = !use_static;
    const coreml_enabled = b.option(bool, "coreml", "Enable CoreML execution provider (macOS only)") orelse default_coreml;

    // Build option for CUDA support (requires dynamically linked CUDA ONNX Runtime)
    const cuda_enabled = b.option(bool, "cuda", "Enable CUDA execution provider") orelse false;

    // Create build options module
    const build_options = b.addOptions();
    build_options.addOption(bool, "coreml_enabled", coreml_enabled);
    build_options.addOption(bool, "cuda_enabled", cuda_enabled);

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
    // For CUDA builds, use ~/.osgrep/onnxruntime-cuda/{include,lib}
    const home = std.posix.getenv("HOME") orelse "/tmp";
    const ort_include: std.Build.LazyPath = if (use_static)
        b.path("deps/onnxruntime-static/include")
    else if (cuda_enabled)
        .{ .cwd_relative = b.fmt("{s}/.osgrep/onnxruntime-cuda/include", .{home}) }
    else
        b.path("deps/onnxruntime/include");

    const ort_lib: std.Build.LazyPath = if (use_static)
        b.path("deps/onnxruntime-static/lib")
    else if (cuda_enabled)
        .{ .cwd_relative = b.fmt("{s}/.osgrep/onnxruntime-cuda/lib", .{home}) }
    else
        b.path("deps/onnxruntime/lib");

    // -------------------------------------------------------------------------
    // Helper to configure a module with ONNX Runtime dependencies
    // -------------------------------------------------------------------------
    const ConfigureOrtContext = struct {
        tgt: std.Build.ResolvedTarget,
        o_include: std.Build.LazyPath,
        o_lib: std.Build.LazyPath,
        static: bool,
        coreml: bool,

        fn configure(ctx: @This(), module: *std.Build.Module) void {
            // ONNX Runtime
            module.addIncludePath(ctx.o_include);
            module.addLibraryPath(ctx.o_lib);

            if (ctx.static) {
                // Link the combined static library
                module.linkSystemLibrary("onnxruntime_all", .{ .preferred_link_mode = .static });
                // Link C++ standard library
                if (ctx.tgt.result.os.tag == .macos) {
                    module.linkSystemLibrary("c++", .{});
                } else {
                    module.linkSystemLibrary("stdc++", .{ .preferred_link_mode = .static });
                }
            } else {
                module.linkSystemLibrary("onnxruntime", .{});
                module.addRPath(ctx.o_lib);
            }

            // System dependencies
            module.link_libc = true;
            if (ctx.tgt.result.os.tag == .macos) {
                module.linkFramework("Foundation", .{});
                // Only link CoreML if enabled
                if (ctx.coreml) {
                    module.linkFramework("CoreML", .{});
                }
            }
        }
    };
    const ort_config = ConfigureOrtContext{
        .tgt = target,
        .o_include = ort_include,
        .o_lib = ort_lib,
        .static = use_static,
        .coreml = coreml_enabled,
    };

    // -------------------------------------------------------------------------
    // Library Module
    // -------------------------------------------------------------------------
    const fastembed_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tokenizer", .module = tokenizer_mod },
            .{ .name = "build_options", .module = build_options.createModule() },
        },
    });
    ort_config.configure(fastembed_mod);

    // Export the configured module for other packages
    _ = b.addModule("fastembed", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "tokenizer", .module = tokenizer_mod },
            .{ .name = "build_options", .module = build_options.createModule() },
        },
    });

    // Also add the dependency paths to the exported module
    const exported_mod = b.modules.get("fastembed").?;
    ort_config.configure(exported_mod);

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
                .{ .name = "build_options", .module = build_options.createModule() },
            },
        }),
    });
    ort_config.configure(lib_tests.root_module);

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
    ort_config.configure(tokenize_example.root_module);
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
    ort_config.configure(embed_example.root_module);
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
    ort_config.configure(benchmark_example.root_module);
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
    ort_config.configure(test_batch_example.root_module);
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
                .{ .name = "build_options", .module = build_options.createModule() },
            },
        }),
    });
    check.root_module.addIncludePath(ort_include);

    const check_step = b.step("check", "Check for compilation errors");
    check_step.dependOn(&check.step);
}
