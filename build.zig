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
    // ONNX Runtime paths - configurable via build option, defaults to ~/.local/share/onnxruntime
    // -------------------------------------------------------------------------
    const home = std.posix.getenv("HOME") orelse "/tmp";
    const default_ort_dir = b.fmt("{s}/.local/share/onnxruntime", .{home});
    const ort_dir = b.option([]const u8, "ort_path", "Path to ONNX Runtime installation") orelse default_ort_dir;
    const ort_include: std.Build.LazyPath = .{ .cwd_relative = b.fmt("{s}/include", .{ort_dir}) };
    const ort_lib: std.Build.LazyPath = .{ .cwd_relative = b.fmt("{s}/lib", .{ort_dir}) };

    // Add download step if ONNX Runtime is not installed
    const download_step = addOrtDownloadStep(b, ort_dir);

    // -------------------------------------------------------------------------
    // Dependencies via build.zig.zon
    // -------------------------------------------------------------------------
    const onnxruntime_dep = b.dependency("onnxruntime_zig", .{
        .target = target,
        .optimize = optimize,
        .cuda_enabled = cuda_enabled,
        .coreml_enabled = coreml_enabled,
        .dynamic_ort = dynamic_ort,
        .ort_path = ort_dir,
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
    fastembed_mod.addIncludePath(ort_include);

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
    lib_tests.root_module.addIncludePath(ort_include);
    lib_tests.root_module.addLibraryPath(ort_lib);
    if (!dynamic_ort) {
        lib_tests.root_module.linkSystemLibrary("onnxruntime", .{});
        lib_tests.root_module.addRPath(ort_lib);
    }
    lib_tests.linkLibC();

    if (target.result.os.tag == .macos) {
        lib_tests.root_module.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            lib_tests.root_module.linkFramework("CoreML", .{});
        }
    }

    const run_tests = b.addRunArtifact(lib_tests);
    run_tests.step.dependOn(download_step);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_tests.step);

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
    embed_example.root_module.addIncludePath(ort_include);
    embed_example.linkLibC();

    if (!dynamic_ort) {
        embed_example.root_module.addLibraryPath(ort_lib);
        embed_example.root_module.linkSystemLibrary("onnxruntime", .{});
        embed_example.root_module.addRPath(ort_lib);
    }

    if (target.result.os.tag == .macos) {
        embed_example.root_module.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            embed_example.root_module.linkFramework("CoreML", .{});
        }
    }

    embed_example.step.dependOn(download_step);
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
    benchmark_example.root_module.addIncludePath(ort_include);
    benchmark_example.linkLibC();
    if (!dynamic_ort) {
        benchmark_example.root_module.addLibraryPath(ort_lib);
        benchmark_example.root_module.linkSystemLibrary("onnxruntime", .{});
        benchmark_example.root_module.addRPath(ort_lib);
    }
    if (target.result.os.tag == .macos) {
        benchmark_example.root_module.linkFramework("Foundation", .{});
        if (coreml_enabled) {
            benchmark_example.root_module.linkFramework("CoreML", .{});
        }
    }
    benchmark_example.step.dependOn(download_step);
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
    check.root_module.addIncludePath(ort_include);
    check.linkLibC();

    const check_step = b.step("check", "Check for compilation errors");
    check_step.dependOn(&check.step);

    // -------------------------------------------------------------------------
    // Setup step (manual download trigger)
    // -------------------------------------------------------------------------
    const setup_step = b.step("setup", "Download ONNX Runtime if not present");
    setup_step.dependOn(download_step);
}

/// Add a step that downloads ONNX Runtime if not present
fn addOrtDownloadStep(b: *std.Build, ort_dir: []const u8) *std.Build.Step {
    const check_and_download = b.addSystemCommand(&.{
        "sh", "-c", b.fmt(
            \\if [ -f "{[0]s}/lib/libonnxruntime.dylib" ] || [ -f "{[0]s}/lib/libonnxruntime.so" ]; then
            \\    echo "ONNX Runtime found at {[0]s}"
            \\else
            \\    echo "Downloading ONNX Runtime to {[0]s}..."
            \\    mkdir -p "{[0]s}"
            \\
            \\    # Detect OS and architecture
            \\    OS=$(uname -s)
            \\    ARCH=$(uname -m)
            \\    VERSION="1.23.2"
            \\
            \\    if [ "$OS" = "Darwin" ]; then
            \\        if [ "$ARCH" = "arm64" ]; then
            \\            URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/onnxruntime-osx-arm64-$VERSION.tgz"
            \\        else
            \\            URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/onnxruntime-osx-x86_64-$VERSION.tgz"
            \\        fi
            \\    elif [ "$OS" = "Linux" ]; then
            \\        if [ "$ARCH" = "aarch64" ]; then
            \\            URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/onnxruntime-linux-aarch64-$VERSION.tgz"
            \\        else
            \\            URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/onnxruntime-linux-x64-$VERSION.tgz"
            \\        fi
            \\    else
            \\        echo "Unsupported OS: $OS"
            \\        exit 1
            \\    fi
            \\
            \\    echo "Downloading from $URL"
            \\    curl -L "$URL" | tar xz -C "{[0]s}" --strip-components=2
            \\
            \\    if [ -f "{[0]s}/lib/libonnxruntime.dylib" ] || [ -f "{[0]s}/lib/libonnxruntime.so" ]; then
            \\        echo "ONNX Runtime installed successfully"
            \\    else
            \\        echo "Failed to install ONNX Runtime"
            \\        exit 1
            \\    fi
            \\fi
        , .{ort_dir}),
    });
    return &check_and_download.step;
}
