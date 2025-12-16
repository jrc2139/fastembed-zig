//! Execution Provider abstraction for ONNX Runtime
//!
//! Provides configurable execution providers (CPU, CoreML, CUDA) with
//! automatic platform detection.

const std = @import("std");
const builtin = @import("builtin");
const c_api = @import("c_api.zig");

/// CoreML compute units for hardware selection
pub const CoreMLComputeUnits = enum {
    /// Enable all available compute units (CPU, GPU, Neural Engine)
    all,
    /// Restrict execution to CPU only
    cpu_only,
    /// Enable CPU and GPU acceleration
    cpu_and_gpu,
    /// Enable CPU and Neural Engine (recommended for precision)
    cpu_and_neural_engine,
};

/// CoreML model format
pub const CoreMLModelFormat = enum {
    /// Legacy NeuralNetwork format (Core ML 3+)
    neural_network,
    /// Modern MLProgram format (Core ML 5+, macOS 12+) - better precision
    ml_program,
};

/// CoreML execution provider options
pub const CoreMLOptions = struct {
    /// Model format - MLProgram has better precision
    model_format: CoreMLModelFormat = .ml_program,
    /// Which hardware units to use
    compute_units: CoreMLComputeUnits = .cpu_and_neural_engine,
    /// Allow low precision accumulation on GPU (may cause NaN issues)
    allow_low_precision_accumulation_on_gpu: bool = false,
    /// Require static input shapes (may improve performance)
    require_static_input_shapes: bool = false,
};

/// Execution provider type
pub const ExecutionProvider = union(enum) {
    /// CPU execution (default, works everywhere)
    cpu: void,
    /// CoreML execution (macOS only, uses Neural Engine + Metal GPU)
    coreml: CoreMLOptions,
    /// CUDA execution (NVIDIA GPU)
    cuda: struct {
        device_id: u32,
    },
    /// Auto-detect best provider for platform
    auto: void,

    /// Default CPU provider
    pub fn cpuProvider() ExecutionProvider {
        return .{ .cpu = {} };
    }

    /// CoreML provider with safe defaults (precision-focused)
    pub fn coremlProvider() ExecutionProvider {
        return .{ .coreml = .{} };
    }

    /// CoreML provider with custom options
    pub fn coremlWithOptions(opts: CoreMLOptions) ExecutionProvider {
        return .{ .coreml = opts };
    }

    /// CoreML with CPU only (for debugging precision issues)
    pub fn coremlSafe() ExecutionProvider {
        return .{ .coreml = .{ .compute_units = .cpu_only } };
    }

    /// CoreML with all compute units (maximum performance)
    pub fn coremlPerformance() ExecutionProvider {
        return .{ .coreml = .{ .compute_units = .all } };
    }

    /// CUDA provider with device ID
    pub fn cudaProvider(device_id: u32) ExecutionProvider {
        return .{ .cuda = .{ .device_id = device_id } };
    }

    /// Auto-detect best provider
    pub fn autoProvider() ExecutionProvider {
        return .{ .auto = {} };
    }

    /// Get the display name for this provider
    pub fn getName(self: ExecutionProvider) []const u8 {
        return switch (self) {
            .cpu => "CPU",
            .coreml => |opts| switch (opts.compute_units) {
                .all => "CoreML:All",
                .cpu_only => "CoreML:CPUOnly",
                .cpu_and_gpu => "CoreML:CPU+GPU",
                .cpu_and_neural_engine => "CoreML:CPU+ANE",
            },
            .cuda => "CUDA",
            .auto => "Auto",
        };
    }

    /// Apply this provider to session options
    pub fn apply(self: ExecutionProvider, opts: *c_api.OrtSessionOptions) !void {
        const resolved = self.resolve();
        switch (resolved) {
            .cpu => {
                // CPU is the default, nothing to configure
            },
            .coreml => |coreml_opts| {
                // CoreML is only available when coreml_enabled is true at compile time
                if (comptime !c_api.coreml_enabled) {
                    return error.ProviderNotAvailable;
                }

                // Build CoreML flags
                var flags: u32 = 0;

                // Model format
                if (coreml_opts.model_format == .ml_program) {
                    flags |= c_api.CoreMLFlags.CREATE_MLPROGRAM;
                }

                // Compute units
                switch (coreml_opts.compute_units) {
                    .cpu_only => flags |= c_api.CoreMLFlags.USE_CPU_ONLY,
                    .cpu_and_gpu => flags |= c_api.CoreMLFlags.USE_CPU_AND_GPU,
                    .cpu_and_neural_engine => {
                        // Default behavior uses CPU + ANE when available
                    },
                    .all => {
                        // Use everything available
                    },
                }

                // Static shapes can improve performance
                if (coreml_opts.require_static_input_shapes) {
                    flags |= c_api.CoreMLFlags.ONLY_ALLOW_STATIC_INPUT_SHAPES;
                }

                const status = c_api.OrtSessionOptionsAppendExecutionProvider_CoreML(opts, flags);
                if (status != null) {
                    const api = c_api.getApi() orelse return error.ApiNotAvailable;
                    api.ReleaseStatus.?(status);
                    return error.ProviderConfigurationFailed;
                }
            },
            .cuda => |cuda_opts| {
                // Check if using dynamic ONNX Runtime loading
                if (comptime c_api.dynamic_ort) {
                    // Dynamic loading mode - use the dynamically discovered CUDA provider
                    const cuda_fn = c_api.getDynamicCudaProvider() orelse {
                        // CUDA provider not available in the loaded runtime
                        return error.ProviderNotAvailable;
                    };
                    const status = cuda_fn(opts, @intCast(cuda_opts.device_id));
                    if (status != null) {
                        const api = c_api.getApi() orelse return error.ApiNotAvailable;
                        api.ReleaseStatus.?(status);
                        return error.ProviderConfigurationFailed;
                    }
                } else if (comptime c_api.cuda_enabled) {
                    // Static/linked mode with CUDA enabled at compile time
                    const status = c_api.OrtSessionOptionsAppendExecutionProvider_CUDA(opts, @intCast(cuda_opts.device_id));
                    if (status != null) {
                        const api = c_api.getApi() orelse return error.ApiNotAvailable;
                        api.ReleaseStatus.?(status);
                        return error.ProviderConfigurationFailed;
                    }
                } else {
                    // Static build without CUDA enabled
                    // To use CUDA, either:
                    // 1. Rebuild with: zig build -Dcuda=true
                    // 2. Rebuild with: zig build -Ddynamic-ort=true (auto-detects CUDA at runtime)
                    return error.ProviderNotAvailable;
                }
            },
            .auto => unreachable, // resolve() handles auto
        }
    }

    /// Resolve auto provider to concrete provider
    pub fn resolve(self: ExecutionProvider) ExecutionProvider {
        if (self != .auto) return self;

        // Platform-specific auto-detection
        if (comptime builtin.os.tag == .macos) {
            // On macOS, use CoreML if enabled at compile time
            if (comptime c_api.coreml_enabled) {
                return ExecutionProvider.coremlProvider();
            }
        }

        // Check for CUDA availability
        if (comptime c_api.dynamic_ort) {
            // Dynamic loading mode - check if CUDA was detected at runtime
            if (c_api.isDynamicCudaLoaded()) {
                return ExecutionProvider.cudaProvider(0);
            }
        } else if (comptime c_api.cuda_enabled) {
            // Static/linked mode with CUDA enabled at compile time
            return ExecutionProvider.cudaProvider(0);
        }

        // Fall back to CPU
        return ExecutionProvider.cpuProvider();
    }
};

pub const ProviderError = error{
    ApiNotAvailable,
    ProviderConfigurationFailed,
    ProviderNotAvailable,
};

test "provider names" {
    try std.testing.expectEqualStrings("CPU", ExecutionProvider.cpuProvider().getName());
    try std.testing.expectEqualStrings("CoreML:CPU+ANE", ExecutionProvider.coremlProvider().getName());
    try std.testing.expectEqualStrings("CoreML:CPUOnly", ExecutionProvider.coremlSafe().getName());
    try std.testing.expectEqualStrings("CoreML:All", ExecutionProvider.coremlPerformance().getName());
}

test "auto resolve on macos" {
    if (comptime builtin.os.tag == .macos) {
        const resolved = ExecutionProvider.autoProvider().resolve();
        if (comptime c_api.coreml_enabled) {
            try std.testing.expect(resolved == .coreml);
        } else {
            // CoreML not enabled, should fall back to CPU
            try std.testing.expect(resolved == .cpu);
        }
    }
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "CoreMLComputeUnits enum completeness" {
    const units = [_]CoreMLComputeUnits{
        .all,
        .cpu_only,
        .cpu_and_gpu,
        .cpu_and_neural_engine,
    };
    try std.testing.expectEqual(@as(usize, 4), units.len);
}

test "CoreMLModelFormat enum" {
    const formats = [_]CoreMLModelFormat{
        .neural_network,
        .ml_program,
    };
    try std.testing.expectEqual(@as(usize, 2), formats.len);
}

test "CoreMLOptions defaults" {
    const opts = CoreMLOptions{};
    try std.testing.expectEqual(CoreMLModelFormat.ml_program, opts.model_format);
    try std.testing.expectEqual(CoreMLComputeUnits.cpu_and_neural_engine, opts.compute_units);
    try std.testing.expectEqual(false, opts.allow_low_precision_accumulation_on_gpu);
    try std.testing.expectEqual(false, opts.require_static_input_shapes);
}

test "CoreMLOptions custom values" {
    const opts = CoreMLOptions{
        .model_format = .neural_network,
        .compute_units = .all,
        .allow_low_precision_accumulation_on_gpu = true,
        .require_static_input_shapes = true,
    };
    try std.testing.expectEqual(CoreMLModelFormat.neural_network, opts.model_format);
    try std.testing.expectEqual(CoreMLComputeUnits.all, opts.compute_units);
    try std.testing.expectEqual(true, opts.allow_low_precision_accumulation_on_gpu);
    try std.testing.expectEqual(true, opts.require_static_input_shapes);
}

test "ExecutionProvider union types" {
    // CPU
    const cpu = ExecutionProvider{ .cpu = {} };
    try std.testing.expect(cpu == .cpu);

    // CoreML
    const coreml = ExecutionProvider{ .coreml = .{} };
    try std.testing.expect(coreml == .coreml);

    // CUDA
    const cuda = ExecutionProvider{ .cuda = .{ .device_id = 0 } };
    try std.testing.expect(cuda == .cuda);

    // Auto
    const auto_prov = ExecutionProvider{ .auto = {} };
    try std.testing.expect(auto_prov == .auto);
}

test "ExecutionProvider factory methods" {
    // cpuProvider
    const cpu = ExecutionProvider.cpuProvider();
    try std.testing.expect(cpu == .cpu);

    // coremlProvider
    const coreml = ExecutionProvider.coremlProvider();
    try std.testing.expect(coreml == .coreml);

    // coremlSafe
    const coreml_safe = ExecutionProvider.coremlSafe();
    try std.testing.expect(coreml_safe == .coreml);
    try std.testing.expectEqual(CoreMLComputeUnits.cpu_only, coreml_safe.coreml.compute_units);

    // coremlPerformance
    const coreml_perf = ExecutionProvider.coremlPerformance();
    try std.testing.expect(coreml_perf == .coreml);
    try std.testing.expectEqual(CoreMLComputeUnits.all, coreml_perf.coreml.compute_units);

    // cudaProvider
    const cuda = ExecutionProvider.cudaProvider(1);
    try std.testing.expect(cuda == .cuda);
    try std.testing.expectEqual(@as(u32, 1), cuda.cuda.device_id);

    // autoProvider
    const auto_prov = ExecutionProvider.autoProvider();
    try std.testing.expect(auto_prov == .auto);
}

test "coremlWithOptions" {
    const opts = CoreMLOptions{
        .compute_units = .cpu_and_gpu,
        .model_format = .neural_network,
    };
    const provider = ExecutionProvider.coremlWithOptions(opts);
    try std.testing.expect(provider == .coreml);
    try std.testing.expectEqual(CoreMLComputeUnits.cpu_and_gpu, provider.coreml.compute_units);
    try std.testing.expectEqual(CoreMLModelFormat.neural_network, provider.coreml.model_format);
}

test "getName for all providers" {
    try std.testing.expectEqualStrings("CPU", ExecutionProvider.cpuProvider().getName());
    try std.testing.expectEqualStrings("CUDA", ExecutionProvider.cudaProvider(0).getName());
    try std.testing.expectEqualStrings("Auto", ExecutionProvider.autoProvider().getName());
}

test "getName for CoreML compute units" {
    // All compute units
    const all = ExecutionProvider{ .coreml = .{ .compute_units = .all } };
    try std.testing.expectEqualStrings("CoreML:All", all.getName());

    // CPU only
    const cpu_only = ExecutionProvider{ .coreml = .{ .compute_units = .cpu_only } };
    try std.testing.expectEqualStrings("CoreML:CPUOnly", cpu_only.getName());

    // CPU + GPU
    const cpu_gpu = ExecutionProvider{ .coreml = .{ .compute_units = .cpu_and_gpu } };
    try std.testing.expectEqualStrings("CoreML:CPU+GPU", cpu_gpu.getName());

    // CPU + ANE
    const cpu_ane = ExecutionProvider{ .coreml = .{ .compute_units = .cpu_and_neural_engine } };
    try std.testing.expectEqualStrings("CoreML:CPU+ANE", cpu_ane.getName());
}

test "resolve non-auto returns self" {
    const cpu = ExecutionProvider.cpuProvider();
    try std.testing.expectEqual(cpu, cpu.resolve());

    const coreml = ExecutionProvider.coremlProvider();
    try std.testing.expectEqual(coreml, coreml.resolve());

    const cuda = ExecutionProvider.cudaProvider(0);
    try std.testing.expectEqual(cuda, cuda.resolve());
}

test "resolve auto returns platform-specific provider" {
    const auto_prov = ExecutionProvider.autoProvider();
    const resolved = auto_prov.resolve();

    // On macOS, should resolve to CoreML if enabled
    if (comptime builtin.os.tag == .macos) {
        if (comptime c_api.coreml_enabled) {
            try std.testing.expect(resolved == .coreml);
        } else {
            try std.testing.expect(resolved == .cpu);
        }
    } else {
        // On other platforms, should resolve to CPU (unless CUDA enabled)
        try std.testing.expect(resolved == .cpu or resolved == .cuda);
    }
}

test "ProviderError enum" {
    const errors = [_]ProviderError{
        ProviderError.ApiNotAvailable,
        ProviderError.ProviderConfigurationFailed,
        ProviderError.ProviderNotAvailable,
    };
    try std.testing.expectEqual(@as(usize, 3), errors.len);
}

test "ExecutionProvider struct size" {
    const size = @sizeOf(ExecutionProvider);
    try std.testing.expect(size > 0);
    // Union should be compact
    try std.testing.expect(size < 64);
}

test "CoreMLOptions struct size" {
    const size = @sizeOf(CoreMLOptions);
    try std.testing.expect(size > 0);
    // Options should be compact
    try std.testing.expect(size < 32);
}

test "CUDA provider with different device IDs" {
    for (0..4) |i| {
        const cuda = ExecutionProvider.cudaProvider(@intCast(i));
        try std.testing.expectEqual(@as(u32, @intCast(i)), cuda.cuda.device_id);
    }
}
