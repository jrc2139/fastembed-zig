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
                _ = cuda_opts;
                // CUDA configuration would go here
                // For now, CUDA is not supported in this build
                return error.ProviderNotAvailable;
            },
            .auto => unreachable, // resolve() handles auto
        }
    }

    /// Resolve auto provider to concrete provider
    pub fn resolve(self: ExecutionProvider) ExecutionProvider {
        if (self != .auto) return self;

        // Platform-specific auto-detection
        if (comptime builtin.os.tag == .macos) {
            // On macOS, use CoreML with safe defaults
            return ExecutionProvider.coremlProvider();
        }

        // On other platforms, fall back to CPU
        // TODO: Add CUDA detection for Linux
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
        try std.testing.expect(resolved == .coreml);
    }
}
