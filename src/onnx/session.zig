//! ONNX Runtime Session for inference
//!
//! Simplified session wrapper focused on embedding model inference.

const std = @import("std");
const c_api = @import("c_api.zig");
const provider = @import("provider.zig");

pub const ExecutionProvider = provider.ExecutionProvider;
pub const CoreMLOptions = provider.CoreMLOptions;
pub const CoreMLComputeUnits = provider.CoreMLComputeUnits;

/// Error type for ONNX operations
pub const OnnxError = error{
    ApiNotAvailable,
    EnvironmentCreationFailed,
    SessionCreationFailed,
    SessionOptionsCreationFailed,
    TensorCreationFailed,
    InferenceFailed,
    InvalidShape,
    AllocatorError,
    OutOfMemory,
    ProviderConfigurationFailed,
    ProviderNotAvailable,
};

/// ONNX Runtime Environment (one per application)
pub const Environment = struct {
    ptr: *c_api.OrtEnv,
    api: *const c_api.OrtApi,

    const Self = @This();

    pub fn init() OnnxError!Self {
        const api = c_api.getApi() orelse return OnnxError.ApiNotAvailable;

        var env: ?*c_api.OrtEnv = null;
        const status = api.CreateEnv.?(
            c_api.LoggingLevel.warning.toC(),
            "fastembed",
            &env,
        );

        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.EnvironmentCreationFailed;
        }

        return Self{
            .ptr = env.?,
            .api = api,
        };
    }

    pub fn deinit(self: *Self) void {
        self.api.ReleaseEnv.?(self.ptr);
    }
};

/// ONNX Runtime Session for running inference
pub const Session = struct {
    ptr: *c_api.OrtSession,
    api: *const c_api.OrtApi,
    allocator: *c_api.OrtAllocator,
    input_names: std.ArrayList([:0]const u8),
    output_names: std.ArrayList([:0]const u8),
    zig_allocator: std.mem.Allocator,
    execution_provider: ExecutionProvider,

    const Self = @This();

    /// Session initialization options
    pub const InitOptions = struct {
        /// Execution provider to use (default: CPU)
        execution_provider: ExecutionProvider = .{ .cpu = {} },
    };

    /// Load a model from file with default options (CPU provider)
    pub fn init(env: Environment, model_path: [:0]const u8, allocator: std.mem.Allocator) OnnxError!Self {
        return initWithOptions(env, model_path, allocator, .{});
    }

    /// Load a model from file with custom options
    pub fn initWithOptions(env: Environment, model_path: [:0]const u8, allocator: std.mem.Allocator, options: InitOptions) OnnxError!Self {
        const api = env.api;

        // Create session options
        var opts: ?*c_api.OrtSessionOptions = null;
        var status = api.CreateSessionOptions.?(&opts);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.SessionOptionsCreationFailed;
        }
        defer api.ReleaseSessionOptions.?(opts.?);

        // Set graph optimization level
        status = api.SetSessionGraphOptimizationLevel.?(opts.?, 99); // ORT_ENABLE_ALL
        if (status != null) {
            api.ReleaseStatus.?(status);
        }

        // Configure execution provider
        const exec_provider = options.execution_provider.resolve();
        exec_provider.apply(opts.?) catch |err| {
            // If provider fails, fall back to CPU
            std.log.warn("Failed to configure {s} provider: {}, falling back to CPU", .{ options.execution_provider.getName(), err });
        };
        std.log.info("Using {s} execution provider", .{exec_provider.getName()});

        // Create session
        var session: ?*c_api.OrtSession = null;
        status = api.CreateSession.?(env.ptr, model_path.ptr, opts.?, &session);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.SessionCreationFailed;
        }

        // Get allocator
        var ort_allocator: ?*c_api.OrtAllocator = null;
        status = api.GetAllocatorWithDefaultOptions.?(&ort_allocator);
        if (status != null) {
            api.ReleaseStatus.?(status);
            api.ReleaseSession.?(session.?);
            return OnnxError.AllocatorError;
        }

        // Get input names
        var input_count: usize = 0;
        _ = api.SessionGetInputCount.?(session.?, &input_count);

        var input_names = std.ArrayList([:0]const u8).initCapacity(allocator, input_count) catch return OnnxError.OutOfMemory;
        errdefer {
            for (input_names.items) |name| allocator.free(name);
            input_names.deinit(allocator);
        }

        for (0..input_count) |i| {
            var name_ptr: [*c]u8 = undefined;
            _ = api.SessionGetInputName.?(session.?, i, ort_allocator.?, &name_ptr);
            const name_len = std.mem.len(name_ptr);
            const name_copy = allocator.allocSentinel(u8, name_len, 0) catch return OnnxError.OutOfMemory;
            @memcpy(name_copy, name_ptr[0..name_len]);
            _ = api.AllocatorFree.?(ort_allocator.?, name_ptr);
            input_names.append(allocator, name_copy) catch return OnnxError.OutOfMemory;
        }

        // Get output names
        var output_count: usize = 0;
        _ = api.SessionGetOutputCount.?(session.?, &output_count);

        var output_names = std.ArrayList([:0]const u8).initCapacity(allocator, output_count) catch return OnnxError.OutOfMemory;
        errdefer {
            for (output_names.items) |name| allocator.free(name);
            output_names.deinit(allocator);
        }

        for (0..output_count) |i| {
            var name_ptr: [*c]u8 = undefined;
            _ = api.SessionGetOutputName.?(session.?, i, ort_allocator.?, &name_ptr);
            const name_len = std.mem.len(name_ptr);
            const name_copy = allocator.allocSentinel(u8, name_len, 0) catch return OnnxError.OutOfMemory;
            @memcpy(name_copy, name_ptr[0..name_len]);
            _ = api.AllocatorFree.?(ort_allocator.?, name_ptr);
            output_names.append(allocator, name_copy) catch return OnnxError.OutOfMemory;
        }

        return Self{
            .ptr = session.?,
            .api = api,
            .allocator = ort_allocator.?,
            .input_names = input_names,
            .output_names = output_names,
            .zig_allocator = allocator,
            .execution_provider = exec_provider,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.input_names.items) |name| self.zig_allocator.free(name);
        self.input_names.deinit(self.zig_allocator);
        for (self.output_names.items) |name| self.zig_allocator.free(name);
        self.output_names.deinit(self.zig_allocator);
        self.api.ReleaseSession.?(self.ptr);
    }

    /// Run inference with input tensors
    /// Returns output tensor data (caller must free the returned slice)
    /// If output_name is provided, use that specific output; otherwise use the first output.
    pub fn run(
        self: *Self,
        input_ids: []const i64,
        attention_mask: []const i64,
        token_type_ids: ?[]const i64,
        batch_size: usize,
        seq_len: usize,
        output_name: ?[]const u8,
    ) OnnxError![]f32 {
        const api = self.api;

        // Create memory info for CPU
        var memory_info: ?*c_api.OrtMemoryInfo = null;
        var status = api.CreateCpuMemoryInfo.?(0, 0, &memory_info); // OrtDeviceAllocator, OrtMemTypeDefault
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.TensorCreationFailed;
        }
        defer api.ReleaseMemoryInfo.?(memory_info.?);

        // Shape for input tensors [batch_size, seq_len]
        var shape = [_]i64{ @intCast(batch_size), @intCast(seq_len) };

        // Create input tensors
        var input_ids_tensor: ?*c_api.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info.?,
            @ptrCast(@constCast(input_ids.ptr)),
            input_ids.len * @sizeOf(i64),
            &shape,
            2,
            c_api.TensorElementType.int64.toC(),
            &input_ids_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.TensorCreationFailed;
        }
        defer api.ReleaseValue.?(input_ids_tensor.?);

        var attention_mask_tensor: ?*c_api.OrtValue = null;
        status = api.CreateTensorWithDataAsOrtValue.?(
            memory_info.?,
            @ptrCast(@constCast(attention_mask.ptr)),
            attention_mask.len * @sizeOf(i64),
            &shape,
            2,
            c_api.TensorElementType.int64.toC(),
            &attention_mask_tensor,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.TensorCreationFailed;
        }
        defer api.ReleaseValue.?(attention_mask_tensor.?);

        // Prepare inputs array
        var inputs: [3]*c_api.OrtValue = undefined;
        var input_name_ptrs: [3][*c]const u8 = undefined;
        var num_inputs: usize = 2;

        inputs[0] = input_ids_tensor.?;
        inputs[1] = attention_mask_tensor.?;
        input_name_ptrs[0] = self.input_names.items[0].ptr;
        input_name_ptrs[1] = self.input_names.items[1].ptr;

        // Optional token_type_ids
        var token_type_tensor: ?*c_api.OrtValue = null;
        if (token_type_ids) |tti| {
            status = api.CreateTensorWithDataAsOrtValue.?(
                memory_info.?,
                @ptrCast(@constCast(tti.ptr)),
                tti.len * @sizeOf(i64),
                &shape,
                2,
                c_api.TensorElementType.int64.toC(),
                &token_type_tensor,
            );
            if (status != null) {
                api.ReleaseStatus.?(status);
                return OnnxError.TensorCreationFailed;
            }
            inputs[2] = token_type_tensor.?;
            if (self.input_names.items.len > 2) {
                input_name_ptrs[2] = self.input_names.items[2].ptr;
                num_inputs = 3;
            }
        }
        defer if (token_type_tensor) |t| api.ReleaseValue.?(t);

        // Prepare output - find the requested output by name or use first
        var output_name_ptr: [*c]const u8 = self.output_names.items[0].ptr;
        if (output_name) |name| {
            for (self.output_names.items) |stored_name| {
                if (std.mem.eql(u8, stored_name, name)) {
                    output_name_ptr = stored_name.ptr;
                    break;
                }
            }
        }
        var output_name_ptrs: [1][*c]const u8 = .{output_name_ptr};
        var outputs: [1]?*c_api.OrtValue = .{null};

        // Run inference
        status = api.Run.?(
            self.ptr,
            null, // run options
            &input_name_ptrs,
            @ptrCast(&inputs),
            num_inputs,
            &output_name_ptrs,
            1,
            @ptrCast(&outputs),
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.InferenceFailed;
        }
        defer api.ReleaseValue.?(outputs[0].?);

        // Get output tensor info
        var type_info: ?*c_api.OrtTensorTypeAndShapeInfo = null;
        _ = api.GetTensorTypeAndShape.?(outputs[0].?, &type_info);
        defer api.ReleaseTensorTypeAndShapeInfo.?(type_info.?);

        var num_dims: usize = 0;
        _ = api.GetDimensionsCount.?(type_info.?, &num_dims);

        var dims: [4]i64 = undefined;
        _ = api.GetDimensions.?(type_info.?, &dims, num_dims);

        // Calculate total elements
        var total_elements: usize = 1;
        for (0..num_dims) |i| {
            total_elements *= @intCast(dims[i]);
        }

        // Get output data
        var output_data: ?*f32 = null;
        _ = api.GetTensorMutableData.?(outputs[0].?, @ptrCast(&output_data));

        // Copy to owned memory
        const result = self.zig_allocator.alloc(f32, total_elements) catch return OnnxError.OutOfMemory;
        const output_ptr: [*]f32 = @ptrCast(output_data.?);
        @memcpy(result, output_ptr[0..total_elements]);

        return result;
    }

    pub fn getInputCount(self: Self) usize {
        return self.input_names.items.len;
    }

    pub fn getOutputCount(self: Self) usize {
        return self.output_names.items.len;
    }
};

test "Environment creation" {
    var env = try Environment.init();
    defer env.deinit();
}
