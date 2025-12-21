//! ONNX Runtime Session for inference
//!
//! Simplified session wrapper focused on embedding model inference.
//! Uses onnxruntime-zig's generic types with fastembed's dynamic-loading c_api.

const std = @import("std");
const c_api = @import("c_api.zig");
const ort = @import("onnxruntime-zig");

// Instantiate onnxruntime-zig types with fastembed's c_api (supports dynamic loading)
const ORT = ort.OnnxRuntime(c_api);

// Re-export execution provider types from onnxruntime-zig
pub const ExecutionProvider = ORT.ExecutionProvider;
pub const CoreMLOptions = ort.CoreMLOptions;
pub const CoreMLComputeUnits = ort.CoreMLComputeUnits;

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

/// ONNX Runtime Environment - uses onnxruntime-zig's Environment with fastembed's c_api
pub const Environment = ORT.Environment;

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
        /// Number of threads for intra-op parallelism (0 = use all cores)
        intra_op_num_threads: u32 = 0,
        /// Number of threads for inter-op parallelism (0 = use all cores)
        inter_op_num_threads: u32 = 0,
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

        // Configure execution provider
        const exec_provider = options.execution_provider.resolve();
        var actual_provider: ExecutionProvider = exec_provider;
        exec_provider.apply(opts.?) catch |err| {
            // If provider fails, fall back to CPU
            std.log.warn("Failed to configure {s} provider: {}, falling back to CPU", .{ options.execution_provider.getName(), err });
            actual_provider = ExecutionProvider.cpuProvider();
        };
        std.log.info("Using {s} execution provider", .{actual_provider.getName()});

        // Set graph optimization level
        // For CoreML, we must disable graph optimization to avoid errors with
        // initializer handling during graph partitioning. Without this, models
        // like Granite fail with "model_path must not be empty" during session
        // creation. The issue is that ONNX RT's optimizer modifies the graph in
        // ways that confuse the CoreML provider's partitioning logic.
        const opt_level: c_uint = switch (actual_provider) {
            .coreml => 0, // ORT_DISABLE_ALL - required for CoreML compatibility
            else => 99, // ORT_ENABLE_ALL for maximum CPU performance
        };
        status = api.SetSessionGraphOptimizationLevel.?(opts.?, opt_level);
        if (status != null) {
            api.ReleaseStatus.?(status);
        }

        // Set thread counts for CPU parallelism
        // intra_op: threads for parallel ops within a single node
        // inter_op: threads for parallel execution of independent nodes
        if (options.intra_op_num_threads > 0) {
            status = api.SetIntraOpNumThreads.?(opts.?, @intCast(options.intra_op_num_threads));
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        }
        if (options.inter_op_num_threads > 0) {
            status = api.SetInterOpNumThreads.?(opts.?, @intCast(options.inter_op_num_threads));
            if (status != null) {
                api.ReleaseStatus.?(status);
            }
        }

        // Create session
        var session: ?*c_api.OrtSession = null;
        status = api.CreateSession.?(env.ptr, model_path.ptr, opts.?, &session);
        if (status != null) {
            // Print the actual error message from ONNX Runtime
            const msg = api.GetErrorMessage.?(status);
            if (msg != null) {
                std.debug.print("ONNX Session error: {s}\n", .{msg});
            }
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

    // =========================================================================
    // Async Inference Support
    // =========================================================================

    /// Callback type for async inference completion
    pub const AsyncCallback = *const fn (
        user_data: ?*anyopaque,
        result: AsyncResult,
    ) void;

    /// Result of async inference
    pub const AsyncResult = union(enum) {
        /// Successful inference with output data (caller must free)
        success: []f32,
        /// Inference failed with error
        err: OnnxError,
    };

    /// Context passed through C callback for async inference
    const AsyncContext = struct {
        callback: AsyncCallback,
        user_data: ?*anyopaque,
        api: *const c_api.OrtApi,
        allocator: std.mem.Allocator,
        // Input tensors that must outlive the async call
        input_ids_tensor: *c_api.OrtValue,
        attention_mask_tensor: *c_api.OrtValue,
        token_type_tensor: ?*c_api.OrtValue,
        memory_info: *c_api.OrtMemoryInfo,
        // Input/output name pointers must also outlive the async call
        input_name_ptrs: [3][*c]const u8,
        output_name_ptrs: [1][*c]const u8,
        inputs: [3]*c_api.OrtValue,
        num_inputs: usize,
    };

    /// C-compatible trampoline that converts to Zig callback
    fn asyncTrampoline(
        ctx_ptr: ?*anyopaque,
        outputs: [*c]?*c_api.OrtValue,
        num_outputs: usize,
        status: ?*c_api.OrtStatus,
    ) callconv(.c) void {
        const ctx: *AsyncContext = @ptrCast(@alignCast(ctx_ptr));
        const api = ctx.api;
        defer {
            // Free input tensors and memory info
            api.ReleaseValue.?(ctx.input_ids_tensor);
            api.ReleaseValue.?(ctx.attention_mask_tensor);
            if (ctx.token_type_tensor) |t| api.ReleaseValue.?(t);
            api.ReleaseMemoryInfo.?(ctx.memory_info);
            ctx.allocator.destroy(ctx);
        }

        // Check for errors
        if (status) |s| {
            api.ReleaseStatus.?(s);
            ctx.callback(ctx.user_data, .{ .err = OnnxError.InferenceFailed });
            return;
        }

        if (num_outputs == 0 or outputs[0] == null) {
            ctx.callback(ctx.user_data, .{ .err = OnnxError.InferenceFailed });
            return;
        }

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
        const result = ctx.allocator.alloc(f32, total_elements) catch {
            ctx.callback(ctx.user_data, .{ .err = OnnxError.OutOfMemory });
            // Release output
            api.ReleaseValue.?(outputs[0].?);
            return;
        };
        const output_ptr: [*]f32 = @ptrCast(output_data.?);
        @memcpy(result, output_ptr[0..total_elements]);

        // Release output tensor
        api.ReleaseValue.?(outputs[0].?);

        ctx.callback(ctx.user_data, .{ .success = result });
    }

    /// Run inference asynchronously
    ///
    /// The callback will be invoked when inference completes (or fails).
    /// On success, caller is responsible for freeing the returned f32 slice.
    pub fn runAsync(
        self: *Self,
        input_ids: []const i64,
        attention_mask: []const i64,
        token_type_ids: ?[]const i64,
        batch_size: usize,
        seq_len: usize,
        callback: AsyncCallback,
        user_data: ?*anyopaque,
    ) OnnxError!void {
        const api = self.api;

        // Allocate context (freed in trampoline)
        const ctx = self.zig_allocator.create(AsyncContext) catch return OnnxError.OutOfMemory;
        errdefer self.zig_allocator.destroy(ctx);

        // Create memory info for CPU
        var memory_info: ?*c_api.OrtMemoryInfo = null;
        var status = api.CreateCpuMemoryInfo.?(0, 0, &memory_info);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return OnnxError.TensorCreationFailed;
        }
        errdefer api.ReleaseMemoryInfo.?(memory_info.?);

        // Shape for input tensors [batch_size, seq_len]
        var shape = [_]i64{ @intCast(batch_size), @intCast(seq_len) };

        // Create input tensors (must outlive async call)
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
        errdefer api.ReleaseValue.?(input_ids_tensor.?);

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
        errdefer api.ReleaseValue.?(attention_mask_tensor.?);

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
        }
        errdefer if (token_type_tensor) |t| api.ReleaseValue.?(t);

        // Prepare output array (on stack, not passed to RunAsync)
        var outputs: [1]?*c_api.OrtValue = .{null};

        // Fill context with all arrays that must outlive the async call
        ctx.* = .{
            .callback = callback,
            .user_data = user_data,
            .api = api,
            .allocator = self.zig_allocator,
            .input_ids_tensor = input_ids_tensor.?,
            .attention_mask_tensor = attention_mask_tensor.?,
            .token_type_tensor = token_type_tensor,
            .memory_info = memory_info.?,
            .input_name_ptrs = undefined,
            .output_name_ptrs = undefined,
            .inputs = undefined,
            .num_inputs = 2,
        };

        // Set up input arrays in context (must persist until callback)
        ctx.inputs[0] = input_ids_tensor.?;
        ctx.inputs[1] = attention_mask_tensor.?;
        ctx.input_name_ptrs[0] = self.input_names.items[0].ptr;
        ctx.input_name_ptrs[1] = self.input_names.items[1].ptr;

        if (token_type_tensor != null and self.input_names.items.len > 2) {
            ctx.inputs[2] = token_type_tensor.?;
            ctx.input_name_ptrs[2] = self.input_names.items[2].ptr;
            ctx.num_inputs = 3;
        }

        // Set up output names in context
        ctx.output_name_ptrs[0] = self.output_names.items[0].ptr;

        // Call RunAsync with pointers to context arrays
        status = api.RunAsync.?(
            self.ptr,
            null, // run options
            &ctx.input_name_ptrs,
            @ptrCast(&ctx.inputs),
            ctx.num_inputs,
            &ctx.output_name_ptrs,
            1,
            @ptrCast(&outputs),
            asyncTrampoline,
            ctx,
        );
        if (status != null) {
            api.ReleaseStatus.?(status);
            // Context will be freed by errdefers
            return OnnxError.InferenceFailed;
        }
        // On success, tensors will be freed in trampoline after async completion
    }
};

test "Environment creation" {
    var env = try Environment.init(.{});
    defer env.deinit();
}

// =============================================================================
// COMPREHENSIVE TESTS
// =============================================================================

test "OnnxError enum completeness" {
    const errors = [_]OnnxError{
        OnnxError.ApiNotAvailable,
        OnnxError.EnvironmentCreationFailed,
        OnnxError.SessionCreationFailed,
        OnnxError.SessionOptionsCreationFailed,
        OnnxError.TensorCreationFailed,
        OnnxError.InferenceFailed,
        OnnxError.InvalidShape,
        OnnxError.AllocatorError,
        OnnxError.OutOfMemory,
        OnnxError.ProviderConfigurationFailed,
        OnnxError.ProviderNotAvailable,
    };

    try std.testing.expectEqual(@as(usize, 11), errors.len);
}

test "Environment struct size" {
    const size = @sizeOf(Environment);
    try std.testing.expect(size > 0);
    try std.testing.expect(size < 1024); // Should be small (just pointers)
}

test "Environment has expected fields" {
    try std.testing.expect(@hasField(Environment, "ptr"));
    try std.testing.expect(@hasField(Environment, "api"));
}

test "Session struct has expected fields" {
    try std.testing.expect(@hasField(Session, "ptr"));
    try std.testing.expect(@hasField(Session, "api"));
    try std.testing.expect(@hasField(Session, "allocator"));
    try std.testing.expect(@hasField(Session, "input_names"));
    try std.testing.expect(@hasField(Session, "output_names"));
    try std.testing.expect(@hasField(Session, "zig_allocator"));
    try std.testing.expect(@hasField(Session, "execution_provider"));
}

test "Session.InitOptions default values" {
    const opts = Session.InitOptions{};
    try std.testing.expectEqual(ExecutionProvider{ .cpu = {} }, opts.execution_provider);
}

test "Session.InitOptions with CoreML" {
    const opts = Session.InitOptions{
        .execution_provider = .{ .coreml = .{} },
    };
    _ = opts;
    try std.testing.expect(true);
}

test "Session.InitOptions with CoreML custom options" {
    const opts = Session.InitOptions{
        .execution_provider = .{
            .coreml = .{
                .compute_units = .cpu_and_neural_engine,
                .require_static_input_shapes = true,
            },
        },
    };
    _ = opts;
    try std.testing.expect(true);
}

test "ExecutionProvider re-export" {
    // Verify re-exported types are accessible
    _ = ExecutionProvider{ .cpu = {} };
    _ = ExecutionProvider{ .coreml = .{} };
    try std.testing.expect(true);
}

test "CoreMLOptions re-export" {
    const opts = CoreMLOptions{
        .compute_units = .all,
    };
    _ = opts;
    try std.testing.expect(true);
}

test "CoreMLComputeUnits re-export" {
    _ = CoreMLComputeUnits.all;
    _ = CoreMLComputeUnits.cpu_only;
    _ = CoreMLComputeUnits.cpu_and_gpu;
    _ = CoreMLComputeUnits.cpu_and_neural_engine;
    try std.testing.expect(true);
}

test "Environment init and deinit multiple times" {
    // Test that we can create and destroy multiple environments
    for (0..3) |_| {
        var env = try Environment.init(.{});
        env.deinit();
    }
}

test "Session struct size is reasonable" {
    const size = @sizeOf(Session);
    try std.testing.expect(size > 0);
    // Session should be reasonable size (contains ArrayLists, allocator, etc.)
    try std.testing.expect(size < 4096);
}

test "Session.InitOptions thread configuration" {
    const opts = Session.InitOptions{
        .intra_op_num_threads = 4,
        .inter_op_num_threads = 2,
    };

    try std.testing.expectEqual(@as(u32, 4), opts.intra_op_num_threads);
    try std.testing.expectEqual(@as(u32, 2), opts.inter_op_num_threads);
}

test "Session.InitOptions all fields" {
    const opts = Session.InitOptions{
        .execution_provider = .{ .cpu = {} },
        .intra_op_num_threads = 8,
        .inter_op_num_threads = 4,
    };

    try std.testing.expectEqual(@as(u32, 8), opts.intra_op_num_threads);
    try std.testing.expectEqual(@as(u32, 4), opts.inter_op_num_threads);
}
