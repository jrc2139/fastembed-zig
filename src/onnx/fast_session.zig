//! Zero-Allocation ONNX Session using IoBinding
//!
//! FastSession provides minimal-allocation inference by pre-allocating buffers
//! and using ONNX Runtime's IoBinding API for efficient tensor binding.
//!
//! After initialization:
//! - Input/output buffers are pre-allocated
//! - Tensors are created at runtime with actual shapes (minimal overhead)
//! - Data buffers are reused across runs (zero allocation for data)
//!
//! This is ideal for embedding models where input shapes vary
//! and you want maximum throughput with minimal allocations.

const std = @import("std");
const c_api = @import("c_api.zig");
const session_mod = @import("session.zig");

pub const Environment = session_mod.Environment;
pub const Session = session_mod.Session;

/// Error type for fast session operations
pub const FastSessionError = error{
    ApiNotAvailable,
    IoBindingCreationFailed,
    BindInputFailed,
    BindOutputFailed,
    TensorCreationFailed,
    InferenceFailed,
    InvalidConfiguration,
    OutOfMemory,
};

/// Configuration for FastSession's pre-allocated buffers
pub const FastSessionConfig = struct {
    /// Maximum batch size (number of sequences)
    max_batch_size: usize = 32,
    /// Maximum sequence length (tokens per sequence)
    max_seq_len: usize = 512,
    /// Model hidden dimension (for output buffer sizing)
    hidden_dim: usize = 384,
    /// Whether the model outputs pooled embeddings directly
    output_is_pooled: bool = false,
    /// Whether the model uses token_type_ids input
    use_token_type_ids: bool = false,
};

/// IoBinding wrapper for efficient inference
pub const IoBinding = struct {
    ptr: *c_api.OrtIoBinding,
    api: *const c_api.OrtApi,

    const Self = @This();

    /// Create IoBinding from a session
    pub fn init(session: *const Session) FastSessionError!Self {
        const api = session.api;

        var binding: ?*c_api.OrtIoBinding = null;
        const status = api.CreateIoBinding.?(session.ptr, &binding);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return FastSessionError.IoBindingCreationFailed;
        }

        return Self{
            .ptr = binding.?,
            .api = api,
        };
    }

    pub fn deinit(self: *Self) void {
        self.api.ReleaseIoBinding.?(self.ptr);
    }

    /// Bind an input tensor by name
    pub fn bindInput(self: *Self, name: [:0]const u8, tensor: *c_api.OrtValue) FastSessionError!void {
        const status = self.api.BindInput.?(self.ptr, name.ptr, tensor);
        if (status != null) {
            self.api.ReleaseStatus.?(status);
            return FastSessionError.BindInputFailed;
        }
    }

    /// Bind an output tensor by name
    pub fn bindOutput(self: *Self, name: [:0]const u8, tensor: *c_api.OrtValue) FastSessionError!void {
        const status = self.api.BindOutput.?(self.ptr, name.ptr, tensor);
        if (status != null) {
            self.api.ReleaseStatus.?(status);
            return FastSessionError.BindOutputFailed;
        }
    }

    /// Bind output to device (ORT allocates output)
    pub fn bindOutputToDevice(self: *Self, name: [:0]const u8, memory_info: *c_api.OrtMemoryInfo) FastSessionError!void {
        const status = self.api.BindOutputToDevice.?(self.ptr, name.ptr, memory_info);
        if (status != null) {
            self.api.ReleaseStatus.?(status);
            return FastSessionError.BindOutputFailed;
        }
    }

    /// Run inference with pre-bound tensors
    pub fn run(self: *Self, session: *const Session, run_options: ?*c_api.OrtRunOptions) FastSessionError!void {
        const status = self.api.RunWithBinding.?(session.ptr, run_options, self.ptr);
        if (status != null) {
            // Get error message for debugging
            const msg = self.api.GetErrorMessage.?(status);
            if (msg != null) {
                std.debug.print("IoBinding run error: {s}\n", .{msg});
            }
            self.api.ReleaseStatus.?(status);
            return FastSessionError.InferenceFailed;
        }
    }

    /// Clear all bindings (for rebinding with different shapes)
    pub fn clearBindings(self: *Self) void {
        self.api.ClearBoundInputs.?(self.ptr);
        self.api.ClearBoundOutputs.?(self.ptr);
    }
};

/// FastSession wraps a Session with pre-allocated buffers and IoBinding
/// for minimal-allocation inference.
///
/// Buffers are pre-allocated at init for max batch/seq size.
/// At runtime, tensors are created with actual shapes but point to
/// the pre-allocated buffers, eliminating data allocation overhead.
pub const FastSession = struct {
    session: Session,
    io_binding: IoBinding,
    config: FastSessionConfig,
    api: *const c_api.OrtApi,
    allocator: std.mem.Allocator,

    // Pre-allocated input buffers
    input_ids: []i64,
    attention_mask: []i64,
    token_type_ids: ?[]i64,

    // Pre-allocated output buffer
    output_buffer: []f32,

    // Memory info for tensor creation
    memory_info: *c_api.OrtMemoryInfo,

    const Self = @This();

    /// Initialize FastSession with pre-allocated buffers
    pub fn init(
        env: Environment,
        model_path: [:0]const u8,
        allocator: std.mem.Allocator,
        config: FastSessionConfig,
        session_opts: Session.InitOptions,
    ) FastSessionError!Self {
        const api = env.api;

        // Create underlying session
        var session = Session.initWithOptions(env, model_path, allocator, session_opts) catch {
            return FastSessionError.InvalidConfiguration;
        };
        errdefer session.deinit();

        // Create IoBinding
        var io_binding = try IoBinding.init(&session);
        errdefer io_binding.deinit();

        // Create CPU memory info
        var memory_info: ?*c_api.OrtMemoryInfo = null;
        const status = api.CreateCpuMemoryInfo.?(0, 0, &memory_info);
        if (status != null) {
            api.ReleaseStatus.?(status);
            return FastSessionError.TensorCreationFailed;
        }
        errdefer api.ReleaseMemoryInfo.?(memory_info.?);

        const max_input_size = config.max_batch_size * config.max_seq_len;

        // Allocate input buffers
        const input_ids = allocator.alloc(i64, max_input_size) catch {
            return FastSessionError.OutOfMemory;
        };
        errdefer allocator.free(input_ids);

        const attention_mask = allocator.alloc(i64, max_input_size) catch {
            return FastSessionError.OutOfMemory;
        };
        errdefer allocator.free(attention_mask);

        var token_type_ids: ?[]i64 = null;
        if (config.use_token_type_ids) {
            token_type_ids = allocator.alloc(i64, max_input_size) catch {
                return FastSessionError.OutOfMemory;
            };
            @memset(token_type_ids.?, 0);
        }
        errdefer if (token_type_ids) |t| allocator.free(t);

        // Allocate output buffer
        const output_size = if (config.output_is_pooled)
            config.max_batch_size * config.hidden_dim
        else
            config.max_batch_size * config.max_seq_len * config.hidden_dim;

        const output_buffer = allocator.alloc(f32, output_size) catch {
            return FastSessionError.OutOfMemory;
        };
        errdefer allocator.free(output_buffer);

        return Self{
            .session = session,
            .io_binding = io_binding,
            .config = config,
            .api = api,
            .allocator = allocator,
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .token_type_ids = token_type_ids,
            .output_buffer = output_buffer,
            .memory_info = memory_info.?,
        };
    }

    pub fn deinit(self: *Self) void {
        self.api.ReleaseMemoryInfo.?(self.memory_info);

        self.allocator.free(self.output_buffer);
        if (self.token_type_ids) |t| self.allocator.free(t);
        self.allocator.free(self.attention_mask);
        self.allocator.free(self.input_ids);

        self.io_binding.deinit();
        self.session.deinit();
    }

    /// Run inference with dynamic tensor binding.
    ///
    /// Prerequisites:
    /// - Fill input_ids and attention_mask buffers with data
    /// - Call with actual batch_size and seq_len (must be <= config max)
    ///
    /// Returns slice of output_buffer containing [batch_size * hidden_dim] floats
    /// if output_is_pooled, or [batch_size * seq_len * hidden_dim] otherwise.
    pub fn run(self: *Self, batch_size: usize, seq_len: usize) FastSessionError![]f32 {
        if (batch_size > self.config.max_batch_size or seq_len > self.config.max_seq_len) {
            return FastSessionError.InvalidConfiguration;
        }

        // Clear any previous bindings
        self.io_binding.clearBindings();

        // Create input tensors with actual shape
        var shape = [_]i64{ @intCast(batch_size), @intCast(seq_len) };
        const input_size = batch_size * seq_len;

        // Create input_ids tensor
        var input_ids_tensor: ?*c_api.OrtValue = null;
        var status = self.api.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(self.input_ids.ptr),
            input_size * @sizeOf(i64),
            &shape,
            2,
            c_api.TensorElementType.int64.toC(),
            &input_ids_tensor,
        );
        if (status != null) {
            self.api.ReleaseStatus.?(status);
            return FastSessionError.TensorCreationFailed;
        }
        defer self.api.ReleaseValue.?(input_ids_tensor.?);

        // Create attention_mask tensor
        var attention_mask_tensor: ?*c_api.OrtValue = null;
        status = self.api.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(self.attention_mask.ptr),
            input_size * @sizeOf(i64),
            &shape,
            2,
            c_api.TensorElementType.int64.toC(),
            &attention_mask_tensor,
        );
        if (status != null) {
            self.api.ReleaseStatus.?(status);
            return FastSessionError.TensorCreationFailed;
        }
        defer self.api.ReleaseValue.?(attention_mask_tensor.?);

        // Create token_type_ids tensor if needed
        var token_type_ids_tensor: ?*c_api.OrtValue = null;
        if (self.config.use_token_type_ids) {
            status = self.api.CreateTensorWithDataAsOrtValue.?(
                self.memory_info,
                @ptrCast(self.token_type_ids.?.ptr),
                input_size * @sizeOf(i64),
                &shape,
                2,
                c_api.TensorElementType.int64.toC(),
                &token_type_ids_tensor,
            );
            if (status != null) {
                self.api.ReleaseStatus.?(status);
                return FastSessionError.TensorCreationFailed;
            }
        }
        defer if (token_type_ids_tensor) |t| self.api.ReleaseValue.?(t);

        // Create output tensor with actual shape
        var output_shape: [3]i64 = undefined;
        var output_shape_dims: usize = undefined;
        var output_size: usize = undefined;

        if (self.config.output_is_pooled) {
            output_shape[0] = @intCast(batch_size);
            output_shape[1] = @intCast(self.config.hidden_dim);
            output_shape_dims = 2;
            output_size = batch_size * self.config.hidden_dim;
        } else {
            output_shape[0] = @intCast(batch_size);
            output_shape[1] = @intCast(seq_len);
            output_shape[2] = @intCast(self.config.hidden_dim);
            output_shape_dims = 3;
            output_size = batch_size * seq_len * self.config.hidden_dim;
        }

        var output_tensor: ?*c_api.OrtValue = null;
        status = self.api.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(self.output_buffer.ptr),
            output_size * @sizeOf(f32),
            &output_shape,
            output_shape_dims,
            c_api.TensorElementType.float32.toC(),
            &output_tensor,
        );
        if (status != null) {
            self.api.ReleaseStatus.?(status);
            return FastSessionError.TensorCreationFailed;
        }
        defer self.api.ReleaseValue.?(output_tensor.?);

        // Bind inputs
        try self.io_binding.bindInput(self.session.input_names.items[0], input_ids_tensor.?);
        try self.io_binding.bindInput(self.session.input_names.items[1], attention_mask_tensor.?);
        if (self.config.use_token_type_ids and self.session.input_names.items.len > 2) {
            try self.io_binding.bindInput(self.session.input_names.items[2], token_type_ids_tensor.?);
        }

        // Bind output
        try self.io_binding.bindOutput(self.session.output_names.items[0], output_tensor.?);

        // Run inference via IoBinding
        try self.io_binding.run(&self.session, null);

        return self.output_buffer[0..output_size];
    }

    /// Get input buffer slices for the given batch/seq size
    pub fn getInputBuffers(self: *Self, batch_size: usize, seq_len: usize) struct {
        input_ids: []i64,
        attention_mask: []i64,
        token_type_ids: ?[]i64,
    } {
        const size = batch_size * seq_len;
        return .{
            .input_ids = self.input_ids[0..size],
            .attention_mask = self.attention_mask[0..size],
            .token_type_ids = if (self.token_type_ids) |t| t[0..size] else null,
        };
    }

    /// Get the underlying session for metadata access
    pub fn getSession(self: *const Self) *const Session {
        return &self.session;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "IoBinding struct size" {
    const size = @sizeOf(IoBinding);
    try std.testing.expect(size > 0);
    try std.testing.expect(size < 256);
}

test "FastSessionConfig defaults" {
    const config = FastSessionConfig{};
    try std.testing.expectEqual(@as(usize, 32), config.max_batch_size);
    try std.testing.expectEqual(@as(usize, 512), config.max_seq_len);
    try std.testing.expectEqual(@as(usize, 384), config.hidden_dim);
    try std.testing.expect(!config.output_is_pooled);
    try std.testing.expect(!config.use_token_type_ids);
}

test "FastSession struct has expected fields" {
    try std.testing.expect(@hasField(FastSession, "session"));
    try std.testing.expect(@hasField(FastSession, "io_binding"));
    try std.testing.expect(@hasField(FastSession, "input_ids"));
    try std.testing.expect(@hasField(FastSession, "attention_mask"));
    try std.testing.expect(@hasField(FastSession, "output_buffer"));
}
