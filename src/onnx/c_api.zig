//! Raw C bindings for ONNX Runtime C API
//!
//! This module provides direct access to the ONNX Runtime C API.
//! When dynamic_ort is enabled, the library is loaded at runtime via dlopen.
//! CoreML support is optional and controlled by the coreml_enabled build option.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const log = std.log.scoped(.onnx_loader);

/// Whether CoreML is enabled (controlled by build.zig)
pub const coreml_enabled = build_options.coreml_enabled;

/// Whether CUDA is enabled (controlled by build.zig)
pub const cuda_enabled = build_options.cuda_enabled;

/// Whether dynamic loading is enabled (controlled by build.zig)
pub const dynamic_ort = if (@hasDecl(build_options, "dynamic_ort")) build_options.dynamic_ort else false;

pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
    if (coreml_enabled) {
        @cInclude("coreml_provider_factory.h");
    }
});

// Re-export commonly used types
pub const OrtApi = c.OrtApi;
pub const OrtApiBase = c.OrtApiBase;
pub const OrtEnv = c.OrtEnv;
pub const OrtSession = c.OrtSession;
pub const OrtSessionOptions = c.OrtSessionOptions;
pub const OrtValue = c.OrtValue;
pub const OrtMemoryInfo = c.OrtMemoryInfo;
pub const OrtAllocator = c.OrtAllocator;
pub const OrtStatus = c.OrtStatus;
pub const OrtRunOptions = c.OrtRunOptions;
pub const OrtLoggingLevel = c.OrtLoggingLevel;
pub const OrtTensorTypeAndShapeInfo = c.OrtTensorTypeAndShapeInfo;
pub const OrtIoBinding = c.OrtIoBinding;

pub const ORT_API_VERSION = c.ORT_API_VERSION;

// =============================================================================
// Dynamic loading support
// =============================================================================

/// Error types for dynamic loading
pub const LoadError = error{
    LibraryNotFound,
    SymbolNotFound,
    ApiNotAvailable,
};

/// Function pointer type for OrtGetApiBase
const GetApiBaseFn = *const fn () callconv(.c) *const OrtApiBase;

/// Global state for dynamic loading
var dynamic_api_base: ?*const OrtApiBase = null;
var dynamic_handle: ?*anyopaque = null;
var dynamic_mutex: std.Thread.Mutex = .{};
var dynamic_cuda_provider_fn: ?*const fn (*OrtSessionOptions, c_int) callconv(.c) ?*OrtStatus = null;

/// Check if CUDA ONNX Runtime library exists (without loading)
pub fn isCudaRuntimeAvailable() bool {
    const home = std.posix.getenv("HOME") orelse return false;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cuda_lib_path = std.fmt.bufPrint(&path_buf, "{s}/.osgrep/onnxruntime-cuda/lib/libonnxruntime.so", .{home}) catch return false;
    std.fs.accessAbsolute(cuda_lib_path, .{}) catch return false;
    return true;
}

/// Load ONNX Runtime dynamically at runtime
fn loadDynamic() LoadError!*const OrtApiBase {
    dynamic_mutex.lock();
    defer dynamic_mutex.unlock();

    // Return cached if already loaded
    if (dynamic_api_base) |base| {
        return base;
    }

    // Try CUDA version first
    if (loadFromCudaPath()) |result| {
        dynamic_api_base = result.base;
        dynamic_handle = result.handle;
        dynamic_cuda_provider_fn = result.cuda_fn;
        log.info("Loaded CUDA-enabled ONNX Runtime dynamically", .{});
        return result.base;
    } else |_| {
        log.debug("CUDA ONNX Runtime not available, trying system version", .{});
    }

    // Fall back to system version
    if (loadFromSystemPath()) |result| {
        dynamic_api_base = result.base;
        dynamic_handle = result.handle;
        dynamic_cuda_provider_fn = null;
        log.info("Loaded system ONNX Runtime (CPU) dynamically", .{});
        return result.base;
    } else |err| {
        log.err("Failed to load any ONNX Runtime library", .{});
        return err;
    }
}

const LoadResult = struct {
    base: *const OrtApiBase,
    handle: *anyopaque,
    cuda_fn: ?*const fn (*OrtSessionOptions, c_int) callconv(.c) ?*OrtStatus,
};

fn loadFromCudaPath() LoadError!LoadResult {
    const home = std.posix.getenv("HOME") orelse return LoadError.LibraryNotFound;
    var path_buf: [std.fs.max_path_bytes:0]u8 = undefined;
    const len = std.fmt.bufPrint(&path_buf, "{s}/.osgrep/onnxruntime-cuda/lib/libonnxruntime.so", .{home}) catch {
        return LoadError.LibraryNotFound;
    };
    path_buf[len.len] = 0;

    return loadFromPath(&path_buf, true);
}

fn loadFromSystemPath() LoadError!LoadResult {
    // Try common library names
    const lib_names = [_][:0]const u8{
        "libonnxruntime.so.1",
        "libonnxruntime.so",
        "libonnxruntime.dylib",
    };

    for (lib_names) |name| {
        if (loadFromPath(name, false)) |result| {
            return result;
        } else |_| {
            continue;
        }
    }

    return LoadError.LibraryNotFound;
}

fn loadFromPath(path: [:0]const u8, is_cuda: bool) LoadError!LoadResult {
    const handle = std.c.dlopen(path.ptr, .{ .LAZY = true }) orelse {
        return LoadError.LibraryNotFound;
    };
    errdefer _ = std.c.dlclose(handle);

    // Get OrtGetApiBase function
    const get_api_base_ptr = std.c.dlsym(handle, "OrtGetApiBase") orelse {
        return LoadError.SymbolNotFound;
    };
    const get_api_base: GetApiBaseFn = @ptrCast(get_api_base_ptr);

    const api_base = get_api_base();

    // Try to get CUDA provider function (optional, only for CUDA builds)
    var cuda_fn: ?*const fn (*OrtSessionOptions, c_int) callconv(.c) ?*OrtStatus = null;
    if (is_cuda) {
        if (std.c.dlsym(handle, "OrtSessionOptionsAppendExecutionProvider_CUDA")) |ptr| {
            cuda_fn = @ptrCast(ptr);
        }
    }

    return LoadResult{
        .base = api_base,
        .handle = handle,
        .cuda_fn = cuda_fn,
    };
}

/// Get the dynamically loaded CUDA provider function (if available)
pub fn getDynamicCudaProvider() ?*const fn (*OrtSessionOptions, c_int) callconv(.c) ?*OrtStatus {
    return dynamic_cuda_provider_fn;
}

/// Check if runtime was loaded with CUDA support
pub fn isDynamicCudaLoaded() bool {
    return dynamic_cuda_provider_fn != null;
}

// =============================================================================
// Public API
// =============================================================================

/// Get the ONNX Runtime API base
pub fn getApiBase() *const OrtApiBase {
    if (comptime dynamic_ort) {
        // Dynamic loading mode - use dlopen
        return loadDynamic() catch |err| {
            std.debug.panic("Failed to load ONNX Runtime dynamically: {}", .{err});
        };
    } else {
        // Static/linked mode - use direct call
        return c.OrtGetApiBase();
    }
}

/// Get the ONNX Runtime API for the current version
pub fn getApi() ?*const OrtApi {
    const base = getApiBase();
    return base.GetApi.?(ORT_API_VERSION);
}

// Tensor element types
pub const TensorElementType = enum(c_uint) {
    undefined = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    float32 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    uint8 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    int8 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    uint16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    int16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    int32 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    int64 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    string = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    bool_ = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    float16 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    float64 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    uint32 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    uint64 = c.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,

    pub fn toC(self: TensorElementType) c_uint {
        return @intFromEnum(self);
    }
};

pub const LoggingLevel = enum(c_uint) {
    verbose = c.ORT_LOGGING_LEVEL_VERBOSE,
    info = c.ORT_LOGGING_LEVEL_INFO,
    warning = c.ORT_LOGGING_LEVEL_WARNING,
    err = c.ORT_LOGGING_LEVEL_ERROR,
    fatal = c.ORT_LOGGING_LEVEL_FATAL,

    pub fn toC(self: LoggingLevel) c_uint {
        return @intFromEnum(self);
    }
};

// CoreML execution provider flags
pub const CoreMLFlags = struct {
    pub const NONE: u32 = 0x000;
    pub const USE_CPU_ONLY: u32 = 0x001;
    pub const ENABLE_ON_SUBGRAPH: u32 = 0x002;
    pub const ONLY_ENABLE_DEVICE_WITH_ANE: u32 = 0x004;
    pub const ONLY_ALLOW_STATIC_INPUT_SHAPES: u32 = 0x008;
    pub const CREATE_MLPROGRAM: u32 = 0x010;
    pub const USE_CPU_AND_GPU: u32 = 0x020;
};

/// Append CoreML execution provider to session options
pub extern fn OrtSessionOptionsAppendExecutionProvider_CoreML(
    options: *OrtSessionOptions,
    coreml_flags: u32,
) ?*OrtStatus;

/// CUDA provider options struct (used when cuda_enabled is true)
pub const OrtCUDAProviderOptions = extern struct {
    device_id: c_int = 0,
    cudnn_conv_algo_search: c_int = 0, // OrtCudnnConvAlgoSearchDefault
    gpu_mem_limit: usize = @as(usize, @bitCast(@as(isize, -1))), // SIZE_MAX = no limit
    arena_extend_strategy: c_int = 0,
    do_copy_in_default_stream: c_int = 1,
    has_user_compute_stream: c_int = 0,
    user_compute_stream: ?*anyopaque = null,
    default_memory_arena_cfg: ?*anyopaque = null,
    tunable_op_enable: c_int = 0,
    tunable_op_tuning_enable: c_int = 0,
    tunable_op_max_tuning_duration_ms: c_int = 0,
};

/// Append CUDA execution provider to session options (only available when cuda_enabled)
/// This extern is only valid when linking against CUDA-enabled ONNX Runtime
pub extern fn OrtSessionOptionsAppendExecutionProvider_CUDA(
    options: *OrtSessionOptions,
    device_id: c_int,
) ?*OrtStatus;

test "can get API" {
    const api = getApi();
    try std.testing.expect(api != null);
}
