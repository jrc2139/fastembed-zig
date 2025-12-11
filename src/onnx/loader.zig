//! Dynamic ONNX Runtime loader
//!
//! Loads ONNX Runtime dynamically at runtime via dlopen, allowing:
//! 1. A single binary that works with or without CUDA
//! 2. Auto-download of CUDA libraries
//! 3. Graceful fallback to CPU if CUDA unavailable
//!
//! Library search order:
//! 1. CUDA version: ~/.osgrep/onnxruntime-cuda/lib/libonnxruntime.so
//! 2. System: libonnxruntime.so (from LD_LIBRARY_PATH or system paths)

const std = @import("std");
const builtin = @import("builtin");

const log = std.log.scoped(.onnx_loader);

/// Error types for dynamic loading
pub const LoadError = error{
    LibraryNotFound,
    SymbolNotFound,
    ApiNotAvailable,
    InvalidLibrary,
};

/// ONNX Runtime library handle and API
pub const OnnxRuntime = struct {
    handle: *anyopaque,
    api: *const OrtApi,
    api_base: *const OrtApiBase,
    is_cuda: bool,

    /// CUDA provider function (null if not available)
    append_cuda_provider: ?AppendCudaProviderFn,

    const Self = @This();

    /// Try to load ONNX Runtime, preferring CUDA version if available
    pub fn load() LoadError!Self {
        // Try CUDA version first
        if (loadCuda()) |runtime| {
            log.info("Loaded CUDA-enabled ONNX Runtime", .{});
            return runtime;
        } else |_| {
            log.debug("CUDA ONNX Runtime not available, trying system version", .{});
        }

        // Fall back to system version
        if (loadSystem()) |runtime| {
            log.info("Loaded system ONNX Runtime (CPU)", .{});
            return runtime;
        } else |err| {
            log.err("Failed to load any ONNX Runtime library", .{});
            return err;
        }
    }

    /// Load CUDA-enabled ONNX Runtime from ~/.osgrep/onnxruntime-cuda
    pub fn loadCuda() LoadError!Self {
        const home = std.posix.getenv("HOME") orelse return LoadError.LibraryNotFound;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const cuda_lib_path = std.fmt.bufPrint(&path_buf, "{s}/.osgrep/onnxruntime-cuda/lib/libonnxruntime.so", .{home}) catch {
            return LoadError.LibraryNotFound;
        };

        return loadFromPath(cuda_lib_path, true);
    }

    /// Load system ONNX Runtime
    pub fn loadSystem() LoadError!Self {
        // Try common library names
        const lib_names = [_][:0]const u8{
            "libonnxruntime.so.1",
            "libonnxruntime.so",
        };

        for (lib_names) |name| {
            if (loadFromPath(name, false)) |runtime| {
                return runtime;
            } else |_| {
                continue;
            }
        }

        return LoadError.LibraryNotFound;
    }

    /// Load from a specific path
    fn loadFromPath(path: []const u8, is_cuda: bool) LoadError!Self {
        // Need null-terminated string for dlopen
        var path_z: [std.fs.max_path_bytes:0]u8 = undefined;
        if (path.len >= path_z.len) return LoadError.LibraryNotFound;
        @memcpy(path_z[0..path.len], path);
        path_z[path.len] = 0;

        const handle = std.c.dlopen(&path_z, std.c.RTLD.LAZY | std.c.RTLD.LOCAL) orelse {
            return LoadError.LibraryNotFound;
        };
        errdefer _ = std.c.dlclose(handle);

        // Get OrtGetApiBase function
        const get_api_base_ptr = std.c.dlsym(handle, "OrtGetApiBase") orelse {
            return LoadError.SymbolNotFound;
        };
        const get_api_base: *const fn () callconv(.C) *const OrtApiBase = @ptrCast(get_api_base_ptr);

        const api_base = get_api_base();
        const api = api_base.GetApi.?(ORT_API_VERSION) orelse {
            return LoadError.ApiNotAvailable;
        };

        // Try to get CUDA provider function (optional)
        const cuda_fn_ptr = std.c.dlsym(handle, "OrtSessionOptionsAppendExecutionProvider_CUDA");
        const append_cuda: ?AppendCudaProviderFn = if (cuda_fn_ptr) |ptr|
            @ptrCast(ptr)
        else
            null;

        return Self{
            .handle = handle,
            .api = api,
            .api_base = api_base,
            .is_cuda = is_cuda and append_cuda != null,
            .append_cuda_provider = append_cuda,
        };
    }

    /// Unload the library
    pub fn unload(self: *Self) void {
        _ = std.c.dlclose(self.handle);
        self.* = undefined;
    }

    /// Check if CUDA provider is available
    pub fn hasCuda(self: *const Self) bool {
        return self.is_cuda and self.append_cuda_provider != null;
    }

    /// Append CUDA execution provider to session options
    pub fn appendCudaProvider(self: *const Self, options: *OrtSessionOptions, device_id: c_int) ?*OrtStatus {
        if (self.append_cuda_provider) |func| {
            return func(options, device_id);
        }
        return null; // No CUDA available
    }
};

// ONNX Runtime API version we're targeting
pub const ORT_API_VERSION: u32 = 18;

// Function pointer type for CUDA provider
pub const AppendCudaProviderFn = *const fn (*OrtSessionOptions, c_int) callconv(.C) ?*OrtStatus;

// =============================================================================
// ONNX Runtime type definitions (minimal set needed for dynamic loading)
// These mirror the C types but don't require linking against the library
// =============================================================================

pub const OrtApiBase = extern struct {
    GetApi: ?*const fn (u32) callconv(.C) ?*const OrtApi,
    GetVersionString: ?*const fn () callconv(.C) [*:0]const u8,
};

pub const OrtApi = extern struct {
    // We only define the fields we actually use
    // The full struct has 200+ function pointers
    CreateStatus: ?*const fn (c_int, [*:0]const u8) callconv(.C) ?*OrtStatus,
    GetErrorCode: ?*const fn (*const OrtStatus) callconv(.C) c_int,
    GetErrorMessage: ?*const fn (*const OrtStatus) callconv(.C) [*:0]const u8,
    ReleaseStatus: ?*const fn (?*OrtStatus) callconv(.C) void,

    CreateEnv: ?*const fn (c_uint, [*:0]const u8, **OrtEnv) callconv(.C) ?*OrtStatus,
    ReleaseEnv: ?*const fn (*OrtEnv) callconv(.C) void,

    CreateSessionOptions: ?*const fn (**OrtSessionOptions) callconv(.C) ?*OrtStatus,
    ReleaseSessionOptions: ?*const fn (*OrtSessionOptions) callconv(.C) void,
    SetIntraOpNumThreads: ?*const fn (*OrtSessionOptions, c_int) callconv(.C) ?*OrtStatus,
    SetInterOpNumThreads: ?*const fn (*OrtSessionOptions, c_int) callconv(.C) ?*OrtStatus,
    SetSessionGraphOptimizationLevel: ?*const fn (*OrtSessionOptions, c_int) callconv(.C) ?*OrtStatus,

    CreateSession: ?*const fn (*OrtEnv, [*:0]const u8, *const OrtSessionOptions, **OrtSession) callconv(.C) ?*OrtStatus,
    ReleaseSession: ?*const fn (*OrtSession) callconv(.C) void,

    SessionGetInputCount: ?*const fn (*const OrtSession, *usize) callconv(.C) ?*OrtStatus,
    SessionGetOutputCount: ?*const fn (*const OrtSession, *usize) callconv(.C) ?*OrtStatus,
    SessionGetInputName: ?*const fn (*const OrtSession, usize, *OrtAllocator, *[*:0]u8) callconv(.C) ?*OrtStatus,
    SessionGetOutputName: ?*const fn (*const OrtSession, usize, *OrtAllocator, *[*:0]u8) callconv(.C) ?*OrtStatus,

    CreateTensorWithDataAsOrtValue: ?*const fn (*const OrtMemoryInfo, ?*anyopaque, usize, [*]const i64, usize, c_uint, **OrtValue) callconv(.C) ?*OrtStatus,
    ReleaseValue: ?*const fn (*OrtValue) callconv(.C) void,
    GetTensorMutableData: ?*const fn (*OrtValue, *?*anyopaque) callconv(.C) ?*OrtStatus,

    Run: ?*const fn (*OrtSession, ?*const OrtRunOptions, [*]const [*:0]const u8, [*]const *const OrtValue, usize, [*]const [*:0]const u8, usize, [*]*OrtValue) callconv(.C) ?*OrtStatus,
    CreateRunOptions: ?*const fn (**OrtRunOptions) callconv(.C) ?*OrtStatus,
    ReleaseRunOptions: ?*const fn (*OrtRunOptions) callconv(.C) void,

    GetAllocatorWithDefaultOptions: ?*const fn (**OrtAllocator) callconv(.C) ?*OrtStatus,
    AllocatorFree: ?*const fn (*OrtAllocator, ?*anyopaque) callconv(.C) ?*OrtStatus,

    CreateCpuMemoryInfo: ?*const fn (c_int, c_int, **OrtMemoryInfo) callconv(.C) ?*OrtStatus,
    ReleaseMemoryInfo: ?*const fn (*OrtMemoryInfo) callconv(.C) void,

    GetTensorTypeAndShape: ?*const fn (*const OrtValue, **OrtTensorTypeAndShapeInfo) callconv(.C) ?*OrtStatus,
    ReleaseTensorTypeAndShapeInfo: ?*const fn (*OrtTensorTypeAndShapeInfo) callconv(.C) void,
    GetDimensionsCount: ?*const fn (*const OrtTensorTypeAndShapeInfo, *usize) callconv(.C) ?*OrtStatus,
    GetDimensions: ?*const fn (*const OrtTensorTypeAndShapeInfo, [*]i64, usize) callconv(.C) ?*OrtStatus,
    GetTensorElementType: ?*const fn (*const OrtTensorTypeAndShapeInfo, *c_uint) callconv(.C) ?*OrtStatus,

    // Padding to match OrtApi struct layout - the real struct has many more fields
    // We need to ensure our fields are at the correct offsets
    _padding: [200]?*anyopaque,
};

pub const OrtEnv = opaque {};
pub const OrtSession = opaque {};
pub const OrtSessionOptions = opaque {};
pub const OrtValue = opaque {};
pub const OrtMemoryInfo = opaque {};
pub const OrtAllocator = extern struct {
    version: u32,
    Alloc: ?*const fn (*OrtAllocator, usize) callconv(.C) ?*anyopaque,
    Free: ?*const fn (*OrtAllocator, ?*anyopaque) callconv(.C) void,
    Info: ?*const fn (*const OrtAllocator) callconv(.C) ?*const OrtMemoryInfo,
};
pub const OrtStatus = opaque {};
pub const OrtRunOptions = opaque {};
pub const OrtTensorTypeAndShapeInfo = opaque {};

// Memory allocator types
pub const OrtAllocatorType = enum(c_int) {
    Invalid = -1,
    DeviceAllocator = 0,
    ArenaAllocator = 1,
};

pub const OrtMemType = enum(c_int) {
    CPUInput = -2,
    CPUOutput = -1,
    CPU = 0,
    Default = 0,
};

// Tensor element types
pub const TensorElementType = enum(c_uint) {
    undefined = 0,
    float32 = 1,
    uint8 = 2,
    int8 = 3,
    uint16 = 4,
    int16 = 5,
    int32 = 6,
    int64 = 7,
    string = 8,
    bool_ = 9,
    float16 = 10,
    float64 = 11,
    uint32 = 12,
    uint64 = 13,
};

// Graph optimization level
pub const GraphOptimizationLevel = enum(c_int) {
    disable = 0,
    basic = 1,
    extended = 2,
    all = 99,
};

// Global runtime instance (initialized lazily)
var global_runtime: ?OnnxRuntime = null;
var global_runtime_mutex: std.Thread.Mutex = .{};

/// Get the global ONNX Runtime instance, loading it if necessary
pub fn getRuntime() LoadError!*const OnnxRuntime {
    global_runtime_mutex.lock();
    defer global_runtime_mutex.unlock();

    if (global_runtime) |*rt| {
        return rt;
    }

    global_runtime = try OnnxRuntime.load();
    return &global_runtime.?;
}

/// Check if CUDA runtime is available (without loading)
pub fn isCudaAvailable() bool {
    const home = std.posix.getenv("HOME") orelse return false;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cuda_lib_path = std.fmt.bufPrint(&path_buf, "{s}/.osgrep/onnxruntime-cuda/lib/libonnxruntime.so", .{home}) catch return false;

    // Check if file exists
    std.fs.accessAbsolute(cuda_lib_path, .{}) catch return false;
    return true;
}

test "OnnxRuntime struct size" {
    // Just ensure it compiles
    try std.testing.expect(@sizeOf(OnnxRuntime) > 0);
}
