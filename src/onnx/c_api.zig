//! Raw C bindings for ONNX Runtime C API
//!
//! This module provides direct access to the ONNX Runtime C API via @cImport.

pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
    @cInclude("coreml_provider_factory.h");
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

pub const ORT_API_VERSION = c.ORT_API_VERSION;

/// Get the ONNX Runtime API base
pub fn getApiBase() *const OrtApiBase {
    return c.OrtGetApiBase();
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

test "can get API" {
    const api = getApi();
    const std = @import("std");
    try std.testing.expect(api != null);
}
