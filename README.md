# fastembed-zig

Fast, lightweight text embeddings in Zig. Generate semantic embeddings for search, RAG, and similarity matching.

Built on [ONNX Runtime](https://onnxruntime.ai/) and [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers).

## Features

- High-performance text embeddings using ONNX models
- HuggingFace tokenizer support (BERT, RoBERTa, GPT-2, etc.)
- Multiple pooling strategies (CLS token, mean pooling)
- L2 normalization for cosine similarity
- Model registry with popular embedding models (BGE, MiniLM, E5)
- Pure Zig with zero garbage collection

## Quick Start

```zig
const fe = @import("fastembed");
const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize embedder
    var embedder = try fe.Embedder.init(allocator, .{
        .model_path = "models/bge-small-en-v1.5",
    });
    defer embedder.deinit();

    // Generate embeddings
    const texts = &[_][]const u8{
        "What is machine learning?",
        "How do neural networks work?",
    };
    const embeddings = try embedder.embed(texts);
    defer allocator.free(embeddings);

    // Compute similarity
    const dim = embedder.getDimension(); // 384 for BGE-small
    const vec1 = embeddings[0..dim];
    const vec2 = embeddings[dim .. 2 * dim];
    const similarity = fe.cosineSimilarity(vec1, vec2);
    std.debug.print("Similarity: {d:.4}\n", .{similarity});
}
```

## Installation

### As a Zig Package

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .fastembed = .{
        .url = "https://github.com/your-org/fastembed-zig/archive/refs/tags/v0.1.0.tar.gz",
        .hash = "...",
    },
},
```

Then in `build.zig`:

```zig
const fastembed = b.dependency("fastembed", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("fastembed", fastembed.module("fastembed"));
```

### Build from Source

```bash
git clone https://github.com/your-org/fastembed-zig.git
cd fastembed-zig

# Build
zig build

# Run example
zig build run -- models/bge-small-en-v1.5 "Hello world"

# Run tests
zig build test
```

## Supported Models

| Model | Dimensions | Size | Use Case |
|-------|-----------|------|----------|
| `bge-small-en-v1.5` | 384 | ~33MB | General purpose, English |
| `bge-base-en-v1.5` | 768 | ~109MB | Higher quality, English |
| `all-MiniLM-L6-v2` | 384 | ~23MB | Fast, multilingual |
| `e5-small-v2` | 384 | ~33MB | Asymmetric retrieval |

Download models from [HuggingFace](https://huggingface.co/models?library=onnx&sort=downloads&search=embedding) or use the model registry:

```zig
// Using model registry
const model = fe.models.get(.bge_small_en_v1_5);
var embedder = try fe.Embedder.init(allocator, .{
    .model = model,
});
```

## API Reference

### Embedder

```zig
// Initialize
var embedder = try fe.Embedder.init(allocator, .{
    .model_path = "path/to/model",      // Local model directory
    .pooling = .mean,                    // .cls or .mean
    .normalize = true,                   // L2 normalize output
    .max_length = 512,                   // Max tokens per input
});
defer embedder.deinit();

// Generate embeddings
const embeddings = try embedder.embed(&[_][]const u8{"text1", "text2"});
defer allocator.free(embeddings);

// Get model info
const dim = embedder.getDimension();     // 384, 768, etc.
const vocab_size = embedder.getVocabSize();
```

### Tokenizer

For direct tokenizer access:

```zig
var tokenizer = try fe.Tokenizer.fromFile(allocator, "tokenizer.json");
defer tokenizer.deinit();

// Encode text to token IDs
const tokens = try tokenizer.encodeAlloc(allocator, "Hello world", true);
defer allocator.free(tokens);
// tokens = [101, 7592, 2088, 102] for BERT

// Decode back to text
if (tokenizer.decode(tokens, true)) |text| {
    std.debug.print("{s}\n", .{text}); // "hello world"
}
```

### Utility Functions

```zig
// Cosine similarity between embeddings
const sim = fe.cosineSimilarity(vec1, vec2);

// Dot product (for normalized vectors)
const dot = fe.dotProduct(vec1, vec2);
```

## Project Structure

```
fastembed-zig/
├── src/
│   ├── lib.zig           # Public API
│   ├── embedding.zig     # Embedder implementation
│   ├── models.zig        # Model registry
│   ├── pooling.zig       # Pooling strategies
│   ├── normalize.zig     # L2 normalization
│   ├── tokenizer/
│   │   ├── tokenizer.zig # High-level tokenizer
│   │   └── c_api.zig     # C bindings
│   └── onnx/
│       └── session.zig   # ONNX Runtime wrapper
├── examples/
│   ├── basic_embed.zig   # Embedding example
│   └── basic_tokenize.zig # Tokenizer example
├── deps/
│   ├── tokenizers/       # Pre-built tokenizers-cpp
│   └── tokenizers-cpp/   # Source for rebuilding
└── models/               # Downloaded models
```

## Dependencies

### Runtime

- ONNX Runtime 1.23+ (dynamic or static linking)
- tokenizers-cpp (HuggingFace tokenizers via C FFI)

### System (macOS)

- Security.framework
- SystemConfiguration.framework

### Building tokenizers-cpp

```bash
cd deps/tokenizers-cpp
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp libtokenizers_c.a ../deps/tokenizers/lib/
```

## Performance

Benchmarks on Apple M4 Pro:

| Operation | Time |
|-----------|------|
| Tokenize (short text) | ~50us |
| Tokenize (512 tokens) | ~200us |
| Embed (single, BGE-small) | ~5ms |
| Embed (batch of 32) | ~40ms |

## Use Cases

### Semantic Search

```zig
// Index documents
const docs = &[_][]const u8{
    "The quick brown fox",
    "Machine learning basics",
    "Neural network architecture",
};
const doc_embeddings = try embedder.embed(docs);

// Query
const query_emb = try embedder.embed(&[_][]const u8{"AI fundamentals"});

// Find most similar
var best_idx: usize = 0;
var best_score: f32 = -1;
for (0..docs.len) |i| {
    const score = fe.cosineSimilarity(
        query_emb[0..dim],
        doc_embeddings[i * dim .. (i + 1) * dim],
    );
    if (score > best_score) {
        best_score = score;
        best_idx = i;
    }
}
```

### RAG (Retrieval Augmented Generation)

```zig
// Embed chunks and store in vector DB
for (chunks) |chunk| {
    const emb = try embedder.embed(&[_][]const u8{chunk});
    try vector_db.insert(chunk, emb[0..dim]);
}

// Retrieve relevant context for LLM
const query_emb = try embedder.embed(&[_][]const u8{user_query});
const relevant = try vector_db.search(query_emb[0..dim], .{ .top_k = 5 });
```

## Comparison to Alternatives

| Library | Language | Binary Size | Startup | Embedding |
|---------|----------|-------------|---------|-----------|
| fastembed-zig | Zig | ~36MB | ~50ms | ~5ms |
| fastembed-py | Python | N/A | ~2s | ~10ms |
| sentence-transformers | Python | N/A | ~3s | ~15ms |

## License

MIT

## Related Projects

- [onnxruntime-zig](../onnxruntime-zig) - ONNX Runtime bindings for Zig
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - C++ wrapper for HuggingFace tokenizers
- [fastembed](https://github.com/qdrant/fastembed) - Python implementation (inspiration)
