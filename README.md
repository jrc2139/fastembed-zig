# fastembed-zig

Fast, lightweight text embeddings and reranking in Zig. Generate semantic embeddings for search, RAG, and similarity matching.

Built on [ONNX Runtime](https://onnxruntime.ai/) and [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers).

## Features

- High-performance text embeddings using ONNX models
- Cross-encoder reranking for two-stage retrieval
- HuggingFace tokenizer support (BERT, RoBERTa, etc.)
- Multiple pooling strategies (CLS token, mean pooling)
- L2 normalization for cosine similarity
- Model registry with popular embedding models (Granite, BGE, MiniLM, Gemma)
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
        .model_path = "models/granite-embedding-small-english-r2-qint8",
        .model = .granite_embedding_small_english_r2_qint8,
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
    const dim = embedder.getDimension(); // 384 for Granite-small
    const vec1 = embeddings[0..dim];
    const vec2 = embeddings[dim .. 2 * dim];
    const similarity = fe.cosineSimilarity(vec1, vec2);
    std.debug.print("Similarity: {d:.4}\n", .{similarity});
}
```

## Reranking

Two-stage retrieval with cross-encoder reranking:

```zig
// Initialize reranker
var reranker = try fe.Reranker.init(allocator, .{
    .model_path = "models/granite-reranker-english-r2-qint8-arm64",
    .model = .granite_reranker_english_r2_qint8_arm64,
});
defer reranker.deinit();

// Rerank documents for a query
const query = "What is machine learning?";
const documents = &[_][]const u8{
    "ML is a subset of AI",
    "The weather is nice today",
    "Neural networks learn from data",
};

const scores = try reranker.rerank(query, documents);
defer allocator.free(scores);
// scores[0] and scores[2] will be high, scores[1] will be low
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

# Run examples
zig build example-embed
zig build example-tokenize

# Run tests
zig build test
```

## Supported Models

### Embedding Models

| Model | Dimensions | Size | Use Case |
|-------|-----------|------|----------|
| `granite_embedding_small_english_r2_qint8` | 384 | ~47MB | Default, ARM64 optimized |
| `granite_embedding_small_english_r2_q4` | 384 | ~47MB | 4-bit quantized |
| `granite_embedding_small_english_r2` | 384 | ~47MB | FP32 |
| `embedding_gemma_300m_q4` | 768 | ~300MB | Higher quality |
| `embedding_gemma_300m_fp16` | 768 | ~600MB | Gemma FP16 |
| `granite_embedding_english_r2` | 768 | ~137MB | Granite large |
| `bge_small_en_v1_5` | 384 | ~33MB | BGE small English |
| `all_minilm_l6_v2` | 384 | ~23MB | Fast, multilingual |
| `multilingual_e5_large` | 1024 | ~1GB | Multilingual |

### Reranker Models

| Model | Size | Use Case |
|-------|------|----------|
| `granite_reranker_english_r2_qint8_arm64` | ~155MB | ARM64 optimized, default |

Download models from [HuggingFace](https://huggingface.co/models?library=onnx&sort=downloads&search=embedding).

## API Reference

### Embedder

```zig
// Initialize
var embedder = try fe.Embedder.init(allocator, .{
    .model_path = "path/to/model",      // Local model directory
    .model = .granite_embedding_small_english_r2_qint8,
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
```

### Reranker

```zig
// Initialize
var reranker = try fe.Reranker.init(allocator, .{
    .model_path = "path/to/reranker",
    .model = .granite_reranker_english_r2_qint8_arm64,
});
defer reranker.deinit();

// Rerank documents
const scores = try reranker.rerank(query, documents);
defer allocator.free(scores);
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
│   ├── reranker.zig      # Reranker implementation
│   ├── models.zig        # Model registry
│   ├── pooling.zig       # Pooling strategies
│   ├── normalize.zig     # L2 normalization
│   ├── tokenizer/
│   │   └── tokenizer.zig # High-level tokenizer
│   └── onnx/
│       ├── session.zig   # ONNX Runtime wrapper
│       ├── c_api.zig     # C bindings
│       └── provider.zig  # Execution providers (CoreML, etc.)
├── examples/
│   ├── basic_embed.zig   # Embedding example
│   ├── basic_tokenize.zig # Tokenizer example
│   ├── benchmark.zig     # Performance benchmarks
│   └── test_batch.zig    # Batch processing tests
├── deps/
│   ├── tokenizers/       # Pre-built tokenizers-cpp
│   └── onnxruntime/      # ONNX Runtime library
└── models/               # Downloaded models
```

## Dependencies

### Runtime

- ONNX Runtime 1.23+ (dynamic linking)
- tokenizers-cpp (HuggingFace tokenizers via C FFI)

### System (macOS)

- Security.framework
- SystemConfiguration.framework
- CoreML.framework (optional, for acceleration)

## Performance

Benchmarks on Apple M4 Pro:

| Operation | Time |
|-----------|------|
| Tokenize (short text) | ~50μs |
| Tokenize (512 tokens) | ~200μs |
| Embed (single, Granite-small) | ~5ms |
| Embed (batch of 32) | ~40ms |
| Rerank (query + 10 docs) | ~50ms |

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

### Two-Stage Retrieval (RAG)

```zig
// Stage 1: Fast embedding search
const candidates = try embedder.embed(all_documents);
const top_k = findTopK(query_embedding, candidates, 100);

// Stage 2: Rerank with cross-encoder
const reranked = try reranker.rerank(query, top_k);
const best_docs = sortByScore(top_k, reranked)[0..10];
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

- [osgrep-zig](../osgrep-zig) - Semantic code search using fastembed
- [onnxruntime-zig](../onnxruntime-zig) - ONNX Runtime bindings for Zig
- [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp) - C++ wrapper for HuggingFace tokenizers
- [fastembed](https://github.com/qdrant/fastembed) - Python implementation (inspiration)
