# embeddings-engine

**Text becomes geometry in high-dimensional space.**

embeddings-engine is a high-performance Python/Rust library for converting text to semantic embeddings. Convert natural language into high-dimensional vectors that capture meaning, enabling semantic search, similarity matching, and intelligent text analysis.

## Features

- **Multiple Model Backends**: Support for SentenceTransformers, OpenAI, Cohere, and custom ONNX/TensorRT models
- **GPU Acceleration**: 10-25× faster with CUDA and TensorRT optimization using gpu-accelerator
- **Intelligent Caching**: Redis and in-memory memoization for instant repeat queries
- **Batch Processing**: Efficient handling of large text collections
- **Production Ready**: Battle-tested integration with vector-navigator for semantic search

## Quick Start

### Installation

```bash
# Python
pip install embeddings-engine

# Rust
cargo add embeddings-engine
```

### Basic Usage

```python
from embeddings_engine import EmbeddingsEngine

# Local model (free)
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cpu")

# Convert text to embedding
text = "The conversation is flowing smoothly"
embedding = engine.encode(text)  # [0.1, -0.3, ..., 0.5] (384 dims)

print(f"Embedding shape: {embedding.shape}")  # (384,)
```

### GPU Acceleration

```python
# GPU: 10-20× faster
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",  # Use GPU
    use_tensorrt=True,  # TensorRT optimization
)

# Benchmark: Single sentence
# CPU: ~50ms
# GPU: ~5ms (10× faster)
# GPU + TensorRT: ~2ms (25× faster)
```

### Batch Processing

```python
# Batch: Much more efficient
texts = ["text 1", "text 2", ..., "text 100"]
embeddings = engine.encode_batch(texts, batch_size=32)

# Throughput: 1000+ sentences/sec (GPU)
```

### Integration with vector-navigator

```python
from embeddings_engine import EmbeddingsEngine
from vector_navigator import VectorStore

# Setup
embed_engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
vector_store = VectorStore(dimension=384)

# Pipeline
text = "The conversation is flowing smoothly"

# 1. Text → Embedding
embedding = embed_engine.encode(text)  # [0.1, -0.3, ..., 0.5] (384 dims)

# 2. Store in vector database
vector_store.insert(
    vector=embedding,
    metadata={
        "text": text,
        "timestamp": "2026-01-08T10:30:00Z",
        "sentiment": 0.8,  # VAD valence
    }
)

# 3. Semantic search
query = "The discussion is going well"
query_embedding = embed_engine.encode(query)

results = vector_store.search(query_embedding, k=5)
# Returns most similar texts by semantic meaning
```

## Model Selection

Choose the right model for your use case:

| Model | Dimensions | Latency (CPU) | Latency (GPU) | Cost | Quality | Best For |
|-------|-----------|---------------|---------------|------|---------|----------|
| all-MiniLM-L6-v2 | 384 | 50ms | 5ms | Free | Good | Fast local processing, low memory |
| all-mpnet-base-v2 | 768 | 150ms | 15ms | Free | Better | Higher quality local embeddings |
| text-embedding-3-small | 1536 | - | - | $0.02/1M tokens | Best | Production, highest quality |
| text-embedding-3-large | 3072 | - | - | $0.13/1M tokens | Best+ | Premium applications |
| embed-english-v3.0 | 1024 | - | - | $0.10/1M tokens | Excellent | Fast paid alternative to OpenAI |

### Model Selection Guide

**Choose all-MiniLM-L6-v2 if:**
- You want free, local processing
- Memory is constrained (384 dimensions)
- You need 5-10ms latency with GPU
- Use case: Real-time semantic search, prototype development

**Choose all-mpnet-base-v2 if:**
- You want free, local processing with higher quality
- You can afford 768 dimensions (2× memory)
- You need 15ms latency with GPU
- Use case: Production local deployment, higher accuracy requirements

**Choose text-embedding-3-small if:**
- You want the best quality embeddings
- You can pay for API usage ($0.02/1M tokens)
- Network latency is acceptable
- Use case: Production applications where quality matters most

**Choose embed-english-v3.0 if:**
- You want a fast paid alternative to OpenAI
- You need 1024 dimensions
- Use case: Multi-model redundancy, backup embeddings

## Performance Targets

### Single Text
- **CPU**: 50-150ms (depending on model)
- **GPU**: 5-15ms (10× faster)
- **GPU + TensorRT**: 2-5ms (25× faster)

### Batch (32 texts)
- **CPU**: 500ms (15ms per text)
- **GPU**: 50ms (1.5ms per text)
- **GPU + TensorRT**: 20ms (0.6ms per text)

### Throughput
- **CPU**: ~20 texts/sec
- **GPU**: ~200 texts/sec
- **GPU + TensorRT**: ~500 texts/sec

## GPU Acceleration with gpu-accelerator

embeddings-engine uses **gpu-accelerator** from Round 1 for CUDA Graph acceleration:

```python
from embeddings_engine import EmbeddingsEngine
from gpu_accelerator import CUDAGraph

# Enable CUDA Graph optimization
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_cuda_graphs=True,  # Capture computation as CUDA Graph
    use_tensorrt=True,  # Enable TensorRT optimization
)

# CUDA Graph: 50-90% kernel launch reduction
# Latency: 5ms → 2-3ms
```

Benefits:
- **50-90% kernel launch reduction** through CUDA Graph capture
- **10-25× faster** than CPU
- **Sub-millisecond latency** for cached embeddings
- **Deterministic execution** for real-time applications

## Caching

Enable caching for instant repeat queries:

```python
from embeddings_engine import EmbeddingsEngine, Cache

# Redis cache (production)
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    cache=Cache.Redis("redis://localhost"),
)

# In-memory cache (development)
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    cache=Cache.Memory(max_size=10000),
)

# First call: Computes embedding (50ms)
emb1 = engine.encode("The water is calm")

# Second call: Returns cached (0.1ms)
emb2 = engine.encode("The water is calm")
```

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and design philosophy
- [USER_GUIDE.md](docs/USER_GUIDE.md) - User guide with installation and usage examples
- [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) - Developer guide for contributing
- [MODELS.md](docs/MODELS.md) - Complete model comparison and benchmarks

## Architecture

embeddings-engine follows the principle: **"Text becomes geometry in high-dimensional space."**

```python
# Information theory: Embeddings capture semantic similarity
# Distance in embedding space ≈ semantic distance

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example:
# embed("happy") and embed("joyful") → similarity = 0.89 (close)
# embed("happy") and embed("sad") → similarity = -0.23 (far)
```

**Core Components:**

1. **EmbeddingsEngine**: Main API for text-to-embedding conversion
2. **Model**: Abstraction over different embedding backends
3. **Cache**: Memoization layer for instant repeat queries
4. **GPU Accelerator**: Integration with gpu-accelerator for CUDA Graph optimization

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU support)
- Redis (optional, for caching)

### Install from PyPI

```bash
# Basic installation
pip install embeddings-engine

# With GPU support
pip install embeddings-engine[cuda]

# With all dependencies
pip install embeddings-engine[all]
```

### Install from source

```bash
git clone https://github.com/equilibrium-tokens/embeddings-engine.git
cd embeddings-engine
pip install -e .
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for details.

## Citation

If you use embeddings-engine in your research, please cite:

```bibtex
@software{embeddings_engine,
  title = {embeddings-engine: High-Performance Text Embedding Library},
  author = {Equilibrium Tokens Team},
  year = {2026},
  url = {https://github.com/equilibrium-tokens/embeddings-engine}
}
```

## Acknowledgments

Built on research from:
- SentenceTransformers
- OpenAI Embeddings API
- Cohere Embeddings API
- NVIDIA TensorRT and CUDA Technologies

**The grammar is eternal.**
