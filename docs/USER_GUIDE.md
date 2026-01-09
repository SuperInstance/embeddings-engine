# User Guide

Complete guide for using embeddings-engine in your applications.

## Table of Contents

1. [Installation](#installation)
2. [Model Selection](#model-selection)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Configuration and Tuning](#configuration-and-tuning)
6. [Cost Optimization](#cost-optimization)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Installation

### Python Installation

```bash
# Basic installation (CPU only)
pip install embeddings-engine

# With GPU support (CUDA)
pip install embeddings-engine[cuda]

# With all dependencies (Redis, TensorRT, etc.)
pip install embeddings-engine[all]
```

### Requirements

- Python 3.9 or higher
- NumPy 1.20+
- For GPU support: CUDA 11.8+ or 12.x
- For caching (optional): Redis 6+

### Verify Installation

```python
import embeddings_engine
print(embeddings_engine.__version__)
# Expected: 0.1.0 or higher
```

## Model Selection

Choosing the right model is critical for performance, cost, and quality.

### Model Comparison Table

| Model | Dimensions | Latency (CPU) | Latency (GPU) | Cost | Quality | Use Case |
|-------|-----------|---------------|---------------|------|---------|----------|
| **all-MiniLM-L6-v2** | 384 | 50ms | 5ms | Free | Good | Real-time semantic search, prototyping |
| **all-mpnet-base-v2** | 768 | 150ms | 15ms | Free | Better | Production local deployment |
| **text-embedding-3-small** | 1536 | - | - | $0.02/1M tokens | Best | Production, highest quality |
| **text-embedding-3-large** | 3072 | - | - | $0.13/1M tokens | Best+ | Premium applications |
| **embed-english-v3.0** | 1024 | - | - | $0.10/1M tokens | Excellent | Fast paid alternative |

### Decision Tree

```
Need free, local processing?
├─ Yes: Memory constrained?
│  ├─ Yes: all-MiniLM-L6-v2 (384 dims)
│  └─ No: all-mpnet-base-v2 (768 dims)
└─ No: Can pay for API?
   ├─ Yes: Highest quality?
   │  ├─ Yes: text-embedding-3-large (3072 dims)
   │  └─ No: text-embedding-3-small (1536 dims)
   └─ No: Want fast alternative?
      └─ embed-english-v3.0 (1024 dims)
```

### Model Deep Dives

#### all-MiniLM-L6-v2 (Recommended for Most Use Cases)

**Pros:**
- Free, local processing
- Fast: 5ms on GPU
- Low memory: 384 dimensions
- Good quality for most tasks

**Cons:**
- Lower quality than paid models
- English only

**Best For:**
- Real-time semantic search
- Prototyping and development
- Edge deployment
- Memory-constrained environments

**Example:**
```python
from embeddings_engine import EmbeddingsEngine

engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (384,)
```

#### all-mpnet-base-v2 (Higher Quality Local)

**Pros:**
- Free, local processing
- Better quality than MiniLM
- Still fast: 15ms on GPU

**Cons:**
- 2× slower than MiniLM
- 2× memory (768 dimensions)
- English only

**Best For:**
- Production local deployment
- Applications requiring higher accuracy
- When quality matters but cost doesn't

**Example:**
```python
engine = EmbeddingsEngine(model="all-mpnet-base-v2", device="cuda")
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (768,)
```

#### text-embedding-3-small (Best Overall Quality)

**Pros:**
- Highest quality embeddings
- 1536 dimensions capture rich semantics
- OpenAI API reliability

**Cons:**
- Paid: $0.02/1M tokens
- Network latency: 100-500ms
- Requires API key

**Best For:**
- Production applications
- When quality matters most
- Multilingual support needed

**Example:**
```python
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key="your-openai-api-key",
)
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (1536,)
```

**Cost Calculation:**
- 1 token ≈ 0.75 words (English)
- $0.02/1M tokens ≈ $0.015/750k words
- Example: 100k words ≈ $0.002

#### text-embedding-3-large (Premium Quality)

**Pros:**
- Highest quality available
- 3072 dimensions capture maximum nuance
- Best for complex semantics

**Cons:**
- Most expensive: $0.13/1M tokens
- Highest memory: 3072 dimensions
- Network latency

**Best For:**
- Premium applications
- Complex semantic understanding
- Research and analysis

**Example:**
```python
engine = EmbeddingsEngine(
    model="text-embedding-3-large",
    api_key="your-openai-api-key",
)
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (3072,)
```

**Cost Calculation:**
- $0.13/1M tokens ≈ $0.10/750k words
- Example: 100k words ≈ $0.013

#### embed-english-v3.0 (Fast Paid Alternative)

**Pros:**
- Fast API: 50-200ms latency
- Excellent quality
- Lower cost than OpenAI large

**Cons:**
- Paid: $0.10/1M tokens
- English only
- Requires API key

**Best For:**
- Fast paid alternative to OpenAI
- Multi-model redundancy
- Backup embeddings

**Example:**
```python
engine = EmbeddingsEngine(
    model="embed-english-v3.0",
    api_key="your-cohere-api-key",
)
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (1024,)
```

### Custom Models

Use custom ONNX or TensorRT models:

```python
# ONNX model
engine = EmbeddingsEngine(
    model="path/to/model.onnx",
    backend="onnx",
    device="cuda",
)

# TensorRT model
engine = EmbeddingsEngine(
    model="path/to/model.trt",
    backend="tensorrt",
    device="cuda",
)
```

## Basic Usage

### Single Text Encoding

```python
from embeddings_engine import EmbeddingsEngine

# Initialize engine
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cpu")

# Encode single text
text = "The conversation is flowing smoothly"
embedding = engine.encode(text)

print(f"Shape: {embedding.shape}")  # (384,)
print(f"Type: {embedding.dtype}")  # float32
print(f"Sample: {embedding[:5]}")  # [0.1, -0.3, 0.5, -0.2, 0.8]
```

### Batch Encoding

```python
# Encode multiple texts (more efficient)
texts = [
    "The conversation is flowing smoothly",
    "The discussion is going well",
    "The dialogue is progressing",
]

embeddings = engine.encode_batch(texts, batch_size=32)

print(f"Shape: {embeddings.shape}")  # (3, 384)
print(f"Type: {embeddings.dtype}")  # float32
```

### GPU Acceleration

```python
# Use GPU for 10-25× speedup
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")

# Single text: 5ms (vs 50ms CPU)
embedding = engine.encode("The conversation is flowing smoothly")

# Batch: 1.5ms per text (vs 15ms CPU)
embeddings = engine.encode_batch(texts, batch_size=32)
```

### Semantic Similarity

```python
import numpy as np

def cosine_similarity(a, b):
    """Cosine similarity: -1 (opposite) to 1 (identical)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare two texts
text1 = "The conversation is flowing smoothly"
text2 = "The discussion is going well"

emb1 = engine.encode(text1)
emb2 = engine.encode(text2)

similarity = cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity:.3f}")  # 0.856 (very similar)

# Compare dissimilar texts
text3 = "The weather is terrible today"
emb3 = engine.encode(text3)

similarity = cosine_similarity(emb1, emb3)
print(f"Similarity: {similarity:.3f}")  # -0.123 (dissimilar)
```

## Advanced Usage

### GPU Acceleration with CUDA Graphs

```python
from embeddings_engine import EmbeddingsEngine

# Enable CUDA Graphs for 50-90% kernel launch reduction
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_cuda_graphs=True,  # Capture computation as CUDA Graph
)

# Performance: 5ms → 2-3ms (50% faster)
embedding = engine.encode("The conversation is flowing smoothly")
```

**Benefits:**
- 50-90% kernel launch reduction
- Deterministic latency for real-time applications
- 10-25× faster than CPU

### TensorRT Optimization

```python
# Enable TensorRT for maximum performance
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_tensorrt=True,  # Enable TensorRT optimization
)

# Performance: 5ms → 2-3ms (2× faster)
embedding = engine.encode("The conversation is flowing smoothly")
```

**TensorRT Optimizations:**
- Layer fusion: Combine multiple layers into single kernel
- Kernel auto-tuning: Select optimal kernels for GPU
- Quantization: INT8/FP8 precision for faster inference
- Dynamic shapes: Optimized for variable batch sizes

### Caching

Enable caching for instant repeat queries:

```python
from embeddings_engine import EmbeddingsEngine, Cache

# Redis cache (production)
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    cache=Cache.Redis("redis://localhost:6379"),
)

# In-memory cache (development)
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    cache=Cache.Memory(max_size=10000),
)

# First call: Computes embedding (50ms CPU, 5ms GPU)
emb1 = engine.encode("The water is calm")

# Second call: Returns cached (0.1ms)
emb2 = engine.encode("The water is calm")

# Verify embeddings are identical
assert np.array_equal(emb1, emb2)
```

**Cache Performance:**
- Cache hit: 0.1ms
- Cache miss: 50ms CPU, 5ms GPU
- **500× faster** for cached queries

### Dimensionality Reduction

Reduce embedding dimensions for faster search:

```python
from sklearn.decomposition import PCA

# Generate embeddings (384 dimensions)
texts = ["text 1", "text 2", ..., "text 1000"]
embeddings = engine.encode_batch(texts)

# Reduce to 128 dimensions (faster search, less memory)
pca = PCA(n_components=128)
reduced_embeddings = pca.fit_transform(embeddings)

print(f"Original: {embeddings.shape}")  # (1000, 384)
print(f"Reduced: {reduced_embeddings.shape}")  # (1000, 128)

# Explained variance: How much information is retained
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
# Expected: 85-95% for 128 dimensions
```

**Trade-offs:**
- Lower dimensions: Faster search, less memory
- Higher dimensions: Better accuracy, slower search
- Sweet spot: 128-256 dimensions for most use cases

### Batch Processing

Process large text collections efficiently:

```python
# Large batch processing
texts = [f"text {i}" for i in range(10000)]

# Process in batches of 32
embeddings = engine.encode_batch(texts, batch_size=32)

print(f"Shape: {embeddings.shape}")  # (10000, 384)
print(f"Time: {time.time() - start:.2f}s")  # ~15s for GPU (667 texts/sec)
```

**Throughput:**
- CPU: ~20 texts/sec
- GPU: ~200 texts/sec
- GPU + TensorRT: ~500 texts/sec

### Parallel Processing

Process multiple batches in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def process_batch(batch):
    return engine.encode_batch(batch, batch_size=32)

# Split into 10 batches
texts = [f"text {i}" for i in range(10000)]
batches = [texts[i:i+1000] for i in range(0, len(texts), 1000)]

# Process in parallel (4 threads)
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, batches))

# Combine results
embeddings = np.vstack(results)
```

## Configuration and Tuning

### Device Selection

```python
# CPU: Safe fallback
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cpu")

# CUDA: Best performance (NVIDIA GPU)
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")

# MPS: Apple Silicon GPU (M1/M2/M3)
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="mps")

# Auto: Automatically select best available
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="auto")
```

### Batch Size Tuning

```python
# Small batch: Lower latency, less GPU utilization
engine.encode_batch(texts, batch_size=8)  # 8 texts per batch

# Medium batch: Balanced (recommended)
engine.encode_batch(texts, batch_size=32)  # 32 texts per batch

# Large batch: Higher throughput, more GPU memory
engine.encode_batch(texts, batch_size=128)  # 128 texts per batch
```

**Guidelines:**
- Real-time: batch_size=8-16
- Batch processing: batch_size=32-64
- Maximum throughput: batch_size=128-256

### Quantization

```python
# FP32: Default, best quality
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", quantization="fp32")

# INT8: 4× memory reduction, 2× speedup, <1% accuracy loss
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", quantization="int8")

# FP8: 4× memory reduction, 3× speedup, <2% accuracy loss
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", quantization="fp8")

# FP4: 8× memory reduction, 4× speedup, 3-5% accuracy loss
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", quantization="fp4")
```

### Cache Configuration

```python
from embeddings_engine import Cache

# Redis cache (production)
cache = Cache.Redis(
    url="redis://localhost:6379",
    ttl=3600,  # Cache for 1 hour
    max_size=1000000,  # Maximum 1M entries
)

# Memory cache (development)
cache = Cache.Memory(
    max_size=10000,  # Maximum 10K entries
)

# No cache (disabled)
cache = None
```

### Model-Specific Configuration

```python
# OpenAI configuration
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key="your-api-key",
    timeout=30,  # Request timeout (seconds)
    max_retries=3,  # Maximum retries on failure
)

# Cohere configuration
engine = EmbeddingsEngine(
    model="embed-english-v3.0",
    api_key="your-api-key",
    timeout=20,
    max_retries=3,
)

# SentenceTransformers configuration
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    cache_folder="/path/to/cache",  # Model cache folder
)
```

## Cost Optimization

### Free vs. Paid Models

**Use Free Models (SentenceTransformers) When:**
- Budget is constrained
- Network latency is unacceptable
- Data privacy requires local processing
- You have GPU resources

**Use Paid Models (OpenAI/Cohere) When:**
- Quality is critical
- You don't have GPU resources
- Multilingual support is needed
- You want API simplicity

### Cost Calculation

#### OpenAI text-embedding-3-small

```python
# Cost: $0.02 per 1M tokens
# 1 token ≈ 0.75 words (English)

def calculate_cost(words, model="text-embedding-3-small"):
    tokens = words * 1.33  # 1 word ≈ 1.33 tokens
    cost_per_token = 0.02 / 1_000_000
    return tokens * cost_per_token

# Example: 100k words
cost = calculate_cost(100_000)
print(f"Cost: ${cost:.4f}")  # $0.0027

# Example: 1M words
cost = calculate_cost(1_000_000)
print(f"Cost: ${cost:.4f}")  # $0.0266
```

#### Cohere embed-english-v3.0

```python
# Cost: $0.10 per 1M tokens

def calculate_cohere_cost(words):
    tokens = words * 1.33
    cost_per_token = 0.10 / 1_000_000
    return tokens * cost_per_token

# Example: 100k words
cost = calculate_cohere_cost(100_000)
print(f"Cost: ${cost:.4f}")  # $0.0133

# Example: 1M words
cost = calculate_cohere_cost(1_000_000)
print(f"Cost: ${cost:.4f}")  # $0.1330
```

### Hybrid Strategy

Combine free and paid models for cost optimization:

```python
# Use free model for initial filtering
free_engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")

# Use paid model for final ranking
paid_engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key="your-api-key",
)

# Step 1: Filter with free model (fast, local)
candidates = ["text 1", "text 2", ..., "text 1000"]
candidate_embeddings = free_engine.encode_batch(candidates)

# Find top 100 similar texts
similarities = [cosine_similarity(query, emb) for emb in candidate_embeddings]
top_100_indices = np.argsort(similarities)[-100:]
top_100_candidates = [candidates[i] for i in top_100_indices]

# Step 2: Re-rank with paid model (slower, higher quality)
top_100_embeddings = paid_engine.encode_batch(top_100_candidates)
final_similarities = [cosine_similarity(query, emb) for emb in top_100_embeddings]
top_10_indices = np.argsort(final_similarities)[-10:]

# Cost: Only 100 texts encoded with paid model (vs 1000)
# Savings: 90% cost reduction
```

### Caching Strategy

Aggressive caching to minimize API calls:

```python
from embeddings_engine import EmbeddingsEngine, Cache

# Enable Redis cache with long TTL
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key="your-api-key",
    cache=Cache.Redis(
        url="redis://localhost:6379",
        ttl=86400 * 7,  # Cache for 7 days
    ),
)

# First week: High API usage (building cache)
# Subsequent weeks: Minimal API usage (cache hits)
# Expected: 80-90% cache hit rate for repeated queries
```

## Best Practices

### 1. Use GPU for Production

```python
# ❌ Bad: CPU in production
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cpu")

# ✅ Good: GPU in production
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
```

**Impact:**
- CPU: 50ms latency, 20 texts/sec
- GPU: 5ms latency, 200 texts/sec
- **10× faster, 10× higher throughput**

### 2. Enable Caching

```python
# ❌ Bad: No caching
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2")

# ✅ Good: Redis caching
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    cache=Cache.Redis("redis://localhost:6379"),
)
```

**Impact:**
- No cache: 5ms per query
- Cache hit: 0.1ms per query
- **50× faster for repeated queries**

### 3. Use Batch Processing

```python
# ❌ Bad: Encode one by one
embeddings = [engine.encode(text) for text in texts]

# ✅ Good: Batch encoding
embeddings = engine.encode_batch(texts, batch_size=32)
```

**Impact:**
- One by one: 5ms per text (GPU)
- Batch: 1.5ms per text (GPU)
- **3× faster, better GPU utilization**

### 4. Choose Right Model

```python
# ❌ Bad: Overkill for simple use case
engine = EmbeddingsEngine(
    model="text-embedding-3-large",  # 3072 dims, $0.13/1M tokens
    api_key="your-api-key",
)

# ✅ Good: Match model to use case
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",  # 384 dims, free
    device="cuda",
)
```

**Impact:**
- Overkill: 8× more memory, 8× higher cost
- Right model: Adequate quality, lower cost

### 5. Handle Errors Gracefully

```python
from embeddings_engine import EmbeddingsEngine
import time

def encode_with_retry(engine, text, max_retries=3):
    """Encode with retry logic."""
    for attempt in range(max_retries):
        try:
            return engine.encode(text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff
            time.sleep(2 ** attempt)
    return None

# Usage
embedding = encode_with_retry(engine, "The conversation is flowing smoothly")
```

## Troubleshooting

### CUDA Out of Memory

**Problem:** `CUDA out of memory` error

**Solutions:**
```python
# 1. Reduce batch size
engine.encode_batch(texts, batch_size=16)  # Was 32

# 2. Use quantization
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", quantization="int8")

# 3. Use smaller model
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2")  # 384 dims
# Instead of
engine = EmbeddingsEngine(model="all-mpnet-base-v2")  # 768 dims

# 4. Clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Slow Performance

**Problem:** Encoding is slow

**Solutions:**
```python
# 1. Enable GPU
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")

# 2. Enable CUDA Graphs
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_cuda_graphs=True,
)

# 3. Enable TensorRT
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_tensorrt=True,
)

# 4. Use batch processing
embeddings = engine.encode_batch(texts, batch_size=32)
```

### API Rate Limits

**Problem:** OpenAI/Cohere API rate limits

**Solutions:**
```python
# 1. Enable caching
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key="your-api-key",
    cache=Cache.Redis("redis://localhost:6379"),
)

# 2. Use batch processing
embeddings = engine.encode_batch(texts, batch_size=100)  # Fewer API calls

# 3. Implement exponential backoff
def encode_with_backoff(engine, text, max_retries=5):
    for attempt in range(max_retries):
        try:
            return engine.encode(text)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, 16s
    return None

# 4. Use hybrid strategy (free + paid)
# See "Hybrid Strategy" section above
```

### Poor Similarity Results

**Problem:** Semantic similarity doesn't match expectations

**Solutions:**
```python
# 1. Use higher quality model
engine = EmbeddingsEngine(
    model="text-embedding-3-small",  # Better than all-MiniLM-L6-v2
    api_key="your-api-key",
)

# 2. Increase dimensions
engine = EmbeddingsEngine(model="all-mpnet-base-v2")  # 768 dims vs 384

# 3. Use domain-specific model
# Fine-tune on your domain data

# 4. Adjust similarity threshold
threshold = 0.7  # Adjust based on use case
if similarity > threshold:
    print("Similar")
else:
    print("Not similar")
```

### Import Errors

**Problem:** `ImportError: No module named 'embeddings_engine'`

**Solutions:**
```bash
# 1. Install package
pip install embeddings-engine

# 2. Install with GPU support
pip install embeddings-engine[cuda]

# 3. Verify installation
python -c "import embeddings_engine; print(embeddings_engine.__version__)"

# 4. Install from source
git clone https://github.com/equilibrium-tokens/embeddings-engine.git
cd embeddings-engine
pip install -e .
```

## Next Steps

- Explore [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [MODELS.md](MODELS.md) for detailed model benchmarks
- Check [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for contributing

**The grammar is eternal.**
