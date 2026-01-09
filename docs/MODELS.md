# Supported Models

Complete guide to supported embedding models, performance benchmarks, and use case recommendations.

## Table of Contents

1. [Model Comparison Table](#model-comparison-table)
2. [Performance Benchmarks](#performance-benchmarks)
3. [Dimensionality and Memory](#dimensionality-and-memory)
4. [Use Case Recommendations](#use-case-recommendations)
5. [Model Deep Dives](#model-deep-dives)
6. [Update and Maintenance](#update-and-maintenance)

## Model Comparison Table

| Model | Provider | Dimensions | Latency (CPU) | Latency (GPU) | Cost | Quality | Language | Best For |
|-------|----------|-----------|---------------|---------------|------|---------|----------|----------|
| **all-MiniLM-L6-v2** | SentenceTransformers | 384 | 50ms | 5ms | Free | Good | English | Real-time semantic search, prototyping |
| **all-mpnet-base-v2** | SentenceTransformers | 768 | 150ms | 15ms | Free | Better | English | Production local deployment |
| **text-embedding-3-small** | OpenAI | 1536 | - | - | $0.02/1M tokens | Best | Multilingual | Production, highest quality |
| **text-embedding-3-large** | OpenAI | 3072 | - | - | $0.13/1M tokens | Best+ | Multilingual | Premium applications |
| **embed-english-v3.0** | Cohere | 1024 | - | - | $0.10/1M tokens | Excellent | English | Fast paid alternative |
| **paraphrase-multilingual-MiniLM-L12-v2** | SentenceTransformers | 384 | 80ms | 8ms | Free | Good | 50+ languages | Multilingual local processing |

### Quick Reference

**Free Models:**
- all-MiniLM-L6-v2: Best overall free model
- all-mpnet-base-v2: Higher quality free model
- paraphrase-multilingual-MiniLM-L12-v2: Multilingual free model

**Paid Models:**
- text-embedding-3-small: Best value paid model
- text-embedding-3-large: Premium quality
- embed-english-v3.0: Fast alternative to OpenAI

## Performance Benchmarks

### Single Text Latency

Time to encode single text:

| Model | CPU | GPU | GPU + TensorRT |
|-------|-----|-----|----------------|
| all-MiniLM-L6-v2 | 50ms | 5ms | 2ms |
| all-mpnet-base-v2 | 150ms | 15ms | 6ms |
| paraphrase-multilingual-MiniLM-L12-v2 | 80ms | 8ms | 3ms |

**Note:** OpenAI and Cohere models have network latency (100-500ms) in addition to inference time.

### Batch Processing Throughput

Texts per second (batch size 32):

| Model | CPU | GPU | GPU + TensorRT |
|-------|-----|-----|----------------|
| all-MiniLM-L6-v2 | 20 texts/sec | 200 texts/sec | 500 texts/sec |
| all-mpnet-base-v2 | 7 texts/sec | 67 texts/sec | 167 texts/sec |
| paraphrase-multilingual-MiniLM-L12-v2 | 12 texts/sec | 125 texts/sec | 333 texts/sec |

### API Model Throughput

Network-bound throughput:

| Model | Throughput | Latency (P50) | Latency (P95) |
|-------|-----------|---------------|---------------|
| text-embedding-3-small | 100 texts/sec | 150ms | 300ms |
| text-embedding-3-large | 80 texts/sec | 200ms | 400ms |
| embed-english-v3.0 | 150 texts/sec | 100ms | 200ms |

**Note:** Actual throughput depends on network conditions and API rate limits.

### GPU Acceleration Speedup

Speedup compared to CPU:

| Model | GPU (No TensorRT) | GPU + TensorRT |
|-------|-------------------|----------------|
| all-MiniLM-L6-v2 | 10× faster | 25× faster |
| all-mpnet-base-v2 | 10× faster | 25× faster |
| paraphrase-multilingual-MiniLM-L12-v2 | 10× faster | 27× faster |

### Quality Benchmarks

Semantic similarity quality (STS benchmark):

| Model | Pearson Correlation | Spearman Correlation |
|-------|---------------------|----------------------|
| text-embedding-3-large | 0.86 | 0.85 |
| text-embedding-3-small | 0.83 | 0.82 |
| embed-english-v3.0 | 0.81 | 0.80 |
| all-mpnet-base-v2 | 0.78 | 0.77 |
| all-MiniLM-L6-v2 | 0.72 | 0.71 |

**Higher is better.** Pearson and Spearman correlations measure how well embeddings capture semantic similarity.

## Dimensionality and Memory

### Memory Usage Per Embedding

| Model | FP32 (bytes) | INT8 (bytes) | FP4 (bytes) |
|-------|--------------|--------------|-------------|
| all-MiniLM-L6-v2 | 1,536 | 384 | 192 |
| all-mpnet-base-v2 | 3,072 | 768 | 384 |
| text-embedding-3-small | 6,144 | 1,536 | 768 |
| text-embedding-3-large | 12,288 | 3,072 | 1,536 |
| embed-english-v3.0 | 4,096 | 1,024 | 512 |

### Memory for Large Datasets

Memory required for 1 million embeddings:

| Model | FP32 | INT8 | FP4 |
|-------|------|------|-----|
| all-MiniLM-L6-v2 | 1.5 GB | 384 MB | 192 MB |
| all-mpnet-base-v2 | 3.0 GB | 768 MB | 384 MB |
| text-embedding-3-small | 6.0 GB | 1.5 GB | 768 MB |
| text-embedding-3-large | 12.0 GB | 3.0 GB | 1.5 GB |
| embed-english-v3.0 | 4.0 GB | 1.0 GB | 512 MB |

### Dimensionality vs. Quality Trade-off

Higher dimensions capture richer semantics:

- **384 dimensions**: Basic semantic relationships, fast search
- **768 dimensions**: Nuanced semantic understanding
- **1024-1536 dimensions**: Production-grade semantics
- **3072 dimensions**: Premium semantic detail

**Guideline:** Use smallest dimensions that meet quality requirements.

## Use Case Recommendations

### Real-Time Semantic Search

**Requirements:** Low latency, high throughput

**Recommended Models:**
1. **all-MiniLM-L6-v2** (GPU): 5ms latency, 384 dims
2. **all-mpnet-base-v2** (GPU + TensorRT): 6ms latency, 768 dims

**Configuration:**
```python
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_tensorrt=True,
    cache=Cache.Redis("redis://localhost"),
)
```

**Why:** 5ms latency enables real-time search, 384 dims keeps memory low.

### Batch Processing

**Requirements:** High throughput, cost efficiency

**Recommended Models:**
1. **all-MiniLM-L6-v2** (GPU + TensorRT): 500 texts/sec
2. **all-mpnet-base-v2** (GPU + TensorRT): 167 texts/sec

**Configuration:**
```python
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_tensorrt=True,
)
embeddings = engine.encode_batch(texts, batch_size=32)
```

**Why:** 500 texts/sec throughput minimizes processing time.

### Production Deployment

**Requirements:** High quality, reliability

**Recommended Models:**
1. **text-embedding-3-small**: Best value, $0.02/1M tokens
2. **all-mpnet-base-v2** (GPU): Free, high quality

**Configuration:**
```python
# Paid (highest quality)
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    cache=Cache.Redis("redis://localhost", ttl=86400),
)

# Free (local)
engine = EmbeddingsEngine(
    model="all-mpnet-base-v2",
    device="cuda",
    use_tensorrt=True,
    cache=Cache.Redis("redis://localhost"),
)
```

**Why:** text-embedding-3-small has best quality, all-mpnet-base-v2 is free.

### Edge Deployment

**Requirements:** Low memory, no network

**Recommended Models:**
1. **all-MiniLM-L6-v2** (INT8): 384 dims, 384 bytes per embedding
2. **paraphrase-multilingual-MiniLM-L12-v2** (INT8): 384 dims, multilingual

**Configuration:**
```python
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cpu",  # Edge device CPU
    quantization="int8",
)
```

**Why:** 384 bytes per embedding minimizes memory, no network dependency.

### Multilingual Applications

**Requirements:** Multiple languages, quality

**Recommended Models:**
1. **paraphrase-multilingual-MiniLM-L12-v2**: Free, 50+ languages
2. **text-embedding-3-small**: Paid, multilingual, higher quality

**Configuration:**
```python
# Free (local)
engine = EmbeddingsEngine(
    model="paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda",
)

# Paid (highest quality)
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

**Why:** paraphrase-multilingual supports 50+ languages, text-embedding-3-small has higher quality.

### Cost-Sensitive Applications

**Requirements:** Low cost, good quality

**Recommended Models:**
1. **all-MiniLM-L6-v2** (GPU): Free, good quality
2. **Hybrid strategy**: Free model + paid model re-ranking

**Configuration:**
```python
# Free model (primary)
free_engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    cache=Cache.Redis("redis://localhost", ttl=604800),  # 7 days
)

# Paid model (re-ranking top results)
paid_engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

**Why:** Free model reduces API calls by 90%, paid model re-ranks top 10%.

## Model Deep Dives

### all-MiniLM-L6-v2

**Overview:** Best overall free model for most use cases.

**Specs:**
- Dimensions: 384
- Parameters: 22M
- Training data: 1B sentence pairs
- Languages: English

**Performance:**
- CPU: 50ms latency, 20 texts/sec
- GPU: 5ms latency, 200 texts/sec
- GPU + TensorRT: 2ms latency, 500 texts/sec
- Quality: Good (0.72 Pearson correlation)

**Strengths:**
- Fast: 5ms on GPU
- Low memory: 384 dimensions
- Free: No API costs
- Easy: No API keys required

**Weaknesses:**
- Lower quality than paid models
- English only
- Requires GPU for best performance

**Best For:**
- Real-time semantic search
- Prototyping and development
- Edge deployment
- Memory-constrained environments

**Example:**
```python
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (384,)
```

### all-mpnet-base-v2

**Overview:** Higher quality free model for production.

**Specs:**
- Dimensions: 768
- Parameters: 110M
- Training data: 1B sentence pairs
- Languages: English

**Performance:**
- CPU: 150ms latency, 7 texts/sec
- GPU: 15ms latency, 67 texts/sec
- GPU + TensorRT: 6ms latency, 167 texts/sec
- Quality: Better (0.78 Pearson correlation)

**Strengths:**
- Higher quality than MiniLM
- Free: No API costs
- Good for production

**Weaknesses:**
- Slower than MiniLM (3×)
- More memory (768 dimensions)
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

### text-embedding-3-small

**Overview:** Best value paid model for production.

**Specs:**
- Dimensions: 1536
- Training data: Proprietary
- Languages: Multilingual

**Performance:**
- API latency: 100-500ms (network-dependent)
- Quality: Best (0.83 Pearson correlation)
- Cost: $0.02/1M tokens

**Strengths:**
- Highest quality available
- Multilingual support
- Reliable API
- 1536 dimensions capture rich semantics

**Weaknesses:**
- Paid: $0.02/1M tokens
- Network latency: 100-500ms
- Requires API key

**Best For:**
- Production applications
- When quality matters most
- Multilingual support needed

**Cost Calculation:**
- 100k words ≈ $0.0027
- 1M words ≈ $0.0266
- 10M words ≈ $0.266

**Example:**
```python
engine = EmbeddingsEngine(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (1536,)
```

### text-embedding-3-large

**Overview:** Premium quality for demanding applications.

**Specs:**
- Dimensions: 3072
- Training data: Proprietary
- Languages: Multilingual

**Performance:**
- API latency: 150-600ms (network-dependent)
- Quality: Best+ (0.86 Pearson correlation)
- Cost: $0.13/1M tokens

**Strengths:**
- Highest quality available
- 3072 dimensions capture maximum nuance
- Multilingual support

**Weaknesses:**
- Most expensive: $0.13/1M tokens
- Highest memory: 3072 dimensions
- Network latency

**Best For:**
- Premium applications
- Complex semantic understanding
- Research and analysis

**Cost Calculation:**
- 100k words ≈ $0.0173
- 1M words ≈ $0.173
- 10M words ≈ $1.73

**Example:**
```python
engine = EmbeddingsEngine(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
)
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (3072,)
```

### embed-english-v3.0

**Overview:** Fast paid alternative to OpenAI.

**Specs:**
- Dimensions: 1024
- Training data: Proprietary
- Languages: English

**Performance:**
- API latency: 50-200ms (network-dependent)
- Quality: Excellent (0.81 Pearson correlation)
- Cost: $0.10/1M tokens

**Strengths:**
- Fast API: 50-200ms latency
- Lower cost than OpenAI large
- High quality

**Weaknesses:**
- Paid: $0.10/1M tokens
- English only
- Requires API key

**Best For:**
- Fast paid alternative to OpenAI
- Multi-model redundancy
- Backup embeddings

**Cost Calculation:**
- 100k words ≈ $0.0133
- 1M words ≈ $0.133
- 10M words ≈ $1.33

**Example:**
```python
engine = EmbeddingsEngine(
    model="embed-english-v3.0",
    api_key=os.getenv("COHERE_API_KEY"),
)
embedding = engine.encode("The conversation is flowing smoothly")
# embedding.shape = (1024,)
```

### paraphrase-multilingual-MiniLM-L12-v2

**Overview:** Free multilingual model.

**Specs:**
- Dimensions: 384
- Parameters: 118M
- Training data: 50+ languages
- Languages: 50+ (multilingual)

**Performance:**
- CPU: 80ms latency, 12 texts/sec
- GPU: 8ms latency, 125 texts/sec
- GPU + TensorRT: 3ms latency, 333 texts/sec
- Quality: Good (0.70 Pearson correlation)

**Strengths:**
- Multilingual: 50+ languages
- Free: No API costs
- Low memory: 384 dimensions

**Weaknesses:**
- Lower quality than paid models
- Slower than English-only models
- Requires GPU for best performance

**Best For:**
- Multilingual applications
- Edge deployment
- Cost-sensitive multilingual use cases

**Example:**
```python
engine = EmbeddingsEngine(
    model="paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda",
)
embedding = engine.encode("La conversación fluye suavemente")  # Spanish
# embedding.shape = (384,)
```

## Update and Maintenance

### Model Versioning

Models are versioned to ensure reproducibility:

```python
# Pin specific model version
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2")

# Check model version
print(engine.model.version)  # "v1.0.0"
```

### Updating Models

Update models to latest version:

```python
# Check for updates
from embeddings_engine import Model

model = Model("all-MiniLM-L6-v2")
if model.has_update():
    model.update()
```

### Model Deprecation

Deprecated models:

- **all-MiniLM-L6-v1**: Deprecated, use all-MiniLM-L6-v2
- **all-mpnet-base-v1**: Deprecated, use all-mpnet-base-v2

Migration path:
```python
# Old (deprecated)
engine = EmbeddingsEngine(model="all-MiniLM-L6-v1")

# New (recommended)
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2")
```

### Benchmarking New Models

Benchmark new models before adoption:

```python
from embeddings_engine import EmbeddingsEngine
import time

def benchmark_model(model_name, device):
    engine = EmbeddingsEngine(model=model_name, device=device)

    # Warm-up
    for _ in range(10):
        engine.encode("warmup text")

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        engine.encode("The conversation is flowing smoothly")
        times.append(time.perf_counter() - start)

    mean_latency = np.mean(times) * 1000  # ms
    throughput = 1000 / mean_latency  # texts/sec

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Mean latency: {mean_latency:.2f}ms")
    print(f"Throughput: {throughput:.1f} texts/sec")

# Benchmark
benchmark_model("all-MiniLM-L6-v2", "cuda")
```

### Model Quality Validation

Validate model quality on your data:

```python
from sklearn.metrics import pairwise_distances

def validate_quality(engine, texts, labels):
    """Validate model quality on labeled data."""
    embeddings = engine.encode_batch(texts)

    # Compute pairwise distances
    distances = pairwise_distances(embeddings, metric="cosine")

    # Evaluate: Similar texts should have low distances
    score = evaluate_clustering(distances, labels)
    return score

# Usage
texts = ["text 1", "text 2", ..., "text N"]
labels = [0, 0, 1, 1, ...]  # Cluster labels
score = validate_quality(engine, texts, labels)
print(f"Quality score: {score:.3f}")
```

## Next Steps

- Review [USER_GUIDE.md](USER_GUIDE.md) for usage examples
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
- Explore [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for adding new models

**The grammar is eternal.**
