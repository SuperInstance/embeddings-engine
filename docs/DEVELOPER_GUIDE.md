# Developer Guide

Guide for contributing to and extending embeddings-engine.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Adding New Model Backends](#adding-new-model-backends)
4. [Testing Strategies](#testing-strategies)
5. [Benchmarking Methodology](#benchmarking-methodology)
6. [Release Process](#release-process)
7. [Code Style Guidelines](#code-style-guidelines)
8. [Performance Optimization](#performance-optimization)

## Development Setup

### Prerequisites

- Python 3.9+
- Rust 1.70+ (for Rust components)
- CUDA 11.8+ or 12.x (for GPU development)
- Redis 6+ (for caching tests)
- Git

### Clone Repository

```bash
git clone https://github.com/equilibrium-tokens/embeddings-engine.git
cd embeddings-engine
```

### Python Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
pip install -e ".[cuda]"  # For GPU support
pip install -e ".[all]"   # For all dependencies

# Install pre-commit hooks
pre-commit install
```

### Rust Development Setup

```bash
# Rust toolchain
cd rust/
cargo build
cargo test
cargo fmt
cargo clippy
```

### Verify Setup

```bash
# Run Python tests
pytest tests/

# Run Rust tests
cd rust/
cargo test

# Run integration tests
pytest tests/integration/
```

## Project Structure

```
embeddings-engine/
├── python/
│   ├── embeddings_engine/
│   │   ├── __init__.py
│   │   ├── engine.py          # EmbeddingsEngine main class
│   │   ├── models/            # Model implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # Model abstract base class
│   │   │   ├── sentencetransformers.py
│   │   │   ├── openai.py
│   │   │   ├── cohere.py
│   │   │   ├── onnx.py
│   │   │   └── tensorrt.py
│   │   ├── cache/             # Cache implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # Cache abstract base class
│   │   │   ├── redis.py
│   │   │   └── memory.py
│   │   └── utils/             # Utility functions
│   │       ├── __init__.py
│   │       ├── tokenization.py
│   │       └── metrics.py
│   ├── tests/
│   │   ├── unit/              # Unit tests
│   │   ├── integration/       # Integration tests
│   │   └── benchmarks/        # Benchmark tests
│   └── pyproject.toml
├── rust/
│   ├── src/
│   │   ├── lib.rs             # Rust library entry point
│   │   ├── engine.rs          # Core engine logic
│   │   ├── models.rs          # Model implementations
│   │   ├── cache.rs           # Cache implementations
│   │   └── cuda.rs            # CUDA kernels
│   ├── Cargo.toml
│   └── benches/               # Rust benchmarks
├── docs/                      # Documentation
├── examples/                  # Example usage
└── README.md
```

### Python Package Structure

```python
# embeddings_engine/__init__.py
from .engine import EmbeddingsEngine
from .models import Model, SentenceTransformersModel, OpenAIModel, CohereModel
from .cache import Cache, RedisCache, MemoryCache

__version__ = "0.1.0"
__all__ = [
    "EmbeddingsEngine",
    "Model",
    "SentenceTransformersModel",
    "OpenAIModel",
    "CohereModel",
    "Cache",
    "RedisCache",
    "MemoryCache",
]
```

## Adding New Model Backends

### Step 1: Create Model Class

Create new model class in `python/embeddings_engine/models/`:

```python
# python/embeddings_engine/models/custom.py
from .base import Model
import numpy as np
from typing import List

class CustomModel(Model):
    """Custom embedding model backend."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._dimension = 768  # Set your model's dimension
        # Initialize your model here
        self.model = self._load_model(model_name, **kwargs)

    def _load_model(self, model_name: str, **kwargs):
        """Load and initialize model."""
        # Implement model loading logic
        pass

    def encode(self, text: str, device: str = "cpu") -> np.ndarray:
        """Convert single text to embedding."""
        # Implement encoding logic
        embedding = self.model.encode(text)
        return np.array(embedding, dtype=np.float32)

    def encode_batch(self, texts: List[str], device: str = "cpu") -> np.ndarray:
        """Convert batch of texts to embeddings."""
        embeddings = self.model.encode_batch(texts)
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
```

### Step 2: Register Model

Register model in `python/embeddings_engine/models/__init__.py`:

```python
# python/embeddings_engine/models/__init__.py
from .base import Model
from .sentencetransformers import SentenceTransformersModel
from .openai import OpenAIModel
from .cohere import CohereModel
from .onnx import ONNXModel
from .tensorrt import TensorRTModel
from .custom import CustomModel  # Add your model

__all__ = [
    "Model",
    "SentenceTransformersModel",
    "OpenAIModel",
    "CohereModel",
    "ONNXModel",
    "TensorRTModel",
    "CustomModel",  # Add your model
]
```

### Step 3: Update Engine

Update `EmbeddingsEngine` to use new model:

```python
# python/embeddings_engine/engine.py
from .models import CustomModel

class EmbeddingsEngine:
    def _load_model(self, model: str, backend: str = "auto"):
        if backend == "auto":
            backend = self._detect_backend(model)

        if backend == "sentencetransformers":
            return SentenceTransformersModel(model)
        elif backend == "openai":
            return OpenAIModel(model, api_key=self.api_key)
        elif backend == "cohere":
            return CohereModel(model, api_key=self.api_key)
        elif backend == "custom":  # Add your backend
            return CustomModel(model, **self.model_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
```

### Step 4: Add Tests

Add tests in `python/tests/unit/test_custom_model.py`:

```python
# python/tests/unit/test_custom_model.py
import pytest
import numpy as np
from embeddings_engine.models import CustomModel

@pytest.fixture
def model():
    return CustomModel("custom-model")

def test_encode(model):
    """Test single text encoding."""
    text = "The conversation is flowing smoothly"
    embedding = model.encode(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.dtype == np.float32
    assert embedding.shape == (768,)  # Model dimension
    assert not np.isnan(embedding).any()

def test_encode_batch(model):
    """Test batch encoding."""
    texts = ["text 1", "text 2", "text 3"]
    embeddings = model.encode_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 768)
    assert not np.isnan(embeddings).any()

def test_dimension(model):
    """Test dimension property."""
    assert model.dimension == 768

def test_deterministic(model):
    """Test encoding is deterministic."""
    text = "The conversation is flowing smoothly"
    emb1 = model.encode(text)
    emb2 = model.encode(text)

    assert np.array_equal(emb1, emb2)
```

### Step 5: Add Documentation

Document in `docs/MODELS.md`:

```markdown
### CustomModel

**Dimensions**: 768
**Latency**: 10ms (GPU)
**Cost**: Free
**Quality**: Excellent

**Best For**: Custom use cases

**Example**:
```python
engine = EmbeddingsEngine(
    model="custom-model",
    backend="custom",
    device="cuda",
)
```
```

## Testing Strategies

### Unit Tests

Test individual components in isolation:

```python
# tests/unit/test_cache.py
import pytest
import numpy as np
from embeddings_engine.cache import MemoryCache

@pytest.fixture
def cache():
    return MemoryCache(max_size=100)

def test_cache_hit(cache):
    """Test cache hit returns cached value."""
    key = "test_key"
    value = np.array([1.0, 2.0, 3.0])

    cache.set(key, value)
    result = cache.get(key)

    assert np.array_equal(result, value)

def test_cache_miss(cache):
    """Test cache miss returns None."""
    result = cache.get("nonexistent_key")
    assert result is None

def test_cache_eviction(cache):
    """Test LRU eviction when max_size exceeded."""
    cache = MemoryCache(max_size=2)

    cache.set("key1", np.array([1.0]))
    cache.set("key2", np.array([2.0]))
    cache.set("key3", np.array([3.0]))  # Evicts key1

    assert cache.get("key1") is None
    assert np.array_equal(cache.get("key2"), np.array([2.0]))
    assert np.array_equal(cache.get("key3"), np.array([3.0]))
```

### Integration Tests

Test component interactions:

```python
# tests/integration/test_engine.py
import pytest
import numpy as np
from embeddings_engine import EmbeddingsEngine, Cache

@pytest.fixture
def engine():
    return EmbeddingsEngine(
        model="all-MiniLM-L6-v2",
        device="cpu",
        cache=Cache.Memory(max_size=100),
    )

def test_encode_with_cache(engine):
    """Test encoding with cache."""
    text = "The conversation is flowing smoothly"

    # First call: Compute and cache
    emb1 = engine.encode(text)

    # Second call: Return cached
    emb2 = engine.encode(text)

    assert np.array_equal(emb1, emb2)

def test_encode_batch(engine):
    """Test batch encoding."""
    texts = ["text 1", "text 2", "text 3"]
    embeddings = engine.encode_batch(texts, batch_size=2)

    assert embeddings.shape == (3, 384)
```

### Performance Tests

Test performance characteristics:

```python
# tests/benchmarks/test_latency.py
import pytest
import time
from embeddings_engine import EmbeddingsEngine

@pytest.mark.benchmark
def test_single_text_latency():
    """Benchmark single text encoding latency."""
    engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
    text = "The conversation is flowing smoothly"

    # Warm-up
    for _ in range(10):
        engine.encode(text)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        engine.encode(text)
        times.append(time.perf_counter() - start)

    mean_latency = np.mean(times) * 1000  # Convert to ms
    p95_latency = np.percentile(times, 95) * 1000

    print(f"Mean latency: {mean_latency:.2f}ms")
    print(f"P95 latency: {p95_latency:.2f}ms")

    assert mean_latency < 10  # Should be < 10ms on GPU
    assert p95_latency < 15  # P95 should be < 15ms
```

### GPU Tests

Test GPU-specific functionality:

```python
# tests/gpu/test_cuda_graphs.py
import pytest
import torch
from embeddings_engine import EmbeddingsEngine

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_graphs():
    """Test CUDA Graph optimization."""
    engine = EmbeddingsEngine(
        model="all-MiniLM-L6-v2",
        device="cuda",
        use_cuda_graphs=True,
    )

    text = "The conversation is flowing smoothly"
    embedding = engine.encode(text)

    assert embedding is not None
    assert embedding.shape == (384,)
```

## Benchmarking Methodology

### Benchmark Suite

Run comprehensive benchmarks:

```python
# scripts/benchmark.py
import time
import numpy as np
from embeddings_engine import EmbeddingsEngine

def benchmark_model(model_name, device, num_iterations=100):
    """Benchmark model performance."""
    engine = EmbeddingsEngine(model=model_name, device=device)
    text = "The conversation is flowing smoothly"

    # Warm-up
    for _ in range(10):
        engine.encode(text)

    # Benchmark single text
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        engine.encode(text)
        times.append(time.perf_counter() - start)

    mean_time = np.mean(times) * 1000  # ms
    p95_time = np.percentile(times, 95) * 1000  # ms
    throughput = 1000 / mean_time  # texts/sec

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Mean latency: {mean_time:.2f}ms")
    print(f"P95 latency: {p95_time:.2f}ms")
    print(f"Throughput: {throughput:.1f} texts/sec")

    # Benchmark batch
    texts = [text] * 32
    start = time.perf_counter()
    embeddings = engine.encode_batch(texts, batch_size=32)
    batch_time = time.perf_counter() - start

    batch_per_text = batch_time / 32 * 1000  # ms per text
    batch_throughput = 32 / batch_time  # texts/sec

    print(f"Batch latency (per text): {batch_per_text:.2f}ms")
    print(f"Batch throughput: {batch_throughput:.1f} texts/sec")

    return {
        "mean_latency": mean_time,
        "p95_latency": p95_time,
        "throughput": throughput,
        "batch_per_text": batch_per_text,
        "batch_throughput": batch_throughput,
    }

if __name__ == "__main__":
    # Benchmark all models
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ]

    for model in models:
        print(f"\n{'='*60}")
        benchmark_model(model, device="cpu")
        if torch.cuda.is_available():
            benchmark_model(model, device="cuda")
```

### Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| Single text (CPU) | < 100ms | < 150ms |
| Single text (GPU) | < 10ms | < 15ms |
| Batch (GPU) | < 2ms per text | < 3ms per text |
| Throughput (GPU) | > 500 texts/sec | > 200 texts/sec |
| Cache hit latency | < 0.5ms | < 1ms |

### Profiling

Profile performance bottlenecks:

```python
# scripts/profile.py
import cProfile
import pstats
from embeddings_engine import EmbeddingsEngine

def profile_encode():
    """Profile encoding performance."""
    engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")
    text = "The conversation is flowing smoothly"

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(100):
        engine.encode(text)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)  # Top 20 functions

if __name__ == "__main__":
    profile_encode()
```

## Release Process

### Version Bump

Update version in multiple files:

```bash
# Update version in __init__.py
vim python/embeddings_engine/__init__.py
# __version__ = "0.2.0"

# Update version in pyproject.toml
vim python/pyproject.toml
# version = "0.2.0"

# Update version in Cargo.toml
vim rust/Cargo.toml
# version = "0.2.0"
```

### Changelog

Update CHANGELOG.md:

```markdown
# Changelog

## [0.2.0] - 2026-01-15

### Added
- New CustomModel backend for custom embeddings
- CUDA Graph optimization support
- TensorRT optimization support

### Changed
- Improved batch processing performance by 2×
- Reduced GPU memory usage by 30%

### Fixed
- Fixed cache eviction bug in MemoryCache
- Fixed dimension mismatch in OpenAIModel

### Removed
- Removed deprecated Python 3.8 support
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=embeddings_engine --cov-report=html

# Run GPU tests
pytest tests/gpu/ -m gpu

# Run benchmarks
pytest tests/benchmarks/ -m benchmark
```

### Build Python Package

```bash
cd python/

# Build wheel
python -m build

# Check package
twine check dist/*

# Upload to PyPI (test)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

### Build Rust Package

```bash
cd rust/

# Build release
cargo build --release

# Run tests
cargo test --release

# Publish to crates.io
cargo publish
```

### Tag Release

```bash
# Create tag
git tag -a v0.2.0 -m "Release v0.2.0"

# Push tag
git push origin v0.2.0

# Create GitHub release
gh release create v0.2.0 --notes "Release v0.2.0"
```

## Code Style Guidelines

### Python Style

Follow PEP 8 with additional guidelines:

```python
# Good: Clear variable names
embedding = engine.encode(text)

# Bad: Unclear abbreviations
emb = eng.enc(txt)

# Good: Type hints
def encode(text: str) -> np.ndarray:
    pass

# Bad: No type hints
def encode(text):
    pass

# Good: Docstrings
def encode(text: str) -> np.ndarray:
    """Convert text to embedding.

    Args:
        text: Input text to encode.

    Returns:
        Embedding vector as numpy array.
    """
    pass

# Bad: No docstrings
def encode(text):
    pass
```

### Rust Style

Follow standard Rust style:

```rust
// Good: Clear naming
pub fn encode(text: &str) -> Vec<f32> {
    // ...
}

// Bad: Unclear naming
pub fn enc(t: &str) -> Vec<f32> {
    // ...
}

// Good: Documentation
/// Converts text to embedding vector.
///
/// # Arguments
///
/// * `text` - Input text to encode.
///
/// # Returns
///
/// Embedding vector as f32 array.
pub fn encode(text: &str) -> Vec<f32> {
    // ...
}
```

### Formatting

```bash
# Python: Use black and isort
black python/
isort python/

# Rust: Use rustfmt
rustfmt rust/

# Check formatting
black --check python/
isort --check-only python/
rustfmt --check rust/
```

### Linting

```bash
# Python: Use pylint and flake8
pylint python/embeddings_engine
flake8 python/embeddings_engine

# Rust: Use clippy
cargo clippy -- -D warnings
```

## Performance Optimization

### Optimization Checklist

- [ ] Enable GPU acceleration
- [ ] Enable CUDA Graphs
- [ ] Enable TensorRT optimization
- [ ] Use batch processing
- [ ] Enable caching
- [ ] Use quantization (INT8/FP8)
- [ ] Profile bottlenecks
- [ ] Optimize data loading
- [ ] Reduce memory copies
- [ ] Use async I/O for network requests

### Optimization Example

```python
# Before: Slow
engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cpu")
embeddings = [engine.encode(text) for text in texts]

# After: Fast (10× faster)
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    device="cuda",
    use_cuda_graphs=True,
    use_tensorrt=True,
    cache=Cache.Redis("redis://localhost"),
)
embeddings = engine.encode_batch(texts, batch_size=32)
```

### Profiling Bottlenecks

```python
import cProfile
import pstats

def profile_bottlenecks():
    engine = EmbeddingsEngine(model="all-MiniLM-L6-v2", device="cuda")

    profiler = cProfile.Profile()
    profiler.enable()

    # Run workload
    texts = ["text"] * 1000
    embeddings = engine.encode_batch(texts, batch_size=32)

    profiler.disable()

    # Analyze
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

profile_bottlenecks()

# Output shows:
# - ncalls: Number of calls
# - tottime: Total time in function
# - cumtime: Cumulative time (including subcalls)
# - filename:lineno(function): Function location
```

### Memory Optimization

```python
# Before: High memory usage
embeddings = engine.encode_batch(texts, batch_size=32)
# Memory: 1000 texts × 384 dims × 4 bytes = 1.5 MB (FP32)

# After: Low memory usage with quantization
engine = EmbeddingsEngine(
    model="all-MiniLM-L6-v2",
    quantization="int8",
)
embeddings = engine.encode_batch(texts, batch_size=32)
# Memory: 1000 texts × 384 dims × 1 byte = 384 KB (INT8)
# Savings: 4× less memory
```

## Next Steps

- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [USER_GUIDE.md](USER_GUIDE.md) for user-facing features
- Explore [MODELS.md](MODELS.md) for model details

**The grammar is eternal.**
