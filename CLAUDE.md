# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llama.cpp is a C/C++ LLM inference engine built on the ggml tensor library. It is the main playground for developing new features for ggml. The public C API is in `include/llama.h`.

### Important Policies

- **Review AGENTS.md before beginning any work.** This project does not accept PRs that are fully or predominantly AI-generated. AI tools may be used only in an assistive capacity. Contributors must be able to explain every line of code they submit. Do not write PR descriptions, commit messages, or responses to reviewers with AI.
- **Review CONTRIBUTING.md before submitting changes.** In particular: search existing issues/PRs first, run the full CI locally before publishing, verify perplexity and performance are not negatively affected, and focus on CPU support first when adding new models or features.
- Squash-merge PRs using the format: `<module> : <commit title> (#<issue_number>)`.

## Build System

The project uses **CMake exclusively**. The root `Makefile` is a redirect that errors and points to CMake.

### Basic Build

```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### Debug Build

For single-config generators (Unix Makefiles, Ninja):
```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

For multi-config generators (Xcode, Visual Studio):
```bash
cmake -B build -G "Xcode"
cmake --build build --config Debug
```

### Common CMake Options

- `-DGGML_CUDA=ON` — NVIDIA CUDA backend
- `-DGGML_METAL=ON` / `OFF` — Apple Metal (enabled by default on macOS)
- `-DGGML_VULKAN=ON` — Vulkan backend
- `-DGGML_SYCL=ON` — Intel SYCL backend
- `-DGGML_HIP=ON` — AMD HIP backend
- `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` — BLAS acceleration
- `-DBUILD_SHARED_LIBS=OFF` — Static build
- `-DGGML_NATIVE=OFF` — Build for all architectures (larger binary, no JIT)
- `-DLLAMA_BUILD_TESTS=ON` — Build tests (default ON when standalone)
- `-DLLAMA_BUILD_SERVER=ON` — Build llama-server (default ON when standalone)
- `-DLLAMA_BUILD_WEBUI=ON` — Build embedded Web UI for server
- `-DLLAMA_FATAL_WARNINGS=ON` — Treat warnings as errors

Backend-specific tuning options are documented in `docs/build.md` and backend-specific docs under `docs/backend/`.

## Running Tests

Tests are registered with CTest. Build first, then run tests from the build directory.

### Run All Tests

```bash
cd build && ctest --verbose --timeout 900
```

### Run Tests by Label

```bash
# Core tests (fast, no model download required)
ctest -L main --verbose --timeout 900

# Tests that require downloading a model
ctest -L model --verbose --timeout 900

# Python tests
ctest -L python --verbose --timeout 900
```

### Run a Specific Test

```bash
# By test name regex
ctest -R test-tokenizer-0-llama-spm --verbose

# Or run the binary directly
./build/bin/test-tokenizer-0 ./models/ggml-vocab-llama-spm.gguf
```

### Debug a Test

Use the provided script for short feedback loops:
```bash
./scripts/debug-test.sh test-tokenizer
```

Or manually find the exact command and run under GDB:
```bash
cd build && ctest -R "test-tokenizer" -V -N  # shows exact command line
gdb --args ./bin/test-tokenizer-0 ../models/ggml-vocab-llama-spm.gguf
```

### Python Tests

```bash
# gguf-py tests
cd gguf-py && pytest

# Or via root project (if installed)
pytest gguf-py/tests/
```

## Local CI

Before publishing changes, run the full CI locally:

```bash
mkdir -p tmp/results tmp/mnt

# CPU-only
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With SYCL (requires oneAPI environment)
source /opt/intel/oneapi/setvars.sh
GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Code Formatting and Linting

- **C/C++**: Format with `clang-format` using `.clang-format` (C++17, 120 column limit, 4-space indent, middle pointer alignment). The CI uses `git-clang-format` to check only changed code.
- **Python**: Flake8 with `.flake8` config. Pre-commit hooks enforce trailing-whitespace, end-of-file-fixer, check-yaml, and check-added-large-files.
- **C++ static analysis**: `.clang-tidy` is present but not enforced in CI.

## High-Level Architecture

### Layer Stack

```
Tools / Examples          (tools/*, examples/*)
    |
Common library            (common/)
    |
Llama library             (src/)
    |
GGML library              (ggml/)
    |
Backends                  (CPU, CUDA, Metal, Vulkan, SYCL, HIP, ...)
```

### GGML (Tensor Engine)

- `ggml/include/ggml.h` — Core tensor API (~2800 lines, defines ops, types, graph)
- `ggml/include/ggml-backend.h` — Backend abstraction layer
- `ggml/include/gguf.h` — GGUF file format
- `ggml/src/ggml.c` — Core tensor operations
- `ggml/src/ggml-backend.cpp` — Backend scheduler and registry
- `ggml/src/ggml-quants.c` — Quantization/dequantization kernels
- `ggml/src/ggml-alloc.c` — Graph memory allocator
- `ggml/src/gguf.cpp` — GGUF serialization/deserialization
- Backends live in `ggml/src/ggml-<backend>/`

Backends can be linked statically (default) or loaded dynamically at runtime (`GGML_BACKEND_DL=ON`). The scheduler distributes ops across available backends.

### Llama Library (`src/`)

- `llama.h` / `llama.cpp` — Public C API surface
- `llama-model.cpp` / `llama-model.h` — Model loading, architecture dispatch, tensor mapping (~546KB)
- `llama-model-loader.cpp` — GGUF file loading
- `llama-context.cpp` / `llama-context.h` — Inference context, KV cache management, decoding loop
- `llama-graph.cpp` / `llama-graph.h` — GGML graph construction for forward pass
- `llama-sampler.cpp` — Sampling pipeline (temperature, top-k, top-p, min-p, etc.)
- `llama-vocab.cpp` — Tokenizer (BPE, SentencePiece, WordPiece)
- `llama-kv-cache.cpp` — KV cache implementation
- `llama-arch.cpp` / `llama-arch.h` — Architecture registry and metadata
- `src/models/*.cpp` — Per-model-architecture graph builders (140+ architectures)

### Common Library (`common/`)

Shared utilities used by tools and examples:
- `arg.cpp` / `arg.h` — Command-line argument parsing
- `common.cpp` / `common.h` — Shared params, system info, progress bars
- `chat.cpp` / `chat.h` — Chat template application
- `sampling.cpp` / `sampling.h` — High-level sampling pipeline wrappers
- `jinja/` — Jinja2 template engine for chat templates
- `json-schema-to-grammar.cpp` — JSON Schema to GBNF grammar conversion

### Tools

- `tools/cli/` — `llama-cli` interactive CLI
- `tools/server/` — `llama-server` OpenAI-compatible HTTP API
- `tools/quantize/` — `llama-quantize` model quantization
- `tools/llama-bench/` — Benchmarking tool
- `tools/perplexity/` — Perplexity evaluation
- `tools/mtmd/` — Multimodal (vision) encoder library

### Server Architecture (`tools/server/`)

The server supports two modes:
- **Inference mode**: Single loaded GGUF model
- **Router mode**: Multiple inference instances behind one API endpoint

Key components (see `tools/server/README-dev.md` for full details):
- `server_context` — Main inference state, llama_context, and active slots
- `server_slot` — Single sequence abstraction
- `server_routes` — JSON parsing/formatting and request routing
- `server_http_context` — HTTP server via `cpp-httplib`
- `server_queue` / `server_response` — Thread-safe queues between HTTP workers and inference thread
- `server_models` — Backend instance management (router mode only)

Server scope rules: backend handles text completion, embeddings, chat completion, tool calling, third-party API compatibility, multimodal I/O, and memory management. Out of scope: server-side agentic loops, exposing internal model state, model-specific API features, third-party frontend plugins, and customizable themes.

**Security**: features that read/write external files must be disabled by default (MCP, model save/load).

## Adding a New Model Architecture

The process spans Python conversion scripts and C++ graph builders. See `docs/development/HOWTO-add-model.md` for the full procedure. In brief:

1. **Python conversion**: Add `ModelBase.register` subclass in `convert_hf_to_gguf.py`, define tensor layout in `gguf-py/gguf/constants.py` (`MODEL_ARCH`, `MODEL_TENSORS`), and map tensor names in `gguf-py/gguf/tensor_mapping.py`.
2. **C++ registration**: Add `llm_arch` enum in `src/llama-arch.h`, register in `src/llama-arch.cpp` (`LLM_ARCH_NAMES`).
3. **Graph builder**: Implement architecture-specific forward pass in `src/models/<arch>.cpp` following existing patterns.
4. **Verify**: Run `llama-cli`, `llama-perplexity`, `llama-bench`, and `test-backend-ops` on CPU, CUDA, and Metal backends.

## Python Tooling

- `gguf-py/` — Standalone Python package for reading/writing GGUF. Installable independently (`pip install ./gguf-py`).
- `convert_hf_to_gguf.py` — Primary HuggingFace-to-GGUF conversion script.
- `convert_lora_to_gguf.py` — LoRA adapter conversion.
- `convert_llama_ggml_to_gguf.py` — Legacy GGML to GGUF conversion.
- Root `pyproject.toml` defines dependencies for conversion scripts. Uses Poetry/PEP 621 with uv support.

## Key Files for Common Tasks

| Task | Primary Files |
|------|---------------|
| Public API changes | `include/llama.h` |
| New model architecture | `src/llama-arch.h`, `src/llama-arch.cpp`, `src/models/<arch>.cpp`, `gguf-py/gguf/constants.py`, `convert_hf_to_gguf.py` |
| New GGML op | `ggml/include/ggml.h`, `ggml/src/ggml.c`, backend-specific implementations |
| Backend bug | `ggml/src/ggml-<backend>/`, `tests/test-backend-ops.cpp` |
| Tokenizer bug | `src/llama-vocab.cpp`, `tests/test-tokenizer-0.cpp`, `tests/test-chat-template.cpp` |
| Server feature | `tools/server/*.cpp`, `tools/server/README-dev.md` |
| Chat template | `common/chat.cpp`, `common/jinja/` |
| Quantization | `ggml/src/ggml-quants.c`, `src/llama-quant.cpp`, `tools/quantize/` |
| Sampling | `src/llama-sampler.cpp`, `common/sampling.cpp` |
