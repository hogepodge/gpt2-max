# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Repository Overview

The Modular Platform is a unified platform for AI development and deployment
that includes:

- **MAX**: High-performance inference server with OpenAI-compatible endpoints
for LLMs and AI models
- **Mojo**: A new programming language that bridges Python and systems
programming, optimized for AI workloads

## Essential Build Commands

### Global Build System (Bazel)

This directory uses pixi to build and run code. For testing, we are using the
command

```bash
pixi run generate
```

This will run the model end-to-end with the inputted text.

### Pixi Environment Management

Many directories include `pixi.toml` files for environment management. Use Pixi
when present:

```bash
# Install Pixi environment (run once per directory)
pixi install

# Run Mojo files through Pixi
pixi run mojo [file.mojo]

# Format Mojo code
pixi run mojo format ./

# Use predefined tasks from pixi.toml
pixi run main              # Run main example
pixi run test              # Run tests
pixi run hello             # Run hello.mojo

# Common Pixi tasks available in different directories:
# - /mojo/: build, tests, examples, benchmarks
# - /max/: llama3, mistral, generate, serve
# - /examples/*/: main, test, hello, dev-server, format

# List available tasks
pixi task list
```

### MAX Server Commands

```bash
# Install the MAX nightly within a Python virtual environment using pip
pip install modular --index-url https://dl.modular.com/public/nightly/python/simple/ --extra-index-url https://download.pytorch.org/whl/cpu

# Install MAX globally using Pixi, an alternative to the above
pixi global install -c conda-forge -c https://conda.modular.com/max-nightly

# Start OpenAI-compatible server
max serve --model-path=modularai/Llama-3.1-8B-Instruct-GGUF

# Run with Docker
docker run --gpus=1 -p 8000:8000 docker.modular.com/modular/max-nvidia-full:latest --model-path modularai/Llama-3.1-8B-Instruct-GGUF
```

## High-Level Architecture

This repository is an attempt to implement GPT2 using the MAX Platform.
Everything in the `gpt2` subdirectory is the implementation code. 
Everything in the `modular` subdirectory is the open source code for
MAX and Mojo.

### Repository Structure

```text
gpt2/
├── gpt2/                        # gpt2 implementation with modular
├──	modular/                     # open source modular repo with MAX and mojo, used for reference
│ 	├── mojo/                    # Mojo programming language
│ 	│   ├── stdlib/              # Standard library implementation
│ 	│   ├── docs/                # User documentation
│ 	│   ├── proposals/           # Language proposals (RFCs)
│ 	│   └── integration-test/    # Integration tests
│ 	├── max/                     # MAX framework
│ 	│   ├── kernels/             # High-performance Mojo kernels (GPU/CPU)
│ 	│   ├── serve/               # Python inference server (OpenAI-compatible)
│ 	│   ├── pipelines/           # Model architectures (Python)
│ 	│   └── nn/                  # Neural network operators (Python)
│ 	├── examples/                # Usage examples
│ 	├── benchmark/               # Benchmarking tools
│ 	└── bazel/                   # Build system configuration
└── tests/                       # tests
```

### Key Architectural Patterns

1. **Language Separation**:
   - Low-level performance kernels in Mojo (`max/kernels/`)
   - High-level orchestration in Python (`max/serve/`, `max/pipelines/`)

2. **Hardware Abstraction**:
   - Platform-specific optimizations via dispatch tables
   - Support for NVIDIA/AMD GPUs, Intel/Apple CPUs
   - Device-agnostic APIs with hardware-specific implementations

3. **Memory Management**:
   - Device contexts for GPU memory management
   - Host/Device buffer abstractions
   - Careful lifetime management in Mojo code

4. **Testing Philosophy**:
   - Tests mirror source structure
   - Use `lit` tool with FileCheck validation
   - Hardware-specific test configurations
   - Migrating to `testing` module assertions

## Development Workflow

### Code Style

- Use `mojo format` for Mojo code
- Follow existing patterns in the codebase
- Add docstrings to public APIs
- Sign commits with `git commit -s`

## Critical Development Notes

### Mojo Development

- Use nightly Mojo builds for development
- Avoid deprecated types like `Tensor` (use modern alternatives)
- Follow value semantics and ownership conventions
- Use `Reference` types with explicit lifetimes in APIs

### MAX Kernel Development

- Fine-grained control over memory layout and parallelism
- Hardware-specific optimizations (tensor cores, SIMD)
- Vendor library integration when beneficial
- Performance improvements must include benchmarks

### Common Pitfalls

- Always check Mojo function return values for errors
- Ensure coalesced memory access patterns on GPU
- Minimize CPU-GPU synchronization points
- Avoid global state in kernels
- Never commit secrets or large binary files

### Environment Variables

## LLM-friendly Documentation

- Docs index: <https://docs.modular.com/llms.txt>
- Mojo API docs: <https://docs.modular.com/llms-mojo.txt>
- Python API docs: <https://docs.modular.com/llms-python.txt>
- Comprehensive docs: <https://docs.modular.com/llms-full.txt>

## Git commit style

- **Atomic Commits:** Keep commits small and focused. Each commit should
address a single, logical change. This makes it easier to understand the
history and revert changes if needed.
- **Descriptive Commit Messages:** Write clear, concise, and informative commit
messages. Explain the *why* behind the change, not just *what* was changed. Use
a consistent format (e.g., imperative mood: "Fix bug", "Add feature").
- **Commit titles:** git commit titles should have the `[Stdlib]` or `[Kernel]`
depending on whether the kernel is modified and if they are modifying GPU
functions then they should use `[GPU]` tag as well.
- The commit messages should be surrounded by BEGIN_PUBLIC and END_PUBLIC
- Here is an example template a git commit

```git
[Kernels] Some new feature

BEGIN_PUBLIC
[Kernels] Some new feature

This add a new feature for [xyz] to enable [abc]
END_PUBLIC
```
