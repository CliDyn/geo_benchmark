# Enhanced Command Line Interface Usage

The climate_llm_benchmark.py script now supports enhanced command line arguments while maintaining full backward compatibility.

## New Argument-Based Usage

### Basic Usage
```bash
# Default configuration
python climate_llm_benchmark.py

# Custom configuration file
python climate_llm_benchmark.py --config=my_config.yaml
```

### Chunk Processing
```bash
# Process specific chunks with new arguments
python climate_llm_benchmark.py --chunk=1
python climate_llm_benchmark.py --chunk=15
python climate_llm_benchmark.py --config=my_config.yaml --chunk=5
```

### Base URL Override
```bash
# Override Ollama base URL
python climate_llm_benchmark.py --base_url=http://localhost:11434

# Override OpenAI base URL (for local OpenAI-compatible servers)
python climate_llm_benchmark.py --base_url=http://localhost:8080

# Combined: custom config, chunk processing, and base URL override
python climate_llm_benchmark.py --config=ollama_config.yaml --chunk=3 --base_url=http://192.168.1.100:11434
```

### Advanced Examples
```bash
# Process chunk 5 with custom Ollama server
python climate_llm_benchmark.py --chunk=5 --base_url=http://remote-server:11434

# Use different config and chunk with OpenAI-compatible server
python climate_llm_benchmark.py --config=openai_config.yaml --chunk=10 --base_url=http://localhost:8080
```

## Legacy Compatibility

All existing usage patterns continue to work:

```bash
# Legacy positional arguments still work
python climate_llm_benchmark.py config.yaml
python climate_llm_benchmark.py 5                    # chunk 5 with default config
python climate_llm_benchmark.py config.yaml 5       # chunk 5 with custom config
```

## Help and Documentation
```bash
python climate_llm_benchmark.py --help
```

Output:
```
usage: climate_llm_benchmark.py [-h] [--config CONFIG] [--chunk CHUNK]
                                [--base_url BASE_URL]
                                [legacy_args ...]

Climate LLM Benchmark

positional arguments:
  legacy_args          Legacy: [config_file] [chunk_number]

options:
  -h, --help           show this help message and exit
  --config CONFIG      Configuration file path (default: config.yaml)
  --chunk CHUNK        Chunk number to process (enables chunk mode)
  --base_url BASE_URL  Override base URL for model provider (e.g.,
                       http://localhost:11434)
```

## Configuration Integration

### Base URL Override Support

The `--base_url` argument overrides the configuration for these providers:

**Ollama**:
```yaml
# config.yaml - will be overridden by --base_url
providers:
  ollama:
    base_url: "http://localhost:11434"  # ← overridden
```

**OpenAI** (for compatible servers):
```yaml
# config.yaml - will be overridden by --base_url
providers:
  openai:
    base_url: null  # ← overridden
```

**Anthropic**:
```yaml
# config.yaml - will be overridden by --base_url  
providers:
  anthropic:
    base_url: null  # ← overridden
```

### Chunk Mode Auto-Detection

When `--chunk` is specified, chunk mode is automatically enabled regardless of config:

```yaml
# config.yaml
benchmark:
  chunk_mode: false  # ← ignored when --chunk is used
```

## Real-World Usage Examples

### Distributed Processing with Different Servers

```bash
# Split work across multiple Ollama servers
python climate_llm_benchmark.py --chunk=1 --base_url=http://server1:11434 &
python climate_llm_benchmark.py --chunk=2 --base_url=http://server2:11434 &  
python climate_llm_benchmark.py --chunk=3 --base_url=http://server3:11434 &
wait
```

### Local OpenAI-Compatible Servers

```bash
# Use with local OpenAI-compatible servers (e.g., vLLM, Text Generation WebUI)
python climate_llm_benchmark.py --config=local_openai.yaml --chunk=1 --base_url=http://localhost:8080
```

### Testing with Different Configurations

```bash
# Test different models/configs on same data chunks
python climate_llm_benchmark.py --config=gpt4_config.yaml --chunk=1 &
python climate_llm_benchmark.py --config=claude_config.yaml --chunk=1 &
python climate_llm_benchmark.py --config=ollama_config.yaml --chunk=1 --base_url=http://localhost:11434 &
wait
```

### Batch Processing Scripts

```bash
#!/bin/bash
# Process all chunks with custom Ollama server
for i in {1..20}; do
    echo "Starting chunk $i..."
    python climate_llm_benchmark.py --chunk=$i --base_url=http://192.168.1.100:11434 &
done
wait
echo "All chunks completed!"
```

## Output Information

When using overrides, the script shows the applied settings:

```
Climate LLM Benchmark
Configuration: config.yaml
Base URL override: http://localhost:8080  ← Shows override
Chunk mode: Enabled
Processing chunk: 5
Mesh file: meshes/chunks/mesh_data_1.0deg_chunk_05_of_20.json
Provider: openai
Model: gpt-5-nano
...
```

## Migration Guide

### From Legacy to New Format

**Old**:
```bash
python climate_llm_benchmark.py config.yaml 5
```

**New (equivalent)**:
```bash
python climate_llm_benchmark.py --config=config.yaml --chunk=5
```

**Benefits of New Format**:
- More explicit and self-documenting
- Support for base URL overrides
- Better help documentation
- Easier to extend with additional options

### Recommended Usage

For **new scripts and workflows**, use the argument-based format:
```bash
python climate_llm_benchmark.py --chunk=5 --base_url=http://localhost:11434
```

For **existing scripts**, no changes needed - legacy format continues to work.

## Error Messages

Enhanced error messages show both new and legacy usage:

```
Error: Chunk mode is enabled but no chunk number specified.
Usage: python climate_llm_benchmark.py --chunk=N
   or: python climate_llm_benchmark.py --config=config.yaml --chunk=N  
Legacy: python climate_llm_benchmark.py [config_file] chunk_number
```

This enhanced CLI provides better usability while maintaining complete backward compatibility for existing workflows.