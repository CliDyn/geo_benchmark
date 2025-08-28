# Chunking System Update Summary

Updated both future scenario files with the new chunking scheme from the main climate_llm_benchmark.py.

## Files Updated

1. **climate_llm_benchmark_future_4.5.py** - SSP2-4.5 scenario (2070-2099)
2. **climate_llm_benchmark_future_8.5.py** - SSP5-8.5 scenario (2070-2099)

## Key Changes Applied

### 1. Enhanced Argument Parsing
```python
# Support multiple argument formats:
# python climate_llm_benchmark_future_4.5.py [config_file] [chunk_number]
# python climate_llm_benchmark_future_8.5.py chunk_number (uses default config)
```

### 2. Chunk Mode Configuration Support
```python
chunk_mode = get_config_value(config, 'benchmark.chunk_mode', False)
chunks_dir = get_config_value(config, 'benchmark.chunks_dir', 'meshes/chunks')
chunks_pattern = get_config_value(config, 'benchmark.chunks_pattern', 'mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json')
```

### 3. Enhanced Intermediate File Naming
**SSP2-4.5**: `climate_results_intermediate_50_gpt-5-nano_chunk_01_simple_scenario_SSP2-4.5.json`
**SSP5-8.5**: `climate_results_intermediate_50_gpt-5-nano_chunk_01_simple_scenario_SSP5-8.5.json`

### 4. Enhanced Final Result File Naming
**SSP2-4.5**: `climate_results_1.0deg_r10_gpt-5-nano_chunk_01_of_20_simple_scenario_SSP2-4.5.json`
**SSP5-8.5**: `climate_results_1.0deg_r10_gpt-5-nano_chunk_01_of_20_simple_scenario_SSP5-8.5.json`

### 5. Enhanced Response Parsing
- Added `extract_first_float()` function for reasoning model support
- Updated `validate_and_parse_response()` with provider/model parameters
- Support for Ollama reasoning models (deepseek-r1, qwen)

### 6. Google API Key Enhanced Configuration
- Support for both direct API key and environment variable patterns
- Backward compatibility with existing configurations

### 7. Chunk Processing Logic
- Automatic chunk detection from mesh file metadata
- Independent resume capability per chunk
- Chunk-specific intermediate file patterns

## Usage Examples

### SSP2-4.5 Scenario
```bash
# Enable chunk mode in config.yaml: chunk_mode: true

# Process individual chunks
python climate_llm_benchmark_future_4.5.py 1    # Chunk 1
python climate_llm_benchmark_future_4.5.py 15   # Chunk 15

# With custom config
python climate_llm_benchmark_future_4.5.py my_config.yaml 5

# Process all chunks in parallel
for i in {1..20}; do python climate_llm_benchmark_future_4.5.py $i & done; wait
```

### SSP5-8.5 Scenario
```bash
# Same pattern with different scenario
python climate_llm_benchmark_future_8.5.py 1    # Chunk 1
python climate_llm_benchmark_future_8.5.py 20   # Chunk 20

# Process all chunks in parallel
for i in {1..20}; do python climate_llm_benchmark_future_8.5.py $i & done; wait
```

## Configuration Requirements

Add to your `config.yaml`:
```yaml
benchmark:
  # ... existing settings ...
  
  # Chunk processing settings
  chunk_mode: true                          # Enable chunk processing
  chunks_dir: "meshes/chunks"               # Directory with chunk files  
  chunks_pattern: "mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json"  # Pattern
```

## Key Differences from Main File

Both future scenario files maintain their scenario-specific features:

### SSP2-4.5 Specific:
- **Scenario**: SSP2-4.5 ("Middle-of-the-Road" socio-economic pathway)
- **Temperature Range**: -100°C to 80°C (expanded for future warming)
- **Forcing**: ~4.5 W/m² effective radiative forcing by 2100
- **File Suffix**: `_scenario_SSP2-4.5.json`

### SSP5-8.5 Specific:
- **Scenario**: SSP5-8.5 ("Fossil-Fueled Development" pathway)
- **Temperature Range**: -100°C to 90°C (even higher for extreme warming)
- **Forcing**: ~8.5 W/m² effective radiative forcing by 2100
- **File Suffix**: `_scenario_SSP5-8.5.json`

## Backward Compatibility

- Regular (non-chunk) processing still works unchanged
- Existing config files work without modification
- Original file naming preserved when chunk_mode is false

Both future scenario files now support the complete distributed processing workflow while maintaining their scenario-specific climate projections and validation logic.