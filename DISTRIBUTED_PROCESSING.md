# Distributed Processing Guide

Instructions for distributed processing with the Climate LLM Benchmark.

## Overview

Chunk mode splits meshes into pieces with equal land point distribution. Features:

- Automatic load balancing: Equal land points per chunk
- Zero configuration per chunk: Set chunk mode once, specify chunk number via command line
- Independent processing: Each chunk runs independently with separate resume capability
- Scalable parallelism: Process across multiple cores/machines
- Fault tolerance: Failed chunks don't affect others

## Quick Start

### 1. Enable Chunk Mode

Edit your `config.yaml`:

```yaml
benchmark:
  # ... other settings ...
  
  # Chunk processing settings
  chunk_mode: true                          # Enable chunk processing mode
  chunks_dir: "meshes/chunks"               # Directory containing chunk files
  chunks_pattern: "mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json"  # Pattern for chunk files
```

### 2. Create Chunks

```bash
# Split your mesh into 20 chunks (or any number)
python split_mesh.py meshes/mesh_data_1.0deg.json 20
```

### 3. Run Individual Chunks

```bash
# Process specific chunks (no config changes needed!)
python climate_llm_benchmark.py 1   # Process chunk 1 (~equal land points)
python climate_llm_benchmark.py 5   # Process chunk 5
python climate_llm_benchmark.py 20  # Process chunk 20

# With custom config file
python climate_llm_benchmark.py my_config.yaml 3

# System automatically:
# - Detects total chunks (e.g., 20) by counting files  
# - Loads correct chunk file (e.g., mesh_data_1.0deg_chunk_01_of_20.json)
# - Processes only that chunk's land points (e.g., 150 points instead of 3000)
# - Creates unique result/intermediate files per chunk
```

### 4. Run All Chunks in Parallel

```bash
# Run all chunks in background
for i in {1..20}; do
  echo "Starting chunk $i..."
  python climate_llm_benchmark.py $i &
done
wait  # Wait for all to complete
```

### 5. Combine Results

```bash
# Combine all chunk results into single file
python combine_results.py "results/climate_results_*_chunk_*_simple.json" results/combined_results.json
```

## File Structure

### Input Files
```
meshes/
├── mesh_data_1.0deg.json                    # Original mesh file
└── chunks/
    ├── mesh_data_1.0deg_chunk_01_of_20.json # Chunk 1
    ├── mesh_data_1.0deg_chunk_02_of_20.json # Chunk 2
    └── ...                                   # Chunks 3-20
```

### Output Files
```
results/
├── climate_results_1.0deg_r10_gpt-5-nano_chunk_01_of_20_simple.json  # Chunk 1 results
├── climate_results_1.0deg_r10_gpt-5-nano_chunk_02_of_20_simple.json  # Chunk 2 results
├── climate_results_intermediate_50_gpt-5-nano_chunk_01_simple.json    # Chunk 1 intermediate
├── climate_results_intermediate_100_gpt-5-nano_chunk_02_simple.json   # Chunk 2 intermediate
└── combined_results.json                                               # Final combined results
```

## Configuration Options

### Chunk Mode Settings

```yaml
benchmark:
  # Standard settings
  mesh_file: "meshes/mesh_data_1.0deg.json"  # Used when chunk_mode is false
  num_repeats: 10
  simple_mode: true
  # ... other settings ...
  
  # Chunk processing settings
  chunk_mode: true                          # Enable/disable chunk processing
  chunks_dir: "meshes/chunks"               # Directory with chunk files
  chunks_pattern: "mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json"  # File pattern
```

The `chunks_pattern` uses Python string formatting:
- `{:02d}` gets replaced with zero-padded chunk numbers
- Pattern must match your actual chunk file names

### Automatic Pattern Detection

The system automatically:
- Detects total number of chunks by counting matching files
- Validates chunk file existence
- Uses appropriate file naming for results and intermediates

## Command Line Usage

### Simple Chunk Processing

The enhanced system eliminates the need to modify config files for each chunk:

```bash
# ONE-TIME SETUP: Enable chunk mode in config.yaml
chunk_mode: true
chunks_pattern: "mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json"

# THEN: Run any chunk with simple command line arguments
python climate_llm_benchmark.py 1     # Chunk 1
python climate_llm_benchmark.py 15    # Chunk 15
python climate_llm_benchmark.py 20    # Chunk 20
```

### Argument Formats

```bash
# Process chunk N with default config
python climate_llm_benchmark.py N

# Process chunk N with custom config
python climate_llm_benchmark.py config_file.yaml N

# Regular processing (no chunks) with custom config
python climate_llm_benchmark.py config_file.yaml
```

### Processing Verification

Each chunk processes only its assigned points:

```bash
# Example output showing chunk processing:
# Chunk mode enabled: Processing chunk 1 of 20
# Loading mesh data from meshes/chunks/mesh_data_1.0deg_chunk_01_of_20.json...
# Found 147 land points  # <- Only chunk 1's points, not full 3000+ points
```

## Resume Functionality

Each chunk has independent resume capability:

```bash
# Enable resume in config.yaml
benchmark:
  resume: true

# Each chunk resumes from its own intermediate files
python climate_llm_benchmark.py 1  # Resumes chunk 1 from latest intermediate
python climate_llm_benchmark.py 5  # Resumes chunk 5 from latest intermediate
```

Intermediate files are automatically named with chunk information to prevent conflicts:
- `climate_results_intermediate_50_gpt-5-nano_chunk_01_simple.json`
- `climate_results_intermediate_25_gpt-5-nano_chunk_05_simple.json`

## Error Handling

### Missing Chunk Files
```
Error: Chunk file 'meshes/chunks/mesh_data_1.0deg_chunk_15_of_20.json' not found.
Available chunks: ['chunk_01_of_20.json', 'chunk_02_of_20.json', ...]
```

### Chunk Mode Issues
```
Error: Chunk mode is enabled but no chunk number specified.
Usage: python climate_llm_benchmark.py [config_file] chunk_number
```

### Pattern Mismatch
```
Error: No chunk files found in meshes/chunks matching pattern mesh_data_*.json
```

## Advanced Usage

### Custom Chunk Distribution

```bash
# Create fewer, larger chunks for faster processing
python split_mesh.py meshes/mesh_data_1.0deg.json 10

# Create many small chunks for fine-grained parallelism
python split_mesh.py meshes/mesh_data_1.0deg.json 50
```

### Multiple Configurations

```bash
# Run different models on different chunks
python climate_llm_benchmark.py openai_config.yaml 1 &
python climate_llm_benchmark.py anthropic_config.yaml 2 &
python climate_llm_benchmark.py google_config.yaml 3 &
```

### Monitoring Progress

```bash
# Monitor chunk processing
for i in {1..20}; do
  echo "Chunk $i status:"
  ls -la results/*chunk_$(printf "%02d" $i)*
  echo
done
```

## Benefits

### Scalability
- **Massive Parallelism**: Process thousands of points across multiple cores/machines
- **Equal Load Distribution**: Each chunk contains ~equal land points for optimal resource utilization  
- **No Shared State**: Chunks run completely independently without conflicts

### Reliability
- **Independent Resume**: Each chunk has separate resume capability with unique intermediate files
- **Fault Isolation**: Failed chunks don't affect others; easy selective reprocessing
- **Data Integrity**: Automatic validation and consistent file naming prevent data corruption

### Efficiency
- **Zero Configuration**: Set chunk mode once, then use simple command line arguments
- **Automatic Detection**: System finds total chunks and validates files automatically
- **Optimal Processing**: Only processes assigned points (e.g., 150 points vs 3000+ full mesh)

### Flexibility
- **Multi-Model Support**: Run different models/configs on different chunks simultaneously
- **Selective Processing**: Reprocess specific chunks without affecting others
- **Custom Scaling**: Adjust chunk count based on available resources (5-100+ chunks)

## Troubleshooting

### Common Issues

1. **Wrong pattern in config**: Ensure `chunks_pattern` matches your actual chunk file names
2. **Chunk not found**: Verify chunk number exists (1-N where N is total chunks)
3. **Permission errors**: Ensure write access to results directory
4. **API limits**: Distribute chunks across different time periods or API keys

### Validation

```bash
# Verify chunk files exist
ls -la meshes/chunks/

# Test chunk detection
python -c "import glob; print(len(glob.glob('meshes/chunks/mesh_data_*_chunk_*_of_*.json')))"

# Verify results combination
python combine_results.py "results/climate_results_*_chunk_*_simple.json" results/test_combined.json
```

## Real-World Usage Examples

### High-Resolution Global Analysis (1° mesh, ~3000 land points)

```bash
# 1. Create high-resolution mesh and chunks
python geo_mesh_processor.py 1
python split_mesh.py meshes/mesh_data_1.0deg.json 20  # ~150 points per chunk

# 2. Configure once
# Edit config.yaml: chunk_mode: true, chunks_pattern: "mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json"

# 3. Process all chunks in parallel  
for i in {1..20}; do python climate_llm_benchmark.py $i & done; wait

# 4. Results: 20 files with ~150 points each, processed simultaneously
# Total: ~3000 land points processed in parallel across 20 processes
```

### Multi-Model Comparison

```bash
# Run different models on same data chunks for comparison
python climate_llm_benchmark.py openai_config.yaml 1 &     # GPT on chunk 1
python climate_llm_benchmark.py claude_config.yaml 1 &     # Claude on chunk 1  
python climate_llm_benchmark.py gemini_config.yaml 1 &     # Gemini on chunk 1
wait
```

### Production Deployment

```bash
# Submit chunks as separate jobs to compute cluster
for i in {1..50}; do
    sbatch --job-name=climate_chunk_$i \
           --wrap="python climate_llm_benchmark.py $i"
done
```

This enhanced distributed processing system enables efficient scaling of climate LLM benchmarks from small tests to massive global analyses across multiple cores, machines, or compute clusters while maintaining data integrity, fault tolerance, and zero-configuration simplicity.