# GEO Benchmark

Framework for evaluating LLM performance on climate data prediction using global geographic meshes.

## Overview

Creates geographic coordinate grids, queries LLMs for climate data, and compares results against ERA5 climatology.

## Features

- Mesh generation with land/ocean detection
- Multi-provider support: OpenAI, Anthropic Claude, Google Gemini, Ollama
- Batch processing with resume capability
- Distributed processing for large meshes
- ERA5 climatology comparison
- YAML configuration
- Spatial analysis with population and bathymetry data
- Temperature visualization and statistical analysis

## Quick Start

Edit `config.yaml` to specify mesh file, model provider, and parameters, then run `python climate_llm_benchmark.py`.

**Standard processing:**
```bash
# Generate mesh
python geo_mesh_processor.py 20
# Edit config.yaml: set mesh_file, provider, model

# Run benchmark
python climate_llm_benchmark.py

# Complete analysis
python run_complete_analysis_pipeline.py results/climate_results_20.0deg_r10_simple.json
```

**Distributed processing:**
```bash
# Generate and split mesh
python geo_mesh_processor.py 1
python split_mesh.py meshes/mesh_data_1.0deg.json 20

# Configure chunk mode in config.yaml
# Run parallel chunks
for i in {1..20}; do python climate_llm_benchmark.py $i & done; wait

# Combine results
python combine_results.py "results/climate_results_*_chunk_*_simple.json" results/combined.json
python run_complete_analysis_pipeline.py results/combined.json
```

## Documentation

- [File Descriptions](file_descriptions.md) - Script and data structure overview
- [Usage Guide](USAGE.md) - Examples and workflows
- [Examples](EXAMPLES.md) - Common use cases and commands
- [Distributed Processing](DISTRIBUTED_PROCESSING.md) - Parallel processing guide

## Requirements

- Python 3.8+
- API keys for chosen provider (OpenAI, Anthropic, Google) or Ollama server for local models
- ERA5 climatology data (NetCDF format)
- Natural Earth 10m Land shapefile in `data/land/`

```bash
pip install -r requirements.txt

# Optional provider dependencies
pip install langchain-anthropic      # For Claude
pip install langchain-google-genai   # For Gemini
pip install langchain-community      # For Ollama
```

## Citation

```bibtex
@software{geo_benchmark,
  title={GEO Benchmark: LLM Climate Data Evaluation Framework},
  year={2024},
  url={https://github.com/CliDyn/geo_benchmark}
}
```