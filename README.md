# GEO Benchmark

A comprehensive framework for evaluating Large Language Model performance on climatological data prediction tasks using global geographic meshes.

## Overview

This tool creates global geographic meshes, queries LLMs for climate data, and compares results against ERA5 climatology. Features include batch processing, resume capability, visualization tools, and statistical analysis.

## Key Features

- **Mesh Generation**: Create global coordinate grids with land/ocean detection
- **Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini, and Ollama local models
- **LLM Benchmarking**: Parallel batch processing with resume functionality  
- **Distributed Processing**: Split large meshes into chunks for parallel execution across multiple processes/machines
- **Climate Comparison**: Compare LLM predictions vs ERA5 climatology
- **Configuration-Based**: YAML configuration for easy setup and reproducibility
- **Enhanced Analysis**: Spatial RMSE, population density, and bathymetry integration
- **Comprehensive Visualization**: Temperature maps, clustering analysis, filtered views, and statistical comparisons
- **Complete Pipeline**: Automated analysis workflow from raw results to publication-ready plots
- **ERA5 Integration**: Process and compare against ERA5 reanalysis data

## Quick Start

The framework uses a simple configuration-based approach where you edit `config.yaml` to specify your mesh file, model provider (OpenAI, Anthropic, Google, or **Ollama for local models**), model name, and benchmark parameters, then run `python climate_llm_benchmark.py` to execute the LLM evaluation. For Ollama local models, simply start the Ollama server (`ollama serve`), pull your desired model (`ollama pull llama3.1:8b`), set `provider: "ollama"` and `name: "llama3.1:8b"` in config.yaml, and run the benchmark - no API keys required, making it ideal for offline research and experimentation.

**For standard processing:**
```bash
# 1. Generate mesh and configure settings
python geo_mesh_processor.py 20
# Edit config.yaml: set mesh_file, provider, and model

# 2. Run LLM benchmark  
python climate_llm_benchmark.py

# 3. Complete analysis with all enhancements and visualizations
python run_complete_analysis_pipeline.py results/climate_results_20.0deg_r10_simple.json
```

**For large-scale distributed processing:**
```bash
# 1. Generate mesh and split into chunks
python geo_mesh_processor.py 1    # High-resolution 1° mesh
python split_mesh.py meshes/mesh_data_1.0deg.json 20

# 2. Configure chunk mode in config.yaml
# Set: chunk_mode: true, chunks_pattern: "mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json"

# 3. Run chunks in parallel (each processes ~equal land points)
for i in {1..20}; do python climate_llm_benchmark.py $i & done; wait

# 4. Combine and analyze
python combine_results.py "results/climate_results_*_chunk_*_simple.json" results/combined.json
python run_complete_analysis_pipeline.py results/combined.json
```

The standard pipeline handles spatial RMSE analysis, population/bathymetry integration, and generates 25+ publication-ready plots. The distributed approach enables processing thousands of points across multiple cores/machines with automatic load balancing.

## Documentation

- **[File Descriptions](file_descriptions.md)** - Complete overview of all scripts and data structures
- **[Usage Guide](USAGE.md)** - Detailed examples and workflows
- **[Examples](EXAMPLES.md)** - Common use cases and commands
- **[Distributed Processing](DISTRIBUTED_PROCESSING.md)** - Complete guide for large-scale parallel processing

## Requirements

- Python 3.8+
- API keys for your chosen provider:
  - OpenAI API key (for GPT models)
  - Anthropic API key (for Claude models) 
  - Google API key (for Gemini models)
  - Ollama server (for local models)
- ERA5 climatology data (NetCDF format)
- Land boundary shapefile: [Natural Earth 10m Land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/) → `data/land/`

Install dependencies: `pip install -r requirements.txt`

**Optional provider dependencies:**
```bash
# For Anthropic Claude
pip install langchain-anthropic

# For Google Gemini
pip install langchain-google-genai

# For Ollama local models
pip install langchain-community
```

## Citation

```bibtex
@software{geo_benchmark,
  title={GEO Benchmark: LLM Climate Data Evaluation Framework},
  year={2024},
  url={https://github.com/CliDyn/geo_benchmark}
}
```