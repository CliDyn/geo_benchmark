# GEO Benchmark

A comprehensive framework for evaluating Large Language Model performance on climatological data prediction tasks using global geographic meshes.

## Overview

This tool creates global geographic meshes, queries LLMs for climate data, and compares results against ERA5 climatology. Features include batch processing, resume capability, visualization tools, and statistical analysis.

## Key Features

- **Mesh Generation**: Create global coordinate grids with land/ocean detection
- **Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini, and Ollama local models
- **LLM Benchmarking**: Parallel batch processing with resume functionality  
- **Climate Comparison**: Compare LLM predictions vs ERA5 climatology
- **Configuration-Based**: YAML configuration for easy setup and reproducibility
- **Enhanced Analysis**: Spatial RMSE, population density, and bathymetry integration
- **Comprehensive Visualization**: Temperature maps, clustering analysis, filtered views, and statistical comparisons
- **Complete Pipeline**: Automated analysis workflow from raw results to publication-ready plots
- **ERA5 Integration**: Process and compare against ERA5 reanalysis data

## Quick Start

The framework uses a simple configuration-based approach where you edit `config.yaml` to specify your mesh file, model provider (OpenAI, Anthropic, Google, or **Ollama for local models**), model name, and benchmark parameters, then run `python climate_llm_benchmark.py` to execute the LLM evaluation. For Ollama local models, simply start the Ollama server (`ollama serve`), pull your desired model (`ollama pull llama3.1:8b`), set `provider: "ollama"` and `name: "llama3.1:8b"` in config.yaml, and run the benchmark - no API keys required, making it ideal for offline research and experimentation.

**For fast complete analysis:**
```bash
# 1. Generate mesh and configure settings
python geo_mesh_processor.py 20
# Edit config.yaml: set mesh_file, provider, and model

# 2. Run LLM benchmark  
python climate_llm_benchmark.py

# 3. Complete analysis with all enhancements and visualizations
python run_complete_analysis_pipeline.py results/climate_results_20.0deg_r10_simple.json
```

This automated pipeline handles spatial RMSE analysis, population/bathymetry integration, and generates 25+ publication-ready plots organized in subfolders.

## Documentation

- **[File Descriptions](file_descriptions.md)** - Complete overview of all scripts and data structures
- **[Usage Guide](USAGE.md)** - Detailed examples and workflows
- **[Examples](EXAMPLES.md)** - Common use cases and commands

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