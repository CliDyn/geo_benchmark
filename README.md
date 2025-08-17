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
- **Visualization**: Generate temperature maps, comparison plots, and statistical charts
- **ERA5 Integration**: Process and compare against ERA5 reanalysis data

## Quick Start

```bash
# 1. Generate mesh
python geo_mesh_processor.py 20

# 2. Configure settings in config.yaml
# Set mesh_file: "meshes/mesh_data_20.0deg.json"
# Set provider: "openai" and model: "gpt-5-nano"

# 3. Run LLM benchmark
python climate_llm_benchmark.py

# 4. Compare with ERA5 climatology  
python compare_llm_era5.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r10_gpt-5-nano_simple.json data/t2m_climatology_1991-2020.nc
```

## Documentation

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