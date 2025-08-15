# GEO Benchmark

A comprehensive framework for evaluating Large Language Model performance on climatological data prediction tasks using global geographic meshes.

## Overview

This tool creates global geographic meshes, queries LLMs for climate data, and compares results against ERA5 climatology. Features include batch processing, resume capability, visualization tools, and statistical analysis.

## Key Features

- **Mesh Generation**: Create global coordinate grids with land/ocean detection
- **LLM Benchmarking**: Parallel batch processing with resume functionality  
- **Climate Comparison**: Compare LLM predictions vs ERA5 climatology
- **Visualization**: Generate temperature maps, comparison plots, and statistical charts
- **ERA5 Integration**: Process and compare against ERA5 reanalysis data

## Quick Start

```bash
# 1. Generate mesh
python geo_mesh_processor.py 10

# 2. Run LLM benchmark (batch mode, 10 repeats)
python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 10 gpt-5-nano

# 3. Compare with ERA5 climatology
python compare_llm_era5.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r10_simple.json data/t2m_climatology_1991-2020.nc
```

## Documentation

- **[Usage Guide](USAGE.md)** - Detailed examples and workflows
- **[Examples](EXAMPLES.md)** - Common use cases and commands

## Requirements

- Python 3.8+
- OpenAI API key  
- ERA5 climatology data (NetCDF format)
- Land boundary shapefile: [Natural Earth 10m Land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/) → `data/land/`

Install dependencies: `pip install -r requirements.txt`

## Citation

```bibtex
@software{geo_benchmark,
  title={GEO Benchmark: LLM Climate Data Evaluation Framework},
  year={2024},
  url={https://github.com/CliDyn/geo_benchmark}
}
```