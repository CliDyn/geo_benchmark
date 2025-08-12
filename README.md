# GEO Benchmark

A framework for evaluating Large Language Model performance on climatological data prediction tasks using global geographic coordinates.

## Overview

This tool generates global geographic meshes, identifies land coordinates, and queries LLMs for temperature and precipitation estimates. The system validates responses and provides visualization tools for analysis.

## Usage

```bash
# Generate mesh
python geo_mesh_processor.py 10

# Run benchmark
python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 1 gpt-4o-mini

# Create visualizations
python plot_mesh.py meshes/mesh_data_10.0deg.json
```

## Requirements

- Python 3.8+
- OpenAI API key
- Land boundary data: Download [Natural Earth 10m Land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/) shapefile to `data/land/`

See `requirements.txt` for dependencies.

## Citation

```bibtex
@software{geo_benchmark,
  title={GEO Benchmark: LLM Climate Data Evaluation Framework},
  year={2024},
  url={https://github.com/CliDyn/geo_benchmark}
}
```