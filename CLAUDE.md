# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive framework for evaluating Large Language Model performance on climatological data prediction tasks using global geographic meshes. The system creates geographic coordinate grids, queries LLMs for climate data, and compares results against ERA5 climatology data.

## Core Architecture

The codebase follows a modular pipeline architecture:

1. **Mesh Generation** (`geo_mesh_processor.py`) - Creates global coordinate grids with land/ocean detection
2. **LLM Benchmarking** (`climate_llm_benchmark.py`) - Queries LLMs with parallel batch processing and resume capability
3. **ERA5 Processing** (`process_era5_climatology.py`) - Processes ERA5 reanalysis data for comparison baselines
4. **Comparison & Analysis** (`compare_llm_era5.py`) - Statistical comparison between LLM predictions and ERA5 data
5. **Visualization** (`plot_*.py`) - Temperature mapping and analysis visualization

## Key Commands

### Setup
```bash
pip install -r requirements.txt
```

### Core Workflow Commands
```bash
# 1. Generate geographic mesh (resolution in degrees)
python geo_mesh_processor.py {resolution}

# 2. Configure your settings in config.yaml, then run:
python climate_llm_benchmark.py

# 3. Or use a custom config file:
python climate_llm_benchmark.py my_custom_config.yaml

# 4. Compare LLM results with ERA5 climatology
python compare_llm_era5.py {mesh_file} {results_file} {era5_climatology_file}

# 5. Visualize temperature results
python plot_temperature_results.py {mesh_file} {results_file}

# 6. Plot mesh visualization
python plot_mesh.py {mesh_file}
```

## Model Provider Support

The framework now supports multiple LLM providers:

### OpenAI (Default)
```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key"

# Configure in config.yaml
model:
  provider: "openai"
  name: "gpt-5-nano"  # or gpt-4o, gpt-4o-mini, gpt-3.5-turbo
```

### Anthropic Claude
```bash
# Install dependencies
pip install langchain-anthropic

# Set environment variable
export ANTHROPIC_API_KEY="your-api-key"

# Configure in config.yaml
model:
  provider: "anthropic"
  name: "claude-3-5-sonnet-20241022"
```

### Google Gemini
```bash
# Install dependencies
pip install langchain-google-genai

# Set environment variable
export GOOGLE_API_KEY="your-api-key"

# Configure in config.yaml
model:
  provider: "google"
  name: "gemini-1.5-pro"
```

### Ollama (Local Models)
```bash
# Install dependencies
pip install langchain-community

# Start Ollama server
ollama serve

# Pull a model
ollama pull llama3.1:8b

# Configure in config.yaml
model:
  provider: "ollama"
  name: "llama3.1:8b"
```

### Example Workflows

#### Quick evaluation with OpenAI
```bash
python geo_mesh_processor.py 20
# Edit config.yaml to set mesh_file: "meshes/mesh_data_20.0deg.json"
python climate_llm_benchmark.py
python compare_llm_era5.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r10_simple.json data/t2m_climatology_1991-2020.nc
```

#### Using different model providers
```bash
# For Anthropic Claude
# Edit config.yaml to set provider: "anthropic", name: "claude-3-5-sonnet-20241022"
python climate_llm_benchmark.py

# For Google Gemini  
# Edit config.yaml to set provider: "google", name: "gemini-1.5-pro"
python climate_llm_benchmark.py

# For Ollama local models
# Edit config.yaml to set provider: "ollama", name: "llama3.1:8b"
python climate_llm_benchmark.py

# High-resolution production run with resume capability
python geo_mesh_processor.py 1
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable
# If interrupted, resume with:
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable resume
```

## Data Dependencies

### Required External Data
- **Land boundaries**: Natural Earth 10m Land shapefiles in `data/land/` directory
- **ERA5 climatology**: NetCDF format climatology data (1991-2020) as `data/t2m_climatology_1991-2020.nc`
- **OpenAI API key**: Required for LLM queries

### Generated Data Structure
```
├── meshes/
│   ├── mesh_data_{resolution}deg.json    # Geographic mesh data
│   └── mesh_data_{resolution}deg.csv     # CSV format for analysis
├── results/
│   ├── climate_results_{resolution}deg_r{repeats}_simple.json        # Final results
│   ├── climate_results_intermediate_{n}_simple.json                  # Intermediate saves
│   └── climate_results_{resolution}deg_r{repeats}_simple_era5.json   # Combined LLM+ERA5
└── png/
    ├── temperature_map_*.png             # Temperature visualizations
    ├── *_comparison_*.png                # LLM vs ERA5 comparisons
    └── mesh_*.png                        # Mesh visualizations
```

## Resume Functionality

The LLM benchmarking system includes robust resume capability:
- Saves intermediate results every 10 processed points
- Automatically detects latest intermediate file with `resume` parameter
- Continues processing from exact interruption point
- Maintains data consistency across resume sessions

## Performance Optimization

### Speed Settings
- Use `batch` processing mode (default) for parallel API calls
- Set `disable` for tracing to reduce overhead
- Use coarser resolution (20°) for testing/validation
- Enable `resume` for fault tolerance in long runs

### Quality Settings
- Increase repeats per point (10-20) for statistical robustness
- Use higher resolution (1-5°) for detailed analysis
- Validate across multiple months/seasons

## Key Dependencies

**Core Libraries**:
- `langchain` + `langchain-openai`: LLM integration and batch processing
- `geopandas` + `shapely`: Geospatial operations and land/ocean detection
- `xarray` + `netcdf4`: ERA5 climatology data processing
- `matplotlib`: Temperature mapping and visualization
- `numpy` + `pandas`: Numerical analysis and data manipulation

**External Requirements**:
- OpenAI API access for LLM queries
- ERA5 climatology NetCDF data
- Natural Earth land boundary shapefiles

## File Processing Patterns

- **Mesh files**: JSON format with coordinate arrays and land detection flags
- **Results files**: JSON with nested LLM response data and statistics
- **ERA5 files**: NetCDF format with time-series temperature data
- **Intermediate files**: Auto-numbered JSON saves for resume functionality

## Output Analysis

Results include comprehensive statistical analysis:
- **Temperature statistics**: Mean, standard deviation across multiple LLM queries
- **Comparison metrics**: RMSE, MAE, bias, correlation with ERA5
- **Visualization outputs**: Temperature maps, difference plots, scatter comparisons
- **Error quantification**: LLM variability vs ERA5 climatological uncertainty