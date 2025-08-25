# Usage Guide

Complete guide for using the GEO Benchmark framework to evaluate LLM climate prediction performance.

## Workflow Overview

1. **Generate Mesh** → Create global coordinate grid
2. **Run LLM Benchmark** → Query LLMs for climate data  
3. **Process ERA5 Data** → Prepare reference climatology
4. **Enhance with Spatial RMSE** → Add neighborhood analysis
5. **Add Population/Bathymetry** → Integrate geographic datasets
6. **Advanced Analysis** → Clustering, multivariate modeling

## 1. Mesh Generation

Create geographic coordinate grids with land/ocean detection.

### Basic Usage
```bash
# Generate 10-degree resolution mesh
python geo_mesh_processor.py 10

# Generate 1-degree high-resolution mesh  
python geo_mesh_processor.py 1

# Generate 20-degree coarse mesh
python geo_mesh_processor.py 20
```

### Output
- `meshes/mesh_data_{resolution}deg.json` - Mesh data with land points
- `meshes/mesh_data_{resolution}deg.csv` - CSV format for analysis

### Visualization
```bash
# Plot mesh with land boundaries
python plot_mesh.py meshes/mesh_data_10.0deg.json
```

## 2. LLM Climate Benchmarking

Query LLMs for temperature data using configuration-based approach with multi-provider support.

### Configuration Setup

Edit `config.yaml` to set up your benchmark parameters:

```yaml
# Basic benchmark settings
benchmark:
  mesh_file: "meshes/mesh_data_10.0deg.json"
  num_repeats: 10
  simple_mode: true
  month: "July"
  use_batch: true
  disable_tracing: false
  resume: false

# Model configuration
model:
  provider: "openai"  # openai, anthropic, google, ollama
  name: "gpt-5-nano"
  temperature: 0
  max_tokens: 300
  max_retries: 3
  timeout: 30
```

### Command Structure
```bash
# Use default config.yaml
python climate_llm_benchmark.py

# Use custom config file
python climate_llm_benchmark.py my_config.yaml
```

### Provider Setup

#### OpenAI (GPT Models)
```bash
export OPENAI_API_KEY="your-api-key"
```
```yaml
model:
  provider: "openai"
  name: "gpt-5-nano"  # or gpt-4o, gpt-4o-mini, gpt-3.5-turbo
```

#### Anthropic Claude
```bash
pip install langchain-anthropic
export ANTHROPIC_API_KEY="your-api-key"
```
```yaml
model:
  provider: "anthropic"
  name: "claude-3-5-sonnet-20241022"  # or claude-3-5-haiku-20241022
```

#### Google Gemini
```bash
pip install langchain-google-genai
export GOOGLE_API_KEY="your-api-key"
```
```yaml
model:
  provider: "google"
  name: "gemini-1.5-pro"  # or gemini-1.5-flash
```

#### Ollama (Local Models)
```bash
pip install langchain-community
ollama serve
ollama pull llama3.1:8b
```
```yaml
model:
  provider: "ollama"
  name: "llama3.1:8b"  # or mistral:7b, qwen2.5:14b
```

### Examples

#### Basic Benchmark
```bash
# Configure in config.yaml, then run
python climate_llm_benchmark.py
```

#### High-Throughput Processing
```yaml
# In config.yaml
benchmark:
  use_batch: true
  disable_tracing: true
  num_repeats: 20
```

#### Resume Interrupted Run
```yaml
# In config.yaml
benchmark:
  resume: true
```

#### Different Models and Months
```yaml
# For January with Claude
model:
  provider: "anthropic"
  name: "claude-3-5-haiku-20241022"
benchmark:
  month: "January"

# For local Ollama model
model:
  provider: "ollama"
  name: "mistral:7b"
benchmark:
  use_batch: false  # Recommended for local models
```

### Output Files
- `results/climate_results_{resolution}deg_r{repeats}_{model}_simple.json` - Final results
- `results/climate_results_intermediate_{n}_{model}_simple.json` - Intermediate saves

## 3. ERA5 Climatology Processing

Process ERA5 NetCDF data to create monthly climatology reference.

### Usage
```bash
# Process ERA5 data to create climatology
python process_era5_climatology.py data/era5_raw_data.nc
```

### Output
- `data/t2m_climatology_1991-2020.nc` - Monthly climatology (1991-2020)

### ERA5 Data Requirements
- NetCDF format with 2m temperature (t2m) variable
- Time series covering 1991-2020 period
- Global coverage with regular grid

## 4. Temperature Visualization

Create temperature maps from LLM results.

### Usage
```bash
# Create temperature maps from LLM results
python plot_temperature_results.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r10_simple.json
```

### Output Maps
- `png/temperature_map_{resolution}deg_mean.png` - Mean temperature
- `png/temperature_map_{resolution}deg_series_{n}.png` - Individual request series
- `png/temperature_map_{resolution}deg_std.png` - Standard deviation

## 5. LLM vs ERA5 Comparison

Compare LLM predictions against ERA5 climatology with comprehensive analysis.

### Usage
```bash
# Full comparison with maps and statistics
python compare_llm_era5.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r10_simple.json data/t2m_climatology_1991-2020.nc
```

### Output Files
- `results/climate_results_{resolution}deg_r{repeats}_simple_era5.json` - Combined data
- `png/llm_era5_comparison_{resolution}deg.png` - Scatter plot with error bars
- `png/llm_temperature_map_{resolution}deg.png` - LLM temperature map
- `png/era5_temperature_map_{resolution}deg.png` - ERA5 temperature map  
- `png/temperature_difference_map_{resolution}deg.png` - Difference map

### Features
- **Error bars**: ERA5 climatological uncertainty (horizontal) + LLM variability (vertical)
- **Statistics**: RMSE, MAE, bias, correlation, request counts
- **Consistent scaling**: Both maps use ERA5 temperature range
- **Difference visualization**: Diverging colormap (blue=LLM<ERA5, red=LLM>ERA5)

## Common Workflows

### Quick Evaluation (Coarse Resolution)
```bash
# 1. Generate coarse mesh
python geo_mesh_processor.py 20

# 2. Configure for quick test
# Edit config.yaml:
# mesh_file: "meshes/mesh_data_20.0deg.json"
# num_repeats: 3

# 3. Run benchmark
python climate_llm_benchmark.py

# 4. Compare with ERA5
python compare_llm_era5.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r3_gpt-5-nano_simple.json data/t2m_climatology_1991-2020.nc
```

### Production Run (High Resolution)
```bash
# 1. Generate fine mesh
python geo_mesh_processor.py 1

# 2. Configure for production
# Edit config.yaml:
# mesh_file: "meshes/mesh_data_1.0deg.json"
# num_repeats: 10
# disable_tracing: true
# resume: false

# 3. Run comprehensive benchmark
python climate_llm_benchmark.py

# 4. If interrupted, enable resume and re-run
# Edit config.yaml: resume: true
python climate_llm_benchmark.py

# 5. Full analysis
python compare_llm_era5.py meshes/mesh_data_1.0deg.json results/climate_results_1.0deg_r10_gpt-5-nano_simple.json data/t2m_climatology_1991-2020.nc
```

### Seasonal Analysis
```bash
# Create configs for different months
for month in January April July October; do
  # Edit config.yaml: month: $month
  python climate_llm_benchmark.py
  python compare_llm_era5.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r10_gpt-5-nano_simple.json data/t2m_climatology_1991-2020.nc
done
```

### Multi-Provider Comparison
```bash
# 1. Test OpenAI GPT
# config.yaml: provider: "openai", name: "gpt-4o"
python climate_llm_benchmark.py

# 2. Test Anthropic Claude  
# config.yaml: provider: "anthropic", name: "claude-3-5-sonnet-20241022"
python climate_llm_benchmark.py

# 3. Test Google Gemini
# config.yaml: provider: "google", name: "gemini-1.5-pro"
python climate_llm_benchmark.py

# 4. Test local Ollama model
# config.yaml: provider: "ollama", name: "llama3.1:8b", use_batch: false
python climate_llm_benchmark.py
```

## Performance Tips

### Speed Optimization
- Use batch processing (default)
- Disable LangSmith tracing: `disable`
- Use coarser resolution for testing (20°)
- Enable resume for long runs

### Quality Improvement
- Increase repeats per point (10-20)
- Use higher resolution mesh (1-5°)
- Validate with multiple months/seasons
- Compare different LLM models

## 6. Advanced Analysis Pipeline

### Spatial RMSE Enhancement
```bash
# Add neighborhood analysis to existing results
python extend_results_with_spatial_rmse.py results/climate_results_1.0deg_r10_simple_era5.json
```

### Population Integration  
```bash
# Add population density data
python add_population_to_results.py results/climate_results_1.0deg_r10_simple_spatial_rmse.json
```

### Bathymetry Integration
```bash
# 1. Aggregate GEBCO data to 1° grid
python aggregate_bathymetry.py

# 2. Add elevation parameters  
python add_bathymetry_to_results.py results/climate_results_1.0deg_r10_simple_spatial_rmse_population.json
```

### Comprehensive Visualization
```bash
# Enhanced spatial analysis with density plots
python plot_spatial_analysis.py

# Population analysis
python plot_population_map.py

# Bathymetry analysis  
python plot_bathymetry_map.py

# Colored comparison plots
python plot_temperature_comparison_colored.py
```

### Clustering Analysis
```bash
# Elevation-based clusters (3×3 grid)
python plot_elevation_clusters.py

# Population-based clusters (3×3 grid)
python plot_population_clusters.py
```

### Multivariate Analysis
```bash
# Comprehensive statistical modeling
python multivariate_rmse_analysis.py
# Outputs: distributions, correlations, GAM/XGBoost (if available), spatial CV
```

## File Structure

```
geo_benchmark/
├── data/
│   ├── land/                    # Natural Earth shapefiles
│   ├── t2m_climatology_*.nc    # ERA5 climatology
│   └── bathymetry_1deg_aggregated.nc  # GEBCO elevation data
├── meshes/
│   └── mesh_data_*.json        # Generated meshes
├── results/
│   ├── climate_results_*.json           # Basic LLM results
│   ├── climate_results_*_era5.json     # + ERA5 data  
│   ├── climate_results_*_spatial_rmse.json  # + spatial analysis
│   ├── climate_results_*_population.json   # + population data
│   └── climate_results_*_bathymetry.json   # + elevation data
├── reports/
│   └── multivariate_rmse_report.txt    # Statistical analysis
└── png/
    ├── mesh_plot_*.png         # Mesh visualizations
    ├── temperature_map_*.png   # Temperature maps
    ├── *_comparison_*.png      # Comparison plots
    ├── *_clusters_*.png        # Clustering analysis
    ├── distributions.png       # Statistical distributions
    └── correlation_matrix.png  # Variable correlations
```