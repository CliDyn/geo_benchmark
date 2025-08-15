# Usage Guide

Complete guide for using the GEO Benchmark framework to evaluate LLM climate prediction performance.

## Workflow Overview

1. **Generate Mesh** → Create global coordinate grid
2. **Run LLM Benchmark** → Query LLMs for climate data
3. **Process ERA5 Data** → Prepare reference climatology
4. **Compare Results** → Generate analysis and visualizations

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

Query LLMs for temperature data with parallel batch processing.

### Command Structure
```bash
python climate_llm_benchmark.py [mesh_file] [num_repeats] [model] [mode] [month] [processing] [tracing] [resume]
```

### Parameters
- **mesh_file**: Path to mesh JSON file
- **num_repeats**: Number of queries per point (default: 10)
- **model**: LLM model name (default: gpt-5-nano)
- **mode**: simple/full (default: simple)
- **month**: Target month (default: July)
- **processing**: batch/individual (default: batch)
- **tracing**: disable/enable LangSmith (default: enable)
- **resume**: resume/fresh from intermediate files (default: fresh)

### Examples

#### Basic Benchmark
```bash
# Simple mode, 10 repeats, batch processing
python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 10 gpt-5-nano
```

#### High-Throughput Processing
```bash
# Batch mode with tracing disabled for speed
python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 20 gpt-5-nano simple July batch disable
```

#### Resume Interrupted Run
```bash
# Resume from latest intermediate file
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable resume
```

#### Different Models and Months
```bash
# GPT-4o-mini for January temperatures
python climate_llm_benchmark.py meshes/mesh_data_20.0deg.json 5 gpt-4o-mini simple January

# GPT-5-nano for December with individual processing
python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 3 gpt-5-nano simple December individual
```

### Output Files
- `results/climate_results_{resolution}deg_r{repeats}_simple.json` - Final results
- `results/climate_results_intermediate_{n}_simple.json` - Intermediate saves (every 10 points)

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

# 2. Run small benchmark
python climate_llm_benchmark.py meshes/mesh_data_20.0deg.json 3 gpt-5-nano

# 3. Compare with ERA5
python compare_llm_era5.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r3_simple.json data/t2m_climatology_1991-2020.nc
```

### Production Run (High Resolution)
```bash
# 1. Generate fine mesh
python geo_mesh_processor.py 1

# 2. Run comprehensive benchmark with resume capability
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable

# 3. If interrupted, resume from intermediate file
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable resume

# 4. Full analysis
python compare_llm_era5.py meshes/mesh_data_1.0deg.json results/climate_results_1.0deg_r10_simple.json data/t2m_climatology_1991-2020.nc
```

### Seasonal Analysis
```bash
# Run benchmarks for different months
for month in January April July October; do
  python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 5 gpt-5-nano simple $month
  python compare_llm_era5.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r5_simple.json data/t2m_climatology_1991-2020.nc
done
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

## File Structure

```
geo_benchmark/
├── data/
│   ├── land/                    # Natural Earth shapefiles
│   └── t2m_climatology_*.nc    # ERA5 climatology
├── meshes/
│   └── mesh_data_*.json        # Generated meshes
├── results/
│   ├── climate_results_*.json  # LLM benchmark results
│   └── climate_results_*_era5.json  # Combined LLM+ERA5
└── png/
    ├── mesh_plot_*.png         # Mesh visualizations
    ├── temperature_map_*.png   # Temperature maps
    └── *_comparison_*.png      # Comparison plots
```