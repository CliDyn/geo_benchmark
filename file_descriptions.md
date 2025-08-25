# File Descriptions for LLM Climatological Benchmark

## Mesh Generation

### geo_mesh_processor.py
Creates global coordinate grids with land/ocean detection. Generates regular lat/lon grids from -60° to 85° latitude and -180° to 180° longitude at specified resolution. Uses Natural Earth shapefiles to classify points as land/ocean. Queries OpenStreetMap API for location information (country, state, city) for land points. **Output**: `meshes/mesh_data_{resolution}deg.json`, `meshes/mesh_data_{resolution}deg.csv`

### plot_mesh.py
Visualizes geographic mesh points on world map. Creates scatter plots showing land points (red) vs ocean points (black) with country boundaries overlay. Supports country-colored visualization when location data available. **Output**: `png/mesh_plot_{resolution}deg.png`, `png/mesh_countries_{resolution}deg.png`

## LLM Benchmarking

### climate_llm_benchmark.py
Core LLM benchmarking system with multi-provider support (OpenAI, Anthropic, Google, Ollama). Queries LLMs for temperature predictions at global land points. Features parallel batch processing, automatic resume capability, configurable via YAML. Supports simple mode (single month temperature) or full mode (12 months + precipitation). **Output**: `results/climate_results_{resolution}deg_r{repeats}_{model}_{mode}.json`, intermediate files during processing

## ERA5 Data Processing

### process_era5_climatology.py
Processes ERA5 reanalysis data to create monthly climatological statistics for 1991-2020 period. Converts temperature from Kelvin to Celsius. Calculates mean, min, max, and standard deviation for each month. Handles time-series aggregation and creates CF-compliant NetCDF output. **Output**: `data/t2m_climatology_1991-2020.nc`

### add_era5_to_results.py
Enhances existing LLM result files by adding ERA5 climatology data. Spatially matches ERA5 grid points to LLM result coordinates using nearest neighbor interpolation. Calculates temperature differences between LLM predictions and ERA5 climatology. **Output**: `results/*_era5.json` (enhanced result files)

## Comparison & Analysis

### compare_llm_era5.py
Comprehensive comparison between LLM predictions and ERA5 climatology. Loads LLM results, extracts ERA5 data for same coordinates, calculates statistical metrics (RMSE, MAE, bias, correlation). Creates scatter plots with error bars, difference histograms, temperature maps. **Output**: `png/llm_era5_comparison_{resolution}deg.png`, `png/*_temperature_map_{resolution}deg.png`

### analyze_country_performance.py
Country-specific performance analysis with geographic visualization. Groups results by country, calculates RMSE per country, creates world map colored by performance. Handles country name matching between data sources and shapefiles. Shows top/worst performing countries with statistical breakdown. **Output**: `png/country_analysis_{file}.png`, `png/country_rmse_map_{file}.png`

### spatial_rmse_analysis.py
Spatial RMSE analysis using neighborhood smoothing. Takes 9×9 grid neighborhoods around each point to calculate smoothed RMSE field. Creates contour maps showing spatial patterns of LLM performance. Handles coordinate grid operations and spatial interpolation for visualization. **Output**: `results/spatial_rmse_*.csv`, `png/spatial_rmse_*_analysis.png`

### point_rmse_analysis.py
Point-wise RMSE calculation between individual LLM realizations and ERA5. Calculates RMSE for each geographic point using all LLM temperature predictions vs single ERA5 value. Creates detailed contour maps with custom color scales emphasizing 0-3°C range. **Output**: `results/point_rmse_*.csv`, `png/point_rmse_*_map.png`

## Visualization

### plot_temperature_results.py
Creates temperature visualization maps from LLM benchmark results. Plots temperature data on world map with land boundaries, supports multiple visualization modes (individual request series, mean temperature, standard deviation). Uses custom blue-to-red temperature colormap. **Output**: `png/temperature_map_*_series_*.png`, `png/temperature_map_*_mean.png`, `png/temperature_map_*_std.png`

## Test/Development

### extend_results_with_spatial_rmse.py
Extends LLM result files with spatial RMSE calculations for each point. Automatically detects and adds ERA5 data if missing. Calculates spatial RMSE using 5×5 (radius 2) and 9×9 (radius 4) neighborhoods around each point. Adds 14 new fields per point including RMSE, MAE, bias, correlation, and neighbor statistics for both radii. **Output**: `results/*_spatial_rmse.json`

### plot_spatial_analysis.py
Creates comprehensive spatial analysis maps and LLM vs ERA5 comparison plots from spatial RMSE-enhanced result files. Generates maps for RMSE, bias, neighborhood temperatures, and individual point temperatures for both radius 2 and 4. Creates density scatter plots with KDE-based point density coloring, 1:1 line, regression line with equation, and statistical analysis. Includes difference histograms with normal distribution overlay. Uses smart colormaps and includes detailed statistics. **Output**: `png/spatial_analysis_{resolution}deg_{field}.png`, `png/llm_era5_comparison_{resolution}deg.png`

### add_population_to_results.py
Adds population density data to existing result files by reading ASCII grid format population data. Reads GPW v4 population density data (1° resolution, 360×180 global grid) and matches coordinates using nearest neighbor interpolation. Adds population density field to each point. Handles NODATA values and provides comprehensive population statistics. **Output**: `results/*_population.json`

### plot_population_map.py
Creates population density maps and comparison plots from population-enhanced result files. Generates population map with log-scale visualization and comprehensive comparison analysis: population vs spatial RMSE/MAE/bias (r2 and r4), population vs LLM-ERA5 temperature differences, spatial RMSE r2 vs r4 comparisons. Uses density scatter plots with population coloring, regression lines, and correlation analysis. Provides detailed population statistics and correlation summaries. **Output**: `png/population_density_map_{resolution}deg.png`, `png/population_comparison_{resolution}deg_*.png`

### aggregate_bathymetry.py
Aggregates GEBCO 1 arc-minute elevation data to 1° × 1° grid matching project mesh system. Calculates area-weighted statistics (mean, min, max, std) and terrain roughness metric (mean absolute deviation). Uses consistent coordinate system: lats [-60°, 84°], lons [-180°, 179°]. Handles pixel area weighting and CF-compliant NetCDF output. **Output**: `data/bathymetry_1deg_aggregated.nc`

### add_bathymetry_to_results.py
Adds bathymetry/elevation parameters to existing result files. Reads aggregated bathymetry NetCDF and matches coordinates using nearest neighbor. Adds mean_elevation, min_elevation, max_elevation, std_elevation, roughness to each point. Sets negative elevations to 0 for land points. **Output**: `results/*_bathymetry.json`

### plot_bathymetry_map.py
Creates elevation and roughness maps using contour visualization. Generates gridded maps (not scatter) with proper coordinate mapping. Uses 'terrain' colormap for elevation, 'plasma' for roughness with log scale. Includes comparison scatter plots: spatial RMSE/MAE vs elevation/roughness, temperature differences vs bathymetry parameters. **Output**: `png/mean_elevation_map_{resolution}deg.png`, `png/terrain_roughness_map_{resolution}deg.png`, comparison plots

### plot_temperature_comparison_colored.py  
Creates combined LLM vs ERA5 scatter plots with three coloring schemes in single 3-panel figure. Colors points by elevation (linear), population density (log), terrain roughness (log). Each panel includes 1:1 line, regression, statistics. Compact format without histograms. **Output**: `png/temperature_comparison_{resolution}deg_combined.png`

### plot_elevation_clusters.py
Generates 3×3 grid of LLM vs ERA5 comparisons clustered by elevation ranges (0-500m, 500-1000m, ..., 4000m+). Each subplot shows cluster-specific scatter plot with statistics. Points colored by elevation within cluster. **Output**: `png/elevation_clusters_{resolution}deg.png`

### plot_population_clusters.py  
Generates 3×3 grid of LLM vs ERA5 comparisons clustered by population density ranges (0-1, 1-5, ..., 500+ people/km²). Log-distributed bins for population clustering. Points colored by population within cluster. **Output**: `png/population_clusters_{resolution}deg.png`

### multivariate_rmse_analysis.py
Comprehensive multivariate analysis explaining spatial_r2_rmse using population, elevation, roughness. Includes distribution analysis, log/standardization transforms, GAM modeling (if available), XGBoost+SHAP (if available), spatial block cross-validation. Handles missing dependencies gracefully. **Output**: `png/distributions.png`, `png/correlation_matrix.png`, `png/rmse_scatter_plots.png`, `reports/multivariate_rmse_report.txt`

### test.py
Simple LangChain test script for Ollama model integration. Tests basic model invocation with location queries in English and Russian. Used for development and debugging LLM connections.

## Result File Structure

### Basic LLM Result File
```json
{
  "mesh_info": {...},
  "resolution": 20.0,
  "total_land_points": 150,
  "results": [
    {
      "point_info": {
        "lat": 45.0, "lon": -120.0,
        "is_land": true,
        "country": "United States",
        "state": "Oregon", "city": "..."
      },
      "llm_responses": [
        {
          "raw_response": "25.4",
          "parsed_data": {"july_temp_mean": 25.4},
          "numpy_arrays": {...}
        }
      ]
    }
  ],
  "metadata": {...}
}
```

### ERA5-Enhanced Result File (after add_era5_to_results.py)
```json
{
  "point_info": {
    // Basic fields...
    "era5_temp_mean": 24.8,
    "era5_temp_min": 12.3,
    "era5_temp_max": 36.7,
    "era5_temp_std": 8.2,
    "llm_temp_mean": 25.4,
    "llm_temp_std": 1.2,
    "llm_temp_count": 10,
    "temp_difference": 0.6,
    "abs_difference": 0.6
  }
}
```

### Spatial RMSE-Enhanced Result File (after extend_results_with_spatial_rmse.py)
```json
{
  "point_info": {
    // Basic + ERA5 fields...
    "spatial_r2_rmse": 2.3,
    "spatial_r2_mae": 1.9,
    "spatial_r2_bias": -0.4,
    "spatial_r2_correlation": 0.87,
    "spatial_r2_n_neighbors": 18,
    "spatial_r2_neighborhood_llm_mean": 24.9,
    "spatial_r2_neighborhood_era5_mean": 25.3,
    "spatial_r4_rmse": 2.7,
    "spatial_r4_mae": 2.2,
    "spatial_r4_bias": -0.3,
    "spatial_r4_correlation": 0.84,
    "spatial_r4_n_neighbors": 45,
    "spatial_r4_neighborhood_llm_mean": 25.1,
    "spatial_r4_neighborhood_era5_mean": 25.4
  }
}
```

### Population-Enhanced Result File (after add_population_to_results.py)
```json
{
  "point_info": {
    // Basic + ERA5 + Spatial RMSE fields...
    "population_density": 145.7
  }
}
```

### Bathymetry-Enhanced Result File (after add_bathymetry_to_results.py)
```json
{
  "point_info": {
    // Basic + ERA5 + Spatial RMSE + Population fields...
    "mean_elevation": 1247.3,
    "min_elevation": 856.0,
    "max_elevation": 1634.0,
    "std_elevation": 187.4,
    "roughness": 142.8,
    "bathymetry_nearest_lat": 45.0,
    "bathymetry_nearest_lon": -120.0,
    "bathymetry_distance_lat": 0.0,
    "bathymetry_distance_lon": 0.0
  }
}
```

## Spatial Metrics Explanation

### spatial_r2_mae (Spatial Mean Absolute Error)
**What it is**: The average absolute difference between LLM predictions and ERA5 values within a 5×5 neighborhood around each point.

**Calculation**: 
1. Take the center point and its 24 surrounding neighbors (5×5 grid)
2. For each neighbor that has both LLM and ERA5 data: calculate |LLM_temp - ERA5_temp|
3. Average all these absolute differences

**Example**: If in a 5×5 neighborhood you have:
- Point 1: LLM=25°C, ERA5=24°C → |25-24| = 1°C
- Point 2: LLM=22°C, ERA5=26°C → |22-26| = 4°C  
- Point 3: LLM=28°C, ERA5=27°C → |28-27| = 1°C
- spatial_r2_mae = (1 + 4 + 1) / 3 = 2.0°C

**Meaning**: How much the LLM typically "misses" ERA5 in the local area, ignoring whether it's too high or too low.

### spatial_r2_bias (Spatial Bias)
**What it is**: The average signed difference (LLM - ERA5) within the 5×5 neighborhood.

**Calculation**:
1. For each neighbor: calculate (LLM_temp - ERA5_temp) 
2. Average all these signed differences

**Example**: Using same neighborhood:
- Point 1: 25 - 24 = +1°C
- Point 2: 22 - 26 = -4°C
- Point 3: 28 - 27 = +1°C
- spatial_r2_bias = (+1 + (-4) + (+1)) / 3 = -0.67°C

**Meaning**: 
- **Positive bias**: LLM tends to predict warmer temperatures than ERA5 in this area
- **Negative bias**: LLM tends to predict cooler temperatures than ERA5 in this area
- **Zero bias**: LLM predictions are unbiased on average (but could still have high MAE)

### spatial_r2_correlation (Spatial Correlation)
**What it correlates**: The correlation between **LLM temperature values** and **ERA5 temperature values** within the 5×5 neighborhood.

**Calculation**:
1. Collect all LLM temperatures in the neighborhood: [25, 22, 28, ...]
2. Collect corresponding ERA5 temperatures: [24, 26, 27, ...]  
3. Calculate Pearson correlation coefficient between these two arrays

**Meaning**:
- **+1.0**: Perfect positive correlation - when ERA5 is high in the area, LLM is also high
- **0.0**: No correlation - LLM and ERA5 patterns are unrelated in this area
- **-1.0**: Perfect negative correlation - when ERA5 is high, LLM is low (very bad!)

**Example**: If in a neighborhood:
- Cold areas (ERA5=10°C) → LLM also predicts cold (LLM=12°C)
- Warm areas (ERA5=30°C) → LLM also predicts warm (LLM=28°C)
- High correlation (~0.9) = LLM captures the spatial temperature pattern well

### Why These Metrics Matter
- **MAE**: Overall accuracy in the local area
- **Bias**: Systematic over/under-prediction in the local area  
- **Correlation**: Whether LLM captures the spatial temperature gradients correctly

A good LLM should have low MAE, low bias, and high correlation in each neighborhood. The same metrics are calculated for both r2 (5×5 grid) and r4 (9×9 grid) neighborhoods to analyze performance at different spatial scales.