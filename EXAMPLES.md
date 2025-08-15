# Examples

Common use cases and example commands for the GEO Benchmark framework.

## Example 1: Quick Model Evaluation

**Goal**: Rapidly test a new LLM model with minimal computational cost.

```bash
# Step 1: Generate coarse mesh (41 land points)
python geo_mesh_processor.py 20

# Step 2: Run small benchmark (3 repeats per point)
python climate_llm_benchmark.py meshes/mesh_data_20.0deg.json 3 gpt-5-nano simple July batch disable

# Step 3: Visualize results
python plot_temperature_results.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r3_simple.json

# Step 4: Compare with ERA5
python compare_llm_era5.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r3_simple.json data/t2m_climatology_1991-2020.nc
```

**Expected Output**:
- ~123 total queries (41 points × 3 repeats)
- Processing time: ~5-10 minutes
- Files: temperature maps, comparison plots, statistics

## Example 2: High-Resolution Comprehensive Study

**Goal**: Detailed evaluation with statistical robustness.

```bash
# Step 1: Generate fine mesh 
python geo_mesh_processor.py 1

# Step 2: Run comprehensive benchmark with resume capability
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable

# If process gets interrupted, resume from where it stopped:
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 10 gpt-5-nano simple July batch disable resume

# Step 3: Generate all visualizations
python plot_temperature_results.py meshes/mesh_data_1.0deg.json results/climate_results_1.0deg_r10_simple.json

python compare_llm_era5.py meshes/mesh_data_1.0deg.json results/climate_results_1.0deg_r10_simple.json data/t2m_climatology_1991-2020.nc
```

**Expected Output**:
- ~150,000 total queries (15,000 points × 10 repeats)
- Processing time: Several hours
- High-resolution global temperature maps
- Robust statistical comparison with ERA5

## Example 3: Resume Interrupted Large-Scale Run

**Goal**: Handle interruptions gracefully in long-running experiments.

```bash
# Start large-scale experiment
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 15 gpt-5-nano simple July batch disable

# Process gets interrupted at point 3,847...
# Check what intermediate files exist:
ls -la results/climate_results_intermediate_*_simple.json

# Resume from latest intermediate file:
python climate_llm_benchmark.py meshes/mesh_data_1.0deg.json 15 gpt-5-nano simple July batch disable resume

# The script will automatically:
# 1. Find: climate_results_intermediate_3840_simple.json (latest)
# 2. Load: 3,840 completed results  
# 3. Resume: from point 3,841 onwards
# 4. Continue: saving intermediate files every 10 points
```

**Benefits**:
- No lost computation time
- Seamless continuation from interruption point
- Automatic intermediate file detection

## Example 4: Performance Optimization

**Goal**: Maximize processing speed for large experiments.

```bash
# Optimized configuration for speed:
python climate_llm_benchmark.py \
    meshes/mesh_data_5.0deg.json \
    20 \
    gpt-5-nano \
    simple \
    July \
    batch \
    disable \
    resume

# This configuration:
# - Processes 20 queries per point in parallel
# - Disables tracing overhead
# - Uses resume-safe processing
# - Optimizes token usage with max_tokens=300
```

**Performance Gains**:
- ~10x faster than individual processing
- Reduced API latency through batching
- Lower overhead from disabled tracing
- Fault tolerance with resume capability

## Expected Processing Times

| Resolution | Land Points | Repeats | Total Queries | Est. Time (batch) |
|------------|------------|---------|---------------|-------------------|
| 20°        | 41         | 3       | 123           | 2-5 min          |
| 15°        | 82         | 5       | 410           | 5-10 min         |
| 10°        | 153        | 10      | 1,530         | 15-30 min        |
| 5°         | 664        | 10      | 6,640         | 1-2 hours        |
| 1°         | 6,478      | 10      | 64,780        | 8-16 hours       |

*Times vary based on model, API performance, and batch size.*

