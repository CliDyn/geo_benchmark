# Examples

Common use cases and example commands for the GEO Benchmark framework.

## Example 1: Quick Model Evaluation

**Goal**: Rapidly test a new LLM model with minimal computational cost.

```bash
# Step 1: Generate coarse mesh (41 land points)
python geo_mesh_processor.py 20

# Step 2: Configure for quick test (edit config.yaml):
benchmark:
  mesh_file: "meshes/mesh_data_20.0deg.json"
  num_repeats: 3
  disable_tracing: true
model:
  provider: "openai"
  name: "gpt-5-nano"

# Step 3: Run benchmark
python climate_llm_benchmark.py

# Step 4: Visualize results
python plot_temperature_results.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r3_gpt-5-nano_simple.json

# Step 5: Compare with ERA5
python compare_llm_era5.py meshes/mesh_data_20.0deg.json results/climate_results_20.0deg_r3_gpt-5-nano_simple.json data/t2m_climatology_1991-2020.nc
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

# Step 2: Configure for comprehensive study (edit config.yaml):
benchmark:
  mesh_file: "meshes/mesh_data_1.0deg.json"
  num_repeats: 10
  disable_tracing: true
  resume: false
model:
  provider: "openai"
  name: "gpt-5-nano"

# Step 3: Run comprehensive benchmark
python climate_llm_benchmark.py

# If process gets interrupted, enable resume and re-run:
# Edit config.yaml: resume: true
python climate_llm_benchmark.py

# Step 4: Generate all visualizations
python plot_temperature_results.py meshes/mesh_data_1.0deg.json results/climate_results_1.0deg_r10_gpt-5-nano_simple.json

python compare_llm_era5.py meshes/mesh_data_1.0deg.json results/climate_results_1.0deg_r10_gpt-5-nano_simple.json data/t2m_climatology_1991-2020.nc
```

**Expected Output**:
- ~150,000 total queries (15,000 points × 10 repeats)
- Processing time: Several hours
- High-resolution global temperature maps
- Robust statistical comparison with ERA5

## Example 3: Resume Interrupted Large-Scale Run

**Goal**: Handle interruptions gracefully in long-running experiments.

```bash
# Step 1: Configure for large-scale experiment (edit config.yaml):
benchmark:
  mesh_file: "meshes/mesh_data_1.0deg.json"
  num_repeats: 15
  disable_tracing: true
  resume: false
batch:
  save_interval: 100  # Save more frequently for safety

# Step 2: Start large-scale experiment
python climate_llm_benchmark.py

# Process gets interrupted at point 3,847...
# Check what intermediate files exist:
ls -la results/climate_results_intermediate_*_gpt-5-nano_simple.json

# Step 3: Enable resume and restart (edit config.yaml):
benchmark:
  resume: true

# Step 4: Resume from latest intermediate file
python climate_llm_benchmark.py

# The script will automatically:
# 1. Find: climate_results_intermediate_3800_gpt-5-nano_simple.json (latest)
# 2. Load: 3,800 completed results  
# 3. Resume: from point 3,801 onwards
# 4. Continue: saving intermediate files every 100 points
```

**Benefits**:
- No lost computation time
- Seamless continuation from interruption point
- Automatic intermediate file detection
- Model name included in filenames for clarity

## Example 4: Multi-Provider Comparison

**Goal**: Compare performance across different LLM providers.

```bash
# Step 1: Generate medium-resolution mesh
python geo_mesh_processor.py 10

# Step 2: Test OpenAI GPT (edit config.yaml):
model:
  provider: "openai"
  name: "gpt-4o"
benchmark:
  mesh_file: "meshes/mesh_data_10.0deg.json"
  num_repeats: 5

python climate_llm_benchmark.py

# Step 3: Test Anthropic Claude (edit config.yaml):
model:
  provider: "anthropic"
  name: "claude-3-5-sonnet-20241022"

python climate_llm_benchmark.py

# Step 4: Test Google Gemini (edit config.yaml):
model:
  provider: "google"
  name: "gemini-1.5-pro"
benchmark:
  use_batch: false  # Gemini may not support batch

python climate_llm_benchmark.py

# Step 5: Test local Ollama model (edit config.yaml):
model:
  provider: "ollama"
  name: "llama3.1:8b"
benchmark:
  use_batch: false
  num_repeats: 3  # Fewer for slower local model

python climate_llm_benchmark.py

# Step 6: Compare results
python compare_llm_era5.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r5_gpt-4o_simple.json data/t2m_climatology_1991-2020.nc
python compare_llm_era5.py meshes/mesh_data_10.0deg.json results/climate_results_10.0deg_r5_claude-3-5-sonnet-20241022_simple.json data/t2m_climatology_1991-2020.nc
```

**Comparison Benefits**:
- Easy provider switching via config
- Consistent evaluation methodology
- Model names in output filenames
- Fair comparison across providers

## Expected Processing Times

| Resolution | Land Points | Repeats | Total Queries | Est. Time (batch) |
|------------|------------|---------|---------------|-------------------|
| 20°        | 41         | 3       | 123           | 2-5 min          |
| 15°        | 82         | 5       | 410           | 5-10 min         |
| 10°        | 153        | 10      | 1,530         | 15-30 min        |
| 5°         | 664        | 10      | 6,640         | 1-2 hours        |
| 1°         | 6,478      | 10      | 64,780        | 8-16 hours       |

*Times vary based on model, API performance, and batch size.*

