# Examples and Usage Guide

This document provides detailed examples and step-by-step instructions for using the GEO Benchmark framework.

## Quick Start Example

### 1. Basic Workflow

```bash
# 1. Generate a 10-degree mesh (fast, good for testing)
python geo_mesh_processor.py 10

# 2. Visualize the mesh
python plot_mesh.py meshes/mesh_data_10.0deg.json

# 3. Run climate benchmark with 1 query per point
python climate_llm_benchmark.py meshes/mesh_data_10.0deg.json 1 gpt-4o-mini


```

### 2. Expected Output Structure

After running the above commands, you'll have:

```
geo_benchmark/
├── meshes/
│   ├── mesh_data_10.0deg.json    # Complete mesh data
│   └── mesh_data_10.0deg.csv     # Simplified CSV format
├── results/
│   └── climate_results_10.0deg_r1.json  # LLM benchmark results
└── png/
    ├── mesh_plot_10.0deg.png     # Land/ocean visualization
    └── mesh_countries_10.0deg.png  # Country-colored map
```

