#!/usr/bin/env python3
"""
Complete analysis pipeline for LLM climate benchmark results.

This script runs the complete analysis pipeline:
1. Extends results with spatial RMSE data
2. Adds bathymetry/elevation data  
3. Adds population density data
4. Generates spatial analysis plots
5. Generates temperature comparison plots
6. Generates elevation clustering plots
7. Generates population clustering plots
8. Generates bathymetry maps and comparisons
9. Generates population maps and comparisons

Usage:
    python run_complete_analysis_pipeline.py [original_results_file]

Default file: climate_results_1.0deg_r10_simple.json
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, args):
    """Run a Python script with arguments"""
    subprocess.run(['python', script_name] + args, check=True)

def main():
    """Main pipeline function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple.json'
    
    # Parse command line arguments
    original_results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    # Define intermediate and final filenames
    original_path = Path(original_results_file)
    base_name = original_path.stem
    parent_dir = original_path.parent
    
    # Generate progressive filenames
    spatial_file = parent_dir / f"{base_name}_spatial_rmse.json"
    bathymetry_file = parent_dir / f"{base_name}_spatial_rmse_bathymetry.json" 
    final_file = parent_dir / f"{base_name}_spatial_rmse_bathymetry_population.json"
    
    # Step 1: Add spatial RMSE data
    run_script('extend_results_with_spatial_rmse.py', [original_results_file])
    
    # Step 2: Add bathymetry data
    run_script('add_bathymetry_to_results.py', [str(spatial_file)])
    
    # Step 3: Add population data
    run_script('add_population_to_results.py', [str(bathymetry_file)])
    
    # Step 4: Generate spatial analysis plots
    run_script('plot_spatial_analysis.py', [str(final_file)])
    
    # Step 5: Generate temperature comparison plots
    run_script('plot_temperature_comparison_colored.py', [str(final_file)])
    
    # Step 6: Generate elevation clustering plots
    run_script('plot_elevation_clusters.py', [str(final_file)])
    
    # Step 7: Generate population clustering plots
    run_script('plot_population_clusters.py', [str(final_file)])
    
    # Step 8: Generate bathymetry maps and comparisons
    run_script('plot_bathymetry_map.py', [str(final_file)])
    
    # Step 9: Generate population maps and comparisons
    run_script('plot_population_map.py', [str(final_file)])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())