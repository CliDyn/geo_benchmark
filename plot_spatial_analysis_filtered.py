#!/usr/bin/env python3
"""
Filtered spatial RMSE analysis plotting script.

This script creates spatial analysis plots similar to plot_spatial_analysis.py but applies filters:
- Excludes points where population density < 5 people/km²
- Excludes points where elevation > 2000m

Creates the same maps as plot_spatial_analysis.py but with filtered data:
- RMSE maps (radius 2 and 4)
- Bias maps (radius 2 and 4) 
- Neighborhood mean temperature maps (LLM and ERA5, radius 2 and 4)
- Individual point temperature maps (LLM and ERA5)
- LLM vs ERA5 comparison plots with density coloring

Usage:
    python plot_spatial_analysis_filtered.py [results_file]

Default: results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
Output: png/{results_filename}/filtered_spatial_analysis_*_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
import pandas as pd
from pathlib import Path
import sys
import os
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_enhanced_results(results_file):
    """Load results file with spatial RMSE, population, and bathymetry data"""
    print(f"Loading enhanced results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check what data is present
    has_era5 = results_data.get('metadata', {}).get('era5_climatology_added', False)
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False)
    has_population = results_data.get('metadata', {}).get('population_added', False)
    has_bathymetry = results_data.get('metadata', {}).get('bathymetry_added', False)
    
    print(f"Data present - ERA5: {has_era5}, Spatial RMSE: {has_spatial}, Population: {has_population}, Bathymetry: {has_bathymetry}")
    
    if not (has_era5 and has_spatial and has_population and has_bathymetry):
        print("Warning: Missing required data for filtering")
    
    return results_data


def extract_filtered_mapping_data(results_data, field_name, min_population=5.0, max_elevation=2000.0):
    """Extract mapping data for a specific field with population and elevation filtering"""
    mapping_data = []
    total_land_points = 0
    filtered_out_pop = 0
    filtered_out_elev = 0
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only consider land points
        if point_info.get('is_land', False):
            total_land_points += 1
            
            # Check if point has required data
            if (field_name in point_info and
                not np.isnan(point_info.get(field_name, np.nan)) and
                'population_density' in point_info and
                'mean_elevation' in point_info and
                not np.isnan(point_info.get('population_density', np.nan)) and
                not np.isnan(point_info.get('mean_elevation', np.nan))):
                
                pop_density = point_info['population_density']
                elevation = point_info['mean_elevation']
                
                # Apply filters
                if pop_density < min_population:
                    filtered_out_pop += 1
                    continue
                    
                if elevation > max_elevation:
                    filtered_out_elev += 1
                    continue
                
                # Point passes all filters
                mapping_data.append({
                    'lat': point_info['lat'],
                    'lon': point_info['lon'],
                    'value': point_info[field_name],
                    'population_density': pop_density,
                    'mean_elevation': elevation,
                    'country': point_info.get('country', ''),
                    'state': point_info.get('state', '')
                })
    
    print(f"Filtering results for {field_name}:")
    print(f"  Total land points: {total_land_points}")
    print(f"  Filtered out (pop < {min_population}): {filtered_out_pop}")
    print(f"  Filtered out (elev > {max_elevation}m): {filtered_out_elev}")
    print(f"  Final valid points: {len(mapping_data)}")
    
    return mapping_data


def create_data_grid(mapping_data, resolution=1.0):
    """Create a regular meshgrid with data mapped to it"""
    
    if not mapping_data:
        return None, None, None
    
    # Define grid resolution based on data resolution
    if resolution >= 20.0:
        # For coarse resolution, use the actual resolution
        grid_res = resolution
        # Define bounds based on data
        lats = [p['lat'] for p in mapping_data]
        lons = [p['lon'] for p in mapping_data]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
    else:
        # For fine resolution, use 1 degree grid
        grid_res = 1.0
        lat_min, lat_max = -60, 84
        lon_min, lon_max = -180, 179
    
    # Create coordinate arrays
    lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Initialize data grid with NaN
    data_grid = np.full(lon_grid.shape, np.nan)
    
    # Map data points to grid
    for point in mapping_data:
        lat_idx = np.argmin(np.abs(lats - point['lat']))
        lon_idx = np.argmin(np.abs(lons - point['lon']))
        data_grid[lat_idx, lon_idx] = point['value']
    
    print(f"Created {lon_grid.shape} grid, {np.sum(~np.isnan(data_grid))} filled cells")
    return lon_grid, lat_grid, data_grid


def load_world_boundaries():
    """Load world boundaries for map visualization"""
    try:
        # Try to load from standard location
        data_path = Path("data/land")
        if data_path.exists():
            world_files = list(data_path.glob("*land*.shp"))
            if world_files:
                world = gpd.read_file(world_files[0])
                print(f"Loaded world boundaries from: {world_files[0]}")
                return world
        
        # Try to use Natural Earth from geopandas
        try:
            import geopandas.datasets
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            print("Loaded world boundaries from geopandas Natural Earth data")
            return world
        except:
            pass
        
        print("Warning: Could not load world boundaries, maps will not show coastlines")
        return None
        
    except Exception as e:
        print(f"Warning: Could not load world boundaries: {e}")
        return None


def create_spatial_map(mapping_data, field_name, field_label, colormap, resolution, 
                      use_log_scale=False, output_file=None):
    """Create a spatial map for a given field"""
    
    if not mapping_data:
        print(f"No data available for {field_name}")
        return None
    
    # Create data grid
    lon_grid, lat_grid, data_grid = create_data_grid(mapping_data, resolution)
    
    if lon_grid is None:
        print(f"Could not create grid for {field_name}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Load world boundaries
    world = load_world_boundaries()
    if world is not None:
        world.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7)
    
    # Create color normalization
    valid_data = data_grid[~np.isnan(data_grid)]
    if len(valid_data) == 0:
        print(f"No valid data for {field_name}")
        return None
    
    if use_log_scale and np.all(valid_data > 0):
        norm = colors.LogNorm(vmin=np.min(valid_data), vmax=np.max(valid_data))
    else:
        norm = colors.Normalize(vmin=np.min(valid_data), vmax=np.max(valid_data))
    
    # Create the map
    im = ax.contourf(lon_grid, lat_grid, data_grid, levels=20, cmap=colormap, norm=norm, extend='both')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(field_label, fontsize=12)
    
    # Formatting
    ax.set_xlim(lon_grid.min(), lon_grid.max())
    ax.set_ylim(lat_grid.min(), lat_grid.max())
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'{field_label} Map (Filtered: pop≥5/km², elev≤2000m)\nLand Points: {len(mapping_data)}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add filter info
    filter_text = f'Filters: Population ≥ 5 people/km²\n         Elevation ≤ 2000m'
    ax.text(0.02, 0.98, filter_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save or show
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved filtered map: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def create_density_scatter_with_filters(results_data, resolution, output_file=None):
    """Create LLM vs ERA5 comparison plots with density coloring - filtered data"""
    
    # Extract comparison data with filters
    comparison_data = []
    total_land_points = 0
    filtered_out_pop = 0
    filtered_out_elev = 0
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only consider land points with complete data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and
            'era5_temp_mean' in point_info and
            'population_density' in point_info and
            'mean_elevation' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_temp_mean', np.nan)) and
            not np.isnan(point_info.get('population_density', np.nan)) and
            not np.isnan(point_info.get('mean_elevation', np.nan))):
            
            total_land_points += 1
            
            pop_density = point_info['population_density']
            elevation = point_info['mean_elevation']
            
            # Apply filters
            if pop_density < 5.0:
                filtered_out_pop += 1
                continue
                
            if elevation > 2000.0:
                filtered_out_elev += 1
                continue
            
            # Point passes all filters
            comparison_data.append({
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0),
                'population_density': pop_density,
                'mean_elevation': elevation
            })
    
    print(f"\nFiltering results for comparison plot:")
    print(f"  Total land points with data: {total_land_points}")
    print(f"  Filtered out (pop < 5): {filtered_out_pop}")
    print(f"  Filtered out (elev > 2000m): {filtered_out_elev}")
    print(f"  Final comparison points: {len(comparison_data)}")
    
    if len(comparison_data) == 0:
        print("No valid comparison data after filtering")
        return None, None
    
    # Extract arrays
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([era5_vals, llm_vals])
    kde = gaussian_kde(xy)
    density = kde(xy)
    
    # Create scatter plot colored by density
    scatter = ax1.scatter(era5_vals, llm_vals, c=density, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
    # Add 1:1 line
    min_temp = min(np.min(llm_vals), np.min(era5_vals))
    max_temp = max(np.max(llm_vals), np.max(era5_vals))
    ax1.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, linewidth=2, 
             label='1:1 line')
    
    # Calculate and add regression line
    coeffs = np.polyfit(era5_vals, llm_vals, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_temp, max_temp, 100)
    ax1.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel('ERA5 Temperature (°C)', fontsize=12)
    ax1.set_ylabel('LLM Temperature (°C)', fontsize=12)
    ax1.set_title(f'LLM vs ERA5 Comparison - Filtered Data\nDensity Scatter Plot ({resolution}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for density
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Point Density', fontsize=11)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # Add statistics text
    stats_text = f'N = {len(llm_vals)} points\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
    stats_text += f'Correlation = {r_corr:.3f}\n\n'
    stats_text += f'Filters Applied:\n'
    stats_text += f'Pop ≥ 5 people/km²\n'
    stats_text += f'Elev ≤ 2000m'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Right plot: Temperature difference histogram ===
    differences = llm_vals - era5_vals
    ax2.hist(differences, bins=25, alpha=0.7, edgecolor='black', color='skyblue', density=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean diff = {np.mean(differences):+.2f}°C')
    
    # Add normal distribution overlay for reference
    x_norm = np.linspace(differences.min(), differences.max(), 100)
    y_norm = (1/np.sqrt(2*np.pi*np.var(differences))) * np.exp(-0.5*((x_norm - np.mean(differences))**2)/np.var(differences))
    ax2.plot(x_norm, y_norm, 'g--', alpha=0.8, linewidth=2, label='Normal fit')
    
    ax2.set_xlabel('Temperature Difference (LLM - ERA5) °C', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Temperature Difference Distribution\n(Filtered Data)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add difference statistics
    diff_stats = f'Difference Statistics:\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Filtered comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def create_all_filtered_spatial_maps(results_data, resolution, output_dir):
    """Create all filtered spatial maps"""
    
    # Define map configurations
    spatial_maps = {
        # RMSE maps  
        'spatial_r2_rmse': ('Spatial RMSE R2 (5×5)', 'plasma'),
        'spatial_r4_rmse': ('Spatial RMSE R4 (9×9)', 'plasma'),
        
        # Bias maps
        'spatial_r2_bias': ('Spatial Bias R2 (5×5)', 'RdBu_r'),
        'spatial_r4_bias': ('Spatial Bias R4 (9×9)', 'RdBu_r'),
        
        # Neighborhood temperatures
        'spatial_r2_llm_temp_mean': ('Spatial LLM Temp R2 (5×5)', 'coolwarm'),
        'spatial_r2_era5_temp_mean': ('Spatial ERA5 Temp R2 (5×5)', 'coolwarm'),
        'spatial_r4_llm_temp_mean': ('Spatial LLM Temp R4 (9×9)', 'coolwarm'),
        'spatial_r4_era5_temp_mean': ('Spatial ERA5 Temp R4 (9×9)', 'coolwarm'),
        
        # Individual point temperatures
        'llm_temp_mean': ('LLM Temp Mean (Individual Points)', 'coolwarm'),
        'era5_temp_mean': ('ERA5 Temp Mean (Individual Points)', 'coolwarm'),
    }
    
    created_maps = []
    
    for field_name, (field_label, colormap) in spatial_maps.items():
        print(f"\nCreating filtered map for {field_name}...")
        
        # Extract filtered mapping data
        mapping_data = extract_filtered_mapping_data(results_data, field_name)
        
        if mapping_data:
            # Create output filename
            output_file = Path(output_dir) / f'filtered_spatial_analysis_{field_name}_{resolution}deg.png'
            
            # Create map
            fig = create_spatial_map(
                mapping_data, 
                field_name, 
                field_label, 
                colormap, 
                resolution,
                use_log_scale=(field_name in ['spatial_r2_rmse', 'spatial_r4_rmse']),
                output_file=str(output_file)
            )
            
            if fig is not None:
                created_maps.append(str(output_file))
    
    return created_maps


def create_all_filtered_comparison_plots(results_data, resolution, output_dir):
    """Create all filtered comparison plots"""
    
    print(f"\nCreating filtered LLM vs ERA5 comparison plots...")
    
    # Create comparison plot
    output_file = Path(output_dir) / f'filtered_llm_era5_comparison_{resolution}deg.png'
    
    fig, stats = create_density_scatter_with_filters(
        results_data,
        resolution, 
        str(output_file)
    )
    
    created_plots = []
    if fig is not None:
        created_plots.append(str(output_file))
        rmse, mae, bias, r_corr = stats
        print(f"Filtered comparison statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
    
    return created_plots


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Filtered Spatial Analysis Plotting")
    print("=" * 50)
    print(f"Results file: {results_file}")
    print(f"Filters: Population ≥ 5 people/km², Elevation ≤ 2000m")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # Load data
        results_data = load_enhanced_results(results_file)
        
        # Get resolution from results data
        resolution = results_data.get('resolution', 1.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create all filtered spatial maps
        print(f"\n{'='*60}")
        print("CREATING FILTERED SPATIAL MAPS")
        print(f"{'='*60}")
        
        created_maps = create_all_filtered_spatial_maps(results_data, resolution, str(output_dir))
        
        print(f"\nCreated {len(created_maps)} filtered spatial maps")
        
        # Create filtered comparison plots
        print(f"\n{'='*60}")
        print("CREATING FILTERED COMPARISON PLOTS")
        print(f"{'='*60}")
        
        created_comparison_plots = create_all_filtered_comparison_plots(results_data, resolution, str(output_dir))
        
        print(f"\nCreated {len(created_comparison_plots)} filtered comparison plots")
        
        # Summary
        total_plots = len(created_maps) + len(created_comparison_plots)
        print(f"\n{'='*60}")
        print("FILTERED SPATIAL ANALYSIS COMPLETED")
        print(f"{'='*60}")
        print(f"Total plots created: {total_plots}")
        print(f"Output directory: {output_dir}")
        
        print(f"\nFiltered plots saved:")
        for plot_file in created_maps + created_comparison_plots:
            print(f"  - {Path(plot_file).name}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()