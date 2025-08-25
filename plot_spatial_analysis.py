#!/usr/bin/env python3
"""
Comprehensive plotting script for spatial RMSE analysis results.

This script creates various maps from spatial RMSE-enhanced result files:
- RMSE maps (radius 2 and 4)
- Bias maps (radius 2 and 4) 
- Neighborhood mean temperature maps (LLM and ERA5, radius 2 and 4)
- Individual point temperature maps (LLM and ERA5)

Usage:
    python plot_spatial_analysis.py [results_file]

Default: results/climate_results_20.0deg_r10_simple_spatial_rmse.json
Output: png/spatial_analysis_*_{resolution}deg.png
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


def load_spatial_rmse_results(results_file):
    """Load spatial RMSE-enhanced results file"""
    print(f"Loading spatial RMSE results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check if spatial RMSE data is present
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False)
    if not has_spatial:
        print("Warning: Spatial RMSE data not found in results file")
        print("Please run extend_results_with_spatial_rmse.py first")
    
    return results_data


def extract_mapping_data(results_data, field_name):
    """Extract mapping data for a specific field from results"""
    mapping_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid data
        if (point_info.get('is_land', False) and 
            field_name in point_info and
            not np.isnan(point_info.get(field_name, np.nan))):
            
            mapping_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'value': point_info[field_name],
                'country': point_info.get('country', ''),
                'state': point_info.get('state', '')
            })
    
    print(f"Found {len(mapping_data)} valid data points for {field_name}")
    return mapping_data


def create_data_grid(mapping_data, resolution=20.0):
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
    data_grid = np.full_like(lon_grid, np.nan, dtype=float)
    
    # Map data to grid
    for data_point in mapping_data:
        lat = data_point['lat']
        lon = data_point['lon']
        value = data_point['value']
        
        # Find grid indices
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        # Check if indices are valid and within tolerance
        tolerance = grid_res / 2
        if (np.abs(lats[lat_idx] - lat) <= tolerance and 
            np.abs(lons[lon_idx] - lon) <= tolerance):
            data_grid[lat_idx, lon_idx] = value
    
    print(f"Mapped {np.sum(~np.isnan(data_grid))} points to {len(lats)}×{len(lons)} grid")
    
    return lon_grid, lat_grid, data_grid


def create_map_plot(mapping_data, field_name, resolution, colormap='viridis', 
                   vmin=None, vmax=None, title_suffix="", output_file=None, figsize=(18, 10)):
    """Create a contour map for the given field"""
    
    if not mapping_data:
        print(f"No data available for {field_name}")
        return None
    
    # Create regular grid
    lon_grid, lat_grid, data_grid = create_data_grid(mapping_data, resolution)
    
    if lon_grid is None:
        print(f"Failed to create grid for {field_name}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract values for contour levels and statistics
    values = [d['value'] for d in mapping_data]
    
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    # Create contour levels
    levels = np.linspace(vmin, vmax, 20)
    
    # Create contour plot
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        # For error metrics, use reversed colormap (blue=good, red=bad)
        cmap = plt.cm.get_cmap('viridis_r')
    elif 'bias' in field_name.lower():
        # For bias, use diverging colormap (blue=negative, red=positive)
        cmap = plt.cm.RdBu_r
        # Make bias symmetric around zero
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        levels = np.linspace(vmin, vmax, 20)
    elif 'temp' in field_name.lower():
        # For temperature, use temperature colormap (blue=cold, red=hot)
        from matplotlib.colors import LinearSegmentedColormap
        temp_colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800000']
        cmap = LinearSegmentedColormap.from_list('temperature', temp_colors, N=256)
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    contour = ax.contourf(lon_grid, lat_grid, data_grid, 
                         levels=levels, cmap=cmap, extend='both')
    
    # Try to load and overlay country boundaries
    try:
        world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
        world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
    except Exception as e:
        print(f"Could not load country boundaries: {e}")
        ax.grid(True, alpha=0.3)
    
    # Customize the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    
    # Create title
    clean_field_name = field_name.replace('_', ' ').title()
    ax.set_title(f'{clean_field_name} Map{title_suffix}\n'
                f'Land Points: {len(mapping_data):,}', 
                fontsize=14, fontweight='bold')
    
    # Add coordinate ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    
    # Set colorbar label based on field type
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        cbar.set_label('Error (°C)', fontsize=12)
    elif 'bias' in field_name.lower():
        cbar.set_label('Bias (°C)', fontsize=12)
    elif 'temp' in field_name.lower():
        cbar.set_label('Temperature (°C)', fontsize=12)
    else:
        cbar.set_label('Value', fontsize=12)
    
    # Add statistics text
    mean_val = np.mean(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_val = np.std(values)
    
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {std_val:.2f}\n'
    stats_text += f'Range: {min_val:.2f} - {max_val:.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{clean_field_name} map saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def extract_comparison_data(results_data):
    """Extract LLM vs ERA5 comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0.0),
                'era5_std': point_info.get('era5_temp_std', np.nan),
                'temp_diff': point_info.get('temp_difference', 
                                          point_info['llm_temp_mean'] - point_info['era5_temp_mean']),
                'llm_count': point_info.get('llm_temp_count', 1),
                'country': point_info.get('country', '')
            })
    
    print(f"Found {len(comparison_data)} valid comparison data points")
    return comparison_data


def create_density_scatter_plot(comparison_data, resolution, output_file=None):
    """Create density scatter plot comparing LLM vs ERA5 with statistical analysis"""
    
    if not comparison_data:
        print("No comparison data available for scatter plot")
        return None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    llm_stds = np.array([d['llm_std'] if not np.isnan(d['llm_std']) else 0 for d in comparison_data])
    era5_stds = np.array([d['era5_std'] if not np.isnan(d['era5_std']) else 0 for d in comparison_data])
    llm_counts = np.array([d['llm_count'] for d in comparison_data])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([era5_vals, llm_vals])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
    except:
        # Fallback if KDE fails (e.g., all points identical)
        density = np.ones_like(era5_vals)
    
    # Sort points by density (plot low density first)
    idx = density.argsort()
    era5_sorted, llm_sorted, density_sorted = era5_vals[idx], llm_vals[idx], density[idx]
    llm_stds_sorted, era5_stds_sorted = llm_stds[idx], era5_stds[idx]
    
    # Create density scatter plot
    scatter = ax1.scatter(era5_sorted, llm_sorted, c=density_sorted, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
    # Error bars removed as requested
    
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
    ax1.set_title(f'LLM vs ERA5 Temperature Comparison\nDensity Scatter Plot ({resolution}°)', 
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
    mean_llm_requests = np.mean(llm_counts)
    total_llm_requests = np.sum(llm_counts)
    stats_text = f'N = {len(llm_vals)} points\n'
    stats_text += f'Total LLM requests: {total_llm_requests}\n'
    stats_text += f'Avg requests/point: {mean_llm_requests:.1f}\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
    stats_text += f'Correlation = {r_corr:.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Right plot: Difference histogram ===
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
    ax2.set_title('Temperature Difference Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add difference statistics
    diff_stats = f'Difference Statistics:\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\n'
    diff_stats += f'IQR: {np.percentile(differences, 75) - np.percentile(differences, 25):.3f}°C\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"LLM vs ERA5 comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def create_spatial_comparison_plots(comparison_data, resolution, output_dir='png'):
    """Create spatial comparison scatter plots for neighborhood data"""
    
    print("Creating spatial neighborhood comparison plots...")
    
    # Create plots for different spatial scales
    spatial_comparisons = [
        ('r2', '5×5 neighborhood'),
        ('r4', '9×9 neighborhood')
    ]
    
    created_plots = []
    
    for radius, description in spatial_comparisons:
        # Extract data for this spatial scale
        llm_field = f'spatial_{radius}_neighborhood_llm_mean'
        era5_field = f'spatial_{radius}_neighborhood_era5_mean'
        
        spatial_data = []
        for result_data in comparison_data:
            # We need to get the original result to access spatial fields
            # For now, skip spatial neighborhood comparisons as we don't have direct access
            # This would require re-extracting from the original results_data
            pass
    
    return created_plots


def create_all_comparison_plots(results_data, resolution, output_dir='png'):
    """Create all LLM vs ERA5 comparison plots"""
    
    print("Creating LLM vs ERA5 comparison plots...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract comparison data
    comparison_data = extract_comparison_data(results_data)
    
    if not comparison_data:
        print("No comparison data available for plotting")
        return []
    
    created_plots = []
    
    # Create main density scatter plot with difference histogram
    scatter_output = output_path / f"llm_era5_comparison_{resolution}deg.png"
    fig, stats = create_density_scatter_plot(comparison_data, resolution, str(scatter_output))
    if fig is not None:
        created_plots.append(str(scatter_output))
        rmse, mae, bias, r_corr = stats
        print(f"Overall comparison statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
    
    return created_plots


def create_all_spatial_maps(results_data, resolution, output_dir='png'):
    """Create all spatial analysis maps from the results data"""
    
    print("Creating comprehensive spatial analysis maps...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base filename pattern
    base_name = f"spatial_analysis_{resolution}deg"
    
    # Maps to create - field_name: (title_suffix, colormap)
    maps_config = {
        # RMSE maps
        'spatial_r2_rmse': ('(5×5 neighborhood)', 'viridis_r'),
        'spatial_r4_rmse': ('(9×9 neighborhood)', 'viridis_r'),
        
        # Bias maps  
        'spatial_r2_bias': ('(5×5 neighborhood)', 'RdBu_r'),
        'spatial_r4_bias': ('(9×9 neighborhood)', 'RdBu_r'),
        
        # Neighborhood LLM mean temperature
        'spatial_r2_neighborhood_llm_mean': ('(5×5 neighborhood)', 'temperature'),
        'spatial_r4_neighborhood_llm_mean': ('(9×9 neighborhood)', 'temperature'),
        
        # Neighborhood ERA5 mean temperature
        'spatial_r2_neighborhood_era5_mean': ('(5×5 neighborhood)', 'temperature'),
        'spatial_r4_neighborhood_era5_mean': ('(9×9 neighborhood)', 'temperature'),
        
        # Individual point temperatures
        'llm_temp_mean': ('(Individual Points)', 'temperature'),
        'era5_temp_mean': ('(Individual Points)', 'temperature'),
    }
    
    created_maps = []
    
    for field_name, (title_suffix, colormap) in maps_config.items():
        print(f"\nCreating map for {field_name}...")
        
        # Extract mapping data for this field
        mapping_data = extract_mapping_data(results_data, field_name)
        
        if not mapping_data:
            print(f"No data found for {field_name}, skipping...")
            continue
        
        # Create output filename
        clean_field_name = field_name.replace('spatial_', '').replace('_', '_')
        output_file = output_path / f"{base_name}_{clean_field_name}.png"
        
        # Determine temperature range for consistency
        vmin, vmax = None, None
        if 'temp' in field_name and 'mean' in field_name:
            # Use consistent temperature range for temperature maps
            values = [d['value'] for d in mapping_data]
            if values:
                vmin = min(values)
                vmax = max(values)
                
                # For neighborhood temperatures, use the same range as individual points
                if 'neighborhood' in field_name:
                    # Get range from individual point temperatures for consistency
                    point_field = 'llm_temp_mean' if 'llm' in field_name else 'era5_temp_mean'
                    point_data = extract_mapping_data(results_data, point_field)
                    if point_data:
                        point_values = [d['value'] for d in point_data]
                        vmin = min(point_values)
                        vmax = max(point_values)
        
        # Create the map
        try:
            fig = create_map_plot(
                mapping_data=mapping_data,
                field_name=field_name,
                resolution=resolution,
                colormap=colormap,
                vmin=vmin,
                vmax=vmax,
                title_suffix=f" {title_suffix}",
                output_file=str(output_file)
            )
            
            if fig is not None:
                created_maps.append(str(output_file))
                
        except Exception as e:
            print(f"Error creating map for {field_name}: {e}")
            continue
    
    print(f"\nSpatial analysis mapping completed!")
    print(f"Created {len(created_maps)} maps in {output_dir}/")
    
    return created_maps


def print_summary_statistics(results_data):
    """Print summary statistics for all spatial metrics"""
    
    print(f"\nSpatial Analysis Summary Statistics:")
    print(f"=" * 70)
    
    # Fields to summarize
    summary_fields = [
        'spatial_r2_rmse', 'spatial_r4_rmse',
        'spatial_r2_mae', 'spatial_r4_mae', 
        'spatial_r2_bias', 'spatial_r4_bias',
        'spatial_r2_correlation', 'spatial_r4_correlation',
        'llm_temp_mean', 'era5_temp_mean'
    ]
    
    for field in summary_fields:
        values = []
        
        for result in results_data['results']:
            point_info = result['point_info']
            if (point_info.get('is_land', False) and 
                field in point_info and
                not np.isnan(point_info.get(field, np.nan))):
                values.append(point_info[field])
        
        if values:
            print(f"\n{field.replace('_', ' ').title()}:")
            print(f"  Points: {len(values)}")
            print(f"  Mean: {np.mean(values):.3f}")
            print(f"  Median: {np.median(values):.3f}")
            print(f"  Std: {np.std(values):.3f}")
            print(f"  Range: {np.min(values):.3f} - {np.max(values):.3f}")


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_20.0deg_r10_simple_spatial_rmse.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Comprehensive Spatial Analysis Plotting")
    print("=" * 70)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run extend_results_with_spatial_rmse.py first to create spatial RMSE data.")
        return
    
    try:
        # Load spatial RMSE results
        results_data = load_spatial_rmse_results(results_file)
        
        # Extract resolution from metadata
        resolution = results_data.get('resolution', 20.0)
        
        # Create all spatial maps
        created_maps = create_all_spatial_maps(results_data, resolution)
        
        # Create comparison plots
        created_comparison_plots = create_all_comparison_plots(results_data, resolution)
        
        # Print summary statistics
        print_summary_statistics(results_data)
        
        total_plots = len(created_maps) + len(created_comparison_plots)
        print(f"\nPlotting completed successfully!")
        print(f"Created {len(created_maps)} spatial maps")
        print(f"Created {len(created_comparison_plots)} comparison plots")
        print(f"Total: {total_plots} plots")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()