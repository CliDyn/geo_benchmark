#!/usr/bin/env python3
"""
Comprehensive plotting script for climate change difference analysis results.

This script creates various maps and plots from climate change spatial RMSE-enhanced result files:
- Climate change signal maps (LLM predictions vs ERA5)
- Spatial RMSE maps for climate change predictions (radius 2 and 4)
- Bias maps showing LLM prediction errors
- Correlation maps between LLM and ERA5 climate signals
- Scatter plots comparing LLM vs ERA5 climate change predictions
- Distribution analysis of climate change signal predictions

Usage:
    python plot_climate_change_analysis.py [results_file]

Default: results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff_climate_rmse.json
Output: png/{results_filename}/climate_change_*_{resolution}deg.png
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


def load_climate_change_results(results_file):
    """Load climate change spatial RMSE-enhanced results file"""
    print(f"Loading climate change results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check if climate change spatial RMSE data is present
    has_climate_rmse = results_data.get('metadata', {}).get('climate_change_spatial_rmse_added', False)
    if not has_climate_rmse:
        print("Warning: Climate change spatial RMSE data not found in results file")
        print("Please run extend_results_with_climate_change_rmse.py first")
    
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


def create_climate_change_map(mapping_data, field_name, resolution, colormap='RdBu_r', 
                             vmin=None, vmax=None, title_suffix="", output_file=None, figsize=(18, 10)):
    """Create a contour map for climate change data"""
    
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
    
    # Choose colormap based on field type
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        # For error metrics, use reversed viridis (blue=good, red=bad)
        cmap = plt.cm.get_cmap('viridis_r')
    elif 'bias' in field_name.lower() or 'change' in field_name.lower() or 'diff' in field_name.lower():
        # For climate change signals and bias, use diverging colormap
        cmap = plt.cm.RdBu_r
        # Make symmetric around zero for change signals
        if 'change' in field_name.lower() or 'era5' in field_name.lower() or 'llm' in field_name.lower():
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            levels = np.linspace(vmin, vmax, 20)
    elif 'correlation' in field_name.lower():
        # For correlation, use diverging colormap centered at 0
        cmap = plt.cm.RdBu
        vmin, vmax = -1, 1
        levels = np.linspace(vmin, vmax, 20)
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
    elif 'change' in field_name.lower() or 'diff' in field_name.lower():
        cbar.set_label('Temperature Change (°C)', fontsize=12)
    elif 'correlation' in field_name.lower():
        cbar.set_label('Correlation Coefficient', fontsize=12)
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


def extract_climate_comparison_data(results_data):
    """Extract LLM vs ERA5 climate change comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 climate change data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_change_signal' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_change_signal', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_change': point_info['llm_temp_mean'],  # LLM predicted change
                'era5_change': point_info['era5_change_signal'],  # ERA5 climate change signal
                'difference': point_info['llm_temp_mean'] - point_info['era5_change_signal'],
                'country': point_info.get('country', ''),
                'spatial_r2_rmse': point_info.get('spatial_r2_rmse', np.nan),
                'spatial_r4_rmse': point_info.get('spatial_r4_rmse', np.nan)
            })
    
    print(f"Found {len(comparison_data)} valid climate change comparison data points")
    return comparison_data


def create_climate_change_scatter_plot(comparison_data, resolution, output_file=None):
    """Create density scatter plot comparing LLM vs ERA5 climate change predictions"""
    
    if not comparison_data:
        print("No climate change comparison data available for scatter plot")
        return None
    
    # Extract arrays for plotting
    llm_changes = np.array([d['llm_change'] for d in comparison_data])
    era5_changes = np.array([d['era5_change'] for d in comparison_data])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # === Top Left: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([era5_changes, llm_changes])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
    except:
        # Fallback if KDE fails (e.g., all points identical)
        density = np.ones_like(era5_changes)
    
    # Sort points by density (plot low density first)
    idx = density.argsort()
    era5_sorted, llm_sorted, density_sorted = era5_changes[idx], llm_changes[idx], density[idx]
    
    # Create density scatter plot
    scatter = ax1.scatter(era5_sorted, llm_sorted, c=density_sorted, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
    # Add 1:1 line
    min_change = min(np.min(llm_changes), np.min(era5_changes))
    max_change = max(np.max(llm_changes), np.max(era5_changes))
    ax1.plot([min_change, max_change], [min_change, max_change], 'r--', alpha=0.8, linewidth=2, 
             label='1:1 line (perfect agreement)')
    
    # Calculate and add regression line
    coeffs = np.polyfit(era5_changes, llm_changes, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_change, max_change, 100)
    ax1.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel('ERA5 Climate Change Signal (°C)', fontsize=12)
    ax1.set_ylabel('LLM Predicted Change (°C)', fontsize=12)
    ax1.set_title(f'LLM vs ERA5 Climate Change Predictions\nDensity Scatter Plot ({resolution}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for density
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Point Density', fontsize=11)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_changes - era5_changes)**2))
    mae = np.mean(np.abs(llm_changes - era5_changes))
    bias = np.mean(llm_changes - era5_changes)
    r_corr = np.corrcoef(llm_changes, era5_changes)[0, 1]
    
    # Add statistics text
    stats_text = f'N = {len(llm_changes)} points\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
    stats_text += f'Correlation = {r_corr:.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Top Right: Difference histogram ===
    differences = llm_changes - era5_changes
    ax2.hist(differences, bins=25, alpha=0.7, edgecolor='black', color='skyblue', density=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean diff = {np.mean(differences):+.2f}°C')
    
    # Add normal distribution overlay for reference
    x_norm = np.linspace(differences.min(), differences.max(), 100)
    y_norm = (1/np.sqrt(2*np.pi*np.var(differences))) * np.exp(-0.5*((x_norm - np.mean(differences))**2)/np.var(differences))
    ax2.plot(x_norm, y_norm, 'g--', alpha=0.8, linewidth=2, label='Normal fit')
    
    ax2.set_xlabel('Prediction Error (LLM - ERA5) °C', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Climate Change Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add difference statistics
    diff_stats = f'Error Statistics:\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\n'
    diff_stats += f'IQR: {np.percentile(differences, 75) - np.percentile(differences, 25):.3f}°C\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Bottom Left: ERA5 vs LLM distribution comparison ===
    ax3.hist(era5_changes, bins=20, alpha=0.6, label='ERA5 Climate Signal', color='blue', density=True)
    ax3.hist(llm_changes, bins=20, alpha=0.6, label='LLM Predictions', color='red', density=True)
    ax3.axvline(np.mean(era5_changes), color='blue', linestyle='--', linewidth=2, 
                label=f'ERA5 Mean = {np.mean(era5_changes):.2f}°C')
    ax3.axvline(np.mean(llm_changes), color='red', linestyle='--', linewidth=2,
                label=f'LLM Mean = {np.mean(llm_changes):.2f}°C')
    
    ax3.set_xlabel('Temperature Change (°C)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Climate Change Signal Distributions', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # === Bottom Right: Spatial RMSE vs Prediction Error ===
    spatial_rmse_r2 = np.array([d['spatial_r2_rmse'] for d in comparison_data if not np.isnan(d['spatial_r2_rmse'])])
    spatial_diffs = np.array([d['difference'] for d in comparison_data if not np.isnan(d['spatial_r2_rmse'])])
    
    if len(spatial_rmse_r2) > 0:
        ax4.scatter(spatial_rmse_r2, np.abs(spatial_diffs), alpha=0.6, s=20, color='purple')
        ax4.set_xlabel('Spatial RMSE (5×5 neighborhood) °C', fontsize=12)
        ax4.set_ylabel('|Prediction Error| (°C)', fontsize=12)
        ax4.set_title('Spatial Consistency vs Prediction Error', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation if possible
        if len(spatial_rmse_r2) > 1:
            corr_spatial = np.corrcoef(spatial_rmse_r2, np.abs(spatial_diffs))[0, 1]
            ax4.text(0.02, 0.98, f'Correlation: {corr_spatial:.3f}', transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax4.text(0.5, 0.5, 'No spatial RMSE data available', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=14)
        ax4.set_title('Spatial Consistency vs Prediction Error', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Climate change comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def create_all_climate_change_maps(results_data, resolution, output_dir='png'):
    """Create all climate change analysis maps from the results data"""
    
    print("Creating comprehensive climate change analysis maps...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base filename pattern
    base_name = f"climate_change_{resolution}deg"
    
    # Maps to create - field_name: (title_suffix, colormap)
    maps_config = {
        # Climate change signals
        'llm_temp_mean': ('LLM Predicted Climate Change', 'RdBu_r'),
        'era5_change_signal': ('ERA5 Climate Change Signal', 'RdBu_r'),
        
        # Spatial RMSE maps for climate change
        'spatial_r2_rmse': ('Spatial RMSE (5×5 neighborhood)', 'viridis_r'),
        'spatial_r4_rmse': ('Spatial RMSE (9×9 neighborhood)', 'viridis_r'),
        
        # Bias maps for climate change predictions
        'spatial_r2_bias': ('Spatial Bias (5×5 neighborhood)', 'RdBu_r'),
        'spatial_r4_bias': ('Spatial Bias (9×9 neighborhood)', 'RdBu_r'),
        
        # Correlation maps
        'spatial_r2_correlation': ('Spatial Correlation (5×5 neighborhood)', 'RdBu'),
        'spatial_r4_correlation': ('Spatial Correlation (9×9 neighborhood)', 'RdBu'),
        
        # Neighborhood mean climate change signals
        'spatial_r2_neighborhood_llm_mean': ('LLM Neighborhood Mean (5×5)', 'RdBu_r'),
        'spatial_r4_neighborhood_llm_mean': ('LLM Neighborhood Mean (9×9)', 'RdBu_r'),
        'spatial_r2_neighborhood_era5_mean': ('ERA5 Neighborhood Mean (5×5)', 'RdBu_r'),
        'spatial_r4_neighborhood_era5_mean': ('ERA5 Neighborhood Mean (9×9)', 'RdBu_r'),
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
        
        # Determine value range for consistency
        vmin, vmax = None, None
        if ('change' in field_name or 'temp_mean' in field_name or 
            'neighborhood' in field_name or 'bias' in field_name):
            # Use symmetric range for climate change signals and bias
            values = [d['value'] for d in mapping_data]
            if values:
                abs_max = max(abs(min(values)), abs(max(values)))
                vmin, vmax = -abs_max, abs_max
        
        # Create the map
        try:
            fig = create_climate_change_map(
                mapping_data=mapping_data,
                field_name=field_name,
                resolution=resolution,
                colormap=colormap,
                vmin=vmin,
                vmax=vmax,
                title_suffix=f" - {title_suffix}",
                output_file=str(output_file)
            )
            
            if fig is not None:
                created_maps.append(str(output_file))
                
        except Exception as e:
            print(f"Error creating map for {field_name}: {e}")
            continue
    
    print(f"\nClimate change mapping completed!")
    print(f"Created {len(created_maps)} maps in {output_dir}/")
    
    return created_maps


def create_all_climate_comparison_plots(results_data, resolution, output_dir='png'):
    """Create all LLM vs ERA5 climate change comparison plots"""
    
    print("Creating LLM vs ERA5 climate change comparison plots...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract comparison data
    comparison_data = extract_climate_comparison_data(results_data)
    
    if not comparison_data:
        print("No climate change comparison data available for plotting")
        return []
    
    created_plots = []
    
    # Create main climate change comparison plot
    scatter_output = output_path / f"climate_change_comparison_{resolution}deg.png"
    fig, stats = create_climate_change_scatter_plot(comparison_data, resolution, str(scatter_output))
    if fig is not None:
        created_plots.append(str(scatter_output))
        rmse, mae, bias, r_corr = stats
        print(f"Climate change comparison statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
    
    return created_plots


def print_climate_change_summary_statistics(results_data):
    """Print summary statistics for all climate change metrics"""
    
    print(f"\nClimate Change Analysis Summary Statistics:")
    print(f"=" * 80)
    
    # Fields to summarize
    summary_fields = [
        'llm_temp_mean', 'era5_change_signal',
        'spatial_r2_rmse', 'spatial_r4_rmse',
        'spatial_r2_mae', 'spatial_r4_mae', 
        'spatial_r2_bias', 'spatial_r4_bias',
        'spatial_r2_correlation', 'spatial_r4_correlation'
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
    
    # Special analysis for climate change signals
    print(f"\nClimate Change Signal Analysis:")
    print(f"-" * 40)
    
    llm_changes = []
    era5_changes = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_change_signal' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_change_signal', np.nan))):
            llm_changes.append(point_info['llm_temp_mean'])
            era5_changes.append(point_info['era5_change_signal'])
    
    if llm_changes and era5_changes:
        llm_array = np.array(llm_changes)
        era5_array = np.array(era5_changes)
        
        print(f"LLM vs ERA5 Climate Change:")
        print(f"  Valid comparison points: {len(llm_changes)}")
        print(f"  LLM mean change: {np.mean(llm_array):.3f}°C")
        print(f"  ERA5 mean change: {np.mean(era5_array):.3f}°C")
        print(f"  Overall bias (LLM - ERA5): {np.mean(llm_array - era5_array):+.3f}°C")
        print(f"  Overall RMSE: {np.sqrt(np.mean((llm_array - era5_array)**2)):.3f}°C")
        print(f"  Overall correlation: {np.corrcoef(llm_array, era5_array)[0, 1]:.3f}")


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff_climate_rmse.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Comprehensive Climate Change Analysis Plotting")
    print("=" * 80)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run extend_results_with_climate_change_rmse.py first to create climate change spatial RMSE data.")
        return
    
    try:
        # Load climate change results
        results_data = load_climate_change_results(results_file)
        
        # Extract resolution from metadata or filename
        resolution = results_data.get('resolution', 1.0)
        if 'deg' in results_file:
            try:
                resolution = float(results_file.split('_')[2].replace('deg', ''))
            except:
                pass
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem  # Get filename without extension
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create all climate change maps
        created_maps = create_all_climate_change_maps(results_data, resolution, str(output_dir))
        
        # Create comparison plots
        created_comparison_plots = create_all_climate_comparison_plots(results_data, resolution, str(output_dir))
        
        # Print summary statistics
        print_climate_change_summary_statistics(results_data)
        
        total_plots = len(created_maps) + len(created_comparison_plots)
        print(f"\nClimate change plotting completed successfully!")
        print(f"Created {len(created_maps)} climate change maps")
        print(f"Created {len(created_comparison_plots)} comparison plots")
        print(f"Total: {total_plots} plots")
        print(f"All files saved in: {output_dir}/")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()