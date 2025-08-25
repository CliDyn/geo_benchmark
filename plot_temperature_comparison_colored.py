#!/usr/bin/env python3
"""
Create LLM vs ERA5 temperature comparison scatter plots with coloring by elevation, population, and roughness.

This script generates three versions of the LLM vs ERA5 temperature comparison plot:
1. Points colored by mean elevation
2. Points colored by population density (log scale)
3. Points colored by terrain roughness (log scale)

Each plot includes 1:1 line, regression line, statistics, and difference histogram.
Output: png/{results_filename}/temperature_comparison_*_{resolution}deg_combined.png

Usage:
    python plot_temperature_comparison_colored.py [results_file]

Default file: climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import json
from pathlib import Path
import sys
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_results(results_file):
    """Load results file with ERA5, spatial, population, and bathymetry data"""
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
    
    if not (has_era5 and has_population and has_bathymetry):
        print("Warning: Missing required data (ERA5, population, or bathymetry)")
    
    return results_data

def extract_comparison_data_with_coloring(results_data):
    """Extract LLM vs ERA5 comparison data with coloring parameters"""
    
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with complete data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and
            'era5_temp_mean' in point_info and
            'population_density' in point_info and
            'mean_elevation' in point_info and
            'roughness' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_temp_mean', np.nan)) and
            not np.isnan(point_info.get('population_density', np.nan)) and
            not np.isnan(point_info.get('mean_elevation', np.nan)) and
            not np.isnan(point_info.get('roughness', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0),
                'era5_std': point_info.get('era5_temp_std', 0),
                'llm_count': point_info.get('llm_temp_count', 1),
                'population_density': point_info['population_density'],
                'mean_elevation': point_info['mean_elevation'],
                'roughness': point_info['roughness']
            })
    
    print(f"Extracted {len(comparison_data)} points with complete comparison and coloring data")
    return comparison_data

def create_single_scatter_plot(comparison_data, color_param, color_label, colormap, 
                              use_log_color, ax):
    """Create a single LLM vs ERA5 scatter plot colored by specified parameter"""
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    llm_counts = np.array([d['llm_count'] for d in comparison_data])
    color_vals = np.array([d[color_param] for d in comparison_data])
    
    # Handle log scale for color values if requested
    if use_log_color:
        # Handle zero or negative values for log scale
        if color_param == 'population_density':
            color_vals_plot = np.log10(color_vals + 1)  # Add 1 to avoid log(0)
        elif color_param == 'roughness':
            min_positive = color_vals[color_vals > 0].min() if (color_vals > 0).any() else 1
            color_vals_plot = color_vals.copy()
            color_vals_plot[color_vals_plot <= 0] = min_positive
            color_vals_plot = np.log10(color_vals_plot)
        else:
            color_vals_plot = color_vals
    else:
        color_vals_plot = color_vals
    
    # Create scatter plot colored by parameter
    scatter = ax.scatter(era5_vals, llm_vals, c=color_vals_plot, s=20, alpha=0.7, 
                        cmap=colormap, edgecolors='black', linewidth=0.2)
    
    # Add 1:1 line
    min_temp = min(np.min(llm_vals), np.min(era5_vals))
    max_temp = max(np.max(llm_vals), np.max(era5_vals))
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, linewidth=1.5, 
            label='1:1 line')
    
    # Calculate and add regression line
    coeffs = np.polyfit(era5_vals, llm_vals, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_temp, max_temp, 100)
    ax.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=1.5, 
            label=f'R: y={coeffs[0]:.2f}x+{coeffs[1]:.1f}')
    
    ax.set_xlabel('ERA5 Temperature (°C)', fontsize=11)
    ax.set_ylabel('LLM Temperature (°C)', fontsize=11)
    ax.set_title(f'Colored by {color_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    
    # Set colorbar labels based on parameter and log scale
    if use_log_color and color_param == 'population_density':
        # Create custom tick labels for population (log scale)
        tick_positions_log = np.linspace(color_vals_plot.min(), color_vals_plot.max(), 6)
        tick_labels_original = 10**tick_positions_log - 1
        cbar.set_ticks(tick_positions_log)
        cbar.set_ticklabels([f'{val:.0f}' for val in tick_labels_original])
        cbar.set_label('Pop. Density\n(people/km²)', fontsize=9)
    elif use_log_color and color_param == 'roughness':
        # Create custom tick labels for roughness (log scale)
        tick_positions_log = np.linspace(color_vals_plot.min(), color_vals_plot.max(), 6)
        tick_labels_original = 10**tick_positions_log
        cbar.set_ticks(tick_positions_log)
        cbar.set_ticklabels([f'{val:.0f}' for val in tick_labels_original])
        cbar.set_label('Roughness\n(m)', fontsize=9)
    else:
        # Linear scale
        cbar.set_label('Elevation\n(m)', fontsize=9)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # Add compact statistics text
    stats_text = f'N={len(llm_vals)}\n'
    stats_text += f'RMSE={rmse:.1f}°C\n'
    stats_text += f'MAE={mae:.1f}°C\n'
    stats_text += f'Bias={bias:+.1f}°C\n'
    stats_text += f'R={r_corr:.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return scatter, (rmse, mae, bias, r_corr)

def create_combined_colored_plots(comparison_data, resolution, output_file=None):
    """Create combined plot with all three coloring schemes"""
    
    if not comparison_data:
        print("No comparison data available for scatter plot")
        return None, None
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Configuration for different coloring schemes
    color_configs = [
        {
            'param': 'mean_elevation',
            'label': 'Elevation',
            'colormap': 'terrain',
            'use_log': False
        },
        {
            'param': 'population_density',
            'label': 'Population',
            'colormap': 'YlOrRd',
            'use_log': True
        },
        {
            'param': 'roughness',
            'label': 'Roughness',
            'colormap': 'plasma',
            'use_log': True
        }
    ]
    
    # Create each subplot
    all_stats = []
    for i, config in enumerate(color_configs):
        scatter, stats = create_single_scatter_plot(
            comparison_data,
            config['param'],
            config['label'],
            config['colormap'],
            config['use_log'],
            axes[i]
        )
        all_stats.append(stats)
    
    # Add main title
    fig.suptitle(f'LLM vs ERA5 Temperature Comparison ({resolution}° resolution)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined temperature comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, all_stats

def create_colored_scatter_plot(comparison_data, color_param, color_label, colormap, 
                               use_log_color, resolution, output_file=None):
    """Create LLM vs ERA5 scatter plot colored by specified parameter"""
    
    if not comparison_data:
        print("No comparison data available for scatter plot")
        return None, None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    llm_stds = np.array([d['llm_std'] if not np.isnan(d['llm_std']) else 0 for d in comparison_data])
    era5_stds = np.array([d['era5_std'] if not np.isnan(d['era5_std']) else 0 for d in comparison_data])
    llm_counts = np.array([d['llm_count'] for d in comparison_data])
    color_vals = np.array([d[color_param] for d in comparison_data])
    
    # Handle log scale for color values if requested
    if use_log_color:
        # Handle zero or negative values for log scale
        if color_param == 'population_density':
            color_vals_plot = np.log10(color_vals + 1)  # Add 1 to avoid log(0)
        elif color_param == 'roughness':
            min_positive = color_vals[color_vals > 0].min() if (color_vals > 0).any() else 1
            color_vals_plot = color_vals.copy()
            color_vals_plot[color_vals_plot <= 0] = min_positive
            color_vals_plot = np.log10(color_vals_plot)
        else:
            color_vals_plot = color_vals
    else:
        color_vals_plot = color_vals
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Colored scatter plot ===
    
    # Create scatter plot colored by parameter
    if use_log_color and color_param in ['population_density', 'roughness']:
        scatter = ax1.scatter(era5_vals, llm_vals, c=color_vals_plot, s=30, alpha=0.7, 
                             cmap=colormap, edgecolors='black', linewidth=0.3)
    else:
        scatter = ax1.scatter(era5_vals, llm_vals, c=color_vals_plot, s=30, alpha=0.7, 
                             cmap=colormap, edgecolors='black', linewidth=0.3)
    
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
    ax1.set_title(f'LLM vs ERA5 Temperature Comparison\nColored by {color_label} ({resolution}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    
    # Set colorbar labels based on parameter and log scale
    if use_log_color and color_param == 'population_density':
        # Create custom tick labels for population (log scale)
        tick_positions_log = np.linspace(color_vals_plot.min(), color_vals_plot.max(), 8)
        tick_labels_original = 10**tick_positions_log - 1
        cbar1.set_ticks(tick_positions_log)
        cbar1.set_ticklabels([f'{val:.1f}' if val < 10 else f'{val:.0f}' for val in tick_labels_original])
        cbar1.set_label(f'{color_label} (people/km²)', fontsize=11)
    elif use_log_color and color_param == 'roughness':
        # Create custom tick labels for roughness (log scale)
        tick_positions_log = np.linspace(color_vals_plot.min(), color_vals_plot.max(), 8)
        tick_labels_original = 10**tick_positions_log
        cbar1.set_ticks(tick_positions_log)
        cbar1.set_ticklabels([f'{val:.0f}' for val in tick_labels_original])
        cbar1.set_label(f'{color_label} (m)', fontsize=11)
    else:
        # Linear scale
        cbar1.set_label(f'{color_label} (m)', fontsize=11)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # Add statistics text
    mean_llm_requests = np.mean(llm_counts)
    total_llm_requests = np.sum(llm_counts)
    stats_text = f'N = {len(llm_vals)} points\\n'
    stats_text += f'Total LLM requests: {total_llm_requests}\\n'
    stats_text += f'Avg requests/point: {mean_llm_requests:.1f}\\n'
    stats_text += f'RMSE = {rmse:.2f}°C\\n'
    stats_text += f'MAE = {mae:.2f}°C\\n'
    stats_text += f'Bias = {bias:+.2f}°C\\n'
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
    diff_stats = f'Difference Statistics:\\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\\n'
    diff_stats += f'IQR: {np.percentile(differences, 75) - np.percentile(differences, 25):.3f}°C\\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Temperature comparison plot ({color_label}) saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)

def print_coloring_parameter_statistics(comparison_data):
    """Print statistics for coloring parameters"""
    
    print(f"\nColoring Parameter Statistics:")
    print(f"=" * 60)
    
    # Extract coloring parameters
    elevations = np.array([d['mean_elevation'] for d in comparison_data])
    populations = np.array([d['population_density'] for d in comparison_data])
    roughness = np.array([d['roughness'] for d in comparison_data])
    
    # Elevation statistics
    print(f"\nMean Elevation (m):")
    print(f"  Count: {len(elevations)}")
    print(f"  Range: {np.min(elevations):.1f} to {np.max(elevations):.1f}")
    print(f"  Mean: {np.mean(elevations):.1f}")
    print(f"  Median: {np.median(elevations):.1f}")
    print(f"  Std: {np.std(elevations):.1f}")
    
    # Population statistics
    print(f"\nPopulation Density (people/km²):")
    print(f"  Count: {len(populations)}")
    print(f"  Range: {np.min(populations):.1f} to {np.max(populations):.1f}")
    print(f"  Mean: {np.mean(populations):.1f}")
    print(f"  Median: {np.median(populations):.1f}")
    print(f"  Std: {np.std(populations):.1f}")
    
    # Roughness statistics
    print(f"\nTerrain Roughness (m):")
    print(f"  Count: {len(roughness)}")
    print(f"  Range: {np.min(roughness):.1f} to {np.max(roughness):.1f}")
    print(f"  Mean: {np.mean(roughness):.1f}")
    print(f"  Median: {np.median(roughness):.1f}")
    print(f"  Std: {np.std(roughness):.1f}")

def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("LLM vs ERA5 Temperature Comparison with Colored Scatter Plots")
    print("=" * 80)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # Load data
        results_data = load_enhanced_results(results_file)
        
        # Extract comparison data
        comparison_data = extract_comparison_data_with_coloring(results_data)
        
        if len(comparison_data) == 0:
            print("Error: No valid comparison data found")
            return
        
        # Print statistics
        print_coloring_parameter_statistics(comparison_data)
        
        # Get resolution from results data
        resolution = results_data.get('resolution', 1.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem  # Get filename without extension
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        print(f"\nCreating combined colored scatter plot...")
        
        # Create single combined plot
        output_file = output_dir / f'temperature_comparison_{resolution}deg_combined.png'
        
        fig, all_stats = create_combined_colored_plots(
            comparison_data,
            resolution,
            str(output_file)
        )
        
        if fig is not None:
            # Print statistics for each coloring scheme
            color_labels = ['Elevation', 'Population', 'Roughness']
            for i, (rmse, mae, bias, r_corr) in enumerate(all_stats):
                print(f"Statistics for {color_labels[i]} - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
        
        print(f"\nAll temperature comparison plots completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()