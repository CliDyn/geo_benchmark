#!/usr/bin/env python3
"""
Plot population density map and comparison plots from population-enhanced result files.

This script creates:
1. Population density map from result files enhanced with population data
2. Population vs spatial RMSE, MAE, bias comparison plots (r2 and r4)
3. Population vs LLM-ERA5 temperature difference comparison
4. Spatial RMSE r2 vs r4 comparison with population coloring
5. Comprehensive correlation analysis

Usage:
    python plot_population_map.py [results_file]

Default: results/climate_results_1.0deg_r10_simple_spatial_rmse_population.json
Output: png/{results_filename}/population_density_map_{resolution}deg.png
        png/{results_filename}/population_comparison_{resolution}deg_*.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
from pathlib import Path
import sys
import os
from scipy.stats import gaussian_kde

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_population_results(results_file):
    """Load population-enhanced results file"""
    print(f"Loading population results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check if population data is present
    has_population = results_data.get('metadata', {}).get('population_added', False)
    if not has_population:
        print("Warning: Population data not found in results file")
        print("Please run add_population_to_results.py first")
    
    return results_data


def extract_population_data(results_data):
    """Extract population density data from results"""
    population_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid population data
        if (point_info.get('is_land', False) and 
            'population_density' in point_info and
            not np.isnan(point_info.get('population_density', np.nan))):
            
            population_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'population_density': point_info['population_density'],
                'country': point_info.get('country', ''),
                'state': point_info.get('state', '')
            })
    
    print(f"Found {len(population_data)} land points with valid population data")
    return population_data


def extract_comparison_data_with_population(results_data):
    """Extract all comparison data including population and spatial metrics"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid population and climate data
        if (point_info.get('is_land', False) and 
            'population_density' in point_info and
            'llm_temp_mean' in point_info and
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('population_density', np.nan)) and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            data_point = {
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'population_density': point_info['population_density'],
                'llm_temp_mean': point_info['llm_temp_mean'],
                'era5_temp_mean': point_info['era5_temp_mean'],
                'temp_difference': point_info.get('temp_difference', 
                                                point_info['llm_temp_mean'] - point_info['era5_temp_mean']),
                'country': point_info.get('country', ''),
                'state': point_info.get('state', '')
            }
            
            # Add spatial metrics if available
            for radius in ['r2', 'r4']:
                for metric in ['rmse', 'mae', 'bias', 'correlation']:
                    field_name = f'spatial_{radius}_{metric}'
                    if field_name in point_info and not np.isnan(point_info.get(field_name, np.nan)):
                        data_point[field_name] = point_info[field_name]
            
            comparison_data.append(data_point)
    
    print(f"Found {len(comparison_data)} points with complete comparison data")
    return comparison_data


def create_population_grid(population_data, resolution=1.0):
    """Create a regular meshgrid with population data mapped to it"""
    
    if not population_data:
        return None, None, None
    
    # Define grid resolution based on data resolution
    if resolution >= 10.0:
        # For coarse resolution, use the actual resolution
        grid_res = resolution
        # Define bounds based on data
        lats = [p['lat'] for p in population_data]
        lons = [p['lon'] for p in population_data]
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
    
    # Initialize population grid with NaN
    pop_grid = np.full_like(lon_grid, np.nan, dtype=float)
    
    # Map population data to grid
    for data_point in population_data:
        lat = data_point['lat']
        lon = data_point['lon']
        pop_density = data_point['population_density']
        
        # Find grid indices
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        # Check if indices are valid and within tolerance
        tolerance = grid_res / 2
        if (np.abs(lats[lat_idx] - lat) <= tolerance and 
            np.abs(lons[lon_idx] - lon) <= tolerance):
            pop_grid[lat_idx, lon_idx] = pop_density
    
    print(f"Mapped {np.sum(~np.isnan(pop_grid))} points to {len(lats)}×{len(lons)} grid")
    
    return lon_grid, lat_grid, pop_grid


def create_population_colormap():
    """Create a colormap for population density (low to high density)"""
    # Population colormap: white/light -> yellow -> orange -> red -> dark red
    pop_colors = ['#ffffff', '#ffffcc', '#ffeda0', '#fed976', '#feb24c', 
                  '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
    n_bins = 256
    cmap = colors.LinearSegmentedColormap.from_list('population', pop_colors, N=n_bins)
    return cmap


def plot_population_map(population_data, resolution, output_file=None, figsize=(18, 10)):
    """Create population density map"""
    
    if not population_data:
        print("No population data available for mapping")
        return None
    
    # Create regular grid
    lon_grid, lat_grid, pop_grid = create_population_grid(population_data, resolution)
    
    if lon_grid is None:
        print("Failed to create population grid")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract values for statistics and color scaling
    pop_values = [d['population_density'] for d in population_data]
    
    # Use log scale for better visualization of population data
    # Add small value to avoid log(0) issues
    pop_values_log = np.log10(np.array(pop_values) + 1)
    pop_grid_log = np.log10(pop_grid + 1)
    
    vmin_log = np.min(pop_values_log)
    vmax_log = np.max(pop_values_log)
    
    # Create population colormap
    pop_cmap = create_population_colormap()
    
    # Create contour plot using log-scaled data
    levels = np.linspace(vmin_log, vmax_log, 20)
    contour = ax.contourf(lon_grid, lat_grid, pop_grid_log, 
                         levels=levels, cmap=pop_cmap, extend='max')
    
    # Try to load and overlay country boundaries
    try:
        world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
        world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
        print("Country boundaries overlaid successfully")
    except Exception as e:
        print(f"Could not load country boundaries: {e}")
        ax.grid(True, alpha=0.3)
    
    # Customize the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'Population Density Map (Year 2000)\n'
                f'Land Points: {len(population_data):,} | Resolution: {resolution}°', 
                fontsize=14, fontweight='bold')
    
    # Add coordinate ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    
    # Add colorbar with original scale labels
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Population Density (people/km²)', fontsize=12)
    
    # Create custom tick positions and labels with proper matching
    n_ticks = 8  # Use fewer ticks for cleaner appearance
    custom_positions = np.linspace(vmin_log, vmax_log, n_ticks)
    custom_labels = []
    
    for pos in custom_positions:
        # Convert back from log scale
        original_value = 10**pos - 1
        if original_value < 1:
            custom_labels.append('0')
        elif original_value < 10:
            custom_labels.append(f'{original_value:.1f}')
        elif original_value < 100:
            custom_labels.append(f'{original_value:.0f}')
        elif original_value < 1000:
            custom_labels.append(f'{original_value:.0f}')
        else:
            custom_labels.append(f'{original_value/1000:.1f}k')
    
    # Set both positions and labels
    cbar.set_ticks(custom_positions)
    cbar.set_ticklabels(custom_labels)
    
    # Add statistics text
    min_pop = np.min(pop_values)
    max_pop = np.max(pop_values)
    mean_pop = np.mean(pop_values)
    median_pop = np.median(pop_values)
    zero_pop = np.sum(np.array(pop_values) == 0)
    
    stats_text = f'Population Statistics:\n'
    stats_text += f'Mean: {mean_pop:.1f} people/km²\n'
    stats_text += f'Median: {median_pop:.1f} people/km²\n'
    stats_text += f'Range: {min_pop:.1f} - {max_pop:.0f} people/km²\n'
    stats_text += f'Zero population areas: {zero_pop}\n'
    stats_text += f'Log scale used for visualization\n'
    stats_text += f'Data source: GPW v4 (2000)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Population density map saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def print_population_statistics(population_data):
    """Print detailed population statistics"""
    
    if not population_data:
        print("No population data to analyze")
        return
    
    pop_values = [d['population_density'] for d in population_data]
    
    print(f"\nDetailed Population Statistics:")
    print(f"=" * 50)
    print(f"Total land points with population data: {len(pop_values)}")
    print(f"Mean population density: {np.mean(pop_values):.2f} people/km²")
    print(f"Median population density: {np.median(pop_values):.2f} people/km²")
    print(f"Standard deviation: {np.std(pop_values):.2f} people/km²")
    print(f"Minimum: {np.min(pop_values):.2f} people/km²")
    print(f"Maximum: {np.max(pop_values):.0f} people/km²")
    
    # Population density categories
    zero_pop = np.sum(np.array(pop_values) == 0)
    low_pop = np.sum((np.array(pop_values) > 0) & (np.array(pop_values) <= 10))
    medium_pop = np.sum((np.array(pop_values) > 10) & (np.array(pop_values) <= 100))
    high_pop = np.sum((np.array(pop_values) > 100) & (np.array(pop_values) <= 1000))
    very_high_pop = np.sum(np.array(pop_values) > 1000)
    
    print(f"\nPopulation Density Categories:")
    print(f"Zero (0 people/km²): {zero_pop} points ({100*zero_pop/len(pop_values):.1f}%)")
    print(f"Low (0-10 people/km²): {low_pop} points ({100*low_pop/len(pop_values):.1f}%)")
    print(f"Medium (10-100 people/km²): {medium_pop} points ({100*medium_pop/len(pop_values):.1f}%)")
    print(f"High (100-1000 people/km²): {high_pop} points ({100*high_pop/len(pop_values):.1f}%)")
    print(f"Very High (>1000 people/km²): {very_high_pop} points ({100*very_high_pop/len(pop_values):.1f}%)")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPopulation Density Percentiles:")
    for p in percentiles:
        value = np.percentile(pop_values, p)
        print(f"{p:2d}th percentile: {value:.1f} people/km²")


def create_density_scatter_with_population(x_vals, y_vals, pop_vals, x_label, y_label, title, output_file=None):
    """Create density scatter plot with population density coloring"""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([x_vals, y_vals])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
    except:
        density = np.ones_like(x_vals)
    
    # Sort points by density (plot low density first)
    idx = density.argsort()
    x_sorted, y_sorted, density_sorted = x_vals[idx], y_vals[idx], density[idx]
    
    # Create density scatter plot
    scatter = ax1.scatter(x_sorted, y_sorted, c=density_sorted, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
    # Add 1:1 line if comparing similar quantities
    if 'RMSE' in x_label and 'RMSE' in y_label:
        min_val = min(np.min(x_vals), np.min(y_vals))
        max_val = max(np.max(x_vals), np.max(y_vals))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, 
                 label='1:1 line')
    
    # Calculate and add regression line
    coeffs = np.polyfit(x_vals, y_vals, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(np.min(x_vals), np.max(x_vals), 100)
    ax1.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    ax1.set_title(f'{title}\nDensity Scatter Plot', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for density
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Point Density', fontsize=11)
    
    # Calculate statistics
    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
    rmse = np.sqrt(np.mean((y_vals - x_vals)**2)) if 'RMSE' in x_label and 'RMSE' in y_label else np.nan
    
    # Add statistics text
    stats_text = f'N = {len(x_vals)} points\n'
    stats_text += f'Correlation = {correlation:.3f}\n'
    if not np.isnan(rmse):
        stats_text += f'RMSE = {rmse:.3f}\n'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Right plot: Population density colored scatter ===
    
    # Use log scale for population
    pop_vals_log = np.log10(pop_vals + 1)
    
    scatter2 = ax2.scatter(x_vals, y_vals, c=pop_vals_log, s=30, alpha=0.7, 
                          cmap='YlOrRd', edgecolors='black', linewidth=0.3)
    
    # Add same reference lines as left plot
    if 'RMSE' in x_label and 'RMSE' in y_label:
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    ax2.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2)
    
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel(y_label, fontsize=12)
    ax2.set_title(f'{title}\nColored by Population Density', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for population
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('log₁₀(Population + 1)', fontsize=11)
    
    # Calculate correlation with population
    pop_corr_x = np.corrcoef(pop_vals_log, x_vals)[0, 1]
    pop_corr_y = np.corrcoef(pop_vals_log, y_vals)[0, 1]
    
    # Add population correlation text
    pop_stats_text = f'Population Correlations:\n'
    pop_stats_text += f'with X-axis: {pop_corr_x:.3f}\n'
    pop_stats_text += f'with Y-axis: {pop_corr_y:.3f}\n'
    
    ax2.text(0.02, 0.98, pop_stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (correlation, pop_corr_x, pop_corr_y)


def create_population_comparison_plots(comparison_data, resolution, output_dir='png'):
    """Create all population vs performance metric comparison plots"""
    
    print("\nCreating population comparison plots...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    created_plots = []
    correlations = {}
    
    # Base filename pattern
    base_name = f"population_comparison_{resolution}deg"
    
    # Population vs spatial RMSE comparisons
    for radius in ['r2', 'r4']:
        for metric in ['rmse', 'mae', 'bias']:
            field_name = f'spatial_{radius}_{metric}'
            
            if field_name not in comparison_data[0]:
                continue
            
            print(f"Creating plot for population vs {field_name}...")
            
            # Extract data
            pop_vals = np.array([d['population_density'] for d in comparison_data])
            metric_vals = np.array([d[field_name] for d in comparison_data if field_name in d])
            pop_vals_subset = np.array([d['population_density'] for d in comparison_data if field_name in d])
            
            if len(metric_vals) == 0:
                continue
            
            # Use log scale for population
            pop_vals_log = np.log10(pop_vals_subset + 1)
            
            # Create plot
            x_label = f'log₁₀(Population + 1)'
            y_label = f'{field_name.replace("_", " ").title()} (°C)'
            title = f'Population vs {field_name.replace("_", " ").title()}'
            
            output_file = output_path / f"{base_name}_{field_name}.png"
            
            _, (corr_main, _, _) = create_density_scatter_with_population(
                pop_vals_log, metric_vals, pop_vals_subset,
                x_label, y_label, title, str(output_file)
            )
            
            created_plots.append(str(output_file))
            correlations[f'pop_vs_{field_name}'] = corr_main
    
    # LLM-ERA5 absolute difference vs population
    print("Creating plot for LLM-ERA5 absolute difference vs population...")
    
    temp_diffs = np.array([abs(d['temp_difference']) for d in comparison_data])
    pop_vals = np.array([d['population_density'] for d in comparison_data])
    pop_vals_log = np.log10(pop_vals + 1)
    
    x_label = 'log₁₀(Population + 1)'
    y_label = 'Absolute Temperature Difference |LLM - ERA5| °C'
    title = 'Population vs Absolute LLM-ERA5 Temperature Difference'
    
    output_file = output_path / f"{base_name}_temp_difference.png"
    
    _, (corr_main, _, _) = create_density_scatter_with_population(
        pop_vals_log, temp_diffs, pop_vals,
        x_label, y_label, title, str(output_file)
    )
    
    created_plots.append(str(output_file))
    correlations['pop_vs_temp_diff'] = corr_main
    
    # Spatial RMSE r2 vs r4 comparison
    if 'spatial_r2_rmse' in comparison_data[0] and 'spatial_r4_rmse' in comparison_data[0]:
        print("Creating plot for spatial RMSE r2 vs r4...")
        
        # Get data for points that have both r2 and r4 RMSE
        valid_data = [d for d in comparison_data if 'spatial_r2_rmse' in d and 'spatial_r4_rmse' in d]
        r2_rmse = np.array([d['spatial_r2_rmse'] for d in valid_data])
        r4_rmse = np.array([d['spatial_r4_rmse'] for d in valid_data])
        pop_vals_subset = np.array([d['population_density'] for d in valid_data])
        
        if len(r2_rmse) > 0 and len(r4_rmse) > 0:
            x_label = 'Spatial RMSE r2 (5×5) °C'
            y_label = 'Spatial RMSE r4 (9×9) °C'
            title = 'Spatial RMSE: 5×5 vs 9×9 Neighborhood'
            
            output_file = output_path / f"{base_name}_rmse_r2_vs_r4.png"
            
            _, (corr_main, _, _) = create_density_scatter_with_population(
                r2_rmse, r4_rmse, pop_vals_subset,
                x_label, y_label, title, str(output_file)
            )
            
            created_plots.append(str(output_file))
            correlations['r2_rmse_vs_r4_rmse'] = corr_main
    
    return created_plots, correlations


def print_correlation_summary(correlations):
    """Print summary of all calculated correlations"""
    
    print(f"\nCorrelation Analysis Summary:")
    print(f"=" * 60)
    
    for metric_pair, correlation in correlations.items():
        metric_name = metric_pair.replace('_', ' ').replace('vs', 'vs').title()
        print(f"{metric_name:<40}: {correlation:+.3f}")
    
    # Group correlations by type
    pop_correlations = {k: v for k, v in correlations.items() if k.startswith('pop_vs_')}
    spatial_correlations = {k: v for k, v in correlations.items() if 'spatial' in k}
    
    if pop_correlations:
        print(f"\nPopulation vs Performance Metrics:")
        for metric, corr in pop_correlations.items():
            clean_name = metric.replace('pop_vs_spatial_', '').replace('pop_vs_', '').replace('_', ' ').title()
            print(f"  {clean_name:<30}: {corr:+.3f}")
    
    if spatial_correlations:
        print(f"\nSpatial Scale Comparisons:")
        for metric, corr in spatial_correlations.items():
            clean_name = metric.replace('_', ' ').title()
            print(f"  {clean_name:<30}: {corr:+.3f}")


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Population Density Mapping")
    print("=" * 50)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run add_population_to_results.py first to create population-enhanced data.")
        return
    
    try:
        # Load population results
        results_data = load_population_results(results_file)
        
        # Extract resolution from metadata
        resolution = results_data.get('resolution', 1.0)
        
        # Extract population data
        population_data = extract_population_data(results_data)
        
        if not population_data:
            print("No valid population data found for mapping.")
            return
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_file = str(output_dir / f"population_density_map_{resolution}deg.png")
        
        # Create population map
        fig = plot_population_map(population_data, resolution, output_file)
        
        if fig is not None:
            print(f"\nPopulation density map completed successfully!")
            
            # Print detailed statistics
            print_population_statistics(population_data)
        
        # Extract comparison data for additional plots
        comparison_data = extract_comparison_data_with_population(results_data)
        
        if comparison_data:
            # Create population comparison plots
            created_plots, correlations = create_population_comparison_plots(comparison_data, resolution, str(output_dir))
            
            if created_plots:
                print(f"\nCreated {len(created_plots)} comparison plots")
                
                # Print correlation summary
                print_correlation_summary(correlations)
            
            total_plots = 1 + len(created_plots)  # population map + comparison plots
            print(f"\nAll plotting completed successfully!")
            print(f"Total plots created: {total_plots}")
        else:
            print(f"\nPopulation density map completed successfully!")
        
    except Exception as e:
        print(f"Error during mapping: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()