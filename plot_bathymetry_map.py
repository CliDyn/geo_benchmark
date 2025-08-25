#!/usr/bin/env python3
"""
Create bathymetry (elevation and roughness) maps and comparison plots from bathymetry-enhanced result files.

This script generates:
1. Mean elevation map with log-scale visualization
2. Terrain roughness map with log-scale visualization
3. Comparison plots: spatial RMSE r2, MAE r2 vs mean elevation and roughness
4. Temperature difference (LLM-ERA5) vs mean elevation and roughness

Usage:
    python plot_bathymetry_map.py [results_file]

Default file: climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, Normalize
import json
from pathlib import Path
import sys
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

def load_bathymetry_results(results_file):
    """Load results file with bathymetry data"""
    print(f"Loading bathymetry results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check what data is present
    has_era5 = results_data.get('metadata', {}).get('era5_climatology_added', False)
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False)
    has_population = results_data.get('metadata', {}).get('population_added', False)
    has_bathymetry = results_data.get('metadata', {}).get('bathymetry_added', False)
    
    print(f"Data present - ERA5: {has_era5}, Spatial RMSE: {has_spatial}, Population: {has_population}, Bathymetry: {has_bathymetry}")
    
    if not has_bathymetry:
        print("Warning: No bathymetry data found in results file")
    
    return results_data

def extract_bathymetry_data(results_data):
    """Extract bathymetry and analysis data from results"""
    
    data_points = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Basic coordinates
        lat = point_info['lat']
        lon = point_info['lon']
        
        # Bathymetry parameters
        mean_elevation = point_info.get('mean_elevation')
        roughness = point_info.get('roughness')
        
        # Spatial RMSE data (r2 only as requested)
        spatial_r2_rmse = point_info.get('spatial_r2_rmse')
        spatial_r2_mae = point_info.get('spatial_r2_mae')
        
        # Temperature difference (LLM - ERA5)
        temp_difference = point_info.get('temp_difference')
        
        # Only include points with valid bathymetry data
        if (mean_elevation is not None and roughness is not None and 
            spatial_r2_rmse is not None and spatial_r2_mae is not None and
            temp_difference is not None):
            
            data_points.append({
                'lat': lat,
                'lon': lon,
                'mean_elevation': mean_elevation,
                'roughness': roughness,
                'spatial_r2_rmse': spatial_r2_rmse,
                'spatial_r2_mae': spatial_r2_mae,
                'temp_difference': temp_difference,
                'abs_temp_difference': abs(temp_difference)
            })
    
    df = pd.DataFrame(data_points)
    print(f"Extracted {len(df)} points with complete bathymetry and spatial data")
    
    return df

def print_bathymetry_statistics(df):
    """Print comprehensive bathymetry statistics"""
    
    print(f"\nBathymetry Data Statistics:")
    print(f"=" * 60)
    print(f"Total points: {len(df)}")
    
    # Elevation statistics
    elevation_stats = df['mean_elevation'].describe()
    print(f"\nMean Elevation (m):")
    print(f"  Count: {elevation_stats['count']:.0f}")
    print(f"  Mean: {elevation_stats['mean']:.1f}")
    print(f"  Median: {elevation_stats['50%']:.1f}")
    print(f"  Range: {elevation_stats['min']:.1f} to {elevation_stats['max']:.1f}")
    print(f"  Std: {elevation_stats['std']:.1f}")
    
    # Roughness statistics
    roughness_stats = df['roughness'].describe()
    print(f"\nTerrain Roughness (m):")
    print(f"  Count: {roughness_stats['count']:.0f}")
    print(f"  Mean: {roughness_stats['mean']:.1f}")
    print(f"  Median: {roughness_stats['50%']:.1f}")
    print(f"  Range: {roughness_stats['min']:.1f} to {roughness_stats['max']:.1f}")
    print(f"  Std: {roughness_stats['std']:.1f}")
    
    # Elevation ranges
    below_sea = len(df[df['mean_elevation'] < 0])
    low_elevation = len(df[(df['mean_elevation'] >= 0) & (df['mean_elevation'] < 500)])
    mid_elevation = len(df[(df['mean_elevation'] >= 500) & (df['mean_elevation'] < 1500)])
    high_elevation = len(df[df['mean_elevation'] >= 1500])
    
    print(f"\nElevation Categories:")
    print(f"  Below sea level: {below_sea} ({100*below_sea/len(df):.1f}%)")
    print(f"  0-500m: {low_elevation} ({100*low_elevation/len(df):.1f}%)")
    print(f"  500-1500m: {mid_elevation} ({100*mid_elevation/len(df):.1f}%)")
    print(f"  Above 1500m: {high_elevation} ({100*high_elevation/len(df):.1f}%)")

def create_bathymetry_grid(bathymetry_data, field_name, resolution=1.0):
    """Create a regular meshgrid with bathymetry data mapped to it"""
    
    if len(bathymetry_data) == 0:
        return None, None, None
    
    # Define grid resolution based on data resolution
    if resolution >= 10.0:
        # For coarse resolution, use the actual resolution
        grid_res = resolution
        # Define bounds based on data
        lats = bathymetry_data['lat'].values
        lons = bathymetry_data['lon'].values
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
    else:
        # For fine resolution, use 1 degree grid
        grid_res = 1.0
        lat_min, lat_max = -60, 84
        lon_min, lon_max = -180, 179
    
    # Create coordinate arrays
    grid_lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    grid_lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
    
    # Initialize data grid with NaN
    data_grid = np.full_like(lon_grid, np.nan, dtype=float)
    
    # Map bathymetry data to grid
    for idx, row in bathymetry_data.iterrows():
        lat = row['lat']
        lon = row['lon']
        value = row[field_name]
        
        # Find grid indices
        lat_idx = np.argmin(np.abs(grid_lats - lat))
        lon_idx = np.argmin(np.abs(grid_lons - lon))
        
        # Check if indices are valid and within tolerance
        tolerance = grid_res / 2
        if (np.abs(grid_lats[lat_idx] - lat) <= tolerance and 
            np.abs(grid_lons[lon_idx] - lon) <= tolerance):
            data_grid[lat_idx, lon_idx] = value
    
    print(f"Mapped {np.sum(~np.isnan(data_grid))} points to {len(grid_lats)}×{len(grid_lons)} grid for {field_name}")
    
    return lon_grid, lat_grid, data_grid

def create_bathymetry_map(df, resolution, output_dir='png'):
    """Create mean elevation map using contour plot"""
    
    print("Creating mean elevation map...")
    
    # Create regular grid
    lon_grid, lat_grid, elevation_grid = create_bathymetry_grid(df, 'mean_elevation', resolution)
    
    if lon_grid is None:
        print("Failed to create elevation grid")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # Get elevation values for color scaling
    elevation_values = df['mean_elevation'].values
    
    # Create contour plot
    levels = np.linspace(np.min(elevation_values), np.max(elevation_values), 20)
    contour = ax.contourf(lon_grid, lat_grid, elevation_grid, 
                         levels=levels, cmap='terrain', extend='both')
    
    # Try to overlay country boundaries
    try:
        import geopandas as gpd
        world = gpd.read_file('./data/land/ne_10m_land.shp')
        world.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.3)
    except:
        print("Country boundaries not available")
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Elevation (m)', fontsize=12)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Mean Elevation Map ({resolution}° resolution)', fontsize=14, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f'mean_elevation_map_{resolution}deg.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Mean elevation map saved to: {output_file}")
    plt.close()

def create_roughness_map(df, resolution, output_dir='png'):
    """Create terrain roughness map using contour plot"""
    
    print("Creating terrain roughness map...")
    
    # Create regular grid
    lon_grid, lat_grid, roughness_grid = create_bathymetry_grid(df, 'roughness', resolution)
    
    if lon_grid is None:
        print("Failed to create roughness grid")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    
    # Get roughness values for color scaling
    roughness_values = df['roughness'].values
    
    # Use log scale for better visualization of roughness data
    # Add small value to avoid log(0) issues
    roughness_values_log = np.log10(roughness_values + 1)
    roughness_grid_log = np.log10(roughness_grid + 1)
    
    vmin_log = np.min(roughness_values_log)
    vmax_log = np.max(roughness_values_log)
    
    # Create contour plot using log-scaled data
    levels = np.linspace(vmin_log, vmax_log, 20)
    contour = ax.contourf(lon_grid, lat_grid, roughness_grid_log, 
                         levels=levels, cmap='plasma', extend='max')
    
    # Try to overlay country boundaries
    try:
        import geopandas as gpd
        world = gpd.read_file('./data/land/ne_10m_land.shp')
        world.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.3)
    except:
        print("Country boundaries not available")
    
    # Colorbar with original roughness values
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
    
    # Create custom colorbar labels showing original roughness values
    tick_positions_log = np.linspace(vmin_log, vmax_log, 8)
    tick_labels_original = 10**tick_positions_log - 1
    cbar.set_ticks(tick_positions_log)
    cbar.set_ticklabels([f'{val:.0f}' for val in tick_labels_original])
    cbar.set_label('Terrain Roughness (m)', fontsize=12)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Terrain Roughness Map ({resolution}° resolution)', fontsize=14, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f'terrain_roughness_map_{resolution}deg.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Terrain roughness map saved to: {output_file}")
    plt.close()

def create_density_scatter_with_bathymetry(df, x_col, y_col, x_label, y_label, title, output_file, 
                                         use_log_x=False, use_log_y=False, use_abs_y=False):
    """Create density scatter plot with bathymetry parameter coloring"""
    
    # Prepare data
    x_data = df[x_col].copy()
    y_data = df[y_col].copy()
    
    if use_abs_y:
        y_data = np.abs(y_data)
        y_label = f"Absolute {y_label}"
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    
    if len(x_data) == 0:
        print(f"Warning: No valid data for {title}")
        return
    
    # Handle log scale for x if requested
    if use_log_x and (x_data <= 0).any():
        min_positive = x_data[x_data > 0].min() if (x_data > 0).any() else 1
        x_data[x_data <= 0] = min_positive
    
    print(f"Creating density scatter plot: {title}")
    print(f"  Data points: {len(x_data)}")
    print(f"  X range: {x_data.min():.2f} to {x_data.max():.2f}")
    print(f"  Y range: {y_data.min():.2f} to {y_data.max():.2f}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate point density using KDE
    try:
        xy = np.vstack([x_data, y_data])
        kde = gaussian_kde(xy)
        density = kde(xy)
    except Exception as e:
        print(f"Warning: Could not calculate KDE density: {e}")
        density = np.ones(len(x_data))
    
    # Create scatter plot colored by density
    scatter = ax.scatter(x_data, y_data, c=density, cmap='viridis', s=15, alpha=0.7)
    
    # Add colorbar for density
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Point Density', fontsize=10)
    
    # Calculate and plot regression line
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        line_x = np.linspace(x_data.min(), x_data.max(), 100)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'red', linewidth=2, alpha=0.8, 
                label=f'R² = {r_value**2:.3f}, y = {slope:.3f}x + {intercept:.3f}')
        ax.legend(fontsize=10)
    except Exception as e:
        print(f"Warning: Could not calculate regression: {e}")
    
    # Set scales
    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')
    
    # Labels and formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Density scatter plot saved to: {output_file}")
    plt.close()
    
    # Print correlation statistics
    corr = np.corrcoef(x_data, y_data)[0, 1]
    print(f"  Correlation: {corr:.3f}")

def create_bathymetry_comparison_plots(df, resolution, output_dir='png'):
    """Create comprehensive bathymetry comparison plots"""
    
    print("\nCreating bathymetry comparison plots...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Spatial R2 RMSE vs Mean Elevation
    create_density_scatter_with_bathymetry(
        df, 'mean_elevation', 'spatial_r2_rmse',
        'Mean Elevation (m)', 'Spatial RMSE r2 (°C)',
        f'Spatial RMSE r2 vs Mean Elevation ({resolution}° resolution)',
        Path(output_dir) / f'bathymetry_comparison_{resolution}deg_rmse_vs_elevation.png'
    )
    
    # 2. Spatial R2 RMSE vs Terrain Roughness  
    create_density_scatter_with_bathymetry(
        df, 'roughness', 'spatial_r2_rmse', 
        'Terrain Roughness (m)', 'Spatial RMSE r2 (°C)',
        f'Spatial RMSE r2 vs Terrain Roughness ({resolution}° resolution)',
        Path(output_dir) / f'bathymetry_comparison_{resolution}deg_rmse_vs_roughness.png',
        use_log_x=True
    )
    
    # 3. Spatial R2 MAE vs Mean Elevation
    create_density_scatter_with_bathymetry(
        df, 'mean_elevation', 'spatial_r2_mae',
        'Mean Elevation (m)', 'Spatial MAE r2 (°C)', 
        f'Spatial MAE r2 vs Mean Elevation ({resolution}° resolution)',
        Path(output_dir) / f'bathymetry_comparison_{resolution}deg_mae_vs_elevation.png'
    )
    
    # 4. Spatial R2 MAE vs Terrain Roughness
    create_density_scatter_with_bathymetry(
        df, 'roughness', 'spatial_r2_mae',
        'Terrain Roughness (m)', 'Spatial MAE r2 (°C)',
        f'Spatial MAE r2 vs Terrain Roughness ({resolution}° resolution)', 
        Path(output_dir) / f'bathymetry_comparison_{resolution}deg_mae_vs_roughness.png',
        use_log_x=True
    )
    
    # 5. Temperature Difference vs Mean Elevation
    create_density_scatter_with_bathymetry(
        df, 'mean_elevation', 'abs_temp_difference',
        'Mean Elevation (m)', 'LLM-ERA5 Temperature Difference (°C)',
        f'Temperature Difference vs Mean Elevation ({resolution}° resolution)',
        Path(output_dir) / f'bathymetry_comparison_{resolution}deg_temp_vs_elevation.png',
        use_abs_y=False
    )
    
    # 6. Temperature Difference vs Terrain Roughness
    create_density_scatter_with_bathymetry(
        df, 'roughness', 'abs_temp_difference', 
        'Terrain Roughness (m)', 'LLM-ERA5 Temperature Difference (°C)',
        f'Temperature Difference vs Terrain Roughness ({resolution}° resolution)',
        Path(output_dir) / f'bathymetry_comparison_{resolution}deg_temp_vs_roughness.png',
        use_log_x=True, use_abs_y=False
    )

def print_correlation_summary(df):
    """Print correlation summary between bathymetry and performance metrics"""
    
    print(f"\nCorrelation Summary:")
    print(f"=" * 50)
    
    # Correlations with elevation
    corr_elev_rmse = np.corrcoef(df['mean_elevation'], df['spatial_r2_rmse'])[0, 1]
    corr_elev_mae = np.corrcoef(df['mean_elevation'], df['spatial_r2_mae'])[0, 1]
    corr_elev_temp = np.corrcoef(df['mean_elevation'], df['abs_temp_difference'])[0, 1]
    
    print(f"\nMean Elevation correlations:")
    print(f"  vs Spatial RMSE r2: {corr_elev_rmse:.3f}")
    print(f"  vs Spatial MAE r2: {corr_elev_mae:.3f}")
    print(f"  vs Abs Temp Diff: {corr_elev_temp:.3f}")
    
    # Correlations with roughness
    corr_rough_rmse = np.corrcoef(df['roughness'], df['spatial_r2_rmse'])[0, 1]
    corr_rough_mae = np.corrcoef(df['roughness'], df['spatial_r2_mae'])[0, 1]
    corr_rough_temp = np.corrcoef(df['roughness'], df['abs_temp_difference'])[0, 1]
    
    print(f"\nTerrain Roughness correlations:")
    print(f"  vs Spatial RMSE r2: {corr_rough_rmse:.3f}")
    print(f"  vs Spatial MAE r2: {corr_rough_mae:.3f}")
    print(f"  vs Abs Temp Diff: {corr_rough_temp:.3f}")

def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Bathymetry Analysis and Visualization")
    print("=" * 60)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # Load data
        results_data = load_bathymetry_results(results_file)
        
        # Extract data
        df = extract_bathymetry_data(results_data)
        
        if len(df) == 0:
            print("Error: No valid data points found with complete bathymetry and spatial data")
            return
        
        # Print statistics
        print_bathymetry_statistics(df)
        
        # Get resolution from results data
        resolution = results_data.get('resolution', 1.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create maps
        create_bathymetry_map(df, resolution, str(output_dir))
        create_roughness_map(df, resolution, str(output_dir))
        
        # Create comparison plots
        create_bathymetry_comparison_plots(df, resolution, str(output_dir))
        
        # Print correlation summary
        print_correlation_summary(df)
        
        print(f"\nBathymetry analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()