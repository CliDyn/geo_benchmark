#!/usr/bin/env python3
"""
Point-wise RMSE Analysis: Calculate RMSE between LLM realizations and ERA5 for each individual point.

This script takes ERA5-enhanced result files and calculates RMSE for each point using
all individual LLM realizations compared to the ERA5 value for that point.

Usage:
    python point_rmse_analysis.py [results_file]

Default: results/climate_results_20.0deg_r10_simple_era5.json
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
import sys
import geopandas as gpd


def load_era5_enhanced_results(results_file):
    """Load ERA5-enhanced results file"""
    print(f"Loading ERA5-enhanced results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    return results_data


def extract_llm_temperatures_for_point(result):
    """Extract all individual LLM temperature realizations for a single point"""
    temps = []
    
    for response in result.get('llm_responses', []):
        if response and 'parsed_data' in response:
            parsed = response['parsed_data']
            # Find temperature key (e.g., 'july_temp_mean', 'january_temp_mean')
            temp_key = next((k for k in parsed.keys() if k.endswith('_temp_mean')), None)
            if temp_key:
                temps.append(parsed[temp_key])
    
    return temps


def calculate_point_rmse(llm_temps, era5_temp):
    """Calculate RMSE between LLM realizations and ERA5 for a single point"""
    if not llm_temps or np.isnan(era5_temp):
        return np.nan, np.nan, np.nan, 0
    
    llm_array = np.array(llm_temps)
    era5_array = np.full_like(llm_array, era5_temp)  # ERA5 value repeated for each realization
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((llm_array - era5_array)**2))
    mae = np.mean(np.abs(llm_array - era5_array))
    bias = np.mean(llm_array - era5_array)
    n_realizations = len(llm_temps)
    
    return rmse, mae, bias, n_realizations


def analyze_point_rmse(results_data):
    """Analyze point-wise RMSE for all points in the results"""
    print("Calculating point-wise RMSE for each location...")
    
    point_rmse_data = []
    
    for i, result in enumerate(results_data['results']):
        point_info = result['point_info']
        lat, lon = point_info['lat'], point_info['lon']
        
        # Skip if not land or missing ERA5 data
        if not point_info.get('is_land', False):
            continue
            
        era5_temp = point_info.get('era5_temp_mean', np.nan)
        if np.isnan(era5_temp):
            continue
        
        # Extract all LLM temperature realizations
        llm_temps = extract_llm_temperatures_for_point(result)
        
        # Calculate RMSE for this point
        rmse, mae, bias, n_realizations = calculate_point_rmse(llm_temps, era5_temp)
        
        if not np.isnan(rmse) and n_realizations > 0:
            point_rmse_data.append({
                'lat': lat,
                'lon': lon,
                'country': point_info.get('country', ''),
                'state': point_info.get('state', ''),
                'era5_temp': era5_temp,
                'llm_temps': llm_temps,
                'llm_mean': np.mean(llm_temps),
                'llm_std': np.std(llm_temps) if len(llm_temps) > 1 else 0.0,
                'point_rmse': rmse,
                'point_mae': mae,
                'point_bias': bias,
                'n_realizations': n_realizations
            })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(results_data['results'])} points")
    
    print(f"Calculated point-wise RMSE for {len(point_rmse_data)} land points")
    return point_rmse_data


def create_rmse_grid(point_rmse_data, resolution=20.0):
    """Create a regular meshgrid with RMSE data mapped to it"""
    
    # Define grid resolution based on input data resolution
    if resolution >= 20.0:
        # For coarse resolution (20deg), use the actual resolution
        grid_res = resolution
        # Define bounds based on data
        lats = [p['lat'] for p in point_rmse_data]
        lons = [p['lon'] for p in point_rmse_data]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
    else:
        # For fine resolution, use 1 degree grid like spatial_rmse_analysis
        grid_res = 1.0
        lat_min, lat_max = -60, 84
        lon_min, lon_max = -180, 179
    
    # Create coordinate arrays
    lats = np.arange(lat_min, lat_max + grid_res, grid_res)
    lons = np.arange(lon_min, lon_max + grid_res, grid_res)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Initialize RMSE grid with NaN
    rmse_grid = np.full_like(lon_grid, np.nan, dtype=float)
    
    # Map RMSE data to grid
    for data_point in point_rmse_data:
        lat = data_point['lat']
        lon = data_point['lon']
        rmse = data_point['point_rmse']
        
        # Find grid indices
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        # Check if indices are valid and within tolerance
        tolerance = grid_res / 2
        if (np.abs(lats[lat_idx] - lat) <= tolerance and 
            np.abs(lons[lon_idx] - lon) <= tolerance):
            rmse_grid[lat_idx, lon_idx] = rmse
    
    print(f"Mapped {np.sum(~np.isnan(rmse_grid))} points to {len(lats)}×{len(lons)} grid")
    
    return lon_grid, lat_grid, rmse_grid


def plot_point_rmse_map(point_rmse_data, resolution=20.0, output_file=None, figsize=(18, 10)):
    """Create a contour map showing point-wise RMSE with country boundaries"""
    
    if not point_rmse_data:
        print("No point RMSE data to plot")
        return
    
    # Create regular grid
    lon_grid, lat_grid, rmse_grid = create_rmse_grid(point_rmse_data, resolution)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define specific contour levels with emphasis on 0-3°C range
    rmse_values = [p['point_rmse'] for p in point_rmse_data]
    vmax = np.max(rmse_values)
    
    # Use detailed levels: 0.5°C steps up to 5°C, then bigger steps
    levels_fine = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    levels_coarse = [7, 9, 12, 15, 20, 30]
    levels = levels_fine + levels_coarse
    # Only use levels up to the maximum value in data
    levels = [l for l in levels if l <= vmax + 1]  # +1 to include max
    
    # Create custom 3-color colormap: blue (low) -> yellow (medium) -> red (high)
    from matplotlib.colors import ListedColormap
    colors = ['#0066CC', '#0099FF', '#00CCFF', '#66FFCC', '#CCFF66', '#FFCC00', '#FF9900', '#FF6600', '#FF3300', '#CC0000', '#990000']
    n_colors = len(levels) - 1
    if n_colors > len(colors):
        # If we need more colors, use the predefined RdYlBu_r colormap
        cmap = plt.cm.get_cmap('RdYlBu_r')
        colors = [cmap(i/n_colors) for i in range(n_colors)]
    else:
        colors = colors[:n_colors]
    
    custom_cmap = ListedColormap(colors)
    
    # Create contour plot with custom levels and colormap
    contour = ax.contourf(lon_grid, lat_grid, rmse_grid, 
                         levels=levels, cmap=custom_cmap, extend='max')
    
    # Try to load and overlay country boundaries
    try:
        # Load world countries shapefile
        world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
        world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
        print("Country boundaries overlaid successfully")
    except Exception as e:
        print(f"Could not load country boundaries: {e}")
        # Add basic grid instead
        ax.grid(True, alpha=0.3)
    
    # Customize the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    
    # Add title with information
    n_realizations = [p['n_realizations'] for p in point_rmse_data]
    total_realizations = sum(n_realizations)
    avg_realizations = np.mean(n_realizations)
    ax.set_title(f'Point-wise RMSE: LLM Realizations vs ERA5 on Land\n'
                f'Points: {len(point_rmse_data):,} | Total realizations: {total_realizations:,} | Avg: {avg_realizations:.1f} per point', 
                fontsize=14, fontweight='bold')
    
    # Add coordinate ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    
    # Add colorbar with same normalization
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('RMSE (°C)', fontsize=12)
    
    # Add statistics text
    mean_rmse = np.mean(rmse_values)
    median_rmse = np.median(rmse_values)
    min_rmse = np.min(rmse_values)
    max_rmse = np.max(rmse_values)
    std_rmse = np.std(rmse_values)
    
    stats_text = f'Point RMSE Statistics:\n'
    stats_text += f'Mean: {mean_rmse:.2f}°C\n'
    stats_text += f'Median: {median_rmse:.2f}°C\n'
    stats_text += f'Std: {std_rmse:.2f}°C\n'
    stats_text += f'Range: {min_rmse:.2f} - {max_rmse:.2f}°C\n'
    stats_text += f'Individual point RMSE\n'
    stats_text += f'Grid resolution: {resolution}°'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Point RMSE contour map saved to: {output_file}")
    else:
        plt.show()
    
    return fig


def save_point_rmse_results(point_rmse_data, output_file):
    """Save point RMSE analysis results to JSON and CSV"""
    
    # Prepare data for saving (convert numpy arrays to lists)
    save_data = []
    for point in point_rmse_data:
        save_point = point.copy()
        save_point['llm_temps'] = [float(t) for t in point['llm_temps']]  # Convert to list
        save_data.append(save_point)
    
    # Save as JSON
    json_file = output_file.replace('.csv', '.json')
    with open(json_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Point RMSE data saved to: {json_file}")
    
    # Save as CSV for easier analysis (without the temperature lists)
    csv_data = []
    for point in point_rmse_data:
        csv_point = {k: v for k, v in point.items() if k != 'llm_temps'}  # Exclude temperature arrays
        csv_data.append(csv_point)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"Point RMSE summary saved to: {output_file}")
    
    return df


def print_detailed_summary(point_rmse_data):
    """Print detailed summary statistics"""
    if not point_rmse_data:
        print("No data to summarize")
        return
    
    rmse_values = [p['point_rmse'] for p in point_rmse_data]
    mae_values = [p['point_mae'] for p in point_rmse_data]
    bias_values = [p['point_bias'] for p in point_rmse_data]
    n_realizations = [p['n_realizations'] for p in point_rmse_data]
    
    print(f"\nDetailed Point-wise RMSE Analysis Summary:")
    print(f"=" * 60)
    print(f"Total land points analyzed: {len(point_rmse_data):,}")
    print(f"Total LLM realizations: {sum(n_realizations):,}")
    print(f"Average realizations per point: {np.mean(n_realizations):.1f}")
    print(f"Realization range: {min(n_realizations)} - {max(n_realizations)} per point")
    
    print(f"\nRMSE Statistics:")
    print(f"  Mean: {np.mean(rmse_values):.3f}°C")
    print(f"  Median: {np.median(rmse_values):.3f}°C")
    print(f"  Std Dev: {np.std(rmse_values):.3f}°C")
    print(f"  Range: {np.min(rmse_values):.3f} - {np.max(rmse_values):.3f}°C")
    
    print(f"\nMAE Statistics:")
    print(f"  Mean: {np.mean(mae_values):.3f}°C")
    print(f"  Median: {np.median(mae_values):.3f}°C")
    
    print(f"\nBias Statistics:")
    print(f"  Mean: {np.mean(bias_values):+.3f}°C")
    print(f"  Median: {np.median(bias_values):+.3f}°C")
    
    # Find points with highest and lowest RMSE
    max_rmse_idx = np.argmax(rmse_values)
    min_rmse_idx = np.argmin(rmse_values)
    
    worst_point = point_rmse_data[max_rmse_idx]
    best_point = point_rmse_data[min_rmse_idx]
    
    print(f"\nHighest RMSE point:")
    print(f"  Location: ({worst_point['lat']:.1f}, {worst_point['lon']:.1f}) - {worst_point['country']}")
    print(f"  RMSE: {worst_point['point_rmse']:.2f}°C | ERA5: {worst_point['era5_temp']:.1f}°C | LLM mean: {worst_point['llm_mean']:.1f}°C")
    print(f"  Realizations: {worst_point['n_realizations']}")
    
    print(f"\nLowest RMSE point:")
    print(f"  Location: ({best_point['lat']:.1f}, {best_point['lon']:.1f}) - {best_point['country']}")
    print(f"  RMSE: {best_point['point_rmse']:.2f}°C | ERA5: {best_point['era5_temp']:.1f}°C | LLM mean: {best_point['llm_mean']:.1f}°C")
    print(f"  Realizations: {best_point['n_realizations']}")


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_20.0deg_r10_simple_era5.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Point-wise RMSE Analysis")
    print("=" * 50)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run add_era5_to_results.py first to create ERA5-enhanced results.")
        return
    
    try:
        # Load ERA5-enhanced results
        results_data = load_era5_enhanced_results(results_file)
        
        # Extract resolution from metadata
        resolution = results_data.get('resolution', 20.0)
        
        # Perform point-wise RMSE analysis
        point_rmse_data = analyze_point_rmse(results_data)
        
        if not point_rmse_data:
            print("No valid point RMSE data generated.")
            return
        
        # Generate output filenames
        results_path = Path(results_file)
        output_base = f"point_rmse_{results_path.stem.replace('_era5', '')}"
        csv_output = f"results/{output_base}.csv"
        plot_output = f"png/{output_base}_map.png"
        
        # Save results
        save_point_rmse_results(point_rmse_data, csv_output)
        
        # Create visualization
        plot_point_rmse_map(point_rmse_data, resolution, plot_output)
        
        # Print detailed summary
        print_detailed_summary(point_rmse_data)
        
        print(f"\nPoint-wise RMSE analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()