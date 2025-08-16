import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
import sys
import os
import geopandas as gpd

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_mesh_and_results(mesh_file, results_file):
    """Load mesh coordinates and LLM/ERA5 results"""
    print(f"Loading mesh from {mesh_file}...")
    with open(mesh_file, 'r') as f:
        mesh_data = json.load(f)
    
    print(f"Loading results from {results_file}...")
    # Try to load comparison results first
    try:
        from analyze_country_performance import load_llm_results_and_add_era5
        comparison_results, _ = load_llm_results_and_add_era5(results_file)
        
        if comparison_results is None:
            print("Failed to load comparison results")
            return None, None
        
        print(f"Loaded {len(comparison_results)} data points")
        return mesh_data, comparison_results
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None

def create_coordinate_grid(mesh_data):
    """Create a coordinate grid from mesh data for spatial operations"""
    mesh_points = mesh_data['mesh_points']
    lats = [point['lat'] for point in mesh_points]
    lons = [point['lon'] for point in mesh_points]
    
    # Get unique latitudes and longitudes
    unique_lats = sorted(list(set(lats)))
    unique_lons = sorted(list(set(lons)))
    
    print(f"Grid dimensions: {len(unique_lats)} latitudes × {len(unique_lons)} longitudes")
    print(f"Latitude range: {min(unique_lats):.1f} to {max(unique_lats):.1f}")
    print(f"Longitude range: {min(unique_lons):.1f} to {max(unique_lons):.1f}")
    
    # Calculate grid spacing
    lat_spacing = unique_lats[1] - unique_lats[0] if len(unique_lats) > 1 else 1.0
    lon_spacing = unique_lons[1] - unique_lons[0] if len(unique_lons) > 1 else 1.0
    
    print(f"Grid spacing: {lat_spacing:.1f}° latitude, {lon_spacing:.1f}° longitude")
    
    return unique_lats, unique_lons, lat_spacing, lon_spacing

def create_results_lookup(comparison_results):
    """Create a fast lookup dictionary for results by coordinates - land points only"""
    results_dict = {}
    
    for result in comparison_results:
        # Only include land points with valid LLM and ERA5 data
        if (not np.isnan(result.get('llm_temp_mean', np.nan)) and 
            not np.isnan(result.get('era5_temp_mean', np.nan)) and
            result.get('country') and result.get('country') != 'N/A'):
            
            lat = result['lat']
            lon = result['lon']
            key = (lat, lon)
            
            results_dict[key] = {
                'llm_temp': result['llm_temp_mean'],
                'era5_temp': result['era5_temp_mean'],
                'temp_diff': result.get('temp_difference', 
                                      result['llm_temp_mean'] - result['era5_temp_mean'])
            }
    
    print(f"Created lookup table for {len(results_dict)} valid land data points")
    return results_dict

def get_neighborhood_coordinates(center_lat, center_lon, unique_lats, unique_lons, radius=4):
    """Get coordinates of neighboring points in a radius around center point"""
    neighbors = []
    
    # Find center indices
    try:
        center_lat_idx = unique_lats.index(center_lat)
        center_lon_idx = unique_lons.index(center_lon)
    except ValueError:
        return neighbors  # Center point not in grid
    
    # Generate neighbor coordinates
    for lat_offset in range(-radius, radius + 1):
        for lon_offset in range(-radius, radius + 1):
            # Calculate neighbor indices
            neighbor_lat_idx = center_lat_idx + lat_offset
            neighbor_lon_idx = center_lon_idx + lon_offset
            
            # Check latitude bounds (no wraparound for latitude)
            if neighbor_lat_idx < 0 or neighbor_lat_idx >= len(unique_lats):
                continue
            
            # Handle longitude wraparound
            neighbor_lon_idx = neighbor_lon_idx % len(unique_lons)
            
            # Get actual coordinates
            neighbor_lat = unique_lats[neighbor_lat_idx]
            neighbor_lon = unique_lons[neighbor_lon_idx]
            
            neighbors.append((neighbor_lat, neighbor_lon))
    
    return neighbors

def calculate_spatial_rmse(center_lat, center_lon, results_dict, unique_lats, unique_lons, radius=4):
    """Calculate RMSE for a spatial neighborhood around a center point"""
    
    # Get neighborhood coordinates
    neighbors = get_neighborhood_coordinates(
        center_lat, center_lon, unique_lats, unique_lons, radius
    )
    
    # Collect valid neighbor data
    llm_temps = []
    era5_temps = []
    
    for neighbor_lat, neighbor_lon in neighbors:
        key = (neighbor_lat, neighbor_lon)
        if key in results_dict:
            data = results_dict[key]
            llm_temps.append(data['llm_temp'])
            era5_temps.append(data['era5_temp'])
    
    # Calculate RMSE if we have enough neighbors
    if len(llm_temps) >= 5:  # Require at least 5 valid neighbors
        llm_array = np.array(llm_temps)
        era5_array = np.array(era5_temps)
        rmse = np.sqrt(np.mean((llm_array - era5_array)**2))
        mae = np.mean(np.abs(llm_array - era5_array))
        bias = np.mean(llm_array - era5_array)
        correlation = np.corrcoef(llm_array, era5_array)[0, 1] if len(llm_temps) > 1 else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'correlation': correlation,
            'n_neighbors': len(llm_temps),
            'llm_mean': np.mean(llm_array),
            'era5_mean': np.mean(era5_array)
        }
    else:
        return None

def analyze_spatial_rmse(mesh_data, comparison_results, radius=4):
    """Analyze spatial RMSE patterns across the entire grid"""
    
    # Create coordinate grid
    unique_lats, unique_lons, _, _ = create_coordinate_grid(mesh_data)
    
    # Create results lookup
    results_dict = create_results_lookup(comparison_results)
    
    # Analyze each grid point
    spatial_rmse_data = []
    
    print(f"Calculating spatial RMSE with radius={radius} (grid size: {2*radius+1}×{2*radius+1})")
    total_points = len(mesh_data['mesh_points'])
    processed = 0
    
    for point in mesh_data['mesh_points']:
        lat, lon = point['lat'], point['lon']
        
        # Only process land points
        if not point.get('is_land', False):
            continue
            
        # Calculate spatial RMSE for this point
        spatial_stats = calculate_spatial_rmse(
            lat, lon, results_dict, unique_lats, unique_lons, radius
        )
        
        if spatial_stats is not None:
            spatial_rmse_data.append({
                'lat': lat,
                'lon': lon,
                'spatial_rmse': spatial_stats['rmse'],
                'spatial_mae': spatial_stats['mae'],
                'spatial_bias': spatial_stats['bias'],
                'spatial_correlation': spatial_stats['correlation'],
                'n_neighbors': spatial_stats['n_neighbors'],
                'neighborhood_llm_mean': spatial_stats['llm_mean'],
                'neighborhood_era5_mean': spatial_stats['era5_mean']
            })
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed}/{total_points} points ({100*processed/total_points:.1f}%)")
    
    print(f"Completed spatial RMSE analysis for {len(spatial_rmse_data)} points")
    return spatial_rmse_data

def create_rmse_grid(spatial_rmse_data):
    """Create a regular meshgrid with RMSE data mapped to it"""
    
    # Define grid resolution (1x1 degree)
    lat_min, lat_max = -60, 84
    lon_min, lon_max = -180, 179
    
    # Create coordinate arrays
    lats = np.arange(lat_min, lat_max + 1, 1.0)
    lons = np.arange(lon_min, lon_max + 1, 1.0)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Initialize RMSE grid with NaN
    rmse_grid = np.full_like(lon_grid, np.nan, dtype=float)
    
    # Map RMSE data to grid
    for data_point in spatial_rmse_data:
        lat = data_point['lat']
        lon = data_point['lon']
        rmse = data_point['spatial_rmse']
        
        # Find grid indices
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        
        # Check if indices are valid and within tolerance (0.5 degrees)
        if (np.abs(lats[lat_idx] - lat) <= 0.5 and 
            np.abs(lons[lon_idx] - lon) <= 0.5):
            rmse_grid[lat_idx, lon_idx] = rmse
    
    print(f"Mapped {np.sum(~np.isnan(rmse_grid))} points to {len(lats)}×{len(lons)} grid")
    
    return lon_grid, lat_grid, rmse_grid

def plot_spatial_rmse_map(spatial_rmse_data, output_file=None, figsize=(18, 10)):
    """Create a contour map showing spatial RMSE with country boundaries"""
    
    if not spatial_rmse_data:
        print("No spatial RMSE data to plot")
        return
    
    # Create regular grid
    lon_grid, lat_grid, rmse_grid = create_rmse_grid(spatial_rmse_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define contour levels
    rmse_values = [d['spatial_rmse'] for d in spatial_rmse_data]
    vmin = np.min(rmse_values)
    vmax = np.max(rmse_values)
    levels = np.linspace(vmin, vmax, 20)
    
    # Create contour plot
    contour = ax.contourf(lon_grid, lat_grid, rmse_grid, 
                         levels=levels, cmap='viridis_r', extend='both')
    
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
    ax.set_title(f'Spatial RMSE (9×9 neighborhood): LLM vs ERA5 on Land\n'
                f'Total land points: {len(spatial_rmse_data):,}', 
                fontsize=14, fontweight='bold')
    
    # Add coordinate ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('RMSE (°C)', fontsize=12)
    
    # Add statistics text
    mean_rmse = np.mean(rmse_values)
    median_rmse = np.median(rmse_values)
    min_rmse = np.min(rmse_values)
    max_rmse = np.max(rmse_values)
    std_rmse = np.std(rmse_values)
    
    stats_text = f'Land Points RMSE Statistics:\n'
    stats_text += f'Mean: {mean_rmse:.2f}°C\n'
    stats_text += f'Median: {median_rmse:.2f}°C\n'
    stats_text += f'Std: {std_rmse:.2f}°C\n'
    stats_text += f'Range: {min_rmse:.2f} - {max_rmse:.2f}°C\n'
    stats_text += f'Neighborhood: 9×9 grid (81 points max)\n'
    stats_text += f'Grid resolution: 1° × 1°'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Spatial RMSE contour map saved to: {output_file}")
    else:
        plt.show()
    
    return fig

def save_spatial_rmse_results(spatial_rmse_data, output_file):
    """Save spatial RMSE analysis results to JSON and CSV"""
    
    # Save as JSON
    json_file = output_file.replace('.csv', '.json')
    with open(json_file, 'w') as f:
        json.dump(spatial_rmse_data, f, indent=2)
    print(f"Spatial RMSE data saved to: {json_file}")
    
    # Save as CSV for easier analysis
    df = pd.DataFrame(spatial_rmse_data)
    df.to_csv(output_file, index=False)
    print(f"Spatial RMSE data saved to: {output_file}")
    
    return df

def main():
    """Main function"""
    
    # Default files - adjust based on available data
    mesh_file = 'meshes/mesh_data_1.0deg.json'
    results_file = 'results/climate_results_1.0deg_r10_simple.json'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        results_file = sys.argv[2]
    
    radius = 4  # ±4 points = 9×9 grid
    if len(sys.argv) > 3:
        radius = int(sys.argv[3])
    
    print(f"Spatial RMSE Analysis")
    print(f"Mesh file: {mesh_file}")
    print(f"Results file: {results_file}")
    print(f"Neighborhood radius: {radius} (grid size: {2*radius+1}×{2*radius+1})")
    
    # Check if files exist
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        return
    
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # Load data
        mesh_data, comparison_results = load_mesh_and_results(mesh_file, results_file)
        
        if mesh_data is None or comparison_results is None:
            print("Failed to load required data.")
            return
        
        # Perform spatial RMSE analysis
        spatial_rmse_data = analyze_spatial_rmse(mesh_data, comparison_results, radius)
        
        if not spatial_rmse_data:
            print("No spatial RMSE data generated.")
            return
        
        # Generate output filenames
        results_path = Path(results_file)
        
        # Create output files
        output_base = f"spatial_rmse_{results_path.stem.replace('_era5', '')}_r{radius}"
        csv_output = f"results/{output_base}.csv"
        plot_output = f"png/{output_base}_analysis.png"
        
        # Save results
        save_spatial_rmse_results(spatial_rmse_data, csv_output)
        
        # Create visualization
        plot_spatial_rmse_map(spatial_rmse_data, plot_output)
        
        # Print summary statistics
        rmse_values = [d['spatial_rmse'] for d in spatial_rmse_data]
        print(f"\\nSpatial RMSE Analysis Summary:")
        print(f"Points analyzed: {len(spatial_rmse_data):,}")
        print(f"Mean spatial RMSE: {np.mean(rmse_values):.3f}°C")
        print(f"Median spatial RMSE: {np.median(rmse_values):.3f}°C")
        print(f"RMSE range: {np.min(rmse_values):.3f} to {np.max(rmse_values):.3f}°C")
        
        print("\\nSpatial RMSE analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()