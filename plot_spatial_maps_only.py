#!/usr/bin/env python3
"""
Simple plotting script for creating spatial analysis maps only.

This script creates spatial analysis maps (RMSE, bias, temperature) from results files
that already contain spatial RMSE data. It focuses only on map generation without
scatter plots or other analysis.

Maps created:
- RMSE maps (radius 2 and 4)
- Bias maps (radius 2 and 4) 
- Neighborhood mean temperature maps (LLM and ERA5, radius 2 and 4)
- Individual point temperature maps (LLM and ERA5)

Usage:
    python plot_spatial_maps_only.py [results_file]

Default: results/climate_results_1.0deg_r10_gpt-5_simple_spatial_rmse_bathymetry_population.json
Output: png/{results_filename}/spatial_analysis_*_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import json
from pathlib import Path
import sys
import os
import cmocean

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_spatial_results(results_file):
    """Load spatial analysis results file"""
    print(f"Loading spatial results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
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


def create_map_plot(mapping_data, field_name, resolution, colormap='viridis', 
                   vmin=None, vmax=None, title_suffix="", output_file=None, figsize=(7.1, 4.0),
                   temp_range=None, temp_colormap='thermal', bias_range=None, bias_colormap='RdBu_r',
                   rmse_range=None, rmse_colormap='viridis_r'):
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
    
    # Check field types
    is_temp_field = ('temp' in field_name.lower() and 
                     not any(x in field_name.lower() for x in ['rmse', 'mae', 'bias', 'correlation']))
    is_bias_field = 'bias' in field_name.lower()
    is_rmse_field = 'rmse' in field_name.lower()
    
    # Set vmin/vmax based on field type
    if is_temp_field and temp_range is not None:
        # Use specified temperature range for absolute temperature fields
        vmin, vmax = temp_range
        # Clip grid data to temperature range to avoid empty areas
        data_grid = np.clip(data_grid, vmin, vmax)
    elif is_bias_field and bias_range is not None:
        # Use specified bias range for bias fields
        vmin, vmax = bias_range
        # Clip grid data to bias range to avoid empty areas
        data_grid = np.clip(data_grid, vmin, vmax)
    elif is_rmse_field and rmse_range is not None:
        # Use specified RMSE range for RMSE fields
        vmin, vmax = rmse_range
        # Clip grid data to RMSE range to avoid empty areas
        data_grid = np.clip(data_grid, vmin, vmax)
    else:
        # Use data range for other fields or if no range specified
        if vmin is None:
            vmin = np.min(values)
        if vmax is None:
            vmax = np.max(values)
    
    # Create contour levels
    levels = np.linspace(vmin, vmax, 20)
    
    # Create contour plot with appropriate colormap
    if is_rmse_field:
        # For RMSE, use specified colormap
        try:
            # Try cmocean first
            cmap = getattr(cmocean.cm, rmse_colormap)
        except AttributeError:
            # Fall back to matplotlib colormap
            cmap = plt.colormaps[rmse_colormap]
    elif 'mae' in field_name.lower():
        # For MAE, use reversed colormap (blue=good, red=bad)
        cmap = plt.colormaps['viridis_r']
    elif is_bias_field:
        # For bias, use specified diverging colormap
        try:
            # Try cmocean first
            cmap = getattr(cmocean.cm, bias_colormap)
        except AttributeError:
            # Fall back to matplotlib colormap
            cmap = plt.colormaps[bias_colormap]
        # Bias range is already set above, no need to make symmetric here
    elif is_temp_field:
        # For absolute temperature, use specified colormap (cmocean or matplotlib)
        try:
            # Try cmocean first
            cmap = getattr(cmocean.cm, temp_colormap)
        except AttributeError:
            # Fall back to matplotlib colormap
            cmap = plt.colormaps[temp_colormap]
    else:
        cmap = plt.colormaps[colormap]
    
    # Use extend='both' only for MAE fields that might exceed ranges
    # RMSE, bias and temperature fields use 'neither' for rectangular bars since data is clipped to range
    if 'mae' in field_name.lower():
        extend_option = 'both'
    else:
        extend_option = 'neither'
    
    contour = ax.contourf(lon_grid, lat_grid, data_grid, 
                         levels=levels, cmap=cmap, extend=extend_option)
    
    # Try to load and overlay country boundaries with subtle coastlines
    try:
        world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
        world.boundary.plot(ax=ax, color='gray', linewidth=0.25, alpha=0.7)
    except Exception as e:
        print(f"Could not load country boundaries: {e}")
        # Add faint gridlines if no coastlines available
        ax.grid(True, color='lightgray', alpha=0.3, linewidth=0.5)
    
    # Add faint gridlines for better orientation
    ax.grid(True, color='lightgray', alpha=0.2, linewidth=0.3, linestyle='-')
    
    # Customize the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude', fontsize=9)
    ax.set_ylabel('Latitude', fontsize=9)
    
    # Minimal title - just the field name
    ax.set_title('', fontsize=12)  # Remove title per journal standards
    
    # Add coordinate ticks with proper geographic notation
    lon_ticks = np.arange(-180, 181, 60)
    lat_ticks = np.arange(-60, 91, 30)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    
    # Format longitude labels: 180°W, 120°W, 60°W, 0°, 60°E, 120°E, 180°E
    lon_labels = []
    for lon in lon_ticks:
        if lon == 0:
            lon_labels.append('0°')
        elif lon > 0:
            lon_labels.append(f'{int(lon)}°E')
        else:
            lon_labels.append(f'{int(abs(lon))}°W')
    ax.set_xticklabels(lon_labels)
    
    # Format latitude labels: 60°S, 30°S, 0°, 30°N, 60°N
    lat_labels = []
    for lat in lat_ticks:
        if lat == 0:
            lat_labels.append('0°')
        elif lat > 0:
            lat_labels.append(f'{int(lat)}°N')
        else:
            lat_labels.append(f'{int(abs(lat))}°S')
    ax.set_yticklabels(lat_labels)
    
    # Add horizontal colorbar beneath the map with appropriate ends
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=0.15)
    
    # Set colorbar label and ticks based on field type
    if is_rmse_field:
        cbar.set_label('RMSE (°C)', fontsize=10)
        # Set nice rounded ticks for RMSE colorbar - every 2 degrees
        if rmse_range is not None:
            rmse_ticks = np.arange(rmse_range[0], rmse_range[1] + 0.1, 2)  # Every 2 degrees: 0, 2, 4, 6, 8, 10
            cbar.set_ticks(rmse_ticks)
            cbar.set_ticklabels([f'{int(tick)}°' for tick in rmse_ticks])
    elif 'mae' in field_name.lower():
        cbar.set_label('MAE (°C)', fontsize=10)
    elif 'bias' in field_name.lower():
        cbar.set_label('Bias (°C)', fontsize=10)
        # Set nice rounded ticks for bias colorbar - every 2.5 degrees
        if bias_range is not None:
            bias_ticks = np.arange(bias_range[0], bias_range[1] + 0.1, 2.5)  # Every 2.5 degrees: -7.5, -5.0, -2.5, 0, 2.5, 5.0, 7.5
            cbar.set_ticks(bias_ticks)
            cbar.set_ticklabels([f'{tick:g}°' for tick in bias_ticks])
    elif is_temp_field:
        cbar.set_label('Temperature (°C)', fontsize=10)
        # Set nice rounded ticks for temperature colorbar - every 5 degrees
        if temp_range is not None:
            temp_ticks = np.arange(temp_range[0], temp_range[1] + 1, 5)  # Every 5 degrees: -10, -5, 0, 5, ..., 40
            cbar.set_ticks(temp_ticks)
            cbar.set_ticklabels([f'{int(tick)}°' for tick in temp_ticks])
    else:
        cbar.set_label('Value', fontsize=10)
    
    # Statistics are removed from figure - should go in caption or supplementary table
    # as per journal standards
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{field_name} map saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def create_all_spatial_maps(results_data, resolution, output_dir='png', temp_range=(-13, 40), temp_colormap='thermal', bias_range=(-7, 7), bias_colormap='RdBu_r', rmse_range=(0, 10), rmse_colormap='viridis_r'):
    """Create all spatial analysis maps from the results data"""
    
    print("Creating comprehensive spatial analysis maps...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base filename pattern
    base_name = f"spatial_analysis_{resolution}deg"
    
    # Maps to create - field_name: (title_suffix, colormap)
    maps_config = {
        # RMSE maps (radius 2 only)
        'spatial_r2_rmse': ('(5×5 neighborhood)', 'viridis_r'),
        
        # Bias maps (radius 2 only)
        'spatial_r2_bias': ('(5×5 neighborhood)', 'RdBu_r'),
        
        # Neighborhood LLM mean temperature (radius 2 only)
        'spatial_r2_neighborhood_llm_mean': ('(5×5 neighborhood)', 'coolwarm'),
        
        # Neighborhood ERA5 mean temperature (radius 2 only)
        'spatial_r2_neighborhood_era5_mean': ('(5×5 neighborhood)', 'coolwarm'),
        
        # Individual point temperatures
        'llm_temp_mean': ('(Individual Points)', 'coolwarm'),
        'era5_temp_mean': ('(Individual Points)', 'coolwarm'),
        
        # MAE maps (radius 2 only)
        'spatial_r2_mae': ('(5×5 neighborhood)', 'viridis_r'),
    }
    
    created_maps = []
    
    for field_name, (title_suffix, colormap) in maps_config.items():
        print(f"\nCreating map for {field_name}...")
        
        # Extract mapping data for this field
        mapping_data = extract_mapping_data(results_data, field_name)
        
        if not mapping_data:
            print(f"No data found for {field_name}, skipping...")
            continue
        
        # Create output filename with publication prefix
        clean_field_name = field_name.replace('spatial_', '').replace('_', '_')
        output_file = output_path / f"pub_{base_name}_{clean_field_name}.png"
        
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
                output_file=str(output_file),
                temp_range=temp_range,
                temp_colormap=temp_colormap,
                bias_range=bias_range,
                bias_colormap=bias_colormap,
                rmse_range=rmse_range,
                rmse_colormap=rmse_colormap
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
        'spatial_r2_rmse',
        'spatial_r2_mae', 
        'spatial_r2_bias',
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
    
    # Temperature range configuration for absolute temperature maps
    # Change these values to control the color scale for temperature fields
    TEMP_MIN = -10  # Minimum temperature in °C
    TEMP_MAX = 40   # Maximum temperature in °C
    temp_range = (TEMP_MIN, TEMP_MAX)
    
    # Bias range configuration for bias maps
    # Change these values to control the color scale for bias fields
    BIAS_MIN = -7.5  # Minimum bias in °C
    BIAS_MAX = 7.5   # Maximum bias in °C
    bias_range = (BIAS_MIN, BIAS_MAX)
    
    # RMSE range configuration for RMSE maps
    # Change these values to control the color scale for RMSE fields
    RMSE_MIN = 0     # Minimum RMSE in °C
    RMSE_MAX = 7    # Maximum RMSE in °C
    rmse_range = (RMSE_MIN, RMSE_MAX)
    
    # Temperature colormap configuration
    # Options: cmocean colormaps (thermal, balance) or matplotlib colormaps (viridis, plasma, inferno)
    TEMP_COLORMAP = 'thermal'  # Colormap name for temperature fields
    #  - 'thermal' - cmocean: Natural heat progression
    #  - 'balance' - cmocean: Diverging blue-white-red
    #  - 'plasma' - matplotlib: Viridis-family purple-pink-yellow
    #  - 'inferno' - matplotlib: Dark red-orange-yellow
    #  - 'viridis' - matplotlib: Blue-green-yellow progression
    
    # Bias colormap configuration
    # Options: diverging colormaps for positive/negative bias
    BIAS_COLORMAP = 'balance'  # Colormap name for bias fields
    #  - 'RdBu_r' - matplotlib: Red-white-blue (reversed)
    #  - 'balance' - cmocean: Blue-white-red diverging
    #  - 'seismic' - matplotlib: Blue-white-red
    #  - 'coolwarm' - matplotlib: Blue-white-red
    
    # RMSE colormap configuration
    # Options: sequential colormaps for error magnitude (low=good, high=bad)
    RMSE_COLORMAP = 'balance'  # Colormap name for RMSE fields
    #  - 'viridis_r' - matplotlib: Purple-blue-green-yellow (reversed, low=yellow/good)
    #  - 'plasma_r' - matplotlib: Purple-pink-yellow (reversed)
    #  - 'inferno_r' - matplotlib: Black-red-yellow (reversed)
    #  - 'Spectral' - matplotlib: Blue-green-yellow-red
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_gpt-5_simple_spatial_rmse_bathymetry_population.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Spatial Analysis Maps Only - Plotting")
    print("=" * 50)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please provide a results file that contains spatial RMSE data.")
        return
    
    try:
        # Load spatial results
        results_data = load_spatial_results(results_file)
        
        # Extract resolution from filename or metadata
        resolution = results_data.get('resolution', 1.0)
        if 'deg' in results_file:
            try:
                # Extract resolution from filename like "1.0deg"
                resolution_str = results_file.split('_')[2]
                if 'deg' in resolution_str:
                    resolution = float(resolution_str.replace('deg', ''))
            except:
                pass
        
        print(f"Detected resolution: {resolution}°")
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem  # Get filename without extension
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create all spatial maps
        created_maps = create_all_spatial_maps(results_data, resolution, str(output_dir), temp_range, TEMP_COLORMAP, bias_range, BIAS_COLORMAP, rmse_range, RMSE_COLORMAP)
        
        # Print summary statistics
        print_summary_statistics(results_data)
        
        print(f"\nMapping completed successfully!")
        print(f"Created {len(created_maps)} spatial analysis maps")
        print(f"All files saved in: {output_dir}/")
        
        # List created files
        if created_maps:
            print(f"\nCreated files:")
            for map_file in created_maps:
                print(f"  - {Path(map_file).name}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()