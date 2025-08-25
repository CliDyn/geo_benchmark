#!/usr/bin/env python3
"""
Aggregate GEBCO bathymetry/elevation data to 1° × 1° global grid.

This script reads high-resolution GEBCO elevation data and aggregates it to a 
1° × 1° global grid, calculating various statistics including a roughness metric.

For each 1° cell, calculates:
1. Area-weighted mean altitude (metres)
2. Min/max/std altitude 
3. Roughness metric: area-weighted mean absolute deviation from cell mean

Input: /Users/ikuznets/work/data/bathymetry/GRIDONE_2D.nc (GEBCO 1 arc-minute grid)
Output: ./data/bathymetry_1deg_aggregated.nc (1° × 1° CF-compliant NetCDF)

Usage:
    python aggregate_bathymetry.py [input_file] [output_file]

Default files:
    - Input: /Users/ikuznets/work/data/bathymetry/GRIDONE_2D.nc
    - Output: data/bathymetry_1deg_aggregated.nc
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
import time
from scipy import ndimage


def load_gebco_data(input_file):
    """Load GEBCO bathymetry data"""
    print(f"Loading GEBCO data from: {input_file}")
    
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    print(f"Input data shape: {ds.elevation.shape}")
    print(f"Latitude range: {ds.lat.min().values:.4f} to {ds.lat.max().values:.4f}")
    print(f"Longitude range: {ds.lon.min().values:.4f} to {ds.lon.max().values:.4f}")
    print(f"Elevation range: {ds.elevation.min().values} to {ds.elevation.max().values} m")
    print(f"Grid resolution: ~{(ds.lat[1] - ds.lat[0]).values:.4f}° ({60 * (ds.lat[1] - ds.lat[0]).values:.2f} arc-minutes)")
    
    return ds


def create_target_grid():
    """Create 1° × 1° global target grid matching project mesh system"""
    print("Creating 1° × 1° target grid (matching project mesh system)...")
    
    # Match the mesh generation system from geo_mesh_processor.py
    # lons = np.arange(-180, 180, resolution) -> for 1°: -180, -179, ..., 179
    # lats = np.arange(-60, 85, resolution) -> for 1°: -60, -59, ..., 84
    target_lons = np.arange(-180, 180, 1.0)  # -180 to 179, 360 points
    target_lats = np.arange(-60, 85, 1.0)    # -60 to 84, 145 points
    
    print(f"Target grid: {len(target_lats)} × {len(target_lons)} = {len(target_lats) * len(target_lons)} cells")
    print(f"Latitude range: {target_lats[0]} to {target_lats[-1]} (matches mesh: -60 to 84)")
    print(f"Longitude range: {target_lons[0]} to {target_lons[-1]} (matches mesh: -180 to 179)")
    
    return target_lats, target_lons


def calculate_pixel_areas(lats, lons):
    """Calculate area of each pixel in square meters"""
    print("Calculating pixel areas...")
    
    # Earth radius in meters
    R = 6371000.0
    
    # Convert to radians
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    
    # Calculate grid spacing
    dlat = lats_rad[1] - lats_rad[0]
    dlon = lons_rad[1] - lons_rad[0]
    
    # Create 2D coordinate arrays
    lat_2d, lon_2d = np.meshgrid(lats_rad, lons_rad, indexing='ij')
    
    # Calculate area for each pixel
    # Area = R² * cos(lat) * dlat * dlon
    areas = R**2 * np.cos(lat_2d) * dlat * dlon
    
    print(f"Pixel areas range: {areas.min():.0f} to {areas.max():.0f} m²")
    
    return areas


def aggregate_to_target_cell(source_elevations, source_areas, source_lats, source_lons, 
                           target_lat, target_lon):
    """Aggregate source pixels to a single target cell"""
    
    # Define target cell boundaries (±0.5° around cell center)
    # This matches the mesh system where coordinates are cell centers
    lat_min, lat_max = target_lat - 0.5, target_lat + 0.5
    lon_min, lon_max = target_lon - 0.5, target_lon + 0.5
    
    # Find indices of source pixels within target cell
    lat_mask = (source_lats >= lat_min) & (source_lats <= lat_max)
    lon_mask = (source_lons >= lon_min) & (source_lons <= lon_max)
    
    # Get indices where both conditions are true
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        # No source pixels in this target cell
        return {
            'mean_elevation': np.nan,
            'min_elevation': np.nan,
            'max_elevation': np.nan,
            'std_elevation': np.nan,
            'roughness': np.nan,
            'pixel_count': 0,
            'total_area': 0.0
        }
    
    # Extract elevation and area data for pixels in this cell
    cell_elevations = source_elevations[np.ix_(lat_indices, lon_indices)]
    cell_areas = source_areas[np.ix_(lat_indices, lon_indices)]
    
    # Flatten for easier processing
    elevations_flat = cell_elevations.flatten()
    areas_flat = cell_areas.flatten()
    
    # Remove any NaN or invalid values
    valid_mask = np.isfinite(elevations_flat) & np.isfinite(areas_flat) & (areas_flat > 0)
    if not np.any(valid_mask):
        return {
            'mean_elevation': np.nan,
            'min_elevation': np.nan,
            'max_elevation': np.nan,
            'std_elevation': np.nan,
            'roughness': np.nan,
            'pixel_count': 0,
            'total_area': 0.0
        }
    
    valid_elevations = elevations_flat[valid_mask]
    valid_areas = areas_flat[valid_mask]
    
    # Calculate area-weighted mean
    total_area = np.sum(valid_areas)
    weights = valid_areas / total_area
    mean_elevation = np.sum(weights * valid_elevations)
    
    # Calculate other statistics
    min_elevation = np.min(valid_elevations)
    max_elevation = np.max(valid_elevations)
    
    # Area-weighted standard deviation
    variance = np.sum(weights * (valid_elevations - mean_elevation)**2)
    std_elevation = np.sqrt(variance)
    
    # Roughness: area-weighted mean absolute deviation from cell mean
    roughness = np.sum(weights * np.abs(valid_elevations - mean_elevation))
    
    return {
        'mean_elevation': mean_elevation,
        'min_elevation': min_elevation,
        'max_elevation': max_elevation,
        'std_elevation': std_elevation,
        'roughness': roughness,
        'pixel_count': len(valid_elevations),
        'total_area': total_area
    }


def aggregate_gebco_data(ds, target_lats, target_lons):
    """Aggregate GEBCO data to target grid"""
    print("Aggregating GEBCO data to 1° grid...")
    
    # Get source data
    source_elevations = ds.elevation.values
    source_lats = ds.lat.values
    source_lons = ds.lon.values
    
    # Calculate pixel areas
    source_areas = calculate_pixel_areas(source_lats, source_lons)
    
    # Initialize output arrays
    n_lat, n_lon = len(target_lats), len(target_lons)
    
    mean_elevations = np.full((n_lat, n_lon), np.nan)
    min_elevations = np.full((n_lat, n_lon), np.nan)
    max_elevations = np.full((n_lat, n_lon), np.nan)
    std_elevations = np.full((n_lat, n_lon), np.nan)
    roughness = np.full((n_lat, n_lon), np.nan)
    pixel_counts = np.full((n_lat, n_lon), 0, dtype=int)
    
    # Process each target cell
    total_cells = n_lat * n_lon
    processed = 0
    start_time = time.time()
    
    print(f"Processing {total_cells} target cells...")
    
    for i, target_lat in enumerate(target_lats):
        for j, target_lon in enumerate(target_lons):
            
            # Aggregate data for this cell
            result = aggregate_to_target_cell(
                source_elevations, source_areas, source_lats, source_lons,
                target_lat, target_lon
            )
            
            # Store results
            mean_elevations[i, j] = result['mean_elevation']
            min_elevations[i, j] = result['min_elevation']
            max_elevations[i, j] = result['max_elevation']
            std_elevations[i, j] = result['std_elevation']
            roughness[i, j] = result['roughness']
            pixel_counts[i, j] = result['pixel_count']
            
            processed += 1
            
            # Progress reporting
            if processed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                remaining = (total_cells - processed) / rate
                print(f"Processed {processed}/{total_cells} cells ({100*processed/total_cells:.1f}%) - "
                      f"Rate: {rate:.1f} cells/s - ETA: {remaining/60:.1f} min")
    
    elapsed = time.time() - start_time
    print(f"Aggregation completed in {elapsed/60:.1f} minutes")
    
    # Create output dataset
    output_ds = xr.Dataset(
        data_vars={
            'mean_elevation': (['lat', 'lon'], mean_elevations, {
                'long_name': 'Area-weighted mean elevation',
                'standard_name': 'surface_altitude',
                'units': 'm',
                'description': 'Area-weighted mean elevation relative to sea level from GEBCO data'
            }),
            'min_elevation': (['lat', 'lon'], min_elevations, {
                'long_name': 'Minimum elevation',
                'units': 'm',
                'description': 'Minimum elevation within 1° cell'
            }),
            'max_elevation': (['lat', 'lon'], max_elevations, {
                'long_name': 'Maximum elevation',
                'units': 'm',
                'description': 'Maximum elevation within 1° cell'
            }),
            'std_elevation': (['lat', 'lon'], std_elevations, {
                'long_name': 'Standard deviation of elevation',
                'units': 'm',
                'description': 'Area-weighted standard deviation of elevation within 1° cell'
            }),
            'roughness': (['lat', 'lon'], roughness, {
                'long_name': 'Terrain roughness',
                'units': 'm',
                'description': 'Area-weighted mean absolute deviation from cell mean elevation'
            }),
            'pixel_count': (['lat', 'lon'], pixel_counts, {
                'long_name': 'Number of source pixels',
                'units': '1',
                'description': 'Number of valid GEBCO pixels aggregated into each 1° cell'
            })
        },
        coords={
            'lat': (['lat'], target_lats, {
                'standard_name': 'latitude',
                'long_name': 'latitude',
                'units': 'degrees_north',
                'axis': 'Y'
            }),
            'lon': (['lon'], target_lons, {
                'standard_name': 'longitude',
                'long_name': 'longitude', 
                'units': 'degrees_east',
                'axis': 'X'
            })
        },
        attrs={
            'Conventions': 'CF-1.8',
            'title': 'GEBCO bathymetry/elevation aggregated to 1° × 1° grid',
            'institution': 'Climate LLM Benchmark Project',
            'source': 'Aggregated from GEBCO One Minute Grid',
            'history': f'Created on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")} - '
                      f'Aggregated from 1 arc-minute GEBCO data to 1° × 1° grid using area-weighted statistics',
            'references': 'GEBCO Compilation Group (2019) GEBCO 2019 Grid',
            'comment': 'Elevation data aggregated using area-weighted statistics. '
                      'Roughness calculated as area-weighted mean absolute deviation from cell mean.',
            'grid_resolution': '1 degree',
            'source_resolution': '1 arc-minute',
            'aggregation_method': 'area-weighted statistics'
        }
    )
    
    return output_ds


def print_aggregation_summary(ds):
    """Print summary of aggregated data"""
    
    print(f"\nAggregation Summary:")
    print(f"=" * 60)
    
    for var_name in ['mean_elevation', 'min_elevation', 'max_elevation', 'std_elevation', 'roughness']:
        var_data = ds[var_name].values
        valid_data = var_data[np.isfinite(var_data)]
        
        if len(valid_data) > 0:
            print(f"\n{var_name}:")
            print(f"  Valid cells: {len(valid_data)}/{var_data.size} ({100*len(valid_data)/var_data.size:.1f}%)")
            print(f"  Range: {np.min(valid_data):.1f} to {np.max(valid_data):.1f} m")
            print(f"  Mean: {np.mean(valid_data):.1f} m")
            print(f"  Median: {np.median(valid_data):.1f} m")
            print(f"  Std: {np.std(valid_data):.1f} m")
    
    # Pixel count statistics
    pixel_counts = ds['pixel_count'].values
    print(f"\nPixel Count Statistics:")
    print(f"  Cells with data: {np.sum(pixel_counts > 0)}/{pixel_counts.size}")
    print(f"  Mean pixels per cell: {np.mean(pixel_counts[pixel_counts > 0]):.0f}")
    print(f"  Min/max pixels per cell: {np.min(pixel_counts[pixel_counts > 0]):.0f} / {np.max(pixel_counts):.0f}")


def save_aggregated_data(ds, output_file):
    """Save aggregated data as CF-compliant NetCDF"""
    print(f"\nSaving aggregated data to: {output_file}")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with compression and proper encoding
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dtype == np.float64:
            encoding[var] = {
                'dtype': 'float32',  # Reduce precision to save space
                'zlib': True,
                'complevel': 6,
                '_FillValue': -9999.0
            }
        else:
            encoding[var] = {
                'zlib': True,
                'complevel': 6
            }
    
    # Save dataset
    ds.to_netcdf(output_file, encoding=encoding)
    
    # Verify file was created and report size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Output file created: {file_size_mb:.1f} MB")


def main():
    """Main function"""
    
    # Default files
    default_input = '/Users/ikuznets/work/data/bathymetry/GRIDONE_2D.nc'
    default_output = 'data/bathymetry_1deg_aggregated.nc'
    
    # Parse command line arguments
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_file = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    print("GEBCO Bathymetry Aggregation to 1° Grid")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    try:
        start_time = time.time()
        
        # Load GEBCO data
        ds = load_gebco_data(input_file)
        
        # Create target grid
        target_lats, target_lons = create_target_grid()
        
        # Aggregate data
        aggregated_ds = aggregate_gebco_data(ds, target_lats, target_lons)
        
        # Print summary
        print_aggregation_summary(aggregated_ds)
        
        # Save results
        save_aggregated_data(aggregated_ds, output_file)
        
        total_time = time.time() - start_time
        print(f"\nProcess completed successfully in {total_time/60:.1f} minutes!")
        
        # Close input dataset
        ds.close()
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()