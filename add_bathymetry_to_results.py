#!/usr/bin/env python3
"""
Add bathymetry/elevation parameters to existing result files.

This script reads aggregated bathymetry/elevation data from NetCDF format and adds it to 
existing result files by matching coordinates using nearest neighbor interpolation.

Bathymetry data includes:
- mean_elevation: Area-weighted mean elevation (m)
- min_elevation: Minimum elevation within cell (m)  
- max_elevation: Maximum elevation within cell (m)
- std_elevation: Standard deviation of elevation (m)
- roughness: Terrain roughness metric (m)

Usage:
    python add_bathymetry_to_results.py [results_file] [bathymetry_file]

Default files:
    - Results: results/climate_results_1.0deg_r10_simple_spatial_rmse_population.json
    - Bathymetry: data/bathymetry_1deg_aggregated.nc

Output: results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
"""

import numpy as np
import pandas as pd
import xarray as xr
import json
from pathlib import Path
import sys


def load_bathymetry_data(bathymetry_file):
    """Load bathymetry NetCDF data"""
    print(f"Loading bathymetry data from: {bathymetry_file}")
    
    # Load the dataset
    ds = xr.open_dataset(bathymetry_file)
    
    print(f"Bathymetry grid shape: {ds.mean_elevation.shape}")
    print(f"Latitude range: {ds.lat.min().values:.1f} to {ds.lat.max().values:.1f}")
    print(f"Longitude range: {ds.lon.min().values:.1f} to {ds.lon.max().values:.1f}")
    
    # Check available variables
    data_vars = list(ds.data_vars)
    print(f"Available variables: {data_vars}")
    
    return ds


def get_bathymetry_at_coordinates(target_lat, target_lon, ds):
    """Get bathymetry data at target coordinates using nearest neighbor"""
    
    try:
        # Select nearest grid point
        point_data = ds.sel(lat=target_lat, lon=target_lon, method='nearest')
        
        # Extract all bathymetry parameters
        bathymetry_params = {}
        
        # List of variables to extract
        variables = ['mean_elevation', 'min_elevation', 'max_elevation', 'std_elevation', 'roughness']
        
        for var in variables:
            if var in ds.data_vars:
                value = float(point_data[var].values)
                # Convert NaN to None for JSON serialization
                if not np.isnan(value):
                    # For land points, elevation cannot be below sea level (set minimum to 0)
                    if var in ['mean_elevation', 'min_elevation'] and value < 0:
                        value = 0.0
                    bathymetry_params[var] = value
                else:
                    bathymetry_params[var] = None
            else:
                print(f"Warning: Variable {var} not found in bathymetry data")
                bathymetry_params[var] = None
        
        # Also get the actual coordinates of the nearest grid point for verification
        nearest_lat = float(point_data.lat.values)
        nearest_lon = float(point_data.lon.values)
        
        return bathymetry_params, (nearest_lat, nearest_lon)
        
    except Exception as e:
        print(f"Warning: Could not extract bathymetry data for ({target_lat:.1f}, {target_lon:.1f}): {e}")
        return {
            'mean_elevation': None,
            'min_elevation': None,
            'max_elevation': None,
            'std_elevation': None,
            'roughness': None
        }, (None, None)


def load_results_file(results_file):
    """Load results file and check its format"""
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check what data is already present
    has_era5 = results_data.get('metadata', {}).get('era5_climatology_added', False)
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False)
    has_population = results_data.get('metadata', {}).get('population_added', False)
    has_bathymetry = results_data.get('metadata', {}).get('bathymetry_added', False)
    
    print(f"Data present - ERA5: {has_era5}, Spatial RMSE: {has_spatial}, Population: {has_population}, Bathymetry: {has_bathymetry}")
    
    if has_bathymetry:
        print("Warning: Bathymetry data already present in results file")
    
    return results_data


def add_bathymetry_to_results(results_data, bathymetry_file):
    """Add bathymetry data to each result point"""
    
    # Load bathymetry data
    ds = load_bathymetry_data(bathymetry_file)
    
    print(f"\nAdding bathymetry data to result points...")
    
    # Process each result point
    bathymetry_stats = {
        'mean_elevation': [],
        'min_elevation': [],
        'max_elevation': [], 
        'std_elevation': [],
        'roughness': []
    }
    processed_count = 0
    land_points_count = 0
    
    for result in results_data['results']:
        point_info = result['point_info']
        target_lat = point_info['lat']
        target_lon = point_info['lon']
        is_land = point_info.get('is_land', False)
        
        # Get bathymetry data at this point
        bathymetry_params, nearest_coords = get_bathymetry_at_coordinates(
            target_lat, target_lon, ds
        )
        
        # Add bathymetry data to point_info
        for param, value in bathymetry_params.items():
            point_info[param] = value
        
        # Also add nearest grid coordinates for reference
        if nearest_coords[0] is not None and nearest_coords[1] is not None:
            point_info['bathymetry_nearest_lat'] = nearest_coords[0]
            point_info['bathymetry_nearest_lon'] = nearest_coords[1]
            point_info['bathymetry_distance_lat'] = abs(target_lat - nearest_coords[0])
            point_info['bathymetry_distance_lon'] = abs(target_lon - nearest_coords[1])
        
        # Collect statistics for valid data
        if is_land:
            land_points_count += 1
            for param, value in bathymetry_params.items():
                if value is not None:
                    bathymetry_stats[param].append(value)
        
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count}/{len(results_data['results'])} points")
    
    # Close the dataset
    ds.close()
    
    # Update metadata
    results_data['metadata']['bathymetry_added'] = True
    results_data['metadata']['bathymetry_file'] = str(bathymetry_file)
    results_data['metadata']['bathymetry_processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    results_data['metadata']['bathymetry_grid_resolution'] = '1 degree'
    
    print(f"\nBathymetry data successfully added to {processed_count} result points")
    print(f"Land points: {land_points_count}")
    
    # Print bathymetry statistics for land points
    for param, values in bathymetry_stats.items():
        if values:
            print(f"\n{param.replace('_', ' ').title()} (land points):")
            print(f"  Valid points: {len(values)}")
            print(f"  Mean: {np.mean(values):.2f} m")
            print(f"  Median: {np.median(values):.2f} m")
            print(f"  Range: {np.min(values):.2f} to {np.max(values):.2f} m")
            print(f"  Std: {np.std(values):.2f} m")
    
    return results_data


def save_results_with_bathymetry(results_data, original_file):
    """Save the enhanced results to a new file with bathymetry suffix"""
    
    # Generate output filename
    original_path = Path(original_file)
    output_file = original_path.parent / (original_path.stem + '_bathymetry.json')
    
    print(f"\nSaving enhanced results to: {output_file}")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"Enhanced results saved successfully!")
    return output_file


def print_bathymetry_summary(results_data):
    """Print summary of bathymetry data integration"""
    
    results = results_data['results']
    
    # Collect bathymetry statistics by land/ocean
    all_stats = {}
    land_stats = {}
    ocean_stats = {}
    
    for param in ['mean_elevation', 'min_elevation', 'max_elevation', 'std_elevation', 'roughness']:
        all_stats[param] = []
        land_stats[param] = []
        ocean_stats[param] = []
    
    for result in results:
        point_info = result['point_info']
        is_land = point_info.get('is_land', False)
        
        for param in all_stats.keys():
            value = point_info.get(param)
            if value is not None:
                all_stats[param].append(value)
                if is_land:
                    land_stats[param].append(value)
                else:
                    ocean_stats[param].append(value)
    
    print(f"\nBathymetry Data Integration Summary:")
    print(f"=" * 70)
    print(f"Total result points: {len(results)}")
    
    for param, values in all_stats.items():
        param_name = param.replace('_', ' ').title()
        print(f"\n{param_name}:")
        
        if values:
            print(f"  All points - Count: {len(values)}, Range: {np.min(values):.1f} to {np.max(values):.1f} m")
            print(f"              Mean: {np.mean(values):.1f} m, Median: {np.median(values):.1f} m")
        
        if land_stats[param]:
            land_values = land_stats[param]
            print(f"  Land points - Count: {len(land_values)}, Range: {np.min(land_values):.1f} to {np.max(land_values):.1f} m")
            print(f"                Mean: {np.mean(land_values):.1f} m, Median: {np.median(land_values):.1f} m")
        
        if ocean_stats[param]:
            ocean_values = ocean_stats[param]
            print(f"  Ocean points - Count: {len(ocean_values)}, Range: {np.min(ocean_values):.1f} to {np.max(ocean_values):.1f} m")
            print(f"                 Mean: {np.mean(ocean_values):.1f} m, Median: {np.median(ocean_values):.1f} m")


def check_coordinate_matching(results_data):
    """Check quality of coordinate matching between results and bathymetry grid"""
    
    print(f"\nCoordinate Matching Quality:")
    print(f"=" * 50)
    
    distances_lat = []
    distances_lon = []
    max_distances = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        if ('bathymetry_distance_lat' in point_info and 
            'bathymetry_distance_lon' in point_info):
            
            lat_dist = point_info['bathymetry_distance_lat']
            lon_dist = point_info['bathymetry_distance_lon']
            
            distances_lat.append(lat_dist)
            distances_lon.append(lon_dist)
            max_distances.append(max(lat_dist, lon_dist))
    
    if distances_lat:
        print(f"Latitude distance - Mean: {np.mean(distances_lat):.4f}°, Max: {np.max(distances_lat):.4f}°")
        print(f"Longitude distance - Mean: {np.mean(distances_lon):.4f}°, Max: {np.max(distances_lon):.4f}°")
        print(f"Maximum distance - Mean: {np.mean(max_distances):.4f}°, Max: {np.max(max_distances):.4f}°")
        
        # Count points within tolerance
        within_half_degree = np.sum(np.array(max_distances) <= 0.5)
        print(f"Points within 0.5°: {within_half_degree}/{len(max_distances)} ({100*within_half_degree/len(max_distances):.1f}%)")


def main():
    """Main function"""
    
    # Default files
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population.json'
    default_bathymetry = 'data/bathymetry_1deg_aggregated.nc'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    bathymetry_file = sys.argv[2] if len(sys.argv) > 2 else default_bathymetry
    
    print("Add Bathymetry Data to Results")
    print("=" * 70)
    print(f"Results file: {results_file}")
    print(f"Bathymetry file: {bathymetry_file}")
    print()
    
    # Check if files exist
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    if not Path(bathymetry_file).exists():
        print(f"Error: Bathymetry file '{bathymetry_file}' not found.")
        print("Please run aggregate_bathymetry.py first to create the bathymetry data.")
        return
    
    try:
        # Load results file
        results_data = load_results_file(results_file)
        
        # Add bathymetry data
        enhanced_results = add_bathymetry_to_results(results_data, bathymetry_file)
        
        # Save enhanced results
        output_file = save_results_with_bathymetry(enhanced_results, results_file)
        
        # Print summary
        print_bathymetry_summary(enhanced_results)
        
        # Check coordinate matching quality
        check_coordinate_matching(enhanced_results)
        
        print(f"\nProcess completed successfully!")
        print(f"Enhanced results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()