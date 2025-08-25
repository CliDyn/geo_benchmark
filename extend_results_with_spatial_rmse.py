#!/usr/bin/env python3
"""
Extend LLM result files with spatial RMSE data for each point.

This script takes an existing LLM result file and adds spatial RMSE calculations
for each point using neighborhoods of radius 2 and 4. If ERA5 data is not present
in the result file, it automatically calls add_era5_to_results.py to add it first.

New fields added to each point_info:
- spatial_r2_rmse: RMSE calculated using 5×5 neighborhood (radius 2)
- spatial_r2_mae: Mean Absolute Error for 5×5 neighborhood
- spatial_r2_bias: Bias (LLM - ERA5) for 5×5 neighborhood
- spatial_r2_correlation: Correlation coefficient for 5×5 neighborhood
- spatial_r2_n_neighbors: Number of valid neighbors used in 5×5 calculation
- spatial_r2_neighborhood_llm_mean: Mean LLM temperature in 5×5 neighborhood
- spatial_r2_neighborhood_era5_mean: Mean ERA5 temperature in 5×5 neighborhood

- spatial_r4_rmse: RMSE calculated using 9×9 neighborhood (radius 4)
- spatial_r4_mae: Mean Absolute Error for 9×9 neighborhood
- spatial_r4_bias: Bias (LLM - ERA5) for 9×9 neighborhood
- spatial_r4_correlation: Correlation coefficient for 9×9 neighborhood
- spatial_r4_n_neighbors: Number of valid neighbors used in 9×9 calculation
- spatial_r4_neighborhood_llm_mean: Mean LLM temperature in 9×9 neighborhood
- spatial_r4_neighborhood_era5_mean: Mean ERA5 temperature in 9×9 neighborhood

Usage:
    python extend_results_with_spatial_rmse.py [results_file]

Default: results/climate_results_20.0deg_r10_simple.json
Output: results/climate_results_20.0deg_r10_simple_spatial_rmse.json
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
import sys
import os
import subprocess

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def check_era5_data_in_results(results_data):
    """Check if ERA5 data is already present in the results file"""
    
    # Check metadata for ERA5 flag
    if results_data.get('metadata', {}).get('era5_climatology_added'):
        return True
    
    # Check if any point has ERA5 data
    for result in results_data['results']:
        point_info = result['point_info']
        if 'era5_temp_mean' in point_info:
            return True
    
    return False


def add_era5_data_if_missing(results_file):
    """Add ERA5 data to results file if not present"""
    print(f"Loading results file: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    if check_era5_data_in_results(results_data):
        print("ERA5 data already present in results file")
        return results_file
    
    print("ERA5 data not found, adding it now...")
    
    # Call add_era5_to_results.py script
    try:
        subprocess.run([
            'python', 'add_era5_to_results.py', results_file
        ], capture_output=True, text=True, check=True)
        
        print("ERA5 data added successfully")
        
        # Return the new filename with _era5 suffix
        results_path = Path(results_file)
        era5_file = results_path.parent / (results_path.stem + '_era5.json')
        
        if era5_file.exists():
            return str(era5_file)
        else:
            print("Warning: ERA5 file not found where expected, using original file")
            return results_file
            
    except subprocess.CalledProcessError as e:
        print(f"Error adding ERA5 data: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return results_file
    except FileNotFoundError:
        print("add_era5_to_results.py not found, continuing without ERA5 data")
        return results_file


def load_results_with_era5(results_file):
    """Load results file ensuring ERA5 data is present"""
    
    # Add ERA5 data if missing
    era5_results_file = add_era5_data_if_missing(results_file)
    
    print(f"Loading results from: {era5_results_file}")
    with open(era5_results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Loaded {len(results_data['results'])} result points")
    
    # Verify ERA5 data is present
    era5_present = check_era5_data_in_results(results_data)
    print(f"ERA5 data present: {era5_present}")
    
    return results_data, era5_results_file


def create_coordinate_grid_from_results(results_data):
    """Create coordinate grid from results data"""
    
    # Extract all coordinates
    lats = []
    lons = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        lats.append(point_info['lat'])
        lons.append(point_info['lon'])
    
    # Get unique coordinates and sort
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


def create_results_lookup_from_data(results_data):
    """Create lookup dictionary from results data - land points with valid LLM and ERA5 data"""
    results_dict = {}
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            lat = point_info['lat']
            lon = point_info['lon']
            key = (lat, lon)
            
            results_dict[key] = {
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'temp_diff': point_info.get('temp_difference', 
                                          point_info['llm_temp_mean'] - point_info['era5_temp_mean'])
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


def calculate_spatial_rmse_for_point(center_lat, center_lon, results_dict, unique_lats, unique_lons, radius=4):
    """Calculate spatial RMSE for a neighborhood around a center point"""
    
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
    min_neighbors = max(5, (2*radius+1)**2 // 4)  # At least 1/4 of the grid or 5 points
    
    if len(llm_temps) >= min_neighbors:
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


def extend_results_with_spatial_rmse(results_data, radii=[2, 4]):
    """Extend each result point with spatial RMSE data"""
    
    print(f"Calculating spatial RMSE for radii: {radii}")
    
    # Create coordinate grid and results lookup
    unique_lats, unique_lons, _, _ = create_coordinate_grid_from_results(results_data)
    results_dict = create_results_lookup_from_data(results_data)
    
    if not results_dict:
        print("No valid data points found for spatial RMSE calculation")
        return results_data
    
    # Process each result point
    processed_count = 0
    total_points = len(results_data['results'])
    
    for result in results_data['results']:
        point_info = result['point_info']
        lat, lon = point_info['lat'], point_info['lon']
        
        # Calculate spatial RMSE for each radius
        for radius in radii:
            spatial_stats = calculate_spatial_rmse_for_point(
                lat, lon, results_dict, unique_lats, unique_lons, radius
            )
            
            if spatial_stats is not None:
                # Add spatial RMSE data to point_info
                prefix = f'spatial_r{radius}_'
                point_info[f'{prefix}rmse'] = spatial_stats['rmse']
                point_info[f'{prefix}mae'] = spatial_stats['mae']
                point_info[f'{prefix}bias'] = spatial_stats['bias']
                point_info[f'{prefix}correlation'] = spatial_stats['correlation']
                point_info[f'{prefix}n_neighbors'] = spatial_stats['n_neighbors']
                point_info[f'{prefix}neighborhood_llm_mean'] = spatial_stats['llm_mean']
                point_info[f'{prefix}neighborhood_era5_mean'] = spatial_stats['era5_mean']
            else:
                # Add NaN values if calculation failed
                prefix = f'spatial_r{radius}_'
                point_info[f'{prefix}rmse'] = np.nan
                point_info[f'{prefix}mae'] = np.nan
                point_info[f'{prefix}bias'] = np.nan
                point_info[f'{prefix}correlation'] = np.nan
                point_info[f'{prefix}n_neighbors'] = 0
                point_info[f'{prefix}neighborhood_llm_mean'] = np.nan
                point_info[f'{prefix}neighborhood_era5_mean'] = np.nan
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{total_points} points ({100*processed_count/total_points:.1f}%)")
    
    # Update metadata
    results_data['metadata']['spatial_rmse_added'] = True
    results_data['metadata']['spatial_rmse_radii'] = radii
    results_data['metadata']['spatial_rmse_processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Successfully added spatial RMSE data to {processed_count} points")
    
    return results_data


def save_extended_results(results_data, original_file):
    """Save the extended results to a new file with spatial_rmse suffix"""
    
    # Generate output filename
    original_path = Path(original_file)
    
    # Remove _era5 suffix if present and add spatial_rmse
    stem = original_path.stem
    if stem.endswith('_era5'):
        stem = stem[:-5]  # Remove _era5
    
    output_file = original_path.parent / (stem + '_spatial_rmse.json')
    
    print(f"Saving extended results to: {output_file}")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"Extended results saved successfully!")
    return output_file


def print_spatial_rmse_summary(results_data, radii=[2, 4]):
    """Print summary statistics of the spatial RMSE calculations"""
    
    results = results_data['results']
    
    print(f"\nSpatial RMSE Extension Summary:")
    print(f"=" * 60)
    print(f"Total points processed: {len(results)}")
    
    for radius in radii:
        prefix = f'spatial_r{radius}_'
        rmse_key = f'{prefix}rmse'
        n_neighbors_key = f'{prefix}n_neighbors'
        
        # Collect valid RMSE values
        rmse_values = []
        neighbor_counts = []
        
        for result in results:
            point_info = result['point_info']
            
            if rmse_key in point_info and not np.isnan(point_info.get(rmse_key, np.nan)):
                rmse_values.append(point_info[rmse_key])
                neighbor_counts.append(point_info.get(n_neighbors_key, 0))
        
        print(f"\nRadius {radius} (grid size: {2*radius+1}×{2*radius+1}):")
        print(f"  Points with valid spatial RMSE: {len(rmse_values)}")
        
        if rmse_values:
            print(f"  Mean spatial RMSE: {np.mean(rmse_values):.3f}°C")
            print(f"  Median spatial RMSE: {np.median(rmse_values):.3f}°C")
            print(f"  RMSE range: {np.min(rmse_values):.3f} - {np.max(rmse_values):.3f}°C")
            print(f"  Mean neighbors used: {np.mean(neighbor_counts):.1f}")
            print(f"  Neighbor range: {min(neighbor_counts)} - {max(neighbor_counts)}")


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_20.0deg_r10_simple.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Extend Results with Spatial RMSE")
    print("=" * 60)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # Load results and ensure ERA5 data is present
        results_data, final_results_file = load_results_with_era5(results_file)
        
        # Extend with spatial RMSE data (radius 2 and 4)
        print("\nCalculating spatial RMSE data...")
        extended_results = extend_results_with_spatial_rmse(results_data, radii=[2, 4])
        
        # Save extended results
        output_file = save_extended_results(extended_results, final_results_file)
        
        # Print summary
        print_spatial_rmse_summary(extended_results, radii=[2, 4])
        
        print(f"\nProcess completed successfully!")
        print(f"Extended results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()