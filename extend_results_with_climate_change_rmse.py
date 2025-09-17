#!/usr/bin/env python3
"""
Extend LLM result files with spatial RMSE data for climate change experiments.

This script takes an LLM result file from climate change difference experiments
and adds spatial RMSE calculations using the ERA5 climate change signal from
t2m_climatology_multi_period.nc (t2m_change_mean field).

The script is designed for results files containing temperature differences between
two time periods (e.g., climate_results_*_diff.json).

New fields added to each point_info:
- spatial_r2_rmse: RMSE calculated using 5×5 neighborhood (radius 2)
- spatial_r2_mae: Mean Absolute Error for 5×5 neighborhood
- spatial_r2_bias: Bias (LLM - ERA5 change signal) for 5×5 neighborhood
- spatial_r2_correlation: Correlation coefficient for 5×5 neighborhood
- spatial_r2_n_neighbors: Number of valid neighbors used in 5×5 calculation
- spatial_r2_neighborhood_llm_mean: Mean LLM temperature difference in 5×5 neighborhood
- spatial_r2_neighborhood_era5_mean: Mean ERA5 change signal in 5×5 neighborhood

- spatial_r4_rmse: RMSE calculated using 9×9 neighborhood (radius 4)
- spatial_r4_mae: Mean Absolute Error for 9×9 neighborhood
- spatial_r4_bias: Bias (LLM - ERA5 change signal) for 9×9 neighborhood
- spatial_r4_correlation: Correlation coefficient for 9×9 neighborhood
- spatial_r4_n_neighbors: Number of valid neighbors used in 9×9 calculation
- spatial_r4_neighborhood_llm_mean: Mean LLM temperature difference in 9×9 neighborhood
- spatial_r4_neighborhood_era5_mean: Mean ERA5 change signal in 9×9 neighborhood

Usage:
    python extend_results_with_climate_change_rmse.py [results_file] [era5_climatology_file]

Default results file: results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff.json
Default ERA5 file: data/t2m_climatology_multi_period.nc
Output: results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff_climate_rmse.json
"""

import numpy as np
import json
import pandas as pd
import xarray as xr
from pathlib import Path
import sys


def load_era5_climate_change_data(era5_file):
    """Load ERA5 climate change signal data (t2m_change_mean)"""
    print(f"Loading ERA5 climate change data from: {era5_file}")
    
    if not Path(era5_file).exists():
        raise FileNotFoundError(f"ERA5 file not found: {era5_file}")
    
    # Load the NetCDF file
    ds = xr.open_dataset(era5_file)
    
    # Check if t2m_change_mean field exists
    if 't2m_change_mean' not in ds:
        raise ValueError(f"t2m_change_mean field not found in {era5_file}")
    
    print(f"ERA5 climate change data loaded successfully")
    print(f"Shape: {ds.t2m_change_mean.shape}")
    print(f"Months available: {ds.month.values}")
    
    return ds


def interpolate_era5_to_point(era5_ds, lat, lon, month=7):
    """Interpolate ERA5 climate change data to a specific point"""
    try:
        # Select the month (default July = 7)
        era5_month = era5_ds.t2m_change_mean.sel(month=month)
        
        # Convert longitude from -180:180 to 0:360 if needed
        lon_era5 = lon
        if lon < 0:
            lon_era5 = lon + 360
        
        # Interpolate to the exact lat/lon
        era5_value = era5_month.interp(latitude=lat, longitude=lon_era5, method='linear')
        
        # Extract the scalar value
        era5_temp = float(era5_value.values)
        
        # Check for valid value
        if np.isnan(era5_temp) or np.isinf(era5_temp):
            return None
        
        return era5_temp
        
    except Exception as e:
        print(f"Error interpolating ERA5 data for lat={lat}, lon={lon}: {e}")
        return None


def calculate_llm_temp_mean_for_point(result):
    """Calculate mean temperature from LLM responses for a result point"""
    temps = []
    
    for response in result.get('llm_responses', []):
        parsed_data = response.get('parsed_data', {})
        if 'july_temp_mean' in parsed_data:
            temp = parsed_data['july_temp_mean']
            if not np.isnan(temp):
                temps.append(temp)
    
    if temps:
        return np.mean(temps)
    else:
        return np.nan


def add_era5_climate_change_to_results(results_data, era5_ds, month=7):
    """Add ERA5 climate change data and LLM temperature means to each result point"""
    print(f"Adding ERA5 climate change data for month {month} to results...")
    
    added_count = 0
    llm_temp_added = 0
    total_points = len(results_data['results'])
    
    for result in results_data['results']:
        point_info = result['point_info']
        lat = point_info['lat']
        lon = point_info['lon']
        
        # Calculate LLM temperature mean and add to point_info
        llm_temp_mean = calculate_llm_temp_mean_for_point(result)
        point_info['llm_temp_mean'] = llm_temp_mean
        if not np.isnan(llm_temp_mean):
            llm_temp_added += 1
        
        # Get ERA5 climate change signal
        era5_change = interpolate_era5_to_point(era5_ds, lat, lon, month)
        
        if era5_change is not None:
            point_info['era5_change_signal'] = era5_change
            added_count += 1
        else:
            point_info['era5_change_signal'] = np.nan
    
    print(f"Added LLM temperature means to {llm_temp_added}/{total_points} points")
    print(f"Added ERA5 climate change data to {added_count}/{total_points} points")
    
    # Update metadata
    results_data['metadata']['era5_climate_change_added'] = True
    results_data['metadata']['era5_climate_change_month'] = month
    results_data['metadata']['llm_temp_mean_calculated'] = True
    results_data['metadata']['era5_climate_change_processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return results_data


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
    """Create lookup dictionary from results data - land points with valid LLM and ERA5 change data"""
    results_dict = {}
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 climate change data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_change_signal' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_change_signal', np.nan))):
            
            lat = point_info['lat']
            lon = point_info['lon']
            key = (lat, lon)
            
            results_dict[key] = {
                'llm_temp_diff': point_info['llm_temp_mean'],  # This is the LLM-predicted temperature difference
                'era5_change_signal': point_info['era5_change_signal'],  # ERA5 climate change signal
                'difference': point_info['llm_temp_mean'] - point_info['era5_change_signal']
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
    """Calculate spatial RMSE for a neighborhood around a center point for climate change signals"""
    
    # Get neighborhood coordinates
    neighbors = get_neighborhood_coordinates(
        center_lat, center_lon, unique_lats, unique_lons, radius
    )
    
    # Collect valid neighbor data
    llm_diffs = []
    era5_changes = []
    
    for neighbor_lat, neighbor_lon in neighbors:
        key = (neighbor_lat, neighbor_lon)
        if key in results_dict:
            data = results_dict[key]
            llm_diffs.append(data['llm_temp_diff'])
            era5_changes.append(data['era5_change_signal'])
    
    # Calculate RMSE if we have enough neighbors
    min_neighbors = max(5, (2*radius+1)**2 // 4)  # At least 1/4 of the grid or 5 points
    
    if len(llm_diffs) >= min_neighbors:
        llm_array = np.array(llm_diffs)
        era5_array = np.array(era5_changes)
        rmse = np.sqrt(np.mean((llm_array - era5_array)**2))
        mae = np.mean(np.abs(llm_array - era5_array))
        bias = np.mean(llm_array - era5_array)
        correlation = np.corrcoef(llm_array, era5_array)[0, 1] if len(llm_diffs) > 1 else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'correlation': correlation,
            'n_neighbors': len(llm_diffs),
            'llm_mean': np.mean(llm_array),
            'era5_mean': np.mean(era5_array)
        }
    else:
        return None


def extend_results_with_climate_change_rmse(results_data, era5_ds, radii=[2, 4], month=7):
    """Extend each result point with spatial RMSE data for climate change signals"""
    
    print(f"Calculating spatial RMSE for climate change signals, radii: {radii}")
    
    # Add ERA5 climate change data to results
    results_data = add_era5_climate_change_to_results(results_data, era5_ds, month)
    
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
    results_data['metadata']['climate_change_spatial_rmse_added'] = True
    results_data['metadata']['climate_change_spatial_rmse_radii'] = radii
    results_data['metadata']['climate_change_spatial_rmse_month'] = month
    results_data['metadata']['climate_change_spatial_rmse_processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Successfully added climate change spatial RMSE data to {processed_count} points")
    
    return results_data


def save_extended_results(results_data, original_file):
    """Save the extended results to a new file with climate_rmse suffix"""
    
    # Generate output filename
    original_path = Path(original_file)
    
    # Add climate_rmse suffix
    stem = original_path.stem
    output_file = original_path.parent / (stem + '_climate_rmse.json')
    
    print(f"Saving extended results to: {output_file}")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"Extended results saved successfully!")
    return output_file


def print_climate_change_rmse_summary(results_data, radii=[2, 4]):
    """Print summary statistics of the climate change spatial RMSE calculations"""
    
    results = results_data['results']
    
    print(f"\nClimate Change Spatial RMSE Extension Summary:")
    print(f"=" * 70)
    print(f"Total points processed: {len(results)}")
    
    # Overall statistics
    era5_changes = []
    llm_diffs = []
    
    for result in results:
        point_info = result['point_info']
        if ('era5_change_signal' in point_info and 
            'llm_temp_mean' in point_info and
            not np.isnan(point_info.get('era5_change_signal', np.nan)) and
            not np.isnan(point_info.get('llm_temp_mean', np.nan))):
            era5_changes.append(point_info['era5_change_signal'])
            llm_diffs.append(point_info['llm_temp_mean'])
    
    if era5_changes:
        print(f"\nOverall Climate Change Signal Statistics:")
        print(f"  ERA5 change signal range: {np.min(era5_changes):.3f} to {np.max(era5_changes):.3f}°C")
        print(f"  ERA5 change signal mean: {np.mean(era5_changes):.3f}°C")
        print(f"  LLM prediction range: {np.min(llm_diffs):.3f} to {np.max(llm_diffs):.3f}°C")
        print(f"  LLM prediction mean: {np.mean(llm_diffs):.3f}°C")
        print(f"  Overall bias (LLM - ERA5): {np.mean(llm_diffs) - np.mean(era5_changes):.3f}°C")
        
        if len(era5_changes) > 1:
            overall_correlation = np.corrcoef(llm_diffs, era5_changes)[0, 1]
            print(f"  Overall correlation: {overall_correlation:.3f}")
    
    # Spatial RMSE statistics by radius
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
    
    # Default files
    default_results = 'results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff.json'
    default_era5 = 'data/t2m_climatology_multi_period.nc'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    era5_file = sys.argv[2] if len(sys.argv) > 2 else default_era5
    
    print("Extend Results with Climate Change Spatial RMSE")
    print("=" * 70)
    print(f"Results file: {results_file}")
    print(f"ERA5 climatology file: {era5_file}")
    print()
    
    # Check if files exist
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    if not Path(era5_file).exists():
        print(f"Error: ERA5 file '{era5_file}' not found.")
        return
    
    try:
        # Load results file
        print(f"Loading results from: {results_file}")
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        print(f"Loaded {len(results_data['results'])} result points")
        
        # Load ERA5 climate change data
        era5_ds = load_era5_climate_change_data(era5_file)
        
        # Extend with climate change spatial RMSE data (radius 2 and 4)
        print("\nCalculating climate change spatial RMSE data...")
        extended_results = extend_results_with_climate_change_rmse(
            results_data, era5_ds, radii=[2, 4], month=7
        )
        
        # Save extended results
        output_file = save_extended_results(extended_results, results_file)
        
        # Print summary
        print_climate_change_rmse_summary(extended_results, radii=[2, 4])
        
        print(f"\nProcess completed successfully!")
        print(f"Extended results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()