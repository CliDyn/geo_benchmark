### File: add_bathymetry_to_results.py
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

### File: add_era5_to_results.py
#!/usr/bin/env python3
"""
Add ERA5 climatology data to existing LLM result files.

This script takes an existing LLM result file and adds ERA5 climatology values
for each point, creating a new file with '_era5' suffix.

Usage:
    python add_era5_to_results.py [results_file] [era5_file]

Default files:
    - Results: results/climate_results_20.0deg_r10_simple.json
    - ERA5: data/t2m_climatology_1991-2020.nc
"""

import numpy as np
import pandas as pd
import xarray as xr
import json
from pathlib import Path
import sys


def load_llm_results(results_file):
    """Load LLM results file and extract basic info"""
    print(f"Loading LLM results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    return results_data


def extract_month_from_results(results_data):
    """Extract the month from LLM results by examining temperature keys"""
    month = None
    
    for result in results_data['results']:
        for response in result.get('llm_responses', []):
            if response and 'parsed_data' in response:
                parsed = response['parsed_data']
                # Find temperature key (e.g., 'july_temp_mean', 'january_temp_mean')
                temp_key = next((k for k in parsed.keys() if k.endswith('_temp_mean')), None)
                if temp_key:
                    month = temp_key.split('_')[0]
                    break
        if month:
            break
    
    print(f"Detected month: {month}")
    return month


def extract_era5_for_point(ds, lat, lon, month_num):
    """Extract ERA5 climatology data for a single point"""
    try:
        # Convert longitude from -180-180 to 0-360 if needed
        era5_lon = lon if lon >= 0 else lon + 360
        
        # Select nearest grid point
        point_data = ds.sel(latitude=lat, longitude=era5_lon, month=month_num, method='nearest')
        
        return {
            'era5_temp_mean': float(point_data['t2m_mean'].values),
            'era5_temp_min': float(point_data['t2m_min'].values),
            'era5_temp_max': float(point_data['t2m_max'].values),
            'era5_temp_std': float(point_data['t2m_std'].values)
        }
        
    except Exception as e:
        print(f"Warning: Could not extract ERA5 data for ({lat}, {lon}): {e}")
        return {
            'era5_temp_mean': np.nan,
            'era5_temp_min': np.nan,
            'era5_temp_max': np.nan,
            'era5_temp_std': np.nan
        }


def add_era5_to_results(results_data, era5_file):
    """Add ERA5 climatology data to each result point"""
    print(f"Loading ERA5 climatology from: {era5_file}")
    
    # Load ERA5 dataset
    ds = xr.open_dataset(era5_file)
    
    # Extract month from results
    month = extract_month_from_results(results_data)
    if not month:
        print("Warning: Could not determine month from results, defaulting to July")
        month = 'july'
    
    # Month name to number mapping
    month_mapping = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    month_num = month_mapping.get(month.lower(), 7)  # Default to July
    print(f"Using month number: {month_num}")
    
    # Process each result point
    print("Adding ERA5 data to result points...")
    for i, result in enumerate(results_data['results']):
        point_info = result['point_info']
        lat, lon = point_info['lat'], point_info['lon']
        
        # Extract ERA5 data for this point
        era5_data = extract_era5_for_point(ds, lat, lon, month_num)
        
        # Add ERA5 data to the point_info
        point_info.update(era5_data)
        
        # Also calculate temperature difference if we have LLM data
        llm_temps = []
        for response in result.get('llm_responses', []):
            if response and 'parsed_data' in response:
                parsed = response['parsed_data']
                temp_key = next((k for k in parsed.keys() if k.endswith('_temp_mean')), None)
                if temp_key:
                    llm_temps.append(parsed[temp_key])
        
        if llm_temps and not np.isnan(era5_data['era5_temp_mean']):
            llm_mean = np.mean(llm_temps)
            point_info['llm_temp_mean'] = llm_mean
            point_info['llm_temp_std'] = np.std(llm_temps) if len(llm_temps) > 1 else 0.0
            point_info['llm_temp_count'] = len(llm_temps)
            point_info['temp_difference'] = llm_mean - era5_data['era5_temp_mean']
            point_info['abs_difference'] = abs(point_info['temp_difference'])
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(results_data['results'])} points")
    
    # Update metadata
    results_data['metadata']['era5_climatology_added'] = True
    results_data['metadata']['era5_period'] = '1991-2020'
    results_data['metadata']['era5_file'] = str(era5_file)
    results_data['metadata']['processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    results_data['metadata']['month_processed'] = month
    
    print("ERA5 data successfully added to all result points")
    return results_data


def save_results_with_era5(results_data, original_file):
    """Save the enhanced results to a new file with '_era5' suffix"""
    # Generate output filename
    original_path = Path(original_file)
    output_file = original_path.parent / (original_path.stem + '_era5.json')
    
    print(f"Saving enhanced results to: {output_file}")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"Enhanced results saved successfully!")
    return output_file


def print_summary(results_data):
    """Print summary statistics of the processed data"""
    results = results_data['results']
    
    # Count valid data points
    valid_llm = 0
    valid_era5 = 0
    valid_both = 0
    temp_differences = []
    
    for result in results:
        point_info = result['point_info']
        
        has_llm = 'llm_temp_mean' in point_info and not np.isnan(point_info.get('llm_temp_mean', np.nan))
        has_era5 = 'era5_temp_mean' in point_info and not np.isnan(point_info.get('era5_temp_mean', np.nan))
        
        if has_llm:
            valid_llm += 1
        if has_era5:
            valid_era5 += 1
        if has_llm and has_era5:
            valid_both += 1
            temp_differences.append(point_info['temp_difference'])
    
    print(f"\nSummary Statistics:")
    print(f"=" * 50)
    print(f"Total points: {len(results)}")
    print(f"Points with LLM data: {valid_llm}")
    print(f"Points with ERA5 data: {valid_era5}")
    print(f"Points with both LLM and ERA5: {valid_both}")
    
    if temp_differences:
        print(f"\nTemperature Comparison (LLM vs ERA5):")
        print(f"Mean difference: {np.mean(temp_differences):+.2f}°C")
        print(f"Std deviation: {np.std(temp_differences):.2f}°C")
        print(f"RMSE: {np.sqrt(np.mean(np.array(temp_differences)**2)):.2f}°C")
        print(f"MAE: {np.mean(np.abs(temp_differences)):.2f}°C")


def main():
    """Main function"""
    # Default files
    default_results = 'results/climate_results_20.0deg_r10_simple.json'
    default_era5 = 'data/t2m_climatology_1991-2020.nc'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    era5_file = sys.argv[2] if len(sys.argv) > 2 else default_era5
    
    print("Add ERA5 to Results Tool")
    print("=" * 50)
    print(f"Results file: {results_file}")
    print(f"ERA5 file: {era5_file}")
    print()
    
    # Check if files exist
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    if not Path(era5_file).exists():
        print(f"Error: ERA5 file '{era5_file}' not found.")
        return
    
    try:
        # Load LLM results
        results_data = load_llm_results(results_file)
        
        # Add ERA5 data
        enhanced_results = add_era5_to_results(results_data, era5_file)
        
        # Save enhanced results
        output_file = save_results_with_era5(enhanced_results, results_file)
        
        # Print summary
        print_summary(enhanced_results)
        
        print(f"\nProcess completed successfully!")
        print(f"Enhanced results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

### File: add_population_to_results.py
#!/usr/bin/env python3
"""
Add population density data to existing result files.

This script reads population density data from ASCII grid format and adds it to 
existing result files by matching coordinates using nearest neighbor interpolation.

Population data format:
- File: data/gpw_v4_population_density_rev11_2000_1_deg.asc  
- Projection: GEOGRAPHIC, Datum: WGS84, Units: DD
- Grid: 1 degree resolution (360x180 cells)
- Coverage: Global (-180 to 180 lon, -90 to 90 lat)
- NODATA_value: -9999

Usage:
    python add_population_to_results.py [results_file] [population_file]

Default files:
    - Results: results/climate_results_20.0deg_r10_simple_spatial_rmse.json
    - Population: data/gpw_v4_population_density_rev11_2000_1_deg.asc

Output: results/climate_results_20.0deg_r10_simple_spatial_rmse_population.json
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import os


def read_ascii_grid(ascii_file):
    """Read ASCII grid file and return header info and data array"""
    print(f"Reading ASCII grid file: {ascii_file}")
    
    # Read header
    header = {}
    with open(ascii_file, 'r') as f:
        for i in range(6):  # Standard ASCII grid has 6 header lines
            line = f.readline().strip().split()
            header[line[0].lower()] = float(line[1]) if '.' in line[1] or 'e' in line[1].lower() else int(line[1])
    
    print(f"Grid header: {header}")
    
    # Read data using numpy
    data = np.loadtxt(ascii_file, skiprows=6)
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {np.min(data)} to {np.max(data)}")
    print(f"NODATA values: {np.sum(data == header['nodata_value'])}")
    
    return header, data


def create_coordinate_arrays(header):
    """Create coordinate arrays from ASCII grid header"""
    ncols = int(header['ncols'])
    nrows = int(header['nrows'])
    xllcorner = header['xllcorner']
    yllcorner = header['yllcorner']
    cellsize = header['cellsize']
    
    # Create coordinate arrays (center of each cell)
    lons = np.arange(xllcorner + cellsize/2, xllcorner + ncols*cellsize, cellsize)
    lats = np.arange(yllcorner + cellsize/2, yllcorner + nrows*cellsize, cellsize)
    
    # Note: ASCII grid data is stored top-to-bottom, but coordinates go bottom-to-top
    # So we need to reverse the latitude array to match the data order
    lats = lats[::-1]
    
    print(f"Longitude range: {lons[0]:.1f} to {lons[-1]:.1f} ({len(lons)} points)")
    print(f"Latitude range: {lats[-1]:.1f} to {lats[0]:.1f} ({len(lats)} points)")
    
    return lons, lats


def get_population_at_coordinates(target_lat, target_lon, lons, lats, data, nodata_value):
    """Get population density at target coordinates using nearest neighbor"""
    
    # Find nearest grid indices
    lon_idx = np.argmin(np.abs(lons - target_lon))
    lat_idx = np.argmin(np.abs(lats - target_lat))
    
    # Extract population value
    pop_value = data[lat_idx, lon_idx]
    
    # Convert NODATA to NaN
    if pop_value == nodata_value:
        pop_value = np.nan
    
    return pop_value


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
    
    print(f"Data present - ERA5: {has_era5}, Spatial RMSE: {has_spatial}, Population: {has_population}")
    
    if has_population:
        print("Warning: Population data already present in results file")
    
    return results_data


def add_population_to_results(results_data, population_file):
    """Add population density data to each result point"""
    
    # Load population data
    header, data = read_ascii_grid(population_file)
    lons, lats = create_coordinate_arrays(header)
    nodata_value = header['nodata_value']
    
    print(f"\nAdding population data to result points...")
    
    # Process each result point
    population_stats = []
    processed_count = 0
    
    for i, result in enumerate(results_data['results']):
        point_info = result['point_info']
        target_lat = point_info['lat']
        target_lon = point_info['lon']
        
        # Get population density at this point
        pop_density = get_population_at_coordinates(
            target_lat, target_lon, lons, lats, data, nodata_value
        )
        
        # Add population data to point_info
        point_info['population_density'] = pop_density
        
        # Collect statistics
        if not np.isnan(pop_density):
            population_stats.append(pop_density)
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count}/{len(results_data['results'])} points")
    
    # Update metadata
    results_data['metadata']['population_added'] = True
    results_data['metadata']['population_file'] = str(population_file)
    results_data['metadata']['population_processing_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    results_data['metadata']['population_grid_resolution'] = header['cellsize']
    results_data['metadata']['population_nodata_value'] = nodata_value
    
    print(f"\nPopulation data successfully added to {processed_count} result points")
    
    # Print population statistics
    if population_stats:
        print(f"\nPopulation Statistics:")
        print(f"Points with population data: {len(population_stats)}")
        print(f"Mean population density: {np.mean(population_stats):.2f} people/km²")
        print(f"Median population density: {np.median(population_stats):.2f} people/km²")
        print(f"Population range: {np.min(population_stats):.2f} - {np.max(population_stats):.2f} people/km²")
        print(f"Points with zero population: {np.sum(np.array(population_stats) == 0)}")
    else:
        print("Warning: No valid population data found for any points")
    
    return results_data


def save_results_with_population(results_data, original_file):
    """Save the enhanced results to a new file with population suffix"""
    
    # Generate output filename
    original_path = Path(original_file)
    output_file = original_path.parent / (original_path.stem + '_population.json')
    
    print(f"\nSaving enhanced results to: {output_file}")
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"Enhanced results saved successfully!")
    return output_file


def print_population_summary(results_data):
    """Print summary of population data integration"""
    
    results = results_data['results']
    
    # Collect population statistics
    all_population = []
    land_population = []
    zero_population = 0
    nodata_population = 0
    
    for result in results:
        point_info = result['point_info']
        pop_density = point_info.get('population_density', np.nan)
        
        if not np.isnan(pop_density):
            all_population.append(pop_density)
            if point_info.get('is_land', False):
                land_population.append(pop_density)
            if pop_density == 0:
                zero_population += 1
        else:
            nodata_population += 1
    
    print(f"\nPopulation Data Integration Summary:")
    print(f"=" * 60)
    print(f"Total result points: {len(results)}")
    print(f"Points with population data: {len(all_population)}")
    print(f"Points with no data (NODATA): {nodata_population}")
    print(f"Points with zero population: {zero_population}")
    
    if all_population:
        print(f"\nPopulation Density Statistics (all points):")
        print(f"  Mean: {np.mean(all_population):.2f} people/km²")
        print(f"  Median: {np.median(all_population):.2f} people/km²")
        print(f"  Standard deviation: {np.std(all_population):.2f} people/km²")
        print(f"  Range: {np.min(all_population):.2f} - {np.max(all_population):.2f} people/km²")
    
    if land_population:
        print(f"\nPopulation Density Statistics (land points only):")
        print(f"  Count: {len(land_population)}")
        print(f"  Mean: {np.mean(land_population):.2f} people/km²")
        print(f"  Median: {np.median(land_population):.2f} people/km²")
        print(f"  Range: {np.min(land_population):.2f} - {np.max(land_population):.2f} people/km²")


def main():
    """Main function"""
    
    # Default files
    default_results = 'results/climate_results_20.0deg_r10_simple_spatial_rmse.json'
    default_population = 'data/gpw_v4_population_density_rev11_2000_1_deg.asc'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    population_file = sys.argv[2] if len(sys.argv) > 2 else default_population
    
    print("Add Population Data to Results")
    print("=" * 60)
    print(f"Results file: {results_file}")
    print(f"Population file: {population_file}")
    print()
    
    # Check if files exist
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    if not Path(population_file).exists():
        print(f"Error: Population file '{population_file}' not found.")
        print("Please ensure the population ASCII grid file is available in the data/ directory.")
        return
    
    try:
        # Load results file
        results_data = load_results_file(results_file)
        
        # Add population data
        enhanced_results = add_population_to_results(results_data, population_file)
        
        # Save enhanced results
        output_file = save_results_with_population(enhanced_results, results_file)
        
        # Print summary
        print_population_summary(enhanced_results)
        
        print(f"\nProcess completed successfully!")
        print(f"Enhanced results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

### File: aggregate_bathymetry.py
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

### File: analyze_country_performance.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
from pathlib import Path
from collections import defaultdict
import sys
import os

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_llm_results_and_add_era5(results_file, climatology_file='data/t2m_climatology_1991-2020.nc'):
    """Load LLM results and add ERA5 comparison on the fly"""
    print(f"Loading LLM results from {results_file}...")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is already an ERA5 comparison file
    if 'comparison_results' in data:
        print("File already contains ERA5 comparison data")
        return data['comparison_results'], data.get('metadata', {})
    
    # Otherwise, load original LLM results and add ERA5 data
    print("Adding ERA5 climatology data...")
    
    try:
        # Import ERA5 comparison functions
        from compare_llm_era5 import load_llm_results, extract_era5_climatology, combine_llm_era5_data
        
        # Load LLM results using the existing function
        llm_results, metadata = load_llm_results(results_file)
        
        # Extract coordinates and month
        coordinates = [(r['lat'], r['lon']) for r in llm_results]
        month = llm_results[0]['month'] if llm_results else 'july'
        
        print(f"Processing {len(coordinates)} points for month: {month}")
        
        # Load ERA5 climatology
        era5_data = extract_era5_climatology(climatology_file, coordinates, month)
        
        # Combine data
        comparison_results = combine_llm_era5_data(llm_results, era5_data)
        
        print(f"Successfully created comparison data for {len(comparison_results)} points")
        return comparison_results, metadata
        
    except ImportError as e:
        print(f"Error importing ERA5 functions: {e}")
        print("Please ensure compare_llm_era5.py is in the same directory")
        return None, None
    except FileNotFoundError:
        print(f"ERA5 climatology file not found: {climatology_file}")
        print("Please ensure ERA5 climatology file is available")
        return None, None

def extract_valid_country_data(comparison_results):
    """Extract valid data points grouped by country"""
    country_data = defaultdict(list)
    
    for point in comparison_results:
        # Check if both LLM and ERA5 data are valid
        if (not np.isnan(point.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point.get('era5_temp_mean', np.nan))):
            
            country = point.get('country', 'Unknown')
            if country and country != 'N/A':
                country_data[country].append({
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'llm_temp': point['llm_temp_mean'],
                    'era5_temp': point['era5_temp_mean'],
                    'llm_std': point.get('llm_temp_std', np.nan),
                    'era5_std': point.get('era5_temp_std', np.nan),
                    'temp_diff': point.get('temp_difference', np.nan),
                    'llm_count': point.get('llm_temp_count', 1)
                })
    
    print(f"Found data for {len(country_data)} countries")
    
    # Sort countries by number of points (for better visualization)
    country_data = dict(sorted(country_data.items(), key=lambda x: len(x[1]), reverse=True))
    
    return country_data

def create_country_colormap(country_data):
    """Create a color map for top 15 countries by point count, others are gray"""
    # Sort countries by number of points (descending)
    sorted_countries = sorted(country_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Take top 15 countries
    top_countries = [country for country, _ in sorted_countries[:15]]
    
    # Create distinct colors for top 15 countries
    if len(top_countries) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    else:
        # Use tab20 for up to 15 countries (more distinct than other colormaps)
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_countries)))
    
    # Create color map
    color_map = {}
    gray_color = (0.6, 0.6, 0.6)  # Gray for other countries
    
    # Assign colors to top countries
    for i, country in enumerate(top_countries):
        color_map[country] = colors[i]
    
    # Assign gray to all other countries
    for country in country_data.keys():
        if country not in top_countries:
            color_map[country] = gray_color
    
    return color_map, top_countries

def plot_country_analysis(country_data, output_file=None, figsize=(16, 12)):
    """Create comprehensive country-based analysis plots"""
    
    if not country_data:
        print("No valid country data for plotting")
        return
    
    # Create color map for countries (top 15 get colors, others gray)
    color_map, top_countries = create_country_colormap(country_data)
    countries = list(country_data.keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Main scatter plot (larger subplot)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    # Collect all data for statistics
    all_llm_temps = []
    all_era5_temps = []
    all_llm_stds = []
    all_era5_stds = []
    country_labels = []
    
    # Plot data for each country
    for country, points in country_data.items():
        llm_temps = [p['llm_temp'] for p in points]
        era5_temps = [p['era5_temp'] for p in points]
        llm_stds = [p['llm_std'] if not np.isnan(p['llm_std']) else 0 for p in points]
        era5_stds = [p['era5_std'] if not np.isnan(p['era5_std']) else 0 for p in points]
        
        # Scatter plot with error bars - use higher alpha and larger markers for better visibility
        ax1.errorbar(era5_temps, llm_temps, 
                    xerr=era5_stds, yerr=llm_stds,
                    fmt='o', color=color_map[country], 
                    alpha=0.8, markersize=8, capsize=3, capthick=1.5,
                    elinewidth=1.5, markeredgewidth=0.5, markeredgecolor='black',
                    label=f'{country} (n={len(points)})')
        
        all_llm_temps.extend(llm_temps)
        all_era5_temps.extend(era5_temps)
        all_llm_stds.extend(llm_stds)
        all_era5_stds.extend(era5_stds)
        country_labels.extend([country] * len(points))
    
    # Add 1:1 line
    min_temp = min(min(all_llm_temps), min(all_era5_temps))
    max_temp = max(max(all_llm_temps), max(all_era5_temps))
    ax1.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.8, linewidth=2, label='1:1 line')
    
    # Add regression line
    coeffs = np.polyfit(all_era5_temps, all_llm_temps, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_temp, max_temp, 100)
    ax1.plot(x_reg, regression_line(x_reg), 'r-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel('ERA5 Temperature (°C)', fontsize=12)
    ax1.set_ylabel('LLM Temperature (°C)', fontsize=12)
    ax1.set_title('LLM vs ERA5 Temperature by Country\n(Top 15 countries by data points colored, others gray)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Calculate overall statistics
    rmse = np.sqrt(np.mean((np.array(all_llm_temps) - np.array(all_era5_temps))**2))
    mae = np.mean(np.abs(np.array(all_llm_temps) - np.array(all_era5_temps)))
    bias = np.mean(np.array(all_llm_temps) - np.array(all_era5_temps))
    r_corr = np.corrcoef(all_llm_temps, all_era5_temps)[0, 1]
    
    # Add statistics text
    stats_text = f'Overall Statistics\\n'
    stats_text += f'N = {len(all_llm_temps)} points\\n'
    stats_text += f'Countries = {len(countries)}\\n'
    stats_text += f'RMSE = {rmse:.2f}°C\\n'
    stats_text += f'MAE = {mae:.2f}°C\\n'
    stats_text += f'Bias = {bias:+.2f}°C\\n'
    stats_text += f'Correlation = {r_corr:.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Country legend (separate subplot)
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax2.axis('off')
    
    # Create legend for top countries only
    legend_handles = []
    legend_labels = []
    
    # Add colored countries (top 15)
    for country in top_countries:
        points = country_data[country]
        handle = plt.Line2D([0], [0], marker='o', color=color_map[country], 
                           linestyle='', markersize=10, alpha=0.8,
                           markeredgewidth=0.5, markeredgecolor='black')
        legend_handles.append(handle)
        legend_labels.append(f'{country} (n={len(points)})')
    
    # Add gray countries entry
    gray_countries_count = len(countries) - len(top_countries)
    if gray_countries_count > 0:
        gray_handle = plt.Line2D([0], [0], marker='o', color=(0.6, 0.6, 0.6), 
                                linestyle='', markersize=10, alpha=0.8,
                                markeredgewidth=0.5, markeredgecolor='black')
        legend_handles.append(gray_handle)
        legend_labels.append(f'Other countries (n={gray_countries_count})')
    
    ax2.legend(legend_handles, legend_labels, loc='center', fontsize=9, 
               title='Countries', title_fontsize=11)
    
    # Country performance summary (bottom left)
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    
    # Calculate RMSE for each country (countries with >2 points)
    country_rmse = {}
    for country, points in country_data.items():
        if len(points) >= 3:  # Only countries with 3+ points
            llm_temps = [p['llm_temp'] for p in points]
            era5_temps = [p['era5_temp'] for p in points]
            rmse_country = np.sqrt(np.mean((np.array(llm_temps) - np.array(era5_temps))**2))
            country_rmse[country] = rmse_country
    
    # Plot top 10 countries by RMSE
    if country_rmse:
        sorted_countries = sorted(country_rmse.items(), key=lambda x: x[1])[:10]
        countries_plot = [x[0] for x in sorted_countries]
        rmse_values = [x[1] for x in sorted_countries]
        colors_plot = [color_map[c] for c in countries_plot]
        
        bars = ax3.barh(range(len(countries_plot)), rmse_values, color=colors_plot, alpha=0.7)
        ax3.set_yticks(range(len(countries_plot)))
        ax3.set_yticklabels([c[:12] + ('...' if len(c) > 12 else '') for c in countries_plot], fontsize=9)
        ax3.set_xlabel('RMSE (°C)', fontsize=10)
        ax3.set_title('Best Countries\\n(Lowest RMSE)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add RMSE values on bars
        for i, (bar, rmse_val) in enumerate(zip(bars, rmse_values)):
            ax3.text(rmse_val + 0.1, i, f'{rmse_val:.1f}', 
                    va='center', ha='left', fontsize=8)
    
    # Difference histogram (bottom center)
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    
    differences = np.array(all_llm_temps) - np.array(all_era5_temps)
    ax4.hist(differences, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax4.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean = {np.mean(differences):+.2f}°C')
    
    ax4.set_xlabel('Temperature Difference (LLM - ERA5) °C', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Difference Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # Country point counts (bottom right)
    ax5 = plt.subplot2grid((3, 3), (2, 2))
    
    # Plot distribution of points per country
    point_counts = [len(points) for points in country_data.values()]
    ax5.hist(point_counts, bins=min(15, len(countries)), alpha=0.7, edgecolor='black', color='lightgreen')
    ax5.set_xlabel('Points per Country', fontsize=10)
    ax5.set_ylabel('Number of Countries', fontsize=10)
    ax5.set_title('Point Distribution', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Country analysis plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig, country_data

def calculate_country_rmse(country_data):
    """Calculate RMSE for each country with sufficient data points"""
    country_rmse = {}
    
    for country, points in country_data.items():
        if len(points) >= 2:  # Only countries with 2+ points
            llm_temps = [p['llm_temp'] for p in points]
            era5_temps = [p['era5_temp'] for p in points]
            rmse = np.sqrt(np.mean((np.array(llm_temps) - np.array(era5_temps))**2))
            country_rmse[country] = rmse
    
    return country_rmse

def get_country_name_variants(country_name):
    """Get possible name variations for country matching"""
    # Create comprehensive mapping from shapefile names to analysis data names
    name_mapping = {
        'United States of America': 'United States',
        'Russian Federation': 'Russia', 
        'Bosnia and Herz.': 'Bosnia and Herzegovina',
        'Central African Rep.': 'Central African Republic',
        'Congo': 'Congo-Brazzaville',
        'Dem. Rep. Congo': 'Democratic Republic of the Congo',
        'Dominican Rep.': 'Dominican Republic',
        'Eq. Guinea': 'Equatorial Guinea',
        'S. Sudan': 'South Sudan',
        'Solomon Is.': 'Solomon Islands',
        'Czech Republic': 'Czechia',
        'Macedonia': 'North Macedonia',
        'eSwatini': 'Eswatini',
        'Côte d\'Ivoire': 'Côte d\'Ivoire',
        'São Tomé and Principe': 'Sao Tome and Principe',
        'Timor-Leste': 'East Timor',
        'W. Sahara': 'Sahrawi Arab Democratic Republic'
    }
    
    # Return list of possible name variations
    return [
        country_name,
        name_mapping.get(country_name, country_name),
        country_name.replace(' ', ''),
        country_name.replace('United States of America', 'United States'),
        country_name.replace('Russian Federation', 'Russia'),
        country_name.replace('United Kingdom', 'United Kingdom')
    ]

def check_country_matching(country_data, countries_shapefile='data/land/ne_10m_admin_0_countries.shp'):
    """Check which countries from shapefile don't have matching data"""
    try:
        world = gpd.read_file(countries_shapefile)
        
        countries_without_data = []
        countries_with_data = []
        
        for _, country_row in world.iterrows():
            country_name = country_row['NAME']
            
            # Get possible name variations for matching
            possible_names = get_country_name_variants(country_name)
            
            found_data = False
            for name_variant in possible_names:
                if name_variant in country_data:
                    found_data = True
                    countries_with_data.append(country_name)
                    break
            
            if not found_data:
                countries_without_data.append(country_name)
        
        print(f"\nCountry Matching Summary:")
        print(f"Countries with data: {len(countries_with_data)}")
        print(f"Countries without data: {len(countries_without_data)}")
        
        if countries_without_data:
            print(f"\nCountries from shapefile that appear in GRAY (no matching data):")
            for i, country in enumerate(sorted(countries_without_data), 1):
                print(f"{i:3d}. {country}")
        
        return countries_with_data, countries_without_data
        
    except Exception as e:
        print(f"Error checking country matching: {e}")
        return [], []

def plot_country_rmse_map(country_data, countries_shapefile='data/land/ne_10m_admin_0_countries.shp', output_file=None, figsize=(16, 10)):
    """Plot world map with countries colored by RMSE values"""
    
    # Calculate RMSE for each country
    country_rmse = calculate_country_rmse(country_data)
    
    if not country_rmse:
        print("No countries with sufficient data for RMSE mapping")
        return
    
    try:
        # Load world countries shapefile
        world = gpd.read_file(countries_shapefile)
        
        # Create a colormap for RMSE values
        rmse_values = list(country_rmse.values())
        vmin = min(rmse_values)
        vmax = max(rmse_values)
        
        # Use a reversed RdYlBu colormap (blue=low RMSE/good, red=high RMSE/bad)
        cmap = plt.cm.RdYlBu_r
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all countries with light gray background
        world.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
        
        # Plot countries with RMSE data
        countries_with_data = []
        rmse_colors = []
        
        for _, country_row in world.iterrows():
            country_name = country_row['NAME']
            
            # Get possible name variations for matching
            possible_names = get_country_name_variants(country_name)
            
            rmse_val = None
            for name_variant in possible_names:
                if name_variant in country_rmse:
                    rmse_val = country_rmse[name_variant]
                    break
            
            if rmse_val is not None:
                countries_with_data.append(country_row.geometry)
                rmse_colors.append(cmap(norm(rmse_val)))
        
        # Plot countries with RMSE data using colors
        if countries_with_data:
            for geom, color in zip(countries_with_data, rmse_colors):
                gpd.GeoSeries([geom]).plot(ax=ax, color=color, edgecolor='white', linewidth=0.5)
        
        # Customize the map
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xlabel('Longitude (degrees)', fontsize=12)
        ax.set_ylabel('Latitude (degrees)', fontsize=12)
        ax.set_title(f'Country Performance: RMSE (LLM vs ERA5)\nRange: {vmin:.2f}°C to {vmax:.2f}°C', fontsize=14, fontweight='bold')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-60, 91, 30))
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=30)
        cbar.set_label('RMSE (°C)', fontsize=12)
        
        # Add statistics text
        n_countries = len(country_rmse)
        mean_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        
        stats_text = f'Countries with data: {n_countries}\n'
        stats_text += f'Mean RMSE: {mean_rmse:.2f}°C\n'
        stats_text += f'Std RMSE: {std_rmse:.2f}°C\n'
        stats_text += f'Best: {min(rmse_values):.2f}°C\n'
        stats_text += f'Worst: {max(rmse_values):.2f}°C'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save or show plot
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Country RMSE map saved to: {output_file}")
        else:
            plt.show()
        
        return fig, country_rmse
        
    except FileNotFoundError:
        print(f"Countries shapefile not found: {countries_shapefile}")
        print("Please ensure Natural Earth countries shapefile is available")
        return None, country_rmse
    except Exception as e:
        print(f"Error creating country map: {e}")
        return None, country_rmse

def print_country_summary(country_data):
    """Print detailed country performance summary"""
    print("\\nCountry Performance Summary:")
    print("=" * 80)
    
    # Calculate statistics for each country
    country_stats = []
    
    for country, points in country_data.items():
        if len(points) >= 2:  # Only countries with 2+ points
            llm_temps = [p['llm_temp'] for p in points]
            era5_temps = [p['era5_temp'] for p in points]
            differences = [p['temp_diff'] for p in points]
            
            rmse = np.sqrt(np.mean((np.array(llm_temps) - np.array(era5_temps))**2))
            mae = np.mean(np.abs(differences))
            bias = np.mean(differences)
            correlation = np.corrcoef(llm_temps, era5_temps)[0, 1] if len(points) > 2 else np.nan
            
            country_stats.append({
                'country': country,
                'points': len(points),
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'correlation': correlation,
                'mean_llm': np.mean(llm_temps),
                'mean_era5': np.mean(era5_temps)
            })
    
    # Sort by RMSE
    country_stats.sort(key=lambda x: x['rmse'])
    
    print(f"{'Country':<20} {'Points':<6} {'RMSE':<6} {'MAE':<6} {'Bias':<7} {'Corr':<6} {'LLM':<6} {'ERA5':<6}")
    print("-" * 80)
    
    for stats in country_stats:
        corr_str = f"{stats['correlation']:.3f}" if not np.isnan(stats['correlation']) else "N/A"
        print(f"{stats['country']:<20} {stats['points']:<6} {stats['rmse']:<6.2f} "
              f"{stats['mae']:<6.2f} {stats['bias']:<+7.2f} {corr_str:<6} "
              f"{stats['mean_llm']:<6.1f} {stats['mean_era5']:<6.1f}")
    
    # Overall summary
    total_points = sum(len(points) for points in country_data.values())
    all_rmse = [s['rmse'] for s in country_stats]
    
    print("\\nSummary:")
    print(f"Total countries: {len(country_data)}")
    print(f"Countries with 2+ points: {len(country_stats)}")
    print(f"Total points: {total_points}")
    print(f"Best performing country (lowest RMSE): {country_stats[0]['country']} ({country_stats[0]['rmse']:.2f}°C)")
    print(f"Worst performing country (highest RMSE): {country_stats[-1]['country']} ({country_stats[-1]['rmse']:.2f}°C)")
    print(f"Mean RMSE across countries: {np.mean(all_rmse):.2f}°C")

def main():
    """Main function"""
    
    # Default file
    results_file = 'results/climate_results_1.0deg_r10_simple.json'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    print(f"Country Performance Analysis")
    print(f"Results file: {results_file}")
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run climate_llm_benchmark.py first to generate results.")
        return
    
    try:
        # Load results and add ERA5 comparison if needed
        comparison_results, _ = load_llm_results_and_add_era5(results_file)
        
        if comparison_results is None:
            print("Failed to load or process results file.")
            return
        
        # Extract country data
        country_data = extract_valid_country_data(comparison_results)
        
        if not country_data:
            print("No valid country data found in results file.")
            return
        
        # Generate output filename
        results_path = Path(results_file)
        output_file = f"png/country_analysis_{results_path.stem.replace('_era5', '')}.png"
        
        # Create country analysis plot
        plot_country_analysis(country_data, output_file)
        
        # Check country matching between data and shapefile
        check_country_matching(country_data)
        
        # Create world map with RMSE coloring
        map_output_file = f"png/country_rmse_map_{results_path.stem.replace('_era5', '')}.png"
        plot_country_rmse_map(country_data, output_file=map_output_file)
        
        # Print detailed summary
        print_country_summary(country_data)
        
        print("\\nCountry analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: climate_llm_benchmark.py
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
from geo_mesh_processor import load_mesh_data

# Core langchain imports
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
import glob
import yaml

# Model provider imports (with optional handling)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_ollama import ChatOllama
    from langchain_community.llms import Ollama
    #from langchain_community.chat_models import ChatOllama
except ImportError:
    Ollama = None
    ChatOllama = None

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}

def get_config_value(config: Dict, key_path: str, default=None):
    """Get nested configuration value using dot notation"""
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def configure_langsmith(disable_tracing: bool = False):
    """Configure LangSmith tracing"""
    if disable_tracing:
        try:
            import langsmith as ls
            ls.configure(enabled=False)
            print("LangSmith tracing disabled")
        except ImportError:
            pass  # LangSmith not installed, ignore
    else:
        print("LangSmith tracing enabled (default)")

def find_latest_intermediate_file(resolution: str, simple_mode: bool = False, chunk_id: Optional[str] = None, with_address: bool = True, periods: bool = False) -> Optional[str]:
    """Find the latest intermediate file for resuming"""
    mode_suffix = "_simple" if simple_mode else ""
    address_suffix = "_noaddress" if not with_address else ""
    periods_suffix = "_climchange" if periods else ""
    chunk_suffix = f"_chunk_{chunk_id}" if chunk_id else ""
    pattern = f"results/climate_results_intermediate_*{chunk_suffix}*{mode_suffix}{address_suffix}{periods_suffix}.json"
    
    # Find all matching intermediate files
    intermediate_files = glob.glob(pattern)
    
    if not intermediate_files:
        return None
    
    # Extract numbers and sort to find the latest
    file_numbers = []
    for file_path in intermediate_files:
        try:
            # Extract number from filename like "climate_results_intermediate_1840_chunk_01_simple.json"
            filename = Path(file_path).stem
            
            # Remove prefixes and suffixes to extract the number
            temp_name = filename.replace("climate_results_intermediate_", "")
            if chunk_id:
                temp_name = temp_name.replace(f"_chunk_{chunk_id}", "")
            if simple_mode:
                temp_name = temp_name.replace("_simple", "")
            if periods:
                temp_name = temp_name.replace("_climchange", "")
            
            # Extract any remaining model name parts and the number
            parts = temp_name.split("_")
            # The number should be the first part
            number = int(parts[0])
            file_numbers.append((number, file_path))
        except (ValueError, IndexError):
            continue
    
    if not file_numbers:
        return None
    
    # Sort by number and return the path of the latest file
    file_numbers.sort(key=lambda x: x[0], reverse=True)
    latest_file = file_numbers[0][1]
    
    return latest_file

def load_intermediate_results(intermediate_file: str) -> tuple[List[Dict], Dict, int]:
    """Load intermediate results and return results, mesh_data, and start_index"""
    print(f"Loading intermediate results from {intermediate_file}...")
    
    with open(intermediate_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Reconstruct mesh_data from the saved data
    mesh_data = {
        'mesh_info': data['mesh_info'],
        'resolution': data['resolution'],
        'mesh_points': []  # Will be loaded from the original mesh file
    }
    
    start_index = len(results)
    print(f"Found {start_index} completed points, resuming from point {start_index + 1}")
    
    return results, mesh_data, start_index

def initialize_llm(config: Dict, model_name: str = None, temperature: float = None, simple_mode: bool = None):
    """Initialize LLM based on configuration and provider"""
    
    # Get values from config or use provided values
    provider = get_config_value(config, 'model.provider', 'openai')
    model_name = model_name or get_config_value(config, 'model.name', 'gpt-5-nano')
    temperature = temperature if temperature is not None else get_config_value(config, 'model.temperature', 0)
    max_tokens = get_config_value(config, 'model.max_tokens', 300 if simple_mode else None)
    timeout = get_config_value(config, 'model.timeout', 30)
    
    print(f"Initializing {provider} model: {model_name}")
    
    if provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai not installed. Install with: pip install langchain-openai")
        
        api_key_env = get_config_value(config, 'providers.openai.api_key_env', 'OPENAI_API_KEY')
        base_url = get_config_value(config, 'providers.openai.base_url')
        organization = get_config_value(config, 'providers.openai.organization')
        
        llm_kwargs = {
            'model': model_name,
            'temperature': temperature,
            'request_timeout': timeout,
        }
        
        if max_tokens:
            llm_kwargs['max_tokens'] = max_tokens
        if base_url:
            llm_kwargs['base_url'] = base_url
        if organization:
            llm_kwargs['organization'] = organization
        
        # Special handling for GPT-5 models
        if "gpt-5" in model_name:
            llm_kwargs.update({
                'verbosity': "low",
                'reasoning_effort': "minimal",
            })
        
        return ChatOpenAI(**llm_kwargs)
    
    elif provider == "anthropic":
        if ChatAnthropic is None:
            raise ImportError("langchain-anthropic not installed. Install with: pip install langchain-anthropic")
        
        api_key_env = get_config_value(config, 'providers.anthropic.api_key_env', 'ANTHROPIC_API_KEY')
        base_url = get_config_value(config, 'providers.anthropic.base_url')
        
        llm_kwargs = {
            'model': model_name,
            'temperature': temperature,
            'timeout': timeout,
        }
        
        if max_tokens:
            llm_kwargs['max_tokens'] = max_tokens
        if base_url:
            llm_kwargs['base_url'] = base_url
        
        return ChatAnthropic(**llm_kwargs)
    
    elif provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai not installed. Install with: pip install langchain-google-genai")
        
        # Retrieve API key configuration.
        # We support two patterns for backward compatibility:
        # 1. providers.google.api_key -> contains the actual key (preferred, NOT committed!)
        # 2. providers.google.api_key_env -> contains either the *name* of the env var (e.g. GOOGLE_API_KEY)
        #    or (legacy / current file) the raw key starting with 'AIza'.
        api_key_direct = get_config_value(config, 'providers.google.api_key')
        api_key_config = get_config_value(config, 'providers.google.api_key_env', 'GOOGLE_API_KEY')

        api_key = None
        if api_key_direct:
            api_key = api_key_direct.strip()
        else:
            # If the value looks like an API key (starts with AIza) treat it as the key, otherwise as env var name
            if isinstance(api_key_config, str) and api_key_config.startswith('AIza'):
                api_key = api_key_config.strip()
                print("Warning: Detected a raw Google API key in 'api_key_env'. Consider moving it to an environment variable 'GOOGLE_API_KEY' and setting providers.google.api_key_env: GOOGLE_API_KEY")
            else:
                env_var_name = api_key_config or 'GOOGLE_API_KEY'
                api_key = os.environ.get(env_var_name)
                if not api_key:
                    raise ValueError(f"Google API key not found. Set env var '{env_var_name}' or add 'providers.google.api_key' in config.yaml (do NOT commit the key).")

        llm_kwargs = {
            'model': model_name,
            'temperature': temperature,
            'api_key': api_key,
        }

        if max_tokens:
            llm_kwargs['max_output_tokens'] = max_tokens

        return ChatGoogleGenerativeAI(**llm_kwargs)
    
    elif provider == "ollama":
        if ChatOllama is None:
            raise ImportError("langchain-community not installed. Install with: pip install langchain-community")
        
        base_url = get_config_value(config, 'providers.ollama.base_url', 'http://localhost:11434')
        ollama_timeout = get_config_value(config, 'providers.ollama.timeout', 60)
        
        llm_kwargs = {
            'model': model_name,
            'temperature': temperature,
            'base_url': base_url,
            'timeout': ollama_timeout,
        }
        
        if max_tokens:
            llm_kwargs['num_predict'] = max_tokens
        
        return ChatOllama(**llm_kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic, google, ollama")

def create_climate_prompt(simple_mode=False, month="July", with_address=True, periods=False):
    """Create prompt template for climate data requests"""
    
    # Check for invalid combination
    if periods and not simple_mode:
        raise ValueError("Error: periods=True is only supported when simple_mode=True")
    
    if simple_mode:
        if periods:
            # Climate change mode - temperature difference between periods
            if with_address:
                prompt_template = f"""You are a climate data expert. Given the location coordinates and address information below, calculate the change in mean {month} temperature between the periods 1950–1974 and 2000–2024.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}
- Country: {{country}}
- State/Region: {{state}}
- City: {{city}}

Provide ONLY the temperature difference (2000–2024 minus 1950–1974) at 2 m above surface (°C) for this location.

IMPORTANT: Return ONLY a single number (float) representing the temperature difference in Celsius. No text, no JSON, just the number.

Example: 1.8"""
            else:
                prompt_template = f"""You are a climate data expert. Given the location coordinates below, calculate the change in mean {month} temperature between the periods 1950–1974 and 2000–2024.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}

Provide ONLY the temperature difference (2000–2024 minus 1950–1974) at 2 m above surface (°C) for this location.

IMPORTANT: Return ONLY a single number (float) representing the temperature difference in Celsius. No text, no JSON, just the number.

Example: 1.8"""
        else:
            # Original climate mode - absolute temperature
            if with_address:
                prompt_template = f"""You are a climate data expert. Given the location coordinates and address information below, provide the mean {month} temperature for the period 1991-2020.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}
- Country: {{country}}
- State/Region: {{state}}
- City: {{city}}

Provide ONLY the mean {month} temperature at 2m above surface (°C) for this location for the climatological period 1991-2020.

IMPORTANT: Return ONLY a single number (float) representing the mean {month} temperature in Celsius. No text, no JSON, just the number.

Example: 25.4"""
            else:
                prompt_template = f"""You are a climate data expert. Given the location coordinates below, provide the mean {month} temperature for the period 1991-2020.

Location Information:
- Longitude: {{longitude}}
- Latitude: {{latitude}}

Provide ONLY the mean {month} temperature at 2m above surface (°C) for this location for the climatological period 1991-2020.

IMPORTANT: Return ONLY a single number (float) representing the mean {month} temperature in Celsius. No text, no JSON, just the number.

Example: 25.4"""
    else:
        if with_address:
            prompt_template = """You are a climate data expert. Given the location coordinates and address information below, provide climatological mean values for temperature and precipitation for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}
- Country: {country}
- State/Region: {state}
- City: {city}

Please provide the following climate data for this location:
1. Temperature at 2m above surface (°C) - monthly climatological means, minimums and maximums for 1991-2020
2. Total precipitation (mm/day) - monthly climatological means, minimums and maximums for 1991-2020

For each month (January through December), provide:
- mean: average value
- min: minimum value 
- max: maximum value

IMPORTANT: return only JSON object nothing else!!!!!!!!!!!!!!!!!!!!!!!

Return ONLY a JSON object with this exact structure (no additional text):
{{
  "temperature_2m_celsius": {{
    "january": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "february": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "march": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "april": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "may": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "june": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "july": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "august": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "september": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "october": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "november": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "december": {{"mean": 0.0, "min": 0.0, "max": 0.0}}
  }},
  "precipitation_mm_per_day": {{
    "january": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "february": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "march": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "april": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "may": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "june": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "july": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "august": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "september": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "october": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "november": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "december": {{"mean": 0.0, "min": 0.0, "max": 0.0}}
  }}
}}"""
        else:
            prompt_template = """You are a climate data expert. Given the location coordinates below, provide climatological mean values for temperature and precipitation for the period 1991-2020.

Location Information:
- Longitude: {longitude}
- Latitude: {latitude}

Please provide the following climate data for this location:
1. Temperature at 2m above surface (°C) - monthly climatological means, minimums and maximums for 1991-2020
2. Total precipitation (mm/day) - monthly climatological means, minimums and maximums for 1991-2020

For each month (January through December), provide:
- mean: average value
- min: minimum value 
- max: maximum value

IMPORTANT: return only JSON object nothing else!!!!!!!!!!!!!!!!!!!!!!!

Return ONLY a JSON object with this exact structure (no additional text):
{{
  "temperature_2m_celsius": {{
    "january": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "february": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "march": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "april": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "may": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "june": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "july": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "august": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "september": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "october": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "november": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "december": {{"mean": 0.0, "min": 0.0, "max": 0.0}}
  }},
  "precipitation_mm_per_day": {{
    "january": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "february": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "march": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "april": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "may": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "june": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "july": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "august": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "september": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "october": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "november": {{"mean": 0.0, "min": 0.0, "max": 0.0}},
    "december": {{"mean": 0.0, "min": 0.0, "max": 0.0}}
  }}
}}"""

    return ChatPromptTemplate.from_template(prompt_template)

def extract_first_float(text: str) -> float:
    """Extracts the first float number from a string after removing any <think>...</think> blocks.
       Returns NaN if no numbers are found."""
    # rimuovi blocchi di reasoning (deepseek-r1 / qwen reasoning ecc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else float("nan")


def extract_last_float_after_newlines(text: str) -> float:
    """Extracts the last float number that appears after double newlines (\\n\\n).
       Special function for Mistral models that put thinking before \\n\\n and the answer after.
       Returns NaN if no numbers are found."""
    # Find the last occurrence of \n\n
    parts = text.split('\n\n')
    if len(parts) > 1:
        # Get the last part after the final \n\n
        last_part = parts[-1].strip()
        # Extract float from this last part
        m = re.search(r"[-+]?\d+(?:\.\d+)?", last_part)
        return float(m.group(0)) if m else float("nan")
    else:
        return float("nan")


def validate_and_parse_response(
    response_text: str,
    simple_mode: bool = False,
    month: str = "July",
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Optional[Dict]:
    """Validate and parse LLM response.
       Updated: for provider 'ollama' with models starting with 'qwen' or 'deepseek' (e.g. deepseek-r1), 
       it cleans and extracts the first number even if the model returns additional text or <think> blocks."""
    try:
        raw = response_text or ""
        response_text = raw.strip()

        if simple_mode:
            temperature: Optional[float] = None

            # Caso speciale: reasoning / output prolisso (qwen, deepseek, mistral via ollama)
            if provider == "ollama" and model_name:
                lowered = model_name.lower()
                if lowered.startswith("qwen") or lowered.startswith("deepseek"):
                    val = extract_first_float(response_text)
                    if not (val != val):  # check not NaN
                        temperature = val
                elif lowered.startswith("mistral"):
                    try:
                        temperature = float(response_text)
                    except ValueError:
                        val = extract_last_float_after_newlines(response_text)
                        if not (val != val):  # check not NaN
                            temperature = val

            # Fallback: prova conversione diretta
            if temperature is None:
                try:
                    temperature = float(response_text)
                except ValueError:
                    pass  # If not a valid float, temperature remains None

            if temperature is None:
                return None

            # Range plausibile
            if -100 <= temperature <= 60:
                return {f"{month.lower()}_temp_mean": temperature}
            return None

        # Full mode JSON
        # Rimuove blocchi markdown json
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        data = json.loads(response_text)
        required_keys = ["temperature_2m_celsius", "precipitation_mm_per_day"]
        months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        for key in required_keys:
            if key not in data:
                return None
            for m in months:
                if m not in data[key]:
                    return None
                if not all(stat in data[key][m] for stat in ["mean", "min", "max"]):
                    return None
        return data
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

def convert_to_numpy_arrays(climate_data: Dict, simple_mode: bool = False) -> Dict:
    """Convert climate data to numpy arrays for easier analysis"""
    if simple_mode:
        # For simple mode, just return the temperature value (any month)
        # Find the key that ends with '_temp_mean'
        temp_key = next((k for k in climate_data.keys() if k.endswith('_temp_mean')), None)
        if temp_key:
            return {temp_key: climate_data.get(temp_key, np.nan)}
        else:
            return {'temp_mean': np.nan}
    
    else:
        # Original conversion for full mode
        months = ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"]
        
        result = {
            "temperature_2m_celsius": {
                "mean": np.array([climate_data["temperature_2m_celsius"][month]["mean"] for month in months]),
                "min": np.array([climate_data["temperature_2m_celsius"][month]["min"] for month in months]),
                "max": np.array([climate_data["temperature_2m_celsius"][month]["max"] for month in months])
            },
            "precipitation_mm_per_day": {
                "mean": np.array([climate_data["precipitation_mm_per_day"][month]["mean"] for month in months]),
                "min": np.array([climate_data["precipitation_mm_per_day"][month]["min"] for month in months]),
                "max": np.array([climate_data["precipitation_mm_per_day"][month]["max"] for month in months])
            }
        }
        
        return result

def query_climate_data(llm, prompt_template, point_data: Dict, max_retries: int = 3, simple_mode: bool = False, month: str = "July", provider: Optional[str] = None, model_name: Optional[str] = None, with_address: bool = True, periods: bool = False) -> Optional[Dict]:
    """Query LLM for climate data with retry logic (single request)"""
    
    # Prepare location info
    longitude = point_data.get('lon', 'N/A')
    latitude = point_data.get('lat', 'N/A')
    country = point_data.get('country', 'N/A') if point_data.get('country') else 'N/A'
    state = point_data.get('state', 'N/A') if point_data.get('state') else 'N/A'
    city = point_data.get('city', 'N/A') if point_data.get('city') else 'N/A'
    
    for attempt in range(max_retries):
        try:
            # Create the prompt
            if with_address:
                messages = prompt_template.format_messages(
                    longitude=longitude,
                    latitude=latitude,
                    country=country,
                    state=state,
                    city=city
                )
            else:
                messages = prompt_template.format_messages(
                    longitude=longitude,
                    latitude=latitude
                )
            
            # Query the LLM
            response = llm.invoke(messages)
            response_text = response.content
            
            # Validate and parse response
            parsed_data = validate_and_parse_response(response_text, simple_mode, month, provider=provider, model_name=model_name)
            
            if parsed_data is not None:
                return {
                    'raw_response': response_text,
                    'parsed_data': parsed_data,
                    'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                    'attempt': attempt + 1
                }
            else:
                validation_msg = "Invalid number response" if simple_mode else "Invalid JSON response"
                print(f"  Attempt {attempt + 1}: {validation_msg}, retrying...")
                
        except Exception as e:
            print(f"  Attempt {attempt + 1}: Error querying LLM: {e}")
        
    
    print(f"  Failed to get valid response after {max_retries} attempts")
    return None

def query_climate_data_batch(llm, prompt_template, point_data: Dict, config: Dict, num_repeats: int = 10, simple_mode: bool = False, month: str = "July", provider: Optional[str] = None, model_name: Optional[str] = None, with_address: bool = True, periods: bool = False) -> List[Optional[Dict]]:
    """Query LLM for climate data using batch processing"""
    
    # Prepare location info
    longitude = point_data.get('lon', 'N/A')
    latitude = point_data.get('lat', 'N/A')
    country = point_data.get('country', 'N/A') if point_data.get('country') else 'N/A'
    state = point_data.get('state', 'N/A') if point_data.get('state') else 'N/A'
    city = point_data.get('city', 'N/A') if point_data.get('city') else 'N/A'
    
    # Create the same prompt for all repeats
    if with_address:
        messages = prompt_template.format_messages(
            longitude=longitude,
            latitude=latitude,
            country=country,
            state=state,
            city=city
        )
    else:
        messages = prompt_template.format_messages(
            longitude=longitude,
            latitude=latitude
        )
    
    # Get max concurrency from config
    max_concurrency = get_config_value(config, 'batch.max_concurrency', num_repeats)
    
    # Check if the LLM supports batch processing
    provider = get_config_value(config, 'model.provider', 'openai')
    
    if hasattr(llm, 'batch') and provider in ['openai', 'anthropic']:
        # Use native batch processing for supported providers
        # Create batch inputs (same prompt repeated)
        inputs = [messages] * num_repeats
        
        # Run batch query
        print(f"  Running {num_repeats} queries in parallel batch...")
        results = llm.batch(inputs, config={"max_concurrency": max_concurrency})
        
        # Process results
        processed_results = []
        successful_responses = 0
        
        for i, response in enumerate(results):
            if response and hasattr(response, 'content'):
                response_text = response.content
                
                # Validate and parse response
                parsed_data = validate_and_parse_response(response_text, simple_mode, month, provider=provider, model_name=model_name)
                
                if parsed_data is not None:
                    processed_results.append({
                        'raw_response': response_text,
                        'parsed_data': parsed_data,
                        'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                        'batch_index': i + 1
                    })
                    successful_responses += 1
                else:
                    processed_results.append(None)
            else:
                processed_results.append(None)
        
        print(f"  ✓ Batch completed: {successful_responses}/{num_repeats} successful responses")
        return processed_results
    
    else:
        # Fallback to individual queries for providers that don't support batch processing
        print(f"  Running {num_repeats} individual queries (batch not supported for {provider})...")
        processed_results = []
        successful_responses = 0
        
        for i in range(num_repeats):
            try:
                response = llm.invoke(messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Validate and parse response
                parsed_data = validate_and_parse_response(response_text, simple_mode, month, provider=provider, model_name=model_name)
                
                if parsed_data is not None:
                    processed_results.append({
                        'raw_response': response_text,
                        'parsed_data': parsed_data,
                        'numpy_arrays': convert_to_numpy_arrays(parsed_data, simple_mode),
                        'batch_index': i + 1
                    })
                    successful_responses += 1
                else:
                    processed_results.append(None)
            
            except Exception as e:
                print(f"    Query {i+1} failed: {e}")
                processed_results.append(None)
        
        print(f"  ✓ Individual queries completed: {successful_responses}/{num_repeats} successful responses")
        return processed_results

def process_climate_benchmark(config: Dict, mesh_file: str = None):
    """Main function to process climate benchmark"""
    
    # Get all values from config
    if mesh_file is None:
        mesh_file = get_config_value(config, 'benchmark.mesh_file', 'meshes/mesh_data_10deg.json')
    num_repeats = get_config_value(config, 'benchmark.num_repeats', 10)
    model_name = get_config_value(config, 'model.name', 'gpt-5-nano')
    simple_mode = get_config_value(config, 'benchmark.simple_mode', True)
    month = get_config_value(config, 'benchmark.month', 'July')
    with_address = get_config_value(config, 'benchmark.with_address', True)
    periods = get_config_value(config, 'benchmark.periods', False)
    use_batch = get_config_value(config, 'benchmark.use_batch', True)
    disable_tracing = get_config_value(config, 'benchmark.disable_tracing', False)
    resume = get_config_value(config, 'benchmark.resume', False)
    
    # Configure LangSmith tracing
    configure_langsmith(disable_tracing)
    
    print(f"Loading mesh data from {mesh_file}...")
    mesh_data = load_mesh_data(mesh_file)
    mesh_points = mesh_data['mesh_points']
    resolution = mesh_data['resolution']
    
    mode_str = f"Simple ({month} temp only)" if simple_mode else "Full (all months)"
    batch_str = "Batch processing" if use_batch else "Individual processing"
    provider = get_config_value(config, 'model.provider', 'openai')
    print(f"Loaded {len(mesh_points)} points with {resolution}° resolution")
    print(f"Provider: {provider}")
    print(f"Mode: {mode_str}")
    print(f"Processing: {batch_str}")
    print(f"Repeats per point: {num_repeats}")
    
    # Filter land points
    land_points = [point for point in mesh_points if point['is_land']]
    print(f"Found {len(land_points)} land points")
    
    # Extract chunk information from mesh file if it's a chunk
    chunk_id = None
    if 'chunk_id' in mesh_data.get('mesh_info', {}):
        chunk_id = f"{mesh_data['mesh_info']['chunk_id']:02d}"
        total_chunks = mesh_data['mesh_info'].get('total_chunks', 'unknown')
        print(f"Processing chunk {chunk_id} of {total_chunks}")
    
    # Check for resuming from intermediate file
    results = []
    start_index = 0
    
    if resume:
        latest_file = find_latest_intermediate_file(resolution, simple_mode, chunk_id, with_address, periods)
        if latest_file:
            results, saved_mesh_data, start_index = load_intermediate_results(latest_file)
            print(f"Resuming from intermediate file: {latest_file}")
            print(f"Will continue from land point {start_index + 1}/{len(land_points)}")
        else:
            print("No intermediate files found, starting from beginning")
    else:
        print("Starting fresh processing")
    
    # Initialize LLM
    llm = initialize_llm(config, model_name, simple_mode=simple_mode)
    prompt_template = create_climate_prompt(simple_mode, month, with_address, periods)
    
    # Get save interval from config
    save_interval = get_config_value(config, 'batch.save_interval', 10)
    
    # Process each land point (starting from start_index if resuming)
    for i, point_data in enumerate(land_points[start_index:], start=start_index):
        print(f"\nProcessing land point {i+1}/{len(land_points)}: ({point_data['lat']:.1f}, {point_data['lon']:.1f})")
        if point_data.get('country'):
            print(f"  Location: {point_data['country']}, {point_data.get('state', 'N/A')}, {point_data.get('city', 'N/A')}")
        
        point_results = {
            'point_info': point_data,
            'llm_responses': []
        }
        
        if use_batch and num_repeats > 1:
            # Use batch processing for multiple repeats
            batch_responses = query_climate_data_batch(llm, prompt_template, point_data, config, num_repeats, simple_mode, month, provider=provider, model_name=model_name, with_address=with_address, periods=periods)
            point_results['llm_responses'] = batch_responses
            
        else:
            # Use individual processing (original method)
            max_retries = get_config_value(config, 'model.max_retries', 3)
            for repeat in range(num_repeats):
                if num_repeats > 1:
                    print(f"  Query {repeat + 1}/{num_repeats}")
                
                climate_response = query_climate_data(llm, prompt_template, point_data, max_retries=max_retries, simple_mode=simple_mode, month=month, provider=provider, model_name=model_name, with_address=with_address, periods=periods)
                
                if climate_response:
                    point_results['llm_responses'].append(climate_response)
                    print(f"  ✓ Successfully got climate data")
                else:
                    print(f"  ✗ Failed to get climate data")
                    point_results['llm_responses'].append(None)
        
        results.append(point_results)
        
        # Save intermediate results at configured interval
        if (i + 1) % save_interval == 0:
            mode_suffix = "_simple" if simple_mode else ""
            address_suffix = "_noaddress" if not with_address else ""
            periods_suffix = "_climchange" if periods else ""
            chunk_suffix = f"_chunk_{chunk_id}" if chunk_id else ""
            # Clean model name for filename (replace special characters)
            clean_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
            results_dir = get_config_value(config, 'output.results_dir', 'results')
            intermediate_file = f"{results_dir}/climate_results_intermediate_{i+1}_{clean_model_name}{chunk_suffix}{mode_suffix}{address_suffix}{periods_suffix}.json"
            # Create results directory if it doesn't exist
            Path(intermediate_file).parent.mkdir(parents=True, exist_ok=True)
            save_results(results, mesh_data, intermediate_file, model_name, simple_mode, month, use_batch, periods)
    
    return results, mesh_data

def save_results(results: List[Dict], mesh_data: Dict, output_file: str, model_name: str, simple_mode: bool = False, month: str = "July", use_batch: bool = True, periods: bool = False):
    """Save climate benchmark results"""
    print(f"Saving results to {output_file}...")
    
    output_data = {
        'mesh_info': mesh_data['mesh_info'],
        'resolution': mesh_data['resolution'],
        'total_land_points': len(results),
        'results': results,
        'metadata': {
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': model_name,
            'num_repeats_per_point': len(results[0]['llm_responses']) if results else 0,
            'simple_mode': simple_mode,
            'query_type': f'{month.lower()}_temp_only' if simple_mode else 'full_climate_data',
            'month': month if simple_mode else None,
            'batch_processing': use_batch
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")

def main():
    """Main function"""
    import sys
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Climate LLM Benchmark')
    parser.add_argument('--config', default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--chunk', type=int, 
                       help='Chunk number to process (enables chunk mode)')
    parser.add_argument('--base_url', 
                       help='Override base URL for model provider (e.g., http://localhost:11434)')
    
    # Support legacy positional arguments for backward compatibility
    parser.add_argument('legacy_args', nargs='*', 
                       help='Legacy: [config_file] [chunk_number]')
    
    args = parser.parse_args()
    
    # Handle legacy argument format for backward compatibility
    config_file = args.config
    chunk_number = args.chunk
    base_url_override = args.base_url
    
    if args.legacy_args:
        # Legacy format: python climate_llm_benchmark.py [config_file] [chunk_number]
        if len(args.legacy_args) == 1:
            # Could be config file or chunk number
            arg = args.legacy_args[0]
            try:
                chunk_number = int(arg)
            except ValueError:
                config_file = arg
        elif len(args.legacy_args) == 2:
            config_file = args.legacy_args[0]
            chunk_number = int(args.legacy_args[1])
    
    config = load_config(config_file)
    
    # Apply base_url override if provided
    if base_url_override:
        provider = get_config_value(config, 'model.provider', 'openai')
        if provider == 'ollama':
            # Override Ollama base_url
            if 'providers' not in config:
                config['providers'] = {}
            if 'ollama' not in config['providers']:
                config['providers']['ollama'] = {}
            config['providers']['ollama']['base_url'] = base_url_override
            print(f"Override Ollama base_url: {base_url_override}")
        elif provider == 'openai':
            # Override OpenAI base_url
            if 'providers' not in config:
                config['providers'] = {}
            if 'openai' not in config['providers']:
                config['providers']['openai'] = {}
            config['providers']['openai']['base_url'] = base_url_override
            print(f"Override OpenAI base_url: {base_url_override}")
        elif provider == 'anthropic':
            # Override Anthropic base_url
            if 'providers' not in config:
                config['providers'] = {}
            if 'anthropic' not in config['providers']:
                config['providers']['anthropic'] = {}
            config['providers']['anthropic']['base_url'] = base_url_override
            print(f"Override Anthropic base_url: {base_url_override}")
        else:
            print(f"Warning: --base_url override not supported for provider '{provider}', ignoring")
    
    # Get all values from config
    chunk_mode = get_config_value(config, 'benchmark.chunk_mode', False) or (chunk_number is not None)
    chunks_dir = get_config_value(config, 'benchmark.chunks_dir', 'meshes/chunks')
    chunks_pattern = get_config_value(config, 'benchmark.chunks_pattern', 'mesh_data_1.0deg_chunk_{:02d}_of_{:02d}.json')
    base_mesh_file = get_config_value(config, 'benchmark.mesh_file', 'meshes/mesh_data_10deg.json')
    
    # Determine mesh file based on chunk mode
    if chunk_mode and chunk_number is not None:
        # Find total chunks by looking for existing chunk files
        import glob
        chunk_pattern_search = chunks_pattern.replace('{:02d}', '*')
        chunk_files = glob.glob(f"{chunks_dir}/{chunk_pattern_search}")
        if not chunk_files:
            print(f"Error: No chunk files found in {chunks_dir} matching pattern {chunk_pattern_search}")
            return
        
        # Extract total chunks from first file found
        total_chunks = len(chunk_files)
        mesh_file = f"{chunks_dir}/{chunks_pattern.format(chunk_number, total_chunks)}"
        
        if not Path(mesh_file).exists():
            print(f"Error: Chunk file '{mesh_file}' not found.")
            print(f"Available chunks: {sorted([Path(f).name for f in chunk_files])}")
            return
            
        print(f"Chunk mode enabled: Processing chunk {chunk_number} of {total_chunks}")
        
    elif chunk_mode and chunk_number is None:
        print("Error: Chunk mode is enabled but no chunk number specified.")
        print("Usage: python climate_llm_benchmark.py --chunk=N")
        print("   or: python climate_llm_benchmark.py --config=config.yaml --chunk=N")
        print("Legacy: python climate_llm_benchmark.py [config_file] chunk_number")
        return
        
    else:
        mesh_file = base_mesh_file
        if chunk_number is not None:
            print("Warning: Chunk number specified but chunk_mode is disabled in config. Using regular mesh file.")
    
    num_repeats = get_config_value(config, 'benchmark.num_repeats', 10)
    model_name = get_config_value(config, 'model.name', 'gpt-5-nano')
    simple_mode = get_config_value(config, 'benchmark.simple_mode', True)
    month = get_config_value(config, 'benchmark.month', 'July')
    use_batch = get_config_value(config, 'benchmark.use_batch', True)
    disable_tracing = get_config_value(config, 'benchmark.disable_tracing', False)
    resume = get_config_value(config, 'benchmark.resume', False)
    with_address = get_config_value(config, 'benchmark.with_address', True)
    provider = get_config_value(config, 'model.provider', 'openai')
    
    print(f"Climate LLM Benchmark")
    print(f"Configuration: {config_file}")
    if base_url_override:
        print(f"Base URL override: {base_url_override}")
    print(f"Chunk mode: {'Enabled' if chunk_mode else 'Disabled'}")
    if chunk_mode and chunk_number is not None:
        print(f"Processing chunk: {chunk_number}")
    print(f"Mesh file: {mesh_file}")
    print(f"Repeats per point: {num_repeats}")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Mode: {f'Simple ({month} temp only)' if simple_mode else 'Full (all months)'}")
    print(f"Address info: {'Included' if with_address else 'Coordinates only'}")
    print(f"Processing: {'Batch' if use_batch else 'Individual'}")
    print(f"LangSmith tracing: {'Disabled' if disable_tracing else 'Enabled'}")
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    
    # Check if mesh file exists
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        print("Please provide a valid mesh file or run geo_mesh_processor.py first.")
        return
    
    # Check if final result already exists when resume is enabled
    if resume:
        # Pre-calculate expected output filename to check if job is already done
        # Load mesh data temporarily to get resolution and chunk info
        temp_mesh_data = load_mesh_data(mesh_file)
        temp_resolution = temp_mesh_data['resolution']
        temp_mode_suffix = "_simple" if simple_mode else ""
        temp_address_suffix = "_noaddress" if not with_address else ""
        temp_periods_suffix = "_climchange" if periods else ""
        
        # Extract chunk info for filename
        temp_chunk_suffix = ""
        if 'chunk_id' in temp_mesh_data.get('mesh_info', {}):
            temp_chunk_id_num = temp_mesh_data['mesh_info']['chunk_id']
            temp_total_chunks = temp_mesh_data['mesh_info']['total_chunks']
            temp_chunk_suffix = f"_chunk_{temp_chunk_id_num:02d}_of_{temp_total_chunks:02d}"
        
        # Clean model name for filename (replace special characters)
        temp_clean_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
        results_dir = get_config_value(config, 'output.results_dir', 'results')
        expected_output_file = f"{results_dir}/climate_results_{temp_resolution}deg_r{num_repeats}_{temp_clean_model_name}{temp_chunk_suffix}{temp_mode_suffix}{temp_address_suffix}{temp_periods_suffix}.json"
        
        if Path(expected_output_file).exists():
            print(f"✓ Final result file already exists: {expected_output_file}")
            print("Job already completed - nothing to do. Use resume=false to force re-processing.")
            return
    
    try:
        # Process the benchmark
        results, mesh_data = process_climate_benchmark(config, mesh_file)
        
        # Save final results
        resolution = mesh_data['resolution']
        mode_suffix = "_simple" if simple_mode else ""
        address_suffix = "_noaddress" if not with_address else ""
        periods_suffix = "_climchange" if periods else ""
        
        # Extract chunk info for filename
        chunk_suffix = ""
        if 'chunk_id' in mesh_data.get('mesh_info', {}):
            chunk_id_num = mesh_data['mesh_info']['chunk_id']
            total_chunks = mesh_data['mesh_info']['total_chunks']
            chunk_suffix = f"_chunk_{chunk_id_num:02d}_of_{total_chunks:02d}"
        
        # Clean model name for filename (replace special characters)
        clean_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
        results_dir = get_config_value(config, 'output.results_dir', 'results')
        output_file = f"{results_dir}/climate_results_{resolution}deg_r{num_repeats}_{clean_model_name}{chunk_suffix}{mode_suffix}{address_suffix}{periods_suffix}.json"
        # Create results directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        save_results(results, mesh_data, output_file, model_name, simple_mode, month, use_batch, periods)
        
        # Print summary
        successful_points = sum(1 for r in results if any(resp for resp in r['llm_responses'] if resp))
        total_queries = len(results) * num_repeats
        successful_queries = sum(sum(1 for resp in r['llm_responses'] if resp) for r in results)
        
        print(f"\nBenchmark completed!")
        print(f"Total land points processed: {len(results)}")
        print(f"Points with successful responses: {successful_points}")
        print(f"Total queries made: {total_queries}")
        print(f"Successful queries: {successful_queries}")
        print(f"Success rate: {successful_queries/total_queries*100:.1f}%")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: combine_results.py
#!/usr/bin/env python3
"""
Combine multiple chunk result files into a single complete result file.

Usage:
    python combine_results.py results_pattern.json output_file.json
    
Example:
    python combine_results.py "results/climate_results_1.0deg_r10_gpt-5-nano_chunk_*_simple.json" results/climate_results_1.0deg_r10_gpt-5-nano_combined_simple.json
"""

import json
import sys
import glob
from pathlib import Path
from typing import List, Dict
import time

def load_chunk_results(chunk_files: List[str]) -> List[Dict]:
    """Load results from multiple chunk files"""
    all_chunk_data = []
    
    print(f"Loading {len(chunk_files)} chunk files...")
    
    for chunk_file in sorted(chunk_files):
        print(f"  Loading {Path(chunk_file).name}...")
        
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        # Extract chunk info
        chunk_id = chunk_data.get('mesh_info', {}).get('chunk_id', 'unknown')
        total_chunks = chunk_data.get('mesh_info', {}).get('total_chunks', 'unknown')
        land_points_in_chunk = len(chunk_data.get('results', []))
        
        print(f"    Chunk {chunk_id} of {total_chunks}: {land_points_in_chunk} points")
        
        all_chunk_data.append(chunk_data)
    
    return all_chunk_data

def combine_chunk_results(chunk_data_list: List[Dict]) -> Dict:
    """Combine chunk results into a single result structure"""
    
    if not chunk_data_list:
        raise ValueError("No chunk data provided")
    
    # Use the first chunk as the base structure
    base_chunk = chunk_data_list[0]
    
    # Initialize combined result structure
    combined_data = {
        'mesh_info': {
            # Remove chunk-specific info and add combined info
            **{k: v for k, v in base_chunk['mesh_info'].items() 
               if k not in ['chunk_id', 'total_chunks', 'land_points_in_chunk']},
            'combined_from_chunks': len(chunk_data_list)
        },
        'resolution': base_chunk['resolution'],
        'total_land_points': 0,  # Will be calculated
        'results': [],           # Will be combined from all chunks
        'metadata': {
            # Keep metadata from first chunk as base
            **base_chunk['metadata'],
            'combined_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chunk_info': {
                'total_chunks_combined': len(chunk_data_list),
                'chunk_files': []
            }
        }
    }
    
    # Combine all results from all chunks
    all_results = []
    total_points = 0
    
    for i, chunk_data in enumerate(chunk_data_list):
        chunk_results = chunk_data.get('results', [])
        chunk_id = chunk_data.get('mesh_info', {}).get('chunk_id', i+1)
        
        # Add chunk info to metadata
        combined_data['metadata']['chunk_info']['chunk_files'].append({
            'chunk_id': chunk_id,
            'points_in_chunk': len(chunk_results),
            'processing_date': chunk_data.get('metadata', {}).get('processing_date', 'unknown')
        })
        
        # Add results from this chunk
        all_results.extend(chunk_results)
        total_points += len(chunk_results)
        
        print(f"  Combined chunk {chunk_id}: {len(chunk_results)} points")
    
    combined_data['results'] = all_results
    combined_data['total_land_points'] = total_points
    
    # Update metadata with combined totals
    if all_results:
        total_queries_per_chunk = []
        successful_queries_per_chunk = []
        
        for chunk_data in chunk_data_list:
            chunk_results = chunk_data.get('results', [])
            if chunk_results:
                num_repeats = len(chunk_results[0].get('llm_responses', []))
                chunk_total_queries = len(chunk_results) * num_repeats
                chunk_successful_queries = sum(
                    sum(1 for resp in r.get('llm_responses', []) if resp) 
                    for r in chunk_results
                )
                total_queries_per_chunk.append(chunk_total_queries)
                successful_queries_per_chunk.append(chunk_successful_queries)
        
        combined_data['metadata']['combined_statistics'] = {
            'total_queries_all_chunks': sum(total_queries_per_chunk),
            'successful_queries_all_chunks': sum(successful_queries_per_chunk),
            'success_rate_percent': (sum(successful_queries_per_chunk) / sum(total_queries_per_chunk) * 100) if sum(total_queries_per_chunk) > 0 else 0,
            'points_with_successful_responses': sum(1 for r in all_results if any(resp for resp in r.get('llm_responses', []) if resp))
        }
    
    return combined_data

def save_combined_results(combined_data: Dict, output_file: str):
    """Save combined results to JSON file"""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving combined results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2, default=str)
    
    print(f"Combined results saved to {output_file}")

def combine_chunk_files(pattern: str, output_file: str):
    """Combine chunk files matching pattern into single output file"""
    
    # Find all matching chunk files
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        print(f"Error: No files found matching pattern '{pattern}'")
        return
    
    print(f"Found {len(chunk_files)} chunk files matching pattern:")
    for chunk_file in sorted(chunk_files):
        print(f"  {chunk_file}")
    
    # Load all chunk data
    chunk_data_list = load_chunk_results(chunk_files)
    
    # Validate chunk consistency
    resolutions = set(chunk['resolution'] for chunk in chunk_data_list)
    if len(resolutions) > 1:
        print(f"Warning: Multiple resolutions found: {resolutions}")
    
    models = set(chunk['metadata']['model_used'] for chunk in chunk_data_list)
    if len(models) > 1:
        print(f"Warning: Multiple models found: {models}")
    
    modes = set(chunk['metadata']['simple_mode'] for chunk in chunk_data_list)
    if len(modes) > 1:
        print(f"Warning: Multiple modes found: {modes}")
    
    # Combine chunk results
    combined_data = combine_chunk_results(chunk_data_list)
    
    # Save combined results
    save_combined_results(combined_data, output_file)
    
    # Print summary
    total_points = combined_data['total_land_points']
    stats = combined_data['metadata'].get('combined_statistics', {})
    
    print(f"\nCombining completed!")
    print(f"Total chunks combined: {len(chunk_files)}")
    print(f"Total land points: {total_points}")
    
    if stats:
        print(f"Total queries: {stats.get('total_queries_all_chunks', 'unknown')}")
        print(f"Successful queries: {stats.get('successful_queries_all_chunks', 'unknown')}")
        print(f"Success rate: {stats.get('success_rate_percent', 0):.1f}%")
        print(f"Points with successful responses: {stats.get('points_with_successful_responses', 'unknown')}")

def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python combine_results.py results_pattern.json output_file.json")
        print()
        print("Example:")
        print('  python combine_results.py "results/climate_results_1.0deg_r10_gpt-5-nano_chunk_*_simple.json" results/climate_results_1.0deg_r10_gpt-5-nano_combined_simple.json')
        return
    
    pattern = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        combine_chunk_files(pattern, output_file)
    except Exception as e:
        print(f"Error combining results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: compare_llm_era5.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import xarray as xr
import json
from pathlib import Path
from geo_mesh_processor import load_mesh_data

def load_llm_results(results_file):
    """Load LLM temperature results and calculate statistics"""
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Process results to extract temperature data with statistics
    processed_results = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        lat, lon = point_info['lat'], point_info['lon']
        
        # Extract temperature values from all LLM responses
        temps = []
        for response in result['llm_responses']:
            if response and 'parsed_data' in response:
                parsed = response['parsed_data']
                # Find temperature key (e.g., 'july_temp_mean', 'january_temp_mean')
                temp_key = next((k for k in parsed.keys() if k.endswith('_temp_mean')), None)
                if temp_key:
                    temps.append(parsed[temp_key])
        
        if temps:
            # Calculate statistics
            mean_temp = np.mean(temps)
            std_temp = np.std(temps) if len(temps) > 1 else np.nan
            
            # Extract month from the first valid response
            month_name = None
            for response in result['llm_responses']:
                if response and 'parsed_data' in response:
                    parsed = response['parsed_data']
                    temp_key = next((k for k in parsed.keys() if k.endswith('_temp_mean')), None)
                    if temp_key:
                        month_name = temp_key.split('_')[0]
                        break
            
            processed_results.append({
                'lat': lat,
                'lon': lon,
                'country': point_info.get('country', ''),
                'state': point_info.get('state', ''),
                'city': point_info.get('city', ''),
                'llm_temp_mean': mean_temp,
                'llm_temp_std': std_temp,
                'llm_temp_count': len(temps),
                'month': month_name,
                'raw_temps': temps
            })
    
    return processed_results, results_data['metadata']

def extract_era5_climatology(climatology_file, coordinates, month):
    """Extract ERA5 climatology data for given coordinates and month"""
    # Load ERA5 climatology
    ds = xr.open_dataset(climatology_file)
    
    # Month name to number mapping
    month_mapping = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    month_num = month_mapping.get(month.lower(), 7)  # Default to July if not found
    
    era5_data = []
    
    for lat, lon in coordinates:
        try:
            # Convert longitude from -180-180 to 0-360 if needed
            era5_lon = lon if lon >= 0 else lon + 360
            
            # Select nearest grid point
            point_data = ds.sel(latitude=lat, longitude=era5_lon, month=month_num, method='nearest')
            
            era5_data.append({
                'era5_temp_mean': float(point_data['t2m_mean'].values),
                'era5_temp_min': float(point_data['t2m_min'].values),
                'era5_temp_max': float(point_data['t2m_max'].values),
                'era5_temp_std': float(point_data['t2m_std'].values)
            })
        except Exception as e:
            print(f"Warning: Could not extract ERA5 data for ({lat}, {lon}): {e}")
            era5_data.append({
                'era5_temp_mean': np.nan,
                'era5_temp_min': np.nan,
                'era5_temp_max': np.nan,
                'era5_temp_std': np.nan
            })
    
    return era5_data

def combine_llm_era5_data(llm_results, era5_data):
    """Combine LLM results with ERA5 climatology data"""
    combined_data = []
    
    for llm_point, era5_point in zip(llm_results, era5_data):
        combined_point = llm_point.copy()
        combined_point.update(era5_point)
        
        # Calculate difference and other metrics
        if not (np.isnan(combined_point['llm_temp_mean']) or np.isnan(combined_point['era5_temp_mean'])):
            combined_point['temp_difference'] = combined_point['llm_temp_mean'] - combined_point['era5_temp_mean']
            combined_point['abs_difference'] = abs(combined_point['temp_difference'])
        else:
            combined_point['temp_difference'] = np.nan
            combined_point['abs_difference'] = np.nan
        
        combined_data.append(combined_point)
    
    return combined_data

def save_combined_results(combined_data, original_metadata, mesh_data, output_file):
    """Save combined LLM + ERA5 results to JSON file"""
    
    # Create output data structure
    output_data = {
        'mesh_info': mesh_data['mesh_info'],
        'resolution': mesh_data['resolution'],
        'total_comparison_points': len(combined_data),
        'comparison_results': combined_data,
        'metadata': {
            **original_metadata,
            'era5_climatology_added': True,
            'era5_period': '1991-2020',
            'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"Combined results saved to: {output_file}")

def plot_llm_era5_comparison(combined_data, output_file=None):
    """Create scatter plot comparing LLM results with ERA5 climatology"""
    
    # Filter valid data and extract all required values
    valid_points = []
    for d in combined_data:
        if not (np.isnan(d.get('llm_temp_mean', np.nan)) or np.isnan(d.get('era5_temp_mean', np.nan))):
            valid_points.append({
                'llm_temp': d['llm_temp_mean'],
                'era5_temp': d['era5_temp_mean'],
                'llm_std': d.get('llm_temp_std', np.nan),
                'era5_std': d.get('era5_temp_std', np.nan),
                'llm_count': d.get('llm_temp_count', 1)
            })
    
    if not valid_points:
        print("No valid data points for comparison plot")
        return
    
    # Extract arrays for plotting
    llm_vals = np.array([p['llm_temp'] for p in valid_points])
    era5_vals = np.array([p['era5_temp'] for p in valid_points])
    llm_stds = np.array([p['llm_std'] if not np.isnan(p['llm_std']) else 0 for p in valid_points])
    era5_stds = np.array([p['era5_std'] if not np.isnan(p['era5_std']) else 0 for p in valid_points])
    llm_counts = np.array([p['llm_count'] for p in valid_points])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot with error bars
    ax1.errorbar(era5_vals, llm_vals, xerr=era5_stds, yerr=llm_stds, 
                 fmt='o', alpha=0.7, markersize=5, capsize=3, capthick=1,
                 elinewidth=1, label=f'Data points (N={len(llm_vals)})')
    
    # Add 1:1 line
    min_temp = min(np.min(llm_vals), np.min(era5_vals))
    max_temp = max(np.max(llm_vals), np.max(era5_vals))
    ax1.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, linewidth=2, label='1:1 line')
    
    # Calculate and add regression line
    coeffs = np.polyfit(era5_vals, llm_vals, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_temp, max_temp, 100)
    ax1.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel('ERA5 Temperature (°C)', fontsize=12)
    ax1.set_ylabel('LLM Temperature (°C)', fontsize=12)
    ax1.set_title('LLM vs ERA5 Temperature Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # Add statistics text with error bar information
    mean_llm_requests = np.mean(llm_counts)
    total_llm_requests = np.sum(llm_counts)
    stats_text = f'N = {len(llm_vals)} points\n'
    stats_text += f'Total LLM requests: {total_llm_requests}\n'
    stats_text += f'Avg requests/point: {mean_llm_requests:.1f}\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
    stats_text += f'Correlation = {r_corr:.3f}\n'
    stats_text += f'\nError bars:\n'
    stats_text += f'Horizontal: ERA5 std dev\n'
    stats_text += f'Vertical: LLM std dev'
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Difference histogram
    differences = llm_vals - era5_vals
    ax2.hist(differences, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean diff = {np.mean(differences):+.2f}°C')
    
    ax2.set_xlabel('Temperature Difference (LLM - ERA5) °C', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Temperature Difference Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig, (rmse, mae, bias, r_corr)

def create_temperature_colormap():
    """Create a temperature colormap from cold (blue) to hot (red)"""
    # Custom temperature colormap: blue -> cyan -> green -> yellow -> orange -> red
    colors_list = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800000']
    n_bins = 256
    cmap = colors.LinearSegmentedColormap.from_list('temperature', colors_list, N=n_bins)
    return cmap

def plot_temperature_map_comparison(mesh_data, combined_data, temp_type='llm', 
                                  land_shapefile_path='./data/land/ne_10m_land.shp',
                                  output_file=None, figsize=(15, 10), vmin=None, vmax=None):
    """Plot temperature map with color-coded temperature values from comparison data"""
    
    # Get all mesh points
    mesh_points = mesh_data['mesh_points']
    
    # Extract temperature data based on type
    temp_data = {}
    if temp_type == 'llm':
        temp_key = 'llm_temp_mean'
        title_prefix = 'LLM Temperature'
    elif temp_type == 'era5':
        temp_key = 'era5_temp_mean'
        title_prefix = 'ERA5 Temperature'
    elif temp_type == 'difference':
        temp_key = 'temp_difference'
        title_prefix = 'Temperature Difference (LLM - ERA5)'
    else:
        raise ValueError("temp_type must be 'llm', 'era5', or 'difference'")
    
    # Create temperature dictionary from combined data
    for point in combined_data:
        if not np.isnan(point.get(temp_key, np.nan)):
            temp_data[(point['lat'], point['lon'])] = point[temp_key]
    
    # Collect only points that have temperature data
    land_coords = []
    temp_colors = []
    
    # Create temperature colormap
    if temp_type == 'difference':
        # Use diverging colormap for differences (blue-white-red)
        temp_cmap = plt.cm.RdBu_r  # Reversed so red is positive (LLM > ERA5)
    else:
        temp_cmap = create_temperature_colormap()
    
    # Set temperature range if not provided - use ERA5 range for consistency
    if vmin is None or vmax is None:
        if temp_type == 'difference':
            # For differences, use symmetric range around zero
            abs_diffs = [abs(temp) for temp in temp_data.values()]
            if abs_diffs:
                max_abs_diff = max(abs_diffs)
                vmin = -max_abs_diff
                vmax = max_abs_diff
            else:
                vmin, vmax = -5, 5  # Default range for differences
        else:
            # For temperature maps, use ERA5 range from the comparison points
            era5_temps = [point['era5_temp_mean'] for point in combined_data 
                         if not np.isnan(point.get('era5_temp_mean', np.nan))]
            if era5_temps:
                if vmin is None:
                    vmin = min(era5_temps)
                if vmax is None:
                    vmax = max(era5_temps)
            else:
                vmin, vmax = 0, 30  # Default range
    
    # Process all mesh points: only keep land points that have temperature data
    for point in mesh_points:
        lat, lon = point['lat'], point['lon']
        if point['is_land'] and (lat, lon) in temp_data:
            temp = temp_data[(lat, lon)]
            # Normalize temperature to [0, 1] range for colormap
            normalized_temp = (temp - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            normalized_temp = max(0, min(1, normalized_temp))  # Clamp to [0, 1]
            land_coords.append([lon, lat])
            temp_colors.append(temp_cmap(normalized_temp))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot land points with temperature colors
    if land_coords:
        land_coords = np.array(land_coords)
        ax.scatter(land_coords[:, 0], land_coords[:, 1], c=temp_colors, s=15,
                  alpha=0.9, label='Temperature data points')
    
    # Load and plot land boundaries (on top)
    try:
        land_gdf = gpd.read_file(land_shapefile_path)
        land_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.8, zorder=10)
    except Exception as e:
        print(f"Could not load land shapefile: {e}")
    
    # Customize the plot
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'{title_prefix} Map\nRange: {vmin:.1f}°C to {vmax:.1f}°C', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=temp_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    if temp_type == 'difference':
        cbar.set_label('Temperature Difference (°C)', fontsize=12)
    else:
        cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Add statistics
    temp_points = len([t for t in temp_data.values() if not np.isnan(t)])
    stats_text = f'Temperature data points: {temp_points}'
    if temp_points > 0:
        temps_list = [t for t in temp_data.values() if not np.isnan(t)]
        mean_temp = np.mean(temps_list)
        std_temp = np.std(temps_list)
        stats_text += f'\nMean: {mean_temp:.1f}°C\nStd: {std_temp:.1f}°C'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        # Create png directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Temperature map saved to {output_file}")
    else:
        plt.show()
    
    return fig, ax, vmin, vmax

def create_comparison_maps(mesh_data, combined_data, resolution, output_dir='png'):
    """Create temperature comparison maps (LLM, ERA5, and difference maps)"""
    
    # Determine temperature range from ERA5 data at comparison points
    era5_temps = [point['era5_temp_mean'] for point in combined_data 
                 if not np.isnan(point.get('era5_temp_mean', np.nan))]
    
    if not era5_temps:
        print("No valid ERA5 temperature data for mapping")
        return
    
    era5_vmin = min(era5_temps)
    era5_vmax = max(era5_temps)
    
    print(f"Temperature maps will use ERA5 range: {era5_vmin:.1f}°C to {era5_vmax:.1f}°C")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. LLM Temperature Map
    print("Creating LLM temperature map...")
    llm_output = output_path / f"llm_temperature_map_{resolution}deg.png"
    plot_temperature_map_comparison(mesh_data, combined_data, temp_type='llm', 
                                   output_file=str(llm_output), 
                                   vmin=era5_vmin, vmax=era5_vmax)
    
    # 2. ERA5 Temperature Map  
    print("Creating ERA5 temperature map...")
    era5_output = output_path / f"era5_temperature_map_{resolution}deg.png"
    plot_temperature_map_comparison(mesh_data, combined_data, temp_type='era5',
                                   output_file=str(era5_output),
                                   vmin=era5_vmin, vmax=era5_vmax)
    
    # 3. Difference Map (LLM - ERA5)
    print("Creating temperature difference map...")
    diff_output = output_path / f"temperature_difference_map_{resolution}deg.png"
    plot_temperature_map_comparison(mesh_data, combined_data, temp_type='difference',
                                   output_file=str(diff_output))
    
    print(f"All comparison maps saved to {output_dir}/")
    
    return era5_vmin, era5_vmax

def print_comparison_summary(combined_data):
    """Print summary statistics of the comparison"""
    
    # Filter valid data
    valid_data = [d for d in combined_data if not (np.isnan(d.get('llm_temp_mean', np.nan)) or 
                                                   np.isnan(d.get('era5_temp_mean', np.nan)))]
    
    if not valid_data:
        print("No valid comparison data found")
        return
    
    # Calculate statistics
    differences = [d['temp_difference'] for d in valid_data]
    llm_temps = [d['llm_temp_mean'] for d in valid_data]
    era5_temps = [d['era5_temp_mean'] for d in valid_data]
    
    print("\nComparison Summary:")
    print("=" * 50)
    print(f"Total comparison points: {len(valid_data)}")
    print(f"Mean LLM temperature: {np.mean(llm_temps):.2f}°C")
    print(f"Mean ERA5 temperature: {np.mean(era5_temps):.2f}°C")
    print(f"Mean difference (LLM - ERA5): {np.mean(differences):+.2f}°C")
    print(f"Standard deviation of differences: {np.std(differences):.2f}°C")
    print(f"RMSE: {np.sqrt(np.mean(np.array(differences)**2)):.2f}°C")
    print(f"MAE: {np.mean(np.abs(differences)):.2f}°C")
    print(f"Correlation coefficient: {np.corrcoef(llm_temps, era5_temps)[0,1]:.3f}")
    
    # Print extreme differences
    abs_diffs = [abs(d) for d in differences]
    max_diff_idx = abs_diffs.index(max(abs_diffs))
    worst_point = valid_data[max_diff_idx]
    
    print(f"\nLargest difference:")
    print(f"Location: ({worst_point['lat']:.1f}, {worst_point['lon']:.1f}) - {worst_point['country']}")
    print(f"LLM: {worst_point['llm_temp_mean']:.1f}°C, ERA5: {worst_point['era5_temp_mean']:.1f}°C")
    print(f"Difference: {worst_point['temp_difference']:+.1f}°C")

def main():
    """Main function"""
    import sys
    
    # Default files
    mesh_file = 'meshes/mesh_data_10.0deg.json'
    results_file = 'results/climate_results_10.0deg_r10_simple.json'
    climatology_file = 'data/t2m_climatology_1991-2020.nc'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        results_file = sys.argv[2]
    if len(sys.argv) > 3:
        climatology_file = sys.argv[3]
    
    print("LLM vs ERA5 Comparison Tool")
    print(f"Mesh file: {mesh_file}")
    print(f"Results file: {results_file}")
    print(f"Climatology file: {climatology_file}")
    
    # Check if files exist
    for file_path, name in [(mesh_file, 'Mesh'), (results_file, 'Results'), (climatology_file, 'Climatology')]:
        if not Path(file_path).exists():
            print(f"Error: {name} file '{file_path}' not found.")
            return
    
    try:
        # Load mesh data
        print("\nLoading mesh data...")
        mesh_data = load_mesh_data(mesh_file)
        
        # Load LLM results
        print("Loading LLM results...")
        llm_results, original_metadata = load_llm_results(results_file)
        print(f"Found {len(llm_results)} LLM result points")
        
        if not llm_results:
            print("No valid LLM results found")
            return
        
        # Extract coordinates and month
        coordinates = [(r['lat'], r['lon']) for r in llm_results]
        month = llm_results[0]['month']
        print(f"Processing for month: {month}")
        
        # Load ERA5 climatology
        print("Extracting ERA5 climatology data...")
        era5_data = extract_era5_climatology(climatology_file, coordinates, month)
        
        # Combine data
        print("Combining LLM and ERA5 data...")
        combined_data = combine_llm_era5_data(llm_results, era5_data)
        
        # Generate output filename
        results_path = Path(results_file)
        output_file = results_path.parent / (results_path.stem + '_era5.json')
        
        # Save combined results
        save_combined_results(combined_data, original_metadata, mesh_data, output_file)
        
        # Print summary
        print_comparison_summary(combined_data)
        
        # Create comparison plot
        plot_filename = f"png/llm_era5_comparison_{mesh_data['resolution']}deg.png"
        plot_llm_era5_comparison(combined_data, plot_filename)
        
        # Create temperature comparison maps
        print("\nCreating temperature comparison maps...")
        create_comparison_maps(mesh_data, combined_data, mesh_data['resolution'])
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: extend_results_with_climate_change_rmse.py
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

### File: extend_results_with_spatial_rmse.py
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

### File: geo_mesh_processor.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
import time
import json
from pathlib import Path

def create_global_mesh(resolution=10):
    """Create a global mesh with specified resolution in degrees"""
    lons = np.arange(-180, 180, resolution)
    lats = np.arange(-60, 85, resolution)
    
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Flatten to get coordinate pairs
    mesh_points = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            mesh_points.append({
                'lon': float(lon_mesh[i, j]),
                'lat': float(lat_mesh[i, j]),
                'is_land': False,
                'location_info': None,
                'address_full': None,
                'address_formatted': None,
                'country': None,
                'state': None,
                'city': None
            })
    
    return mesh_points, lon_mesh, lat_mesh

def check_land_points(mesh_points, land_shapefile_path):
    """Check which points are on land using geopandas"""
    try:
        land_shp = gpd.read_file(land_shapefile_path)
        
        for point_data in mesh_points:
            lon, lat = point_data['lon'], point_data['lat']
            point = Point(lon, lat)
            is_on_land = land_shp.contains(point).any()
            point_data['is_land'] = bool(is_on_land)
            
        return mesh_points
    except Exception as e:
        print(f"Error loading land shapefile: {e}")
        return mesh_points

def get_location_info(lat, lon):
    """Get location information from Nominatim API"""
    url = "https://nominatim.openstreetmap.org/reverse"
    
    params = {
        "lat": lat,
        "lon": lon,
        "format": "geojson",
        "extratags": 1,
        "namedetails": 1,
        "zoom": 18
    }
    headers = {
        "User-Agent": "climsight",
        "accept-language": "en"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} {response.reason}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

def parse_address_info(location):
    """Parse address information from Nominatim response"""
    if not location or 'features' not in location or len(location['features']) == 0:
        return None, None, None, None, None
    
    try:
        address = location['features'][0]['properties']['address']
        
        location_str = "Address: "
        for key, value in address.items():
            location_str += f"{key}: {value}, "
        location_str = location_str.rstrip(', ')
        
        location_str_for_print = "**Address:** "
        if "country" in address:
            location_str_for_print += f"{address['country']}, "
        if "state" in address:
            location_str_for_print += f"{address['state']}, "
        if "city" in address:
            location_str_for_print += f"{address['city']}, "
        if "road" in address:
            location_str_for_print += f"{address['road']} "
        if "house_number" in address:
            location_str_for_print += f"{address['house_number']}"
        
        location_str_for_print = location_str_for_print.rstrip(', ')
        country = address.get("country", "")
        state = address.get("state", "")
        city = address.get("city", "")
        
        return location_str, location_str_for_print, country, state, city
    except Exception as e:
        print(f"Error parsing address: {e}")
        return None, None, None, None, None

def process_mesh(resolution=10):
    """Main function to process the global mesh"""
    print(f"Creating global mesh with {resolution}x{resolution} degree resolution...")
    mesh_points, lon_mesh, lat_mesh = create_global_mesh(resolution)
    
    print(f"Created mesh with {len(mesh_points)} points")
    
    # Check for land points
    land_path = './data/land/ne_10m_land.shp'
    print("Checking land points...")
    mesh_points = check_land_points(mesh_points, land_path)
    
    land_points = [p for p in mesh_points if p['is_land']]
    print(f"Found {len(land_points)} land points")
    
    # Get location information for land points
    print("Getting location information for land points...")
    for i, point_data in enumerate(mesh_points):
        if point_data['is_land']:
            print(f"Processing land point {i+1}/{len(land_points)}: ({point_data['lat']}, {point_data['lon']})")
            
            location = get_location_info(point_data['lat'], point_data['lon'])
            if location:
                point_data['location_info'] = location
                
                address_full, address_formatted, country, state, city = parse_address_info(location)
                point_data['address_full'] = address_full
                point_data['address_formatted'] = address_formatted
                point_data['country'] = country
                point_data['state'] = state
                point_data['city'] = city
            
            # Rate limiting - sleep to respect Nominatim terms of use
            time.sleep(1)
    
    return mesh_points, lon_mesh, lat_mesh

def save_mesh_data(mesh_points, lon_mesh, lat_mesh, output_file='mesh_data.json', resolution=10):
    """Save mesh data to file"""
    # Create meshes directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving data to {output_file}...")
    
    # Convert numpy arrays to lists for JSON serialization
    data = {
        'mesh_points': mesh_points,
        'lon_mesh': lon_mesh.tolist(),
        'lat_mesh': lat_mesh.tolist(),
        'mesh_shape': lon_mesh.shape,
        'is_on_regular': True,
        'resolution': resolution,
        'mesh_info': {
            'type': 'regular_grid',
            'resolution_degrees': resolution,
            'lon_range': [-180, 180],
            'lat_range': [-60, 85],
            'total_points': len(mesh_points)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Also save as CSV for easier reading
    df_data = []
    for point in mesh_points:
        df_data.append({
            'lon': point['lon'],
            'lat': point['lat'],
            'is_land': point['is_land'],
            'country': point['country'] if point['country'] else np.nan,
            'state': point['state'] if point['state'] else np.nan,
            'city': point['city'] if point['city'] else np.nan,
            'address_formatted': point['address_formatted'] if point['address_formatted'] else np.nan,
            'address_full': point['address_full'] if point['address_full'] else np.nan,
            'is_on_regular': True,
            'resolution': resolution
        })
    
    df = pd.DataFrame(df_data)
    csv_filename = output_file.replace('.json', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {output_file} and {csv_filename}")

def load_mesh_data(input_file='mesh_data.json'):
    """Load mesh data from file and return all saved parameters"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    mesh_points = data['mesh_points']
    resolution = data.get('resolution', 10)
    mesh_info = data.get('mesh_info', {})
    
    # Check if this is a chunk file (doesn't have lon_mesh/lat_mesh)
    if 'lon_mesh' in data and 'lat_mesh' in data:
        # Full mesh file
        lon_mesh = np.array(data['lon_mesh'])
        lat_mesh = np.array(data['lat_mesh'])
        mesh_shape = data.get('mesh_shape')
        is_on_regular = data.get('is_on_regular', True)
    else:
        # Chunk file - create minimal arrays for compatibility
        lon_mesh = np.array([])
        lat_mesh = np.array([])
        mesh_shape = None
        is_on_regular = True
    
    return {
        'mesh_points': mesh_points,
        'lon_mesh': lon_mesh,
        'lat_mesh': lat_mesh,
        'mesh_shape': mesh_shape,
        'resolution': resolution,
        'is_on_regular': is_on_regular,
        'mesh_info': mesh_info
    }

if __name__ == "__main__":
    import sys
    
    # Get resolution from command line argument if provided
    resolution = 10  # default
    if len(sys.argv) > 1:
        try:
            resolution = float(sys.argv[1])
        except ValueError:
            print("Invalid resolution value. Using default 10 degrees.")
            resolution = 10
    
    # Process the mesh
    mesh_points, lon_mesh, lat_mesh = process_mesh(resolution)
    
    # Save the data
    save_mesh_data(mesh_points, lon_mesh, lat_mesh, f'meshes/mesh_data_{resolution}deg.json', resolution)
    
    # Print summary
    land_points = [p for p in mesh_points if p['is_land']]
    points_with_info = [p for p in land_points if p['location_info'] is not None]
    
    print(f"\nSummary:")
    print(f"Total points: {len(mesh_points)}")
    print(f"Land points: {len(land_points)}")
    print(f"Land points with location info: {len(points_with_info)}")

### File: multivariate_rmse_analysis.py
#!/usr/bin/env python3
"""
Multivariate Analysis of Spatial RMSE for LLM Temperature Predictions

This script performs comprehensive multivariate analysis to explain spatial_r2_rmse using:
- Population density
- Mean elevation (altitude)  
- Terrain roughness

Methods:
1. GAM (Generalized Additive Model) with smooth terms and interactions
2. Gradient Boosted Model (XGBoost) with SHAP analysis
3. Spatial block cross-validation for robust evaluation

Usage:
    python multivariate_rmse_analysis.py [results_file]

Default file: climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
Outputs: 
    - png/multivariate_analysis_*.png (visualizations)
    - reports/multivariate_rmse_report.txt (detailed text report)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Some visualizations will be simplified.")

# Statistical modeling - basic libraries
from scipy import stats
from scipy.stats import pearsonr

# Optional advanced libraries
HAS_SKLEARN = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:
    print("Warning: scikit-learn not available. Using basic implementations.")
    HAS_SKLEARN = False

try:
    from pygam import LinearGAM, s
    HAS_PYGAM = True
except ImportError:
    print("Warning: pygam not available. GAM analysis will be limited.")
    HAS_PYGAM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("Warning: xgboost not available. Gradient boosting analysis will be skipped.")
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("Warning: shap not available. SHAP analysis will be skipped.")
    HAS_SHAP = False

# Basic implementations for missing libraries
def standardize_data(data):
    """Basic standardization (z-score)"""
    return (data - np.mean(data)) / np.std(data)

def r2_score_basic(y_true, y_pred):
    """Basic R² calculation"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def load_and_prepare_data(results_file):
    """Load and prepare data for multivariate analysis"""
    
    print(f"Loading data from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check data availability
    has_era5 = results_data.get('metadata', {}).get('era5_climatology_added', False)
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False) 
    has_population = results_data.get('metadata', {}).get('population_added', False)
    has_bathymetry = results_data.get('metadata', {}).get('bathymetry_added', False)
    
    print(f"Data availability - ERA5: {has_era5}, Spatial RMSE: {has_spatial}, Population: {has_population}, Bathymetry: {has_bathymetry}")
    
    if not (has_spatial and has_population and has_bathymetry):
        raise ValueError("Missing required data components for analysis")
    
    # Extract variables
    data_points = []
    for result in results_data['results']:
        point_info = result['point_info']
        
        if (point_info.get('is_land', False) and
            all(key in point_info and not pd.isna(point_info.get(key, np.nan)) 
                for key in ['spatial_r2_rmse', 'population_density', 'mean_elevation', 'roughness'])):
            
            data_points.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'], 
                'rmse': point_info['spatial_r2_rmse'],
                'population': point_info['population_density'],
                'elevation': point_info['mean_elevation'],
                'roughness': point_info['roughness']
            })
    
    df = pd.DataFrame(data_points)
    print(f"Prepared dataset with {len(df)} complete observations")
    
    return df

def analyze_distributions(df):
    """Analyze variable distributions and determine transformations"""
    
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    variables = ['rmse', 'population', 'elevation', 'roughness']
    
    # Test for normality and skewness
    distribution_stats = {}
    
    for var in variables:
        data = df[var].values
        
        # Basic statistics
        stats_dict = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'min': np.min(data),
            'max': np.max(data)
        }
        
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(data[:5000] if len(data) > 5000 else data)
        stats_dict['shapiro_p'] = shapiro_p
        stats_dict['is_normal'] = shapiro_p > 0.05
        stats_dict['is_skewed'] = abs(stats_dict['skewness']) > 1.0
        
        distribution_stats[var] = stats_dict
        
        print(f"\n{var.upper()}:")
        print(f"  Mean: {stats_dict['mean']:.3f}, Median: {stats_dict['median']:.3f}")
        print(f"  Skewness: {stats_dict['skewness']:.3f} ({'skewed' if stats_dict['is_skewed'] else 'normal'})")
        print(f"  Shapiro-Wilk p-value: {stats_dict['shapiro_p']:.6f}")
    
    return distribution_stats

def apply_transformations(df, distribution_stats):
    """Apply appropriate transformations based on distribution analysis"""
    
    print("\n" + "="*60) 
    print("APPLYING TRANSFORMATIONS")
    print("="*60)
    
    df_transformed = df.copy()
    transformations = {}
    
    # Log-transform RMSE if skewed
    if distribution_stats['rmse']['is_skewed']:
        df_transformed['log_rmse'] = np.log(df['rmse'])
        transformations['rmse'] = 'log'
        print("Applied log transformation to RMSE (target variable)")
    else:
        df_transformed['log_rmse'] = df['rmse']
        transformations['rmse'] = 'none'
        print("No transformation applied to RMSE")
    
    # Log-transform population (typically heavily skewed)
    df_transformed['log_population'] = np.log(df['population'] + 1)  # +1 for zeros
    transformations['population'] = 'log'
    print("Applied log(x+1) transformation to population density")
    
    # Standardize all predictors
    predictor_vars = ['log_population', 'elevation', 'roughness']
    
    if HAS_SKLEARN:
        scaler = StandardScaler()
        df_transformed[['population_std', 'elevation_std', 'roughness_std']] = scaler.fit_transform(
            df_transformed[predictor_vars]
        )
        transformations['scaler'] = scaler
    else:
        # Use basic standardization
        df_transformed['population_std'] = standardize_data(df_transformed['log_population'])
        df_transformed['elevation_std'] = standardize_data(df_transformed['elevation'])
        df_transformed['roughness_std'] = standardize_data(df_transformed['roughness'])
        transformations['scaler'] = 'basic'
    
    transformations['predictor_vars'] = predictor_vars
    
    print("Standardized all predictors (z-score normalization)")
    
    return df_transformed, transformations

def create_spatial_blocks(df, n_blocks=5):
    """Create spatial blocks for cross-validation"""
    
    print(f"\nCreating {n_blocks}x{n_blocks} spatial blocks for cross-validation...")
    
    # Create spatial blocks based on lat/lon quantiles
    lat_bins = pd.cut(df['lat'], bins=n_blocks, labels=False)
    lon_bins = pd.cut(df['lon'], bins=n_blocks, labels=False)
    
    # Combine lat/lon bins to create spatial blocks
    df['spatial_block'] = lat_bins * n_blocks + lon_bins
    
    print(f"Created {len(df['spatial_block'].unique())} spatial blocks")
    print(f"Block sizes: min={df['spatial_block'].value_counts().min()}, max={df['spatial_block'].value_counts().max()}")
    
    return df

def fit_gam_model(df):
    """Fit Generalized Additive Model"""
    
    print("\n" + "="*60)
    print("GAM MODEL FITTING") 
    print("="*60)
    
    if not HAS_PYGAM:
        print("Skipping GAM analysis - pygam not available")
        return None, {}
    
    # Prepare data
    X = df[['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']].values
    y = df['log_rmse'].values
    
    try:
        # Fit GAM with smooth terms for each variable plus spatial smooth
        gam = LinearGAM(s(0) + s(1) + s(2) + s(3, 4)).fit(X, y)
        
        # Calculate predictions and R²
        y_pred = gam.predict(X)
        if HAS_SKLEARN:
            r2_total = r2_score(y, y_pred)
        else:
            r2_total = r2_score_basic(y, y_pred)
        
        print(f"GAM fitted successfully")
        print(f"Total R²: {r2_total:.4f}")
        
        # Try to estimate partial R² by removing each term
        partial_r2 = {}
        
        # This is an approximation - ideally we'd refit models without each term
        feature_names = ['population_std', 'elevation_std', 'roughness_std', 'spatial']
        
        for i, feature in enumerate(feature_names[:-1]):  # Skip spatial for now
            try:
                # Create reduced feature set
                X_reduced = np.delete(X, i, axis=1)
                
                # Fit reduced model (approximate)
                if i == 0:  # Remove population
                    gam_reduced = LinearGAM(s(0) + s(1) + s(2, 3)).fit(X_reduced, y)
                elif i == 1:  # Remove elevation  
                    gam_reduced = LinearGAM(s(0) + s(1) + s(2, 3)).fit(X_reduced, y)
                elif i == 2:  # Remove roughness
                    gam_reduced = LinearGAM(s(0) + s(1) + s(2, 3)).fit(X_reduced, y)
                
                y_pred_reduced = gam_reduced.predict(X_reduced)
                if HAS_SKLEARN:
                    r2_reduced = r2_score(y, y_pred_reduced)
                else:
                    r2_reduced = r2_score_basic(y, y_pred_reduced)
                partial_r2[feature] = r2_total - r2_reduced
                
            except Exception as e:
                partial_r2[feature] = np.nan
                print(f"Could not calculate partial R² for {feature}: {e}")
        
        print("\nPartial R² estimates:")
        for feature, r2_val in partial_r2.items():
            if not np.isnan(r2_val):
                print(f"  {feature}: {r2_val:.4f}")
        
        gam_results = {
            'model': gam,
            'r2_total': r2_total,
            'partial_r2': partial_r2,
            'predictions': y_pred
        }
        
        return gam, gam_results
        
    except Exception as e:
        print(f"GAM fitting failed: {e}")
        return None, {}

def fit_xgb_model(df):
    """Fit XGBoost model with SHAP analysis"""
    
    print("\n" + "="*60)
    print("XGBOOST MODEL WITH SHAP ANALYSIS")
    print("="*60)
    
    if not HAS_XGB:
        print("Skipping XGBoost analysis - xgboost not available")
        return None, {}
    
    # Prepare data
    feature_cols = ['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']
    X = df[feature_cols].values
    y = df['log_rmse'].values
    
    # Add interaction term
    interaction = df['population_std'] * df['elevation_std']
    X_with_interaction = np.column_stack([X, interaction])
    feature_names = feature_cols + ['pop_elev_interaction']
    
    try:
        # Fit XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(X_with_interaction, y)
        
        # Calculate R²
        y_pred = xgb_model.predict(X_with_interaction)
        if HAS_SKLEARN:
            r2_xgb = r2_score(y, y_pred)
        else:
            r2_xgb = r2_score_basic(y, y_pred)
        
        print(f"XGBoost R²: {r2_xgb:.4f}")
        
        # SHAP analysis
        shap_values = None
        interaction_shap = None
        
        if HAS_SHAP:
            print("Computing SHAP values...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_with_interaction)
            
            # Calculate feature importance from SHAP
            feature_importance = np.abs(shap_values).mean(0)
            
            print("\nSHAP Feature Importance:")
            for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                print(f"  {name}: {importance:.4f}")
            
            # Interaction SHAP for population × elevation
            try:
                interaction_shap = explainer.shap_interaction_values(X_with_interaction)
                pop_elev_interaction = interaction_shap[:, 0, 1]  # Population-elevation interaction
                interaction_strength = np.abs(pop_elev_interaction).mean()
                print(f"\nPopulation-Elevation Interaction Strength (SHAP): {interaction_strength:.4f}")
            except Exception as e:
                print(f"Could not compute interaction SHAP: {e}")
        
        xgb_results = {
            'model': xgb_model,
            'r2': r2_xgb,
            'predictions': y_pred,
            'shap_values': shap_values,
            'interaction_shap': interaction_shap,
            'feature_names': feature_names
        }
        
        return xgb_model, xgb_results
        
    except Exception as e:
        print(f"XGBoost fitting failed: {e}")
        return None, {}

def block_cross_validation(df, gam_model=None, xgb_model=None):
    """Perform spatial block cross-validation"""
    
    print("\n" + "="*60)
    print("SPATIAL BLOCK CROSS-VALIDATION")
    print("="*60)
    
    # Prepare data
    feature_cols = ['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']
    X = df[feature_cols].values
    y = df['log_rmse'].values
    
    # Add interaction for XGBoost
    interaction = df['population_std'] * df['elevation_std'] 
    X_xgb = np.column_stack([X, interaction])
    
    # Get spatial blocks
    blocks = df['spatial_block'].values
    unique_blocks = np.unique(blocks)
    n_folds = len(unique_blocks)
    
    print(f"Performing {n_folds}-fold spatial block cross-validation...")
    
    cv_results = {
        'gam_scores': [],
        'xgb_scores': [],
        'gam_rmse': [],
        'xgb_rmse': []
    }
    
    for fold, test_block in enumerate(unique_blocks):
        # Create train/test split based on spatial blocks
        test_mask = (blocks == test_block)
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        X_xgb_train, X_xgb_test = X_xgb[train_mask], X_xgb[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"  Fold {fold+1}: {np.sum(train_mask)} train, {np.sum(test_mask)} test")
        
        # GAM cross-validation
        if gam_model is not None and HAS_PYGAM:
            try:
                gam_fold = LinearGAM(s(0) + s(1) + s(2) + s(3, 4)).fit(X_train, y_train)
                y_pred_gam = gam_fold.predict(X_test)
                if HAS_SKLEARN:
                    r2_gam = r2_score(y_test, y_pred_gam)
                    rmse_gam = np.sqrt(mean_squared_error(y_test, y_pred_gam))
                else:
                    r2_gam = r2_score_basic(y_test, y_pred_gam)
                    rmse_gam = np.sqrt(np.mean((y_test - y_pred_gam) ** 2))
                
                cv_results['gam_scores'].append(r2_gam)
                cv_results['gam_rmse'].append(rmse_gam)
            except:
                pass
        
        # XGBoost cross-validation
        if xgb_model is not None and HAS_XGB:
            try:
                xgb_fold = xgb.XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
                xgb_fold.fit(X_xgb_train, y_train)
                y_pred_xgb = xgb_fold.predict(X_xgb_test)
                if HAS_SKLEARN:
                    r2_xgb = r2_score(y_test, y_pred_xgb)
                    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
                else:
                    r2_xgb = r2_score_basic(y_test, y_pred_xgb)
                    rmse_xgb = np.sqrt(np.mean((y_test - y_pred_xgb) ** 2))
                
                cv_results['xgb_scores'].append(r2_xgb)
                cv_results['xgb_rmse'].append(rmse_xgb)
            except:
                pass
    
    # Calculate CV statistics
    cv_stats = {}
    
    if cv_results['gam_scores']:
        cv_stats['gam'] = {
            'mean_r2': np.mean(cv_results['gam_scores']),
            'std_r2': np.std(cv_results['gam_scores']),
            'mean_rmse': np.mean(cv_results['gam_rmse']),
            'std_rmse': np.std(cv_results['gam_rmse'])
        }
        print(f"\nGAM CV Results:")
        print(f"  R² = {cv_stats['gam']['mean_r2']:.4f} ± {cv_stats['gam']['std_r2']:.4f}")
        print(f"  RMSE = {cv_stats['gam']['mean_rmse']:.4f} ± {cv_stats['gam']['std_rmse']:.4f}")
    
    if cv_results['xgb_scores']:
        cv_stats['xgb'] = {
            'mean_r2': np.mean(cv_results['xgb_scores']),
            'std_r2': np.std(cv_results['xgb_scores']),
            'mean_rmse': np.mean(cv_results['xgb_rmse']),
            'std_rmse': np.std(cv_results['xgb_rmse'])
        }
        print(f"\nXGBoost CV Results:")
        print(f"  R² = {cv_stats['xgb']['mean_r2']:.4f} ± {cv_stats['xgb']['std_r2']:.4f}")
        print(f"  RMSE = {cv_stats['xgb']['mean_rmse']:.4f} ± {cv_stats['xgb']['std_rmse']:.4f}")
    
    return cv_stats

def create_visualizations(df, gam_results=None, xgb_results=None, output_dir='png'):
    """Create comprehensive visualization plots"""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use seaborn style if available
    if HAS_SEABORN:
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    else:
        plt.style.use('default')
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    variables = [('rmse', 'RMSE'), ('population', 'Population Density'), 
                ('elevation', 'Elevation'), ('roughness', 'Roughness')]
    
    for i, (var, title) in enumerate(variables):
        row, col = i // 2, i % 2
        axes[row, col].hist(df[var], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Distribution of {title}')
        axes[row, col].set_xlabel(title)
        axes[row, col].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    corr_vars = ['rmse', 'population', 'elevation', 'roughness']
    corr_matrix = df[corr_vars].corr()
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    else:
        # Fallback visualization without seaborn
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_vars)))
        ax.set_yticks(range(len(corr_vars)))
        ax.set_xticklabels(corr_vars)
        ax.set_yticklabels(corr_vars)
        
        # Add correlation values as text
        for i in range(len(corr_vars)):
            for j in range(len(corr_vars)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
    
    ax.set_title('Variable Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots with RMSE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    predictors = [('population', 'Population Density'), 
                 ('elevation', 'Elevation (m)'), 
                 ('roughness', 'Roughness (m)')]
    
    for i, (var, title) in enumerate(predictors):
        axes[i].scatter(df[var], df['rmse'], alpha=0.6, s=10)
        axes[i].set_xlabel(title)
        axes[i].set_ylabel('Spatial RMSE (°C)')
        axes[i].set_title(f'RMSE vs {title}')
        
        # Add correlation coefficient
        corr, p_val = pearsonr(df[var], df['rmse'])
        axes[i].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3e}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'rmse_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SHAP plots if available
    if xgb_results and xgb_results.get('shap_values') is not None and HAS_SHAP:
        
        # SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(xgb_results['shap_values'], 
                         df[['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']].values,
                         feature_names=xgb_results['feature_names'][:-1],  # Exclude interaction
                         show=False)
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP interaction plot if available
        if xgb_results.get('interaction_shap') is not None:
            try:
                plt.figure(figsize=(8, 6))
                interaction_values = xgb_results['interaction_shap'][:, 0, 1]  # Pop-elev interaction
                plt.scatter(df['population_std'], interaction_values, alpha=0.6)
                plt.xlabel('Population (standardized)')
                plt.ylabel('SHAP Interaction Value\n(Population × Elevation)')
                plt.title('Population-Elevation Interaction Effects')
                plt.tight_layout()
                plt.savefig(output_path / 'shap_interaction.png', dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass
    
    # 5. Spatial residuals map (if we have predictions)
    if gam_results and 'predictions' in gam_results:
        residuals = df['log_rmse'] - gam_results['predictions']
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['lon'], df['lat'], c=residuals, 
                            cmap='RdBu_r', s=15, alpha=0.7)
        plt.colorbar(scatter, label='Residuals (log scale)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Spatial Distribution of GAM Residuals')
        plt.tight_layout()
        plt.savefig(output_path / 'spatial_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_path}/")

def generate_report(df, distribution_stats, gam_results, xgb_results, cv_stats, output_file='reports/multivariate_rmse_report.txt'):
    """Generate comprehensive text report"""
    
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("MULTIVARIATE ANALYSIS OF SPATIAL RMSE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {len(df)} observations\n\n")
        
        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Variables analyzed:\n")
        f.write(f"  - Target: spatial_r2_rmse (spatial RMSE for LLM temperature)\n")
        f.write(f"  - Predictors: population_density, mean_elevation, roughness\n")
        f.write(f"  - Spatial coordinates: lat, lon\n\n")
        
        # Distribution analysis
        f.write("DISTRIBUTION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for var, stats in distribution_stats.items():
            f.write(f"{var.upper()}:\n")
            f.write(f"  Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}\n")
            f.write(f"  Std: {stats['std']:.3f}, Skewness: {stats['skewness']:.3f}\n")
            f.write(f"  Range: {stats['min']:.3f} to {stats['max']:.3f}\n")
            f.write(f"  Normal distribution: {stats['is_normal']}\n")
            f.write(f"  Skewed: {stats['is_skewed']}\n\n")
        
        # GAM Results
        if gam_results:
            f.write("GAM MODEL RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model: log(RMSE) ~ s(population) + s(elevation) + s(roughness) + s(lat,lon)\n")
            f.write(f"Total R²: {gam_results.get('r2_total', 'N/A'):.4f}\n\n")
            
            if 'partial_r2' in gam_results:
                f.write("Partial R² by predictor:\n")
                for predictor, r2_val in gam_results['partial_r2'].items():
                    if not np.isnan(r2_val):
                        f.write(f"  {predictor}: {r2_val:.4f}\n")
                f.write("\n")
        
        # XGBoost Results
        if xgb_results:
            f.write("XGBOOST MODEL RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total R²: {xgb_results.get('r2', 'N/A'):.4f}\n")
            
            if HAS_SHAP and xgb_results.get('shap_values') is not None:
                f.write("\nSHAP Feature Importance:\n")
                shap_values = xgb_results['shap_values']
                feature_names = xgb_results['feature_names']
                importance = np.abs(shap_values).mean(0)
                
                for name, imp in zip(feature_names, importance):
                    f.write(f"  {name}: {imp:.4f}\n")
            f.write("\n")
        
        # Cross-validation results
        if cv_stats:
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-" * 30 + "\n")
            
            if 'gam' in cv_stats:
                stats = cv_stats['gam']
                f.write(f"GAM (Block CV):\n")
                f.write(f"  R² = {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}\n")
                f.write(f"  RMSE = {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}\n\n")
            
            if 'xgb' in cv_stats:
                stats = cv_stats['xgb']
                f.write(f"XGBoost (Block CV):\n")
                f.write(f"  R² = {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}\n")
                f.write(f"  RMSE = {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}\n\n")
        
        # Interpretation
        f.write("KEY FINDINGS & INTERPRETATION\n")
        f.write("-" * 30 + "\n")
        f.write("1. The analysis explains spatial variation in LLM temperature prediction RMSE\n")
        f.write("   using population density, elevation, and terrain roughness.\n\n")
        f.write("2. Both GAM and XGBoost models provide consistent results for robustness.\n\n")
        f.write("3. Spatial block cross-validation accounts for spatial autocorrelation\n")
        f.write("   and provides unbiased performance estimates.\n\n")
        
        if gam_results and 'partial_r2' in gam_results:
            f.write("4. Partial R² values indicate the unique contribution of each predictor\n")
            f.write("   to explaining RMSE variation.\n\n")
        
        if HAS_SHAP and xgb_results and xgb_results.get('shap_values') is not None:
            f.write("5. SHAP values provide interpretable feature importance rankings\n")
            f.write("   and reveal interaction effects between variables.\n\n")
    
    print(f"Report saved to {output_file}")

def main():
    """Main analysis function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("MULTIVARIATE ANALYSIS OF SPATIAL RMSE")
    print("=" * 80)
    print(f"Input file: {results_file}")
    print(f"Analysis target: spatial_r2_rmse")
    print(f"Predictors: population_density, mean_elevation, roughness")
    print()
    
    # Check file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data(results_file)
        
        # 2. Analyze distributions
        distribution_stats = analyze_distributions(df)
        
        # 3. Apply transformations
        df_transformed, transformations = apply_transformations(df, distribution_stats)
        
        # 4. Create spatial blocks
        df_transformed = create_spatial_blocks(df_transformed, n_blocks=5)
        
        # 5. Fit GAM model
        gam_model, gam_results = fit_gam_model(df_transformed)
        
        # 6. Fit XGBoost model with SHAP
        xgb_model, xgb_results = fit_xgb_model(df_transformed)
        
        # 7. Cross-validation
        cv_stats = block_cross_validation(df_transformed, gam_model, xgb_model)
        
        # 8. Create visualizations
        create_visualizations(df_transformed, gam_results, xgb_results)
        
        # 9. Generate report
        generate_report(df_transformed, distribution_stats, gam_results, xgb_results, cv_stats)
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Outputs generated:")
        print("  - Visualizations: png/multivariate_analysis_*.png")
        print("  - Report: reports/multivariate_rmse_report.txt")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: plot_bathymetry_map.py
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

### File: plot_climate_change_analysis.py
#!/usr/bin/env python3
"""
Comprehensive plotting script for climate change difference analysis results.

This script creates various maps and plots from climate change spatial RMSE-enhanced result files:
- Climate change signal maps (LLM predictions vs ERA5)
- Spatial RMSE maps for climate change predictions (radius 2 and 4)
- Bias maps showing LLM prediction errors
- Correlation maps between LLM and ERA5 climate signals
- Scatter plots comparing LLM vs ERA5 climate change predictions
- Distribution analysis of climate change signal predictions

Usage:
    python plot_climate_change_analysis.py [results_file]

Default: results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff_climate_rmse.json
Output: png/{results_filename}/climate_change_*_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
import pandas as pd
from pathlib import Path
import sys
import os
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_climate_change_results(results_file):
    """Load climate change spatial RMSE-enhanced results file"""
    print(f"Loading climate change results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check if climate change spatial RMSE data is present
    has_climate_rmse = results_data.get('metadata', {}).get('climate_change_spatial_rmse_added', False)
    if not has_climate_rmse:
        print("Warning: Climate change spatial RMSE data not found in results file")
        print("Please run extend_results_with_climate_change_rmse.py first")
    
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


def create_climate_change_map(mapping_data, field_name, resolution, colormap='RdBu_r', 
                             vmin=None, vmax=None, title_suffix="", output_file=None, figsize=(18, 10)):
    """Create a contour map for climate change data"""
    
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
    
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    # Create contour levels
    levels = np.linspace(vmin, vmax, 20)
    
    # Choose colormap based on field type
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        # For error metrics, use reversed viridis (blue=good, red=bad)
        cmap = plt.cm.get_cmap('viridis_r')
    elif 'bias' in field_name.lower() or 'change' in field_name.lower() or 'diff' in field_name.lower():
        # For climate change signals and bias, use diverging colormap
        cmap = plt.cm.RdBu_r
        # Make symmetric around zero for change signals
        if 'change' in field_name.lower() or 'era5' in field_name.lower() or 'llm' in field_name.lower():
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            levels = np.linspace(vmin, vmax, 20)
    elif 'correlation' in field_name.lower():
        # For correlation, use diverging colormap centered at 0
        cmap = plt.cm.RdBu
        vmin, vmax = -1, 1
        levels = np.linspace(vmin, vmax, 20)
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    contour = ax.contourf(lon_grid, lat_grid, data_grid, 
                         levels=levels, cmap=cmap, extend='both')
    
    # Try to load and overlay country boundaries
    try:
        world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
        world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
    except Exception as e:
        print(f"Could not load country boundaries: {e}")
        ax.grid(True, alpha=0.3)
    
    # Customize the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    
    # Create title
    clean_field_name = field_name.replace('_', ' ').title()
    ax.set_title(f'{clean_field_name} Map{title_suffix}\n'
                f'Land Points: {len(mapping_data):,}', 
                fontsize=14, fontweight='bold')
    
    # Add coordinate ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    
    # Set colorbar label based on field type
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        cbar.set_label('Error (°C)', fontsize=12)
    elif 'bias' in field_name.lower():
        cbar.set_label('Bias (°C)', fontsize=12)
    elif 'change' in field_name.lower() or 'diff' in field_name.lower():
        cbar.set_label('Temperature Change (°C)', fontsize=12)
    elif 'correlation' in field_name.lower():
        cbar.set_label('Correlation Coefficient', fontsize=12)
    else:
        cbar.set_label('Value', fontsize=12)
    
    # Add statistics text
    mean_val = np.mean(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_val = np.std(values)
    
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {std_val:.2f}\n'
    stats_text += f'Range: {min_val:.2f} - {max_val:.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{clean_field_name} map saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def extract_climate_comparison_data(results_data):
    """Extract LLM vs ERA5 climate change comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 climate change data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_change_signal' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_change_signal', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_change': point_info['llm_temp_mean'],  # LLM predicted change
                'era5_change': point_info['era5_change_signal'],  # ERA5 climate change signal
                'difference': point_info['llm_temp_mean'] - point_info['era5_change_signal'],
                'country': point_info.get('country', ''),
                'spatial_r2_rmse': point_info.get('spatial_r2_rmse', np.nan),
                'spatial_r4_rmse': point_info.get('spatial_r4_rmse', np.nan)
            })
    
    print(f"Found {len(comparison_data)} valid climate change comparison data points")
    return comparison_data


def create_climate_change_scatter_plot(comparison_data, resolution, output_file=None):
    """Create density scatter plot comparing LLM vs ERA5 climate change predictions"""
    
    if not comparison_data:
        print("No climate change comparison data available for scatter plot")
        return None
    
    # Extract arrays for plotting
    llm_changes = np.array([d['llm_change'] for d in comparison_data])
    era5_changes = np.array([d['era5_change'] for d in comparison_data])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # === Top Left: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([era5_changes, llm_changes])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
    except:
        # Fallback if KDE fails (e.g., all points identical)
        density = np.ones_like(era5_changes)
    
    # Sort points by density (plot low density first)
    idx = density.argsort()
    era5_sorted, llm_sorted, density_sorted = era5_changes[idx], llm_changes[idx], density[idx]
    
    # Create density scatter plot
    scatter = ax1.scatter(era5_sorted, llm_sorted, c=density_sorted, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
    # Add 1:1 line
    min_change = min(np.min(llm_changes), np.min(era5_changes))
    max_change = max(np.max(llm_changes), np.max(era5_changes))
    ax1.plot([min_change, max_change], [min_change, max_change], 'r--', alpha=0.8, linewidth=2, 
             label='1:1 line (perfect agreement)')
    
    # Calculate and add regression line
    coeffs = np.polyfit(era5_changes, llm_changes, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_change, max_change, 100)
    ax1.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel('ERA5 Climate Change Signal (°C)', fontsize=12)
    ax1.set_ylabel('LLM Predicted Change (°C)', fontsize=12)
    ax1.set_title(f'LLM vs ERA5 Climate Change Predictions\nDensity Scatter Plot ({resolution}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for density
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Point Density', fontsize=11)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_changes - era5_changes)**2))
    mae = np.mean(np.abs(llm_changes - era5_changes))
    bias = np.mean(llm_changes - era5_changes)
    r_corr = np.corrcoef(llm_changes, era5_changes)[0, 1]
    
    # Add statistics text
    stats_text = f'N = {len(llm_changes)} points\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
    stats_text += f'Correlation = {r_corr:.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Top Right: Difference histogram ===
    differences = llm_changes - era5_changes
    ax2.hist(differences, bins=25, alpha=0.7, edgecolor='black', color='skyblue', density=True)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax2.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean diff = {np.mean(differences):+.2f}°C')
    
    # Add normal distribution overlay for reference
    x_norm = np.linspace(differences.min(), differences.max(), 100)
    y_norm = (1/np.sqrt(2*np.pi*np.var(differences))) * np.exp(-0.5*((x_norm - np.mean(differences))**2)/np.var(differences))
    ax2.plot(x_norm, y_norm, 'g--', alpha=0.8, linewidth=2, label='Normal fit')
    
    ax2.set_xlabel('Prediction Error (LLM - ERA5) °C', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Climate Change Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add difference statistics
    diff_stats = f'Error Statistics:\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\n'
    diff_stats += f'IQR: {np.percentile(differences, 75) - np.percentile(differences, 25):.3f}°C\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Bottom Left: ERA5 vs LLM distribution comparison ===
    ax3.hist(era5_changes, bins=20, alpha=0.6, label='ERA5 Climate Signal', color='blue', density=True)
    ax3.hist(llm_changes, bins=20, alpha=0.6, label='LLM Predictions', color='red', density=True)
    ax3.axvline(np.mean(era5_changes), color='blue', linestyle='--', linewidth=2, 
                label=f'ERA5 Mean = {np.mean(era5_changes):.2f}°C')
    ax3.axvline(np.mean(llm_changes), color='red', linestyle='--', linewidth=2,
                label=f'LLM Mean = {np.mean(llm_changes):.2f}°C')
    
    ax3.set_xlabel('Temperature Change (°C)', fontsize=12)
    ax3.set_ylabel('Probability Density', fontsize=12)
    ax3.set_title('Climate Change Signal Distributions', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # === Bottom Right: Spatial RMSE vs Prediction Error ===
    spatial_rmse_r2 = np.array([d['spatial_r2_rmse'] for d in comparison_data if not np.isnan(d['spatial_r2_rmse'])])
    spatial_diffs = np.array([d['difference'] for d in comparison_data if not np.isnan(d['spatial_r2_rmse'])])
    
    if len(spatial_rmse_r2) > 0:
        ax4.scatter(spatial_rmse_r2, np.abs(spatial_diffs), alpha=0.6, s=20, color='purple')
        ax4.set_xlabel('Spatial RMSE (5×5 neighborhood) °C', fontsize=12)
        ax4.set_ylabel('|Prediction Error| (°C)', fontsize=12)
        ax4.set_title('Spatial Consistency vs Prediction Error', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add correlation if possible
        if len(spatial_rmse_r2) > 1:
            corr_spatial = np.corrcoef(spatial_rmse_r2, np.abs(spatial_diffs))[0, 1]
            ax4.text(0.02, 0.98, f'Correlation: {corr_spatial:.3f}', transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax4.text(0.5, 0.5, 'No spatial RMSE data available', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=14)
        ax4.set_title('Spatial Consistency vs Prediction Error', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Climate change comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def create_all_climate_change_maps(results_data, resolution, output_dir='png'):
    """Create all climate change analysis maps from the results data"""
    
    print("Creating comprehensive climate change analysis maps...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base filename pattern
    base_name = f"climate_change_{resolution}deg"
    
    # Maps to create - field_name: (title_suffix, colormap)
    maps_config = {
        # Climate change signals
        'llm_temp_mean': ('LLM Predicted Climate Change', 'RdBu_r'),
        'era5_change_signal': ('ERA5 Climate Change Signal', 'RdBu_r'),
        
        # Spatial RMSE maps for climate change
        'spatial_r2_rmse': ('Spatial RMSE (5×5 neighborhood)', 'viridis_r'),
        'spatial_r4_rmse': ('Spatial RMSE (9×9 neighborhood)', 'viridis_r'),
        
        # Bias maps for climate change predictions
        'spatial_r2_bias': ('Spatial Bias (5×5 neighborhood)', 'RdBu_r'),
        'spatial_r4_bias': ('Spatial Bias (9×9 neighborhood)', 'RdBu_r'),
        
        # Correlation maps
        'spatial_r2_correlation': ('Spatial Correlation (5×5 neighborhood)', 'RdBu'),
        'spatial_r4_correlation': ('Spatial Correlation (9×9 neighborhood)', 'RdBu'),
        
        # Neighborhood mean climate change signals
        'spatial_r2_neighborhood_llm_mean': ('LLM Neighborhood Mean (5×5)', 'RdBu_r'),
        'spatial_r4_neighborhood_llm_mean': ('LLM Neighborhood Mean (9×9)', 'RdBu_r'),
        'spatial_r2_neighborhood_era5_mean': ('ERA5 Neighborhood Mean (5×5)', 'RdBu_r'),
        'spatial_r4_neighborhood_era5_mean': ('ERA5 Neighborhood Mean (9×9)', 'RdBu_r'),
    }
    
    created_maps = []
    
    for field_name, (title_suffix, colormap) in maps_config.items():
        print(f"\nCreating map for {field_name}...")
        
        # Extract mapping data for this field
        mapping_data = extract_mapping_data(results_data, field_name)
        
        if not mapping_data:
            print(f"No data found for {field_name}, skipping...")
            continue
        
        # Create output filename
        clean_field_name = field_name.replace('spatial_', '').replace('_', '_')
        output_file = output_path / f"{base_name}_{clean_field_name}.png"
        
        # Determine value range for consistency
        vmin, vmax = None, None
        if ('change' in field_name or 'temp_mean' in field_name or 
            'neighborhood' in field_name or 'bias' in field_name):
            # Use symmetric range for climate change signals and bias
            values = [d['value'] for d in mapping_data]
            if values:
                abs_max = max(abs(min(values)), abs(max(values)))
                vmin, vmax = -abs_max, abs_max
        
        # Create the map
        try:
            fig = create_climate_change_map(
                mapping_data=mapping_data,
                field_name=field_name,
                resolution=resolution,
                colormap=colormap,
                vmin=vmin,
                vmax=vmax,
                title_suffix=f" - {title_suffix}",
                output_file=str(output_file)
            )
            
            if fig is not None:
                created_maps.append(str(output_file))
                
        except Exception as e:
            print(f"Error creating map for {field_name}: {e}")
            continue
    
    print(f"\nClimate change mapping completed!")
    print(f"Created {len(created_maps)} maps in {output_dir}/")
    
    return created_maps


def create_all_climate_comparison_plots(results_data, resolution, output_dir='png'):
    """Create all LLM vs ERA5 climate change comparison plots"""
    
    print("Creating LLM vs ERA5 climate change comparison plots...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract comparison data
    comparison_data = extract_climate_comparison_data(results_data)
    
    if not comparison_data:
        print("No climate change comparison data available for plotting")
        return []
    
    created_plots = []
    
    # Create main climate change comparison plot
    scatter_output = output_path / f"climate_change_comparison_{resolution}deg.png"
    fig, stats = create_climate_change_scatter_plot(comparison_data, resolution, str(scatter_output))
    if fig is not None:
        created_plots.append(str(scatter_output))
        rmse, mae, bias, r_corr = stats
        print(f"Climate change comparison statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
    
    return created_plots


def print_climate_change_summary_statistics(results_data):
    """Print summary statistics for all climate change metrics"""
    
    print(f"\nClimate Change Analysis Summary Statistics:")
    print(f"=" * 80)
    
    # Fields to summarize
    summary_fields = [
        'llm_temp_mean', 'era5_change_signal',
        'spatial_r2_rmse', 'spatial_r4_rmse',
        'spatial_r2_mae', 'spatial_r4_mae', 
        'spatial_r2_bias', 'spatial_r4_bias',
        'spatial_r2_correlation', 'spatial_r4_correlation'
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
    
    # Special analysis for climate change signals
    print(f"\nClimate Change Signal Analysis:")
    print(f"-" * 40)
    
    llm_changes = []
    era5_changes = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_change_signal' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_change_signal', np.nan))):
            llm_changes.append(point_info['llm_temp_mean'])
            era5_changes.append(point_info['era5_change_signal'])
    
    if llm_changes and era5_changes:
        llm_array = np.array(llm_changes)
        era5_array = np.array(era5_changes)
        
        print(f"LLM vs ERA5 Climate Change:")
        print(f"  Valid comparison points: {len(llm_changes)}")
        print(f"  LLM mean change: {np.mean(llm_array):.3f}°C")
        print(f"  ERA5 mean change: {np.mean(era5_array):.3f}°C")
        print(f"  Overall bias (LLM - ERA5): {np.mean(llm_array - era5_array):+.3f}°C")
        print(f"  Overall RMSE: {np.sqrt(np.mean((llm_array - era5_array)**2)):.3f}°C")
        print(f"  Overall correlation: {np.corrcoef(llm_array, era5_array)[0, 1]:.3f}")


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_mistral-small3_1_24b_simple_diff_climate_rmse.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Comprehensive Climate Change Analysis Plotting")
    print("=" * 80)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run extend_results_with_climate_change_rmse.py first to create climate change spatial RMSE data.")
        return
    
    try:
        # Load climate change results
        results_data = load_climate_change_results(results_file)
        
        # Extract resolution from metadata or filename
        resolution = results_data.get('resolution', 1.0)
        if 'deg' in results_file:
            try:
                resolution = float(results_file.split('_')[2].replace('deg', ''))
            except:
                pass
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem  # Get filename without extension
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create all climate change maps
        created_maps = create_all_climate_change_maps(results_data, resolution, str(output_dir))
        
        # Create comparison plots
        created_comparison_plots = create_all_climate_comparison_plots(results_data, resolution, str(output_dir))
        
        # Print summary statistics
        print_climate_change_summary_statistics(results_data)
        
        total_plots = len(created_maps) + len(created_comparison_plots)
        print(f"\nClimate change plotting completed successfully!")
        print(f"Created {len(created_maps)} climate change maps")
        print(f"Created {len(created_comparison_plots)} comparison plots")
        print(f"Total: {total_plots} plots")
        print(f"All files saved in: {output_dir}/")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

### File: plot_density_comparison.py
#!/usr/bin/env python3
"""
Density plot comparison between LLM and ERA5 temperature predictions.

This script creates density plots comparing LLM predictions vs ERA5 climatology
with color-coded point density and difference histogram analysis.

Usage:
    python plot_density_comparison.py results_file.json

Output: pub_f2_density_comparison_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

# Configuration section
# Font configuration for publication quality
FONT_FAMILY = 'Helvetica'  # Font family for all text elements
FONT_SIZE_LABEL = 12       # Font size for axis labels
FONT_SIZE_TICK = 10        # Font size for tick labels

# Axis display configuration
SHOW_AXES = True           # Show axis labels and tick marks (True) or clean plot (False)
SHOW_FRAME = True          # Show frame/box around the plot (True) or remove it (False)
SHOW_LEFT_AXIS = True      # Show left Y-axis spine and ticks (True) or hide (False)
SHOW_RIGHT_AXIS = False    # Show right Y-axis spine and ticks (True) or hide (False)
SHOW_TOP_AXIS = False      # Show top X-axis spine and ticks (True) or hide (False)
SHOW_BOTTOM_AXIS = True    # Show bottom X-axis spine and ticks (True) or hide (False)

# Colorbar configuration (for consistency with scatter plot)
COLORBAR_PAD = 0.08        # Distance between plot and colorbar (0.08 = closer, 0.15 = further)

# X-axis limits for density plot
X_AXIS_MIN = -20           # Minimum X-axis value (temperature difference)
X_AXIS_MAX = 20            # Maximum X-axis value (temperature difference)


def extract_comparison_data(results_data):
    """Extract LLM vs ERA5 comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0.0),
                'era5_std': point_info.get('era5_temp_std', np.nan),
                'temp_diff': point_info.get('temp_difference', 
                                          point_info['llm_temp_mean'] - point_info['era5_temp_mean']),
                'llm_count': point_info.get('llm_temp_count', 1),
                'country': point_info.get('country', '')
            })
    
    print(f"Found {len(comparison_data)} valid comparison data points")
    return comparison_data


def create_density_plot(comparison_data, output_file=None, font_family='Helvetica', font_size_label=12, font_size_tick=10, show_axes=True, show_frame=True, show_left_axis=True, show_right_axis=False, show_top_axis=False, show_bottom_axis=True, colorbar_pad=0.08, x_axis_min=-20, x_axis_max=20):
    """Create difference histogram (density distribution) comparing LLM vs ERA5"""
    
    if not comparison_data:
        print("No comparison data available for density plot")
        return None, None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    
    # Set font for matplotlib
    plt.rcParams['font.family'] = font_family
    
    # Create figure with square plot area (width x height)
    # Extra height for potential future colorbar
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 7.0))
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # === Difference histogram ===
    differences = llm_vals - era5_vals
    ax.hist(differences, bins=25, alpha=0.7, edgecolor='black', color='skyblue', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean diff = {np.mean(differences):+.2f}°C')
    
    # Add normal distribution overlay for reference
    x_norm = np.linspace(differences.min(), differences.max(), 100)
    y_norm = (1/np.sqrt(2*np.pi*np.var(differences))) * np.exp(-0.5*((x_norm - np.mean(differences))**2)/np.var(differences))
    ax.plot(x_norm, y_norm, 'g--', alpha=0.8, linewidth=2, label='Normal fit')
    
    # Configure axes and labels based on options
    if show_axes:
        ax.set_xlabel('Temperature Difference (LLM - ERA5) °C', fontsize=font_size_label, fontfamily=font_family)
        ax.set_ylabel('Probability Density', fontsize=font_size_label, fontfamily=font_family)
        # Configure tick labels
        ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
    
    # Configure frame and individual axis spines
    if not show_frame:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        # Configure individual axis spines
        ax.spines['left'].set_visible(show_left_axis)
        ax.spines['right'].set_visible(show_right_axis)
        ax.spines['top'].set_visible(show_top_axis)
        ax.spines['bottom'].set_visible(show_bottom_axis)
        
        # Configure tick marks to match visible spines
        if show_axes:
            ax.tick_params(left=show_left_axis, right=show_right_axis, 
                          top=show_top_axis, bottom=show_bottom_axis,
                          labelleft=show_left_axis, labelright=show_right_axis,
                          labeltop=show_top_axis, labelbottom=show_bottom_axis)
    
    ax.grid(True, alpha=0.3)
    
    # Set X-axis limits for temperature difference range
    ax.set_xlim(x_axis_min, x_axis_max)
    
    # Note: Not using square aspect ratio for histogram as X and Y axes have different units
    # X-axis: Temperature difference (°C), Y-axis: Probability density (dimensionless)
    
    # Configure legend font
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontfamily(font_family)
        text.set_fontsize(font_size_tick)
    
    # Statistics calculated for return values only (not displayed on plot)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"LLM vs ERA5 density plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def main():
    """Main function to create density plot comparison"""
    
    if len(sys.argv) < 2:
        print("Usage: python plot_density_comparison.py results_file.json")
        print("Example: python plot_density_comparison.py results/climate_results_1.0deg_r10_gpt-5_simple_spatial_rmse_bathymetry_population.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Check if results file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found")
        sys.exit(1)
    
    # Load results data
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract resolution from filename or results data
    resolution = results_data.get('resolution', '1.0')
    if isinstance(resolution, str) and 'deg' in resolution:
        resolution = resolution.replace('deg', '')
    
    # Extract comparison data
    comparison_data = extract_comparison_data(results_data)
    
    if not comparison_data:
        print("No comparison data available for plotting")
        return
    
    # Create subfolder based on results filename
    results_filename = Path(results_file).stem  # Get filename without extension
    output_dir = Path('png') / results_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create output filename with pub_f2_ prefix
    output_file = output_dir / f"pub_f2_density_comparison_{resolution}deg.png"
    
    # Create density plot
    fig, stats = create_density_plot(comparison_data, output_file, FONT_FAMILY, FONT_SIZE_LABEL, FONT_SIZE_TICK, SHOW_AXES, SHOW_FRAME, SHOW_LEFT_AXIS, SHOW_RIGHT_AXIS, SHOW_TOP_AXIS, SHOW_BOTTOM_AXIS, COLORBAR_PAD, X_AXIS_MIN, X_AXIS_MAX)
    
    if fig is not None:
        rmse, mae, bias, r_corr = stats
        print(f"Density plot statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")


if __name__ == "__main__":
    main()

### File: plot_elevation_clusters.py
#!/usr/bin/env python3
"""
Create LLM vs ERA5 temperature comparison scatter plots clustered by elevation ranges.

This script creates elevation-based clusters and plots LLM vs ERA5 temperature comparisons
for each cluster in a 3×3 grid layout. Each subplot shows data for a specific elevation range.

Elevation clusters (9 bins):
- 0-500m, 500-1000m, 1000-1500m, 1500-2000m, 2000-2500m, 
- 2500-3000m, 3000-3500m, 3500-4000m, 4000m+

Usage:
    python plot_elevation_clusters.py [results_file]

Default file: climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
Output: png/{results_filename}/elevation_clusters_{resolution}deg.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json
from pathlib import Path
import sys
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
    
    if not (has_era5 and has_bathymetry):
        print("Warning: Missing required data (ERA5 or bathymetry)")
    
    return results_data

def extract_elevation_data(results_data):
    """Extract LLM vs ERA5 comparison data with elevation information"""
    
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with complete data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and
            'era5_temp_mean' in point_info and
            'mean_elevation' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_temp_mean', np.nan)) and
            not np.isnan(point_info.get('mean_elevation', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0),
                'era5_std': point_info.get('era5_temp_std', 0),
                'llm_count': point_info.get('llm_temp_count', 1),
                'mean_elevation': point_info['mean_elevation']
            })
    
    print(f"Extracted {len(comparison_data)} points with complete elevation and temperature data")
    return comparison_data

def create_elevation_clusters(comparison_data):
    """Create elevation-based clusters"""
    
    # Define elevation bins (9 clusters)
    elevation_bins = [
        (0, 500, '0-500m'),
        (500, 1000, '500-1000m'),
        (1000, 1500, '1000-1500m'),
        (1500, 2000, '1500-2000m'),
        (2000, 2500, '2000-2500m'),
        (2500, 3000, '2500-3000m'),
        (3000, 3500, '3000-3500m'),
        (3500, 4000, '3500-4000m'),
        (4000, np.inf, '4000m+')
    ]
    
    # Create clusters
    elevation_clusters = {}
    
    for min_elev, max_elev, label in elevation_bins:
        cluster_data = []
        for data_point in comparison_data:
            elevation = data_point['mean_elevation']
            if min_elev <= elevation < max_elev:
                cluster_data.append(data_point)
        
        elevation_clusters[label] = {
            'data': cluster_data,
            'range': (min_elev, max_elev),
            'count': len(cluster_data)
        }
        
        print(f"Elevation cluster {label}: {len(cluster_data)} points")
    
    return elevation_clusters

def create_single_cluster_plot(cluster_data, cluster_label, ax):
    """Create a single LLM vs ERA5 scatter plot for one elevation cluster"""
    
    if len(cluster_data) == 0:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(f'{cluster_label}\n(N=0)', fontsize=10, fontweight='bold')
        return None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in cluster_data])
    era5_vals = np.array([d['era5_temp'] for d in cluster_data])
    elevations = np.array([d['mean_elevation'] for d in cluster_data])
    
    # Create scatter plot colored by elevation within cluster
    scatter = ax.scatter(era5_vals, llm_vals, c=elevations, s=15, alpha=0.7, 
                        cmap='terrain', edgecolors='black', linewidth=0.1)
    
    # Add 1:1 line
    if len(llm_vals) > 0:
        min_temp = min(np.min(llm_vals), np.min(era5_vals))
        max_temp = max(np.max(llm_vals), np.max(era5_vals))
        ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, linewidth=1, 
                label='1:1')
        
        # Calculate and add regression line
        if len(llm_vals) > 1:  # Need at least 2 points for regression
            coeffs = np.polyfit(era5_vals, llm_vals, 1)
            regression_line = np.poly1d(coeffs)
            x_reg = np.linspace(min_temp, max_temp, 100)
            ax.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=1, 
                    label=f'R: {coeffs[0]:.2f}x+{coeffs[1]:.1f}')
    
    ax.set_xlabel('ERA5 Temp (°C)', fontsize=9)
    ax.set_ylabel('LLM Temp (°C)', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Calculate statistics
    if len(llm_vals) > 0:
        rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
        mae = np.mean(np.abs(llm_vals - era5_vals))
        bias = np.mean(llm_vals - era5_vals)
        r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1] if len(llm_vals) > 1 else np.nan
        
        # Add compact statistics text
        stats_text = f'N={len(llm_vals)}\n'
        stats_text += f'RMSE={rmse:.1f}°C\n'
        stats_text += f'Bias={bias:+.1f}°C'
        if not np.isnan(r_corr):
            stats_text += f'\nR={r_corr:.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set title with elevation range and count
        elev_range = f"{np.min(elevations):.0f}-{np.max(elevations):.0f}m" if len(elevations) > 1 else f"{elevations[0]:.0f}m"
        ax.set_title(f'{cluster_label}\n(N={len(llm_vals)}, {elev_range})', fontsize=10, fontweight='bold')
        
        stats = (rmse, mae, bias, r_corr)
    else:
        ax.set_title(f'{cluster_label}\n(N=0)', fontsize=10, fontweight='bold')
        stats = (np.nan, np.nan, np.nan, np.nan)
    
    # Add legend only for first subplot
    if len(llm_vals) > 1:
        ax.legend(fontsize=7, loc='lower right')
    
    return stats

def create_elevation_cluster_plots(elevation_clusters, resolution, output_file):
    """Create 3x3 grid of elevation cluster plots"""
    
    print("Creating elevation cluster plots...")
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Define cluster order for 3x3 grid
    cluster_labels = [
        '0-500m', '500-1000m', '1000-1500m',
        '1500-2000m', '2000-2500m', '2500-3000m', 
        '3000-3500m', '3500-4000m', '4000m+'
    ]
    
    all_stats = []
    
    # Create each subplot
    for i, label in enumerate(cluster_labels):
        if label in elevation_clusters:
            cluster_data = elevation_clusters[label]['data']
            stats = create_single_cluster_plot(cluster_data, label, axes[i])
            all_stats.append((label, stats))
        else:
            axes[i].text(0.5, 0.5, 'No Data', transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{label}\n(N=0)', fontsize=10, fontweight='bold')
    
    # Add main title
    fig.suptitle(f'LLM vs ERA5 Temperature by Elevation Clusters ({resolution}° resolution)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Elevation cluster plot saved to: {output_file}")
    plt.close()
    
    return fig, all_stats

def print_elevation_cluster_summary(elevation_clusters):
    """Print summary statistics for elevation clusters"""
    
    print(f"\nElevation Cluster Summary:")
    print(f"=" * 70)
    
    total_points = sum(cluster['count'] for cluster in elevation_clusters.values())
    print(f"Total points distributed across clusters: {total_points}")
    
    for label, cluster_info in elevation_clusters.items():
        count = cluster_info['count']
        percentage = (count / total_points * 100) if total_points > 0 else 0
        print(f"{label:12s}: {count:5d} points ({percentage:5.1f}%)")

def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("LLM vs ERA5 Temperature Comparison - Elevation Clusters")
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
        comparison_data = extract_elevation_data(results_data)
        
        if len(comparison_data) == 0:
            print("Error: No valid comparison data found")
            return
        
        # Create elevation clusters
        elevation_clusters = create_elevation_clusters(comparison_data)
        
        # Print cluster summary
        print_elevation_cluster_summary(elevation_clusters)
        
        # Get resolution from results data
        resolution = results_data.get('resolution', 1.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cluster plots
        output_file = output_dir / f'elevation_clusters_{resolution}deg.png'
        
        fig, all_stats = create_elevation_cluster_plots(
            elevation_clusters,
            resolution,
            str(output_file)
        )
        
        if fig is not None:
            print(f"\nCluster Statistics:")
            print(f"-" * 50)
            for label, stats in all_stats:
                if stats and not np.isnan(stats[0]):
                    rmse, mae, bias, r_corr = stats
                    corr_str = f", R={r_corr:.3f}" if not np.isnan(r_corr) else ""
                    print(f"{label:12s}: RMSE={rmse:.2f}°C, MAE={mae:.2f}°C, Bias={bias:+.2f}°C{corr_str}")
        
        print(f"\nElevation cluster analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: plot_mesh.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from geo_mesh_processor import load_mesh_data

def plot_mesh_with_land(mesh_points, resolution, land_shapefile_path='./data/land/ne_10m_land.shp', 
                       output_file=None, figsize=(15, 10)):
    """Plot mesh points with land boundaries"""
    
    # Extract coordinates and land status
    lons = [point['lon'] for point in mesh_points]
    lats = [point['lat'] for point in mesh_points]
    is_land = [point['is_land'] for point in mesh_points]
    
    # Separate land and ocean points
    land_lons = [lon for lon, land in zip(lons, is_land) if land]
    land_lats = [lat for lat, land in zip(lats, is_land) if land]
    ocean_lons = [lon for lon, land in zip(lons, is_land) if not land]
    ocean_lats = [lat for lat, land in zip(lats, is_land) if not land]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ocean points (black)
    if ocean_lons:
        ax.scatter(ocean_lons, ocean_lats, c='black', s=10, alpha=0.6, label='Ocean points')
    
    # Plot land points (colored)
    if land_lons:
        ax.scatter(land_lons, land_lats, c='red', s=15, alpha=0.8, label='Land points')
    
    # Load and plot land boundaries from shapefile (on top)
    try:
        land_gdf = gpd.read_file(land_shapefile_path)
        land_gdf.plot(ax=ax, color='none', edgecolor='green', linewidth=0.8, alpha=0.9, zorder=10)
        print("Land boundaries loaded and plotted")
    except Exception as e:
        print(f"Could not load land shapefile: {e}")
        print("Plotting without land boundaries")
    
    # Customize the plot
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'Global Mesh Points ({resolution}° resolution)\nRed: Land, Black: Ocean, Green: Land boundaries', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text
    total_points = len(mesh_points)
    land_points = len(land_lons)
    ocean_points = len(ocean_lons)
    
    stats_text = f'Total points: {total_points}\nLand points: {land_points}\nOcean points: {ocean_points}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        # Create png directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    return fig, ax

def plot_mesh_countries(mesh_points, resolution, land_shapefile_path='./data/land/ne_10m_land.shp',
                       output_file=None, figsize=(15, 10)):
    """Plot mesh points colored by country"""
    
    # Extract data
    lons = []
    lats = []
    countries = []
    is_land = []
    
    for point in mesh_points:
        lons.append(point['lon'])
        lats.append(point['lat'])
        countries.append(point['country'] if point['country'] else 'Ocean')
        is_land.append(point['is_land'])
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'lon': lons,
        'lat': lats,
        'country': countries,
        'is_land': is_land
    })
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ocean points
    ocean_df = df[~df['is_land']]
    if not ocean_df.empty:
        ax.scatter(ocean_df['lon'], ocean_df['lat'], c='black', s=10, alpha=0.6, label='Ocean')
    
    # Plot land points by country
    land_df = df[df['is_land']]
    if not land_df.empty:
        unique_countries = land_df['country'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_countries)))
        
        for country, color in zip(unique_countries, colors):
            if country and country != 'Ocean':
                country_df = land_df[land_df['country'] == country]
                ax.scatter(country_df['lon'], country_df['lat'], c=[color], s=15, 
                          alpha=0.8, label=country if len(country) < 15 else country[:12] + '...')
    
    # Load and plot land boundaries (on top)
    try:
        land_gdf = gpd.read_file(land_shapefile_path)
        land_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.8, zorder=10)
    except Exception as e:
        print(f"Could not load land shapefile: {e}")
    
    # Customize the plot
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'Global Mesh Points by Country ({resolution}° resolution)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend (limit to avoid overcrowding)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 20:
        ax.legend(handles[:20], labels[:20], bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_file:
        # Create png directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """Main function to create plots"""
    import sys
    
    # Get input file from command line or use default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Look for mesh data files
        possible_files = list(Path('meshes').glob('mesh_data_*deg.json'))
        if possible_files:
            input_file = str(possible_files[0])
            print(f"Using mesh file: {input_file}")
        else:
            input_file = 'meshes/mesh_data_10deg.json'
            print(f"Using default file: {input_file}")
    
    try:
        # Load mesh data
        mesh_data = load_mesh_data(input_file)
        mesh_points = mesh_data['mesh_points']
        resolution = mesh_data['resolution']
        mesh_info = mesh_data['mesh_info']
        
        print(f"Loaded {len(mesh_points)} mesh points with {resolution}° resolution")
        print(f"Mesh info: {mesh_info}")
        
        # Create land/ocean plot
        print("Creating land/ocean plot...")
        plot_mesh_with_land(mesh_points, resolution, 
                           output_file=f'png/mesh_plot_{resolution}deg.png')
        
        # Create country plot if we have country data
        has_country_data = any(point['country'] for point in mesh_points if point['is_land'])
        if has_country_data:
            print("Creating country plot...")
            plot_mesh_countries(mesh_points, resolution,
                               output_file=f'png/mesh_countries_{resolution}deg.png')
        else:
            print("No country data found, skipping country plot")
    
    except FileNotFoundError:
        print(f"Error: Could not find mesh data file '{input_file}'")
        print("Please run geo_mesh_processor.py first to generate the data")
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    main()

### File: plot_population_clusters.py
#!/usr/bin/env python3
"""
Create LLM vs ERA5 temperature comparison scatter plots clustered by population density ranges.

This script creates population-based clusters and plots LLM vs ERA5 temperature comparisons
for each cluster in a 3×3 grid layout. Each subplot shows data for a specific population density range.

Population clusters (9 bins with log-like distribution):
- 0-1, 1-5, 5-10, 10-25, 25-50, 50-100, 100-250, 250-500, 500+ people/km²

Usage:
    python plot_population_clusters.py [results_file]

Default file: climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
Output: png/{results_filename}/population_clusters_{resolution}deg.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import json
from pathlib import Path
import sys
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
    
    if not (has_era5 and has_population):
        print("Warning: Missing required data (ERA5 or population)")
    
    return results_data

def extract_population_data(results_data):
    """Extract LLM vs ERA5 comparison data with population information"""
    
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with complete data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and
            'era5_temp_mean' in point_info and
            'population_density' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_temp_mean', np.nan)) and
            not np.isnan(point_info.get('population_density', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0),
                'era5_std': point_info.get('era5_temp_std', 0),
                'llm_count': point_info.get('llm_temp_count', 1),
                'population_density': point_info['population_density']
            })
    
    print(f"Extracted {len(comparison_data)} points with complete population and temperature data")
    return comparison_data

def create_population_clusters(comparison_data):
    """Create population density-based clusters"""
    
    # Define population bins (9 clusters with log-like distribution)
    population_bins = [
        (0, 1, '0-1'),
        (1, 5, '1-5'),
        (5, 10, '5-10'),
        (10, 25, '10-25'),
        (25, 50, '25-50'),
        (50, 100, '50-100'),
        (100, 250, '100-250'),
        (250, 500, '250-500'),
        (500, np.inf, '500+')
    ]
    
    # Create clusters
    population_clusters = {}
    
    for min_pop, max_pop, label in population_bins:
        cluster_data = []
        for data_point in comparison_data:
            population = data_point['population_density']
            if min_pop <= population < max_pop:
                cluster_data.append(data_point)
        
        population_clusters[label] = {
            'data': cluster_data,
            'range': (min_pop, max_pop),
            'count': len(cluster_data)
        }
        
        print(f"Population cluster {label} people/km²: {len(cluster_data)} points")
    
    return population_clusters

def create_single_cluster_plot(cluster_data, cluster_label, ax):
    """Create a single LLM vs ERA5 scatter plot for one population cluster"""
    
    if len(cluster_data) == 0:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(f'{cluster_label} people/km²\n(N=0)', fontsize=10, fontweight='bold')
        return None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in cluster_data])
    era5_vals = np.array([d['era5_temp'] for d in cluster_data])
    populations = np.array([d['population_density'] for d in cluster_data])
    
    # Handle color mapping for population within cluster
    if len(np.unique(populations)) > 1:
        # Use log scale if there's variation in population within cluster
        color_vals = np.log10(populations + 1)  # Add 1 to avoid log(0)
        colormap = 'YlOrRd'
    else:
        # Use linear scale if all populations are similar
        color_vals = populations
        colormap = 'YlOrRd'
    
    # Create scatter plot colored by population within cluster
    scatter = ax.scatter(era5_vals, llm_vals, c=color_vals, s=15, alpha=0.7, 
                        cmap=colormap, edgecolors='black', linewidth=0.1)
    
    # Add 1:1 line
    if len(llm_vals) > 0:
        min_temp = min(np.min(llm_vals), np.min(era5_vals))
        max_temp = max(np.max(llm_vals), np.max(era5_vals))
        ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, linewidth=1, 
                label='1:1')
        
        # Calculate and add regression line
        if len(llm_vals) > 1:  # Need at least 2 points for regression
            coeffs = np.polyfit(era5_vals, llm_vals, 1)
            regression_line = np.poly1d(coeffs)
            x_reg = np.linspace(min_temp, max_temp, 100)
            ax.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=1, 
                    label=f'R: {coeffs[0]:.2f}x+{coeffs[1]:.1f}')
    
    ax.set_xlabel('ERA5 Temp (°C)', fontsize=9)
    ax.set_ylabel('LLM Temp (°C)', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Calculate statistics
    if len(llm_vals) > 0:
        rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
        mae = np.mean(np.abs(llm_vals - era5_vals))
        bias = np.mean(llm_vals - era5_vals)
        r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1] if len(llm_vals) > 1 else np.nan
        
        # Add compact statistics text
        stats_text = f'N={len(llm_vals)}\n'
        stats_text += f'RMSE={rmse:.1f}°C\n'
        stats_text += f'Bias={bias:+.1f}°C'
        if not np.isnan(r_corr):
            stats_text += f'\nR={r_corr:.3f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set title with population range and count
        pop_range = f"{np.min(populations):.1f}-{np.max(populations):.1f}" if len(populations) > 1 else f"{populations[0]:.1f}"
        ax.set_title(f'{cluster_label} people/km²\n(N={len(llm_vals)}, {pop_range})', fontsize=10, fontweight='bold')
        
        stats = (rmse, mae, bias, r_corr)
    else:
        ax.set_title(f'{cluster_label} people/km²\n(N=0)', fontsize=10, fontweight='bold')
        stats = (np.nan, np.nan, np.nan, np.nan)
    
    # Add legend only for first subplot
    if len(llm_vals) > 1:
        ax.legend(fontsize=7, loc='lower right')
    
    return stats

def create_population_cluster_plots(population_clusters, resolution, output_file):
    """Create 3x3 grid of population cluster plots"""
    
    print("Creating population cluster plots...")
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Define cluster order for 3x3 grid
    cluster_labels = [
        '0-1', '1-5', '5-10',
        '10-25', '25-50', '50-100',
        '100-250', '250-500', '500+'
    ]
    
    all_stats = []
    
    # Create each subplot
    for i, label in enumerate(cluster_labels):
        if label in population_clusters:
            cluster_data = population_clusters[label]['data']
            stats = create_single_cluster_plot(cluster_data, label, axes[i])
            all_stats.append((label, stats))
        else:
            axes[i].text(0.5, 0.5, 'No Data', transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{label} people/km²\n(N=0)', fontsize=10, fontweight='bold')
    
    # Add main title
    fig.suptitle(f'LLM vs ERA5 Temperature by Population Density Clusters ({resolution}° resolution)', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Population cluster plot saved to: {output_file}")
    plt.close()
    
    return fig, all_stats

def print_population_cluster_summary(population_clusters):
    """Print summary statistics for population clusters"""
    
    print(f"\nPopulation Cluster Summary:")
    print(f"=" * 70)
    
    total_points = sum(cluster['count'] for cluster in population_clusters.values())
    print(f"Total points distributed across clusters: {total_points}")
    
    for label, cluster_info in population_clusters.items():
        count = cluster_info['count']
        percentage = (count / total_points * 100) if total_points > 0 else 0
        print(f"{label:12s} people/km²: {count:5d} points ({percentage:5.1f}%)")

def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("LLM vs ERA5 Temperature Comparison - Population Density Clusters")
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
        comparison_data = extract_population_data(results_data)
        
        if len(comparison_data) == 0:
            print("Error: No valid comparison data found")
            return
        
        # Create population clusters
        population_clusters = create_population_clusters(comparison_data)
        
        # Print cluster summary
        print_population_cluster_summary(population_clusters)
        
        # Get resolution from results data
        resolution = results_data.get('resolution', 1.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cluster plots
        output_file = output_dir / f'population_clusters_{resolution}deg.png'
        
        fig, all_stats = create_population_cluster_plots(
            population_clusters,
            resolution,
            str(output_file)
        )
        
        if fig is not None:
            print(f"\nCluster Statistics:")
            print(f"-" * 50)
            for label, stats in all_stats:
                if stats and not np.isnan(stats[0]):
                    rmse, mae, bias, r_corr = stats
                    corr_str = f", R={r_corr:.3f}" if not np.isnan(r_corr) else ""
                    print(f"{label:12s}: RMSE={rmse:.2f}°C, MAE={mae:.2f}°C, Bias={bias:+.2f}°C{corr_str}")
        
        print(f"\nPopulation cluster analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: plot_population_map.py
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

### File: plot_scatter_comparison.py
#!/usr/bin/env python3
"""
Scatter plot comparison between LLM and ERA5 temperature predictions.

This script creates scatter plots comparing LLM predictions vs ERA5 climatology
with statistical analysis and regression lines.

Usage:
    python plot_scatter_comparison.py results_file.json

Output: pub_f2_scatter_comparison_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
from scipy.stats import gaussian_kde

# Configuration section
# Font configuration for publication quality
FONT_FAMILY = 'Helvetica'  # Font family for all text elements
FONT_SIZE_LABEL = 12       # Font size for axis labels and colorbar labels
FONT_SIZE_TICK = 10        # Font size for tick labels (axis and colorbar)

# Axis display configuration
SHOW_AXES = True           # Show axis labels and tick marks (True) or clean plot (False)
SHOW_FRAME = True          # Show frame/box around the plot (True) or remove it (False)
SHOW_LEFT_AXIS = True      # Show left Y-axis spine and ticks (True) or hide (False)
SHOW_RIGHT_AXIS = False    # Show right Y-axis spine and ticks (True) or hide (False)
SHOW_TOP_AXIS = False      # Show top X-axis spine and ticks (True) or hide (False)
SHOW_BOTTOM_AXIS = True    # Show bottom X-axis spine and ticks (True) or hide (False)

# Colorbar configuration
COLORBAR_HORIZONTAL = True # Use horizontal colorbar at bottom (True) or vertical on right (False)
COLORBAR_UNITS = True      # Show units in colorbar label (True) or no units (False)
COLORBAR_PAD = 0.08        # Distance between plot and colorbar (0.05 = very close, 0.08 = close, 0.15 = further)
COLORBAR_MIN = 0.0         # Minimum colorbar value for density scale
COLORBAR_MAX = 0.012       # Maximum colorbar value for density scale

# Point appearance configuration
POINT_COLOR = 'density'    # Point color: 'density' for density-colored, or color name like 'blue', 'red', 'steelblue'
POINT_EDGE_COLOR = 'none' # Edge color around points: 'black', 'white', 'none', or any color
POINT_EDGE_WIDTH = 0.3     # Width of edge lines around points (0 = no edge)
POINT_SIZE = 30            # Size of scatter points


def extract_comparison_data(results_data):
    """Extract LLM vs ERA5 comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0.0),
                'era5_std': point_info.get('era5_temp_std', np.nan),
                'temp_diff': point_info.get('temp_difference', 
                                          point_info['llm_temp_mean'] - point_info['era5_temp_mean']),
                'llm_count': point_info.get('llm_temp_count', 1),
                'country': point_info.get('country', '')
            })
    
    print(f"Found {len(comparison_data)} valid comparison data points")
    return comparison_data


def create_scatter_plot(comparison_data, output_file=None, font_family='Helvetica', font_size_label=12, font_size_tick=10, show_axes=True, show_frame=True, show_left_axis=True, show_right_axis=False, show_top_axis=False, show_bottom_axis=True, colorbar_horizontal=True, colorbar_units=True, colorbar_pad=0.08, colorbar_min=0.0, colorbar_max=0.012, point_color='density', point_edge_color='black', point_edge_width=0.3, point_size=30):
    """Create density-colored scatter plot comparing LLM vs ERA5 with statistical analysis"""
    
    if not comparison_data:
        print("No comparison data available for scatter plot")
        return None, None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    llm_counts = np.array([d['llm_count'] for d in comparison_data])
    
    # Set font for matplotlib
    plt.rcParams['font.family'] = font_family
    
    # Create figure with square plot area (width x height)
    # Extra height for horizontal colorbar below
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 7.5))
    
    # Configure point appearance based on color option
    if point_color == 'density':
        # Calculate point density using KDE
        xy = np.vstack([era5_vals, llm_vals])
        try:
            kde = gaussian_kde(xy)
            density = kde(xy)
        except:
            # Fallback if KDE fails (e.g., all points identical)
            density = np.ones_like(era5_vals)
        
        # Sort points by density (plot low density first)
        idx = density.argsort()
        era5_sorted, llm_sorted, density_sorted = era5_vals[idx], llm_vals[idx], density[idx]
        
        # Create density scatter plot with specified colorbar range
        edge_colors = point_edge_color if point_edge_width > 0 else 'none'
        scatter = ax.scatter(era5_sorted, llm_sorted, c=density_sorted, s=point_size, alpha=0.7, 
                            cmap='viridis', edgecolors=edge_colors, linewidth=point_edge_width,
                            vmin=colorbar_min, vmax=colorbar_max)
        
        # Add colorbar for density
        colorbar_needed = True
    else:
        # Use solid color for all points
        edge_colors = point_edge_color if point_edge_width > 0 else 'none'
        scatter = ax.scatter(era5_vals, llm_vals, s=point_size, alpha=0.7, 
                            color=point_color, edgecolors=edge_colors, linewidth=point_edge_width)
        colorbar_needed = False
    
    # Add 1:1 line
    min_temp = min(np.min(llm_vals), np.min(era5_vals))
    max_temp = max(np.max(llm_vals), np.max(era5_vals))
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 'r--', alpha=0.8, linewidth=2, 
            label='1:1 line')
    
    # Calculate and add regression line
    coeffs = np.polyfit(era5_vals, llm_vals, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_temp, max_temp, 100)
    ax.plot(x_reg, regression_line(x_reg), 'b-', alpha=0.8, linewidth=2, 
            label=f'Regression: y = {coeffs[0]:.3f}x + {coeffs[1]:.2f}')
    
    # Configure axes and labels based on options
    if show_axes:
        ax.set_xlabel('ERA5 Temperature (°C)', fontsize=font_size_label, fontfamily=font_family)
        ax.set_ylabel('LLM Temperature (°C)', fontsize=font_size_label, fontfamily=font_family)
        # Configure tick labels
        ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
    
    # Configure frame and individual axis spines
    if not show_frame:
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        # Configure individual axis spines
        ax.spines['left'].set_visible(show_left_axis)
        ax.spines['right'].set_visible(show_right_axis)
        ax.spines['top'].set_visible(show_top_axis)
        ax.spines['bottom'].set_visible(show_bottom_axis)
        
        # Configure tick marks to match visible spines
        if show_axes:
            ax.tick_params(left=show_left_axis, right=show_right_axis, 
                          top=show_top_axis, bottom=show_bottom_axis,
                          labelleft=show_left_axis, labelright=show_right_axis,
                          labeltop=show_top_axis, labelbottom=show_bottom_axis)
    
    ax.grid(True, alpha=0.3)
    
    # Set square aspect ratio for the plot
    ax.set_aspect('equal', adjustable='box')
    
    # Configure legend font
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontfamily(font_family)
        text.set_fontsize(font_size_tick)
    
    # Add colorbar for density (only if using density coloring)
    if colorbar_needed:
        if colorbar_horizontal:
            cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=colorbar_pad)
        else:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=colorbar_pad)
        
        if show_axes:
            if colorbar_units:
                cbar.set_label('Point Density (°C⁻²)', fontsize=font_size_label, fontfamily=font_family)
            else:
                cbar.set_label('Point Density', fontsize=font_size_label, fontfamily=font_family)
            cbar.ax.tick_params(labelsize=font_size_tick)
        else:
            cbar.set_label('')
            if colorbar_horizontal:
                cbar.ax.set_xticklabels([])
            else:
                cbar.ax.set_yticklabels([])
    
    # Calculate statistics for return values
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"LLM vs ERA5 scatter plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def main():
    """Main function to create scatter plot comparison"""
    
    if len(sys.argv) < 2:
        print("Usage: python plot_scatter_comparison.py results_file.json")
        print("Example: python plot_scatter_comparison.py results/climate_results_1.0deg_r10_gpt-5_simple_spatial_rmse_bathymetry_population.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Check if results file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found")
        sys.exit(1)
    
    # Load results data
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract resolution from filename or results data
    resolution = results_data.get('resolution', '1.0')
    if isinstance(resolution, str) and 'deg' in resolution:
        resolution = resolution.replace('deg', '')
    
    # Extract comparison data
    comparison_data = extract_comparison_data(results_data)
    
    if not comparison_data:
        print("No comparison data available for plotting")
        return
    
    # Create subfolder based on results filename
    results_filename = Path(results_file).stem  # Get filename without extension
    output_dir = Path('png') / results_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create output filename with pub_f2_ prefix
    output_file = output_dir / f"pub_f2_scatter_comparison_{resolution}deg.png"
    
    # Create scatter plot
    fig, stats = create_scatter_plot(comparison_data, output_file, FONT_FAMILY, FONT_SIZE_LABEL, FONT_SIZE_TICK, SHOW_AXES, SHOW_FRAME, SHOW_LEFT_AXIS, SHOW_RIGHT_AXIS, SHOW_TOP_AXIS, SHOW_BOTTOM_AXIS, COLORBAR_HORIZONTAL, COLORBAR_UNITS, COLORBAR_PAD, COLORBAR_MIN, COLORBAR_MAX, POINT_COLOR, POINT_EDGE_COLOR, POINT_EDGE_WIDTH, POINT_SIZE)
    
    if fig is not None:
        rmse, mae, bias, r_corr = stats
        print(f"Scatter plot statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")


if __name__ == "__main__":
    main()

### File: plot_spatial_analysis.py
#!/usr/bin/env python3
"""
Comprehensive plotting script for spatial RMSE analysis results.

This script creates various maps from spatial RMSE-enhanced result files:
- RMSE maps (radius 2 and 4)
- Bias maps (radius 2 and 4) 
- Neighborhood mean temperature maps (LLM and ERA5, radius 2 and 4)
- Individual point temperature maps (LLM and ERA5)

Usage:
    python plot_spatial_analysis.py [results_file]

Default: results/climate_results_20.0deg_r10_simple_spatial_rmse.json
Output: png/{results_filename}/spatial_analysis_*_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
import pandas as pd
from pathlib import Path
import sys
import os
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_spatial_rmse_results(results_file):
    """Load spatial RMSE-enhanced results file"""
    print(f"Loading spatial RMSE results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check if spatial RMSE data is present
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False)
    if not has_spatial:
        print("Warning: Spatial RMSE data not found in results file")
        print("Please run extend_results_with_spatial_rmse.py first")
    
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


def create_data_grid(mapping_data, resolution=20.0):
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
                   vmin=None, vmax=None, title_suffix="", output_file=None, figsize=(18, 10)):
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
    
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    # Create contour levels
    levels = np.linspace(vmin, vmax, 20)
    
    # Create contour plot
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        # For error metrics, use reversed colormap (blue=good, red=bad)
        cmap = plt.cm.get_cmap('viridis_r')
    elif 'bias' in field_name.lower():
        # For bias, use diverging colormap (blue=negative, red=positive)
        cmap = plt.cm.RdBu_r
        # Make bias symmetric around zero
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        levels = np.linspace(vmin, vmax, 20)
    elif 'temp' in field_name.lower():
        # For temperature, use temperature colormap (blue=cold, red=hot)
        from matplotlib.colors import LinearSegmentedColormap
        temp_colors = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800000']
        cmap = LinearSegmentedColormap.from_list('temperature', temp_colors, N=256)
    else:
        cmap = plt.cm.get_cmap(colormap)
    
    contour = ax.contourf(lon_grid, lat_grid, data_grid, 
                         levels=levels, cmap=cmap, extend='both')
    
    # Try to load and overlay country boundaries
    try:
        world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
        world.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
    except Exception as e:
        print(f"Could not load country boundaries: {e}")
        ax.grid(True, alpha=0.3)
    
    # Customize the map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    
    # Create title
    clean_field_name = field_name.replace('_', ' ').title()
    ax.set_title(f'{clean_field_name} Map{title_suffix}\n'
                f'Land Points: {len(mapping_data):,}', 
                fontsize=14, fontweight='bold')
    
    # Add coordinate ticks
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 91, 30))
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    
    # Set colorbar label based on field type
    if 'rmse' in field_name.lower() or 'mae' in field_name.lower():
        cbar.set_label('Error (°C)', fontsize=12)
    elif 'bias' in field_name.lower():
        cbar.set_label('Bias (°C)', fontsize=12)
    elif 'temp' in field_name.lower():
        cbar.set_label('Temperature (°C)', fontsize=12)
    else:
        cbar.set_label('Value', fontsize=12)
    
    # Add statistics text
    mean_val = np.mean(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    std_val = np.std(values)
    
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean_val:.2f}\n'
    stats_text += f'Median: {median_val:.2f}\n'
    stats_text += f'Std: {std_val:.2f}\n'
    stats_text += f'Range: {min_val:.2f} - {max_val:.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"{clean_field_name} map saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def extract_comparison_data(results_data):
    """Extract LLM vs ERA5 comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0.0),
                'era5_std': point_info.get('era5_temp_std', np.nan),
                'temp_diff': point_info.get('temp_difference', 
                                          point_info['llm_temp_mean'] - point_info['era5_temp_mean']),
                'llm_count': point_info.get('llm_temp_count', 1),
                'country': point_info.get('country', '')
            })
    
    print(f"Found {len(comparison_data)} valid comparison data points")
    return comparison_data


def create_density_scatter_plot(comparison_data, resolution, output_file=None):
    """Create density scatter plot comparing LLM vs ERA5 with statistical analysis"""
    
    if not comparison_data:
        print("No comparison data available for scatter plot")
        return None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    llm_stds = np.array([d['llm_std'] if not np.isnan(d['llm_std']) else 0 for d in comparison_data])
    era5_stds = np.array([d['era5_std'] if not np.isnan(d['era5_std']) else 0 for d in comparison_data])
    llm_counts = np.array([d['llm_count'] for d in comparison_data])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([era5_vals, llm_vals])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
    except:
        # Fallback if KDE fails (e.g., all points identical)
        density = np.ones_like(era5_vals)
    
    # Sort points by density (plot low density first)
    idx = density.argsort()
    era5_sorted, llm_sorted, density_sorted = era5_vals[idx], llm_vals[idx], density[idx]
    llm_stds_sorted, era5_stds_sorted = llm_stds[idx], era5_stds[idx]
    
    # Create density scatter plot
    scatter = ax1.scatter(era5_sorted, llm_sorted, c=density_sorted, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
    # Error bars removed as requested
    
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
    ax1.set_title(f'LLM vs ERA5 Temperature Comparison\nDensity Scatter Plot ({resolution}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for density
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Point Density', fontsize=11)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # Add statistics text
    mean_llm_requests = np.mean(llm_counts)
    total_llm_requests = np.sum(llm_counts)
    stats_text = f'N = {len(llm_vals)} points\n'
    stats_text += f'Total LLM requests: {total_llm_requests}\n'
    stats_text += f'Avg requests/point: {mean_llm_requests:.1f}\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
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
    diff_stats = f'Difference Statistics:\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\n'
    diff_stats += f'IQR: {np.percentile(differences, 75) - np.percentile(differences, 25):.3f}°C\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"LLM vs ERA5 comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def create_spatial_comparison_plots(comparison_data, resolution, output_dir='png'):
    """Create spatial comparison scatter plots for neighborhood data"""
    
    print("Creating spatial neighborhood comparison plots...")
    
    # Create plots for different spatial scales
    spatial_comparisons = [
        ('r2', '5×5 neighborhood'),
        ('r4', '9×9 neighborhood')
    ]
    
    created_plots = []
    
    for radius, description in spatial_comparisons:
        # Extract data for this spatial scale
        llm_field = f'spatial_{radius}_neighborhood_llm_mean'
        era5_field = f'spatial_{radius}_neighborhood_era5_mean'
        
        spatial_data = []
        for result_data in comparison_data:
            # We need to get the original result to access spatial fields
            # For now, skip spatial neighborhood comparisons as we don't have direct access
            # This would require re-extracting from the original results_data
            pass
    
    return created_plots


def create_all_comparison_plots(results_data, resolution, output_dir='png'):
    """Create all LLM vs ERA5 comparison plots"""
    
    print("Creating LLM vs ERA5 comparison plots...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract comparison data
    comparison_data = extract_comparison_data(results_data)
    
    if not comparison_data:
        print("No comparison data available for plotting")
        return []
    
    created_plots = []
    
    # Create main density scatter plot with difference histogram
    scatter_output = output_path / f"llm_era5_comparison_{resolution}deg.png"
    fig, stats = create_density_scatter_plot(comparison_data, resolution, str(scatter_output))
    if fig is not None:
        created_plots.append(str(scatter_output))
        rmse, mae, bias, r_corr = stats
        print(f"Overall comparison statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
    
    return created_plots


def create_all_spatial_maps(results_data, resolution, output_dir='png'):
    """Create all spatial analysis maps from the results data"""
    
    print("Creating comprehensive spatial analysis maps...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Base filename pattern
    base_name = f"spatial_analysis_{resolution}deg"
    
    # Maps to create - field_name: (title_suffix, colormap)
    maps_config = {
        # RMSE maps
        'spatial_r2_rmse': ('(5×5 neighborhood)', 'viridis_r'),
        'spatial_r4_rmse': ('(9×9 neighborhood)', 'viridis_r'),
        
        # Bias maps  
        'spatial_r2_bias': ('(5×5 neighborhood)', 'RdBu_r'),
        'spatial_r4_bias': ('(9×9 neighborhood)', 'RdBu_r'),
        
        # Neighborhood LLM mean temperature
        'spatial_r2_neighborhood_llm_mean': ('(5×5 neighborhood)', 'coolwarm'),
        'spatial_r4_neighborhood_llm_mean': ('(9×9 neighborhood)', 'coolwarm'),
        
        # Neighborhood ERA5 mean temperature
        'spatial_r2_neighborhood_era5_mean': ('(5×5 neighborhood)', 'coolwarm'),
        'spatial_r4_neighborhood_era5_mean': ('(9×9 neighborhood)', 'coolwarm'),
        
        # Individual point temperatures
        'llm_temp_mean': ('(Individual Points)', 'coolwarm'),
        'era5_temp_mean': ('(Individual Points)', 'coolwarm'),
    }
    
    created_maps = []
    
    for field_name, (title_suffix, colormap) in maps_config.items():
        print(f"\nCreating map for {field_name}...")
        
        # Extract mapping data for this field
        mapping_data = extract_mapping_data(results_data, field_name)
        
        if not mapping_data:
            print(f"No data found for {field_name}, skipping...")
            continue
        
        # Create output filename
        clean_field_name = field_name.replace('spatial_', '').replace('_', '_')
        output_file = output_path / f"{base_name}_{clean_field_name}.png"
        
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
                output_file=str(output_file)
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
        'spatial_r2_rmse', 'spatial_r4_rmse',
        'spatial_r2_mae', 'spatial_r4_mae', 
        'spatial_r2_bias', 'spatial_r4_bias',
        'spatial_r2_correlation', 'spatial_r4_correlation',
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
    
    # Default file
    default_results = 'results/climate_results_20.0deg_r10_simple_spatial_rmse.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Comprehensive Spatial Analysis Plotting")
    print("=" * 70)
    print(f"Results file: {results_file}")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run extend_results_with_spatial_rmse.py first to create spatial RMSE data.")
        return
    
    try:
        # Load spatial RMSE results
        results_data = load_spatial_rmse_results(results_file)
        
        # Extract resolution from metadata
        resolution = results_data.get('resolution', 20.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem  # Get filename without extension
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create all spatial maps
        created_maps = create_all_spatial_maps(results_data, resolution, str(output_dir))
        
        # Create comparison plots
        created_comparison_plots = create_all_comparison_plots(results_data, resolution, str(output_dir))
        
        # Print summary statistics
        print_summary_statistics(results_data)
        
        total_plots = len(created_maps) + len(created_comparison_plots)
        print(f"\nPlotting completed successfully!")
        print(f"Created {len(created_maps)} spatial maps")
        print(f"Created {len(created_comparison_plots)} comparison plots")
        print(f"Total: {total_plots} plots")
        print(f"All files saved in: {output_dir}/")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

### File: plot_spatial_analysis_filtered.py
#!/usr/bin/env python3
"""
Filtered spatial RMSE analysis plotting script.

This script creates spatial analysis plots similar to plot_spatial_analysis.py but applies filters:
- Excludes points where population density < 5 people/km²
- Excludes points where elevation > 2000m

Creates the same maps as plot_spatial_analysis.py but with filtered data:
- RMSE maps (radius 2 and 4)
- Bias maps (radius 2 and 4) 
- Neighborhood mean temperature maps (LLM and ERA5, radius 2 and 4)
- Individual point temperature maps (LLM and ERA5)
- LLM vs ERA5 comparison plots with density coloring

Usage:
    python plot_spatial_analysis_filtered.py [results_file]

Default: results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
Output: png/{results_filename}/filtered_spatial_analysis_*_{resolution}deg.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
import pandas as pd
from pathlib import Path
import sys
import os
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_enhanced_results(results_file):
    """Load results file with spatial RMSE, population, and bathymetry data"""
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
    
    if not (has_era5 and has_spatial and has_population and has_bathymetry):
        print("Warning: Missing required data for filtering")
    
    return results_data


def extract_filtered_mapping_data(results_data, field_name, min_population=5.0, max_elevation=2000.0):
    """Extract mapping data for a specific field with population and elevation filtering"""
    mapping_data = []
    total_land_points = 0
    filtered_out_pop = 0
    filtered_out_elev = 0
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only consider land points
        if point_info.get('is_land', False):
            total_land_points += 1
            
            # Check if point has required data
            if (field_name in point_info and
                not np.isnan(point_info.get(field_name, np.nan)) and
                'population_density' in point_info and
                'mean_elevation' in point_info and
                not np.isnan(point_info.get('population_density', np.nan)) and
                not np.isnan(point_info.get('mean_elevation', np.nan))):
                
                pop_density = point_info['population_density']
                elevation = point_info['mean_elevation']
                
                # Apply filters
                if pop_density < min_population:
                    filtered_out_pop += 1
                    continue
                    
                if elevation > max_elevation:
                    filtered_out_elev += 1
                    continue
                
                # Point passes all filters
                mapping_data.append({
                    'lat': point_info['lat'],
                    'lon': point_info['lon'],
                    'value': point_info[field_name],
                    'population_density': pop_density,
                    'mean_elevation': elevation,
                    'country': point_info.get('country', ''),
                    'state': point_info.get('state', '')
                })
    
    print(f"Filtering results for {field_name}:")
    print(f"  Total land points: {total_land_points}")
    print(f"  Filtered out (pop < {min_population}): {filtered_out_pop}")
    print(f"  Filtered out (elev > {max_elevation}m): {filtered_out_elev}")
    print(f"  Final valid points: {len(mapping_data)}")
    
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
    data_grid = np.full(lon_grid.shape, np.nan)
    
    # Map data points to grid
    for point in mapping_data:
        lat_idx = np.argmin(np.abs(lats - point['lat']))
        lon_idx = np.argmin(np.abs(lons - point['lon']))
        data_grid[lat_idx, lon_idx] = point['value']
    
    print(f"Created {lon_grid.shape} grid, {np.sum(~np.isnan(data_grid))} filled cells")
    return lon_grid, lat_grid, data_grid


def load_world_boundaries():
    """Load world boundaries for map visualization"""
    try:
        # Try to load from standard location
        data_path = Path("data/land")
        if data_path.exists():
            world_files = list(data_path.glob("*land*.shp"))
            if world_files:
                world = gpd.read_file(world_files[0])
                print(f"Loaded world boundaries from: {world_files[0]}")
                return world
        
        # Try to use Natural Earth from geopandas
        try:
            import geopandas.datasets
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            print("Loaded world boundaries from geopandas Natural Earth data")
            return world
        except:
            pass
        
        print("Warning: Could not load world boundaries, maps will not show coastlines")
        return None
        
    except Exception as e:
        print(f"Warning: Could not load world boundaries: {e}")
        return None


def create_spatial_map(mapping_data, field_name, field_label, colormap, resolution, 
                      use_log_scale=False, output_file=None):
    """Create a spatial map for a given field"""
    
    if not mapping_data:
        print(f"No data available for {field_name}")
        return None
    
    # Create data grid
    lon_grid, lat_grid, data_grid = create_data_grid(mapping_data, resolution)
    
    if lon_grid is None:
        print(f"Could not create grid for {field_name}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Load world boundaries
    world = load_world_boundaries()
    if world is not None:
        world.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7)
    
    # Create color normalization
    valid_data = data_grid[~np.isnan(data_grid)]
    if len(valid_data) == 0:
        print(f"No valid data for {field_name}")
        return None
    
    if use_log_scale and np.all(valid_data > 0):
        norm = colors.LogNorm(vmin=np.min(valid_data), vmax=np.max(valid_data))
    else:
        norm = colors.Normalize(vmin=np.min(valid_data), vmax=np.max(valid_data))
    
    # Create the map
    im = ax.contourf(lon_grid, lat_grid, data_grid, levels=20, cmap=colormap, norm=norm, extend='both')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(field_label, fontsize=12)
    
    # Formatting
    ax.set_xlim(lon_grid.min(), lon_grid.max())
    ax.set_ylim(lat_grid.min(), lat_grid.max())
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'{field_label} Map (Filtered: pop≥5/km², elev≤2000m)\nLand Points: {len(mapping_data)}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add filter info
    filter_text = f'Filters: Population ≥ 5 people/km²\n         Elevation ≤ 2000m'
    ax.text(0.02, 0.98, filter_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save or show
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved filtered map: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig


def create_density_scatter_with_filters(results_data, resolution, output_file=None):
    """Create LLM vs ERA5 comparison plots with density coloring - filtered data"""
    
    # Extract comparison data with filters
    comparison_data = []
    total_land_points = 0
    filtered_out_pop = 0
    filtered_out_elev = 0
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only consider land points with complete data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and
            'era5_temp_mean' in point_info and
            'population_density' in point_info and
            'mean_elevation' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_temp_mean', np.nan)) and
            not np.isnan(point_info.get('population_density', np.nan)) and
            not np.isnan(point_info.get('mean_elevation', np.nan))):
            
            total_land_points += 1
            
            pop_density = point_info['population_density']
            elevation = point_info['mean_elevation']
            
            # Apply filters
            if pop_density < 5.0:
                filtered_out_pop += 1
                continue
                
            if elevation > 2000.0:
                filtered_out_elev += 1
                continue
            
            # Point passes all filters
            comparison_data.append({
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0),
                'population_density': pop_density,
                'mean_elevation': elevation
            })
    
    print(f"\nFiltering results for comparison plot:")
    print(f"  Total land points with data: {total_land_points}")
    print(f"  Filtered out (pop < 5): {filtered_out_pop}")
    print(f"  Filtered out (elev > 2000m): {filtered_out_elev}")
    print(f"  Final comparison points: {len(comparison_data)}")
    
    if len(comparison_data) == 0:
        print("No valid comparison data after filtering")
        return None, None
    
    # Extract arrays
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # === Left plot: Density scatter plot ===
    
    # Calculate point density using KDE
    xy = np.vstack([era5_vals, llm_vals])
    kde = gaussian_kde(xy)
    density = kde(xy)
    
    # Create scatter plot colored by density
    scatter = ax1.scatter(era5_vals, llm_vals, c=density, s=30, alpha=0.7, 
                         cmap='viridis', edgecolors='black', linewidth=0.3)
    
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
    ax1.set_title(f'LLM vs ERA5 Comparison - Filtered Data\nDensity Scatter Plot ({resolution}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar for density
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Point Density', fontsize=11)
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # Add statistics text
    stats_text = f'N = {len(llm_vals)} points\n'
    stats_text += f'RMSE = {rmse:.2f}°C\n'
    stats_text += f'MAE = {mae:.2f}°C\n'
    stats_text += f'Bias = {bias:+.2f}°C\n'
    stats_text += f'Correlation = {r_corr:.3f}\n\n'
    stats_text += f'Filters Applied:\n'
    stats_text += f'Pop ≥ 5 people/km²\n'
    stats_text += f'Elev ≤ 2000m'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # === Right plot: Temperature difference histogram ===
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
    ax2.set_title('Temperature Difference Distribution\n(Filtered Data)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add difference statistics
    diff_stats = f'Difference Statistics:\n'
    diff_stats += f'Mean: {np.mean(differences):+.3f}°C\n'
    diff_stats += f'Std: {np.std(differences):.3f}°C\n'
    diff_stats += f'Median: {np.median(differences):+.3f}°C\n'
    diff_stats += f'Range: {np.min(differences):+.2f} to {np.max(differences):+.2f}°C'
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Filtered comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def create_all_filtered_spatial_maps(results_data, resolution, output_dir):
    """Create all filtered spatial maps"""
    
    # Define map configurations
    spatial_maps = {
        # RMSE maps  
        'spatial_r2_rmse': ('Spatial RMSE R2 (5×5)', 'plasma'),
        'spatial_r4_rmse': ('Spatial RMSE R4 (9×9)', 'plasma'),
        
        # Bias maps
        'spatial_r2_bias': ('Spatial Bias R2 (5×5)', 'RdBu_r'),
        'spatial_r4_bias': ('Spatial Bias R4 (9×9)', 'RdBu_r'),
        
        # Neighborhood temperatures
        'spatial_r2_llm_temp_mean': ('Spatial LLM Temp R2 (5×5)', 'coolwarm'),
        'spatial_r2_era5_temp_mean': ('Spatial ERA5 Temp R2 (5×5)', 'coolwarm'),
        'spatial_r4_llm_temp_mean': ('Spatial LLM Temp R4 (9×9)', 'coolwarm'),
        'spatial_r4_era5_temp_mean': ('Spatial ERA5 Temp R4 (9×9)', 'coolwarm'),
        
        # Individual point temperatures
        'llm_temp_mean': ('LLM Temp Mean (Individual Points)', 'coolwarm'),
        'era5_temp_mean': ('ERA5 Temp Mean (Individual Points)', 'coolwarm'),
    }
    
    created_maps = []
    
    for field_name, (field_label, colormap) in spatial_maps.items():
        print(f"\nCreating filtered map for {field_name}...")
        
        # Extract filtered mapping data
        mapping_data = extract_filtered_mapping_data(results_data, field_name)
        
        if mapping_data:
            # Create output filename
            output_file = Path(output_dir) / f'filtered_spatial_analysis_{field_name}_{resolution}deg.png'
            
            # Create map
            fig = create_spatial_map(
                mapping_data, 
                field_name, 
                field_label, 
                colormap, 
                resolution,
                use_log_scale=(field_name in ['spatial_r2_rmse', 'spatial_r4_rmse']),
                output_file=str(output_file)
            )
            
            if fig is not None:
                created_maps.append(str(output_file))
    
    return created_maps


def create_all_filtered_comparison_plots(results_data, resolution, output_dir):
    """Create all filtered comparison plots"""
    
    print(f"\nCreating filtered LLM vs ERA5 comparison plots...")
    
    # Create comparison plot
    output_file = Path(output_dir) / f'filtered_llm_era5_comparison_{resolution}deg.png'
    
    fig, stats = create_density_scatter_with_filters(
        results_data,
        resolution, 
        str(output_file)
    )
    
    created_plots = []
    if fig is not None:
        created_plots.append(str(output_file))
        rmse, mae, bias, r_corr = stats
        print(f"Filtered comparison statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")
    
    return created_plots


def main():
    """Main function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("Filtered Spatial Analysis Plotting")
    print("=" * 50)
    print(f"Results file: {results_file}")
    print(f"Filters: Population ≥ 5 people/km², Elevation ≤ 2000m")
    print()
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # Load data
        results_data = load_enhanced_results(results_file)
        
        # Get resolution from results data
        resolution = results_data.get('resolution', 1.0)
        
        # Create subfolder based on results filename
        results_filename = Path(results_file).stem
        output_dir = Path('png') / results_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Create all filtered spatial maps
        print(f"\n{'='*60}")
        print("CREATING FILTERED SPATIAL MAPS")
        print(f"{'='*60}")
        
        created_maps = create_all_filtered_spatial_maps(results_data, resolution, str(output_dir))
        
        print(f"\nCreated {len(created_maps)} filtered spatial maps")
        
        # Create filtered comparison plots
        print(f"\n{'='*60}")
        print("CREATING FILTERED COMPARISON PLOTS")
        print(f"{'='*60}")
        
        created_comparison_plots = create_all_filtered_comparison_plots(results_data, resolution, str(output_dir))
        
        print(f"\nCreated {len(created_comparison_plots)} filtered comparison plots")
        
        # Summary
        total_plots = len(created_maps) + len(created_comparison_plots)
        print(f"\n{'='*60}")
        print("FILTERED SPATIAL ANALYSIS COMPLETED")
        print(f"{'='*60}")
        print(f"Total plots created: {total_plots}")
        print(f"Output directory: {output_dir}")
        
        print(f"\nFiltered plots saved:")
        for plot_file in created_maps + created_comparison_plots:
            print(f"  - {Path(plot_file).name}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

### File: plot_spatial_maps_only.py
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

# Optional Cartopy import (graceful fallback if not available)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Install with: pip install cartopy")
    print("Falling back to matplotlib-only plotting.")

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_cartopy_projection(projection_name, central_longitude=0):
    """Get Cartopy projection object from name with optional central longitude"""
    if not CARTOPY_AVAILABLE:
        return None
    
    # Projections that support central_longitude parameter
    central_longitude_projections = {
        'PlateCarree': lambda: ccrs.PlateCarree(central_longitude=central_longitude),
        'Robinson': lambda: ccrs.Robinson(central_longitude=central_longitude),
        'Mollweide': lambda: ccrs.Mollweide(central_longitude=central_longitude),
        'Mercator': lambda: ccrs.Mercator(central_longitude=central_longitude),
        'InterruptedGoodeHomolosine': lambda: ccrs.InterruptedGoodeHomolosine(central_longitude=central_longitude),
    }
    
    # Projections that don't support central_longitude (use default)
    fixed_projections = {
        'Orthographic': ccrs.Orthographic(),
        'NorthPolarStereo': ccrs.NorthPolarStereo(),
        'SouthPolarStereo': ccrs.SouthPolarStereo(),
    }
    
    if projection_name in central_longitude_projections:
        return central_longitude_projections[projection_name]()
    elif projection_name in fixed_projections:
        return fixed_projections[projection_name]
    else:
        # Default fallback
        return ccrs.PlateCarree(central_longitude=central_longitude)


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
                   rmse_range=None, rmse_colormap='viridis_r', font_family='Helvetica', 
                   font_size_label=10, font_size_tick=8, show_axes=True, show_frame=True,
                   use_cartopy=False, projection='PlateCarree', central_longitude=0, color_levels=20):
    """Create a contour map for the given field"""
    
    if not mapping_data:
        print(f"No data available for {field_name}")
        return None
    
    # Create regular grid
    lon_grid, lat_grid, data_grid = create_data_grid(mapping_data, resolution)
    
    if lon_grid is None:
        print(f"Failed to create grid for {field_name}")
        return None
    
    # Create figure with appropriate projection
    if use_cartopy and CARTOPY_AVAILABLE:
        proj = get_cartopy_projection(projection, central_longitude)
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
        cartopy_mode = True
    else:
        if use_cartopy and not CARTOPY_AVAILABLE:
            print(f"Warning: Cartopy requested but not available. Using matplotlib plotting.")
        fig, ax = plt.subplots(figsize=figsize)
        cartopy_mode = False
    
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
    levels = np.linspace(vmin, vmax, color_levels)
    
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
        # Update levels in case bias colormap handling changed them
        levels = np.linspace(vmin, vmax, color_levels)
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
    
    # Create contour plot with appropriate transform
    if cartopy_mode:
        contour = ax.contourf(lon_grid, lat_grid, data_grid, 
                             levels=levels, cmap=cmap, extend=extend_option,
                             transform=ccrs.PlateCarree())
    else:
        contour = ax.contourf(lon_grid, lat_grid, data_grid, 
                             levels=levels, cmap=cmap, extend=extend_option)
    
    # Add geographic features (coastlines and boundaries)
    if cartopy_mode:
        # Use Cartopy's built-in features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, color='gray', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, color='gray', alpha=0.5)
        # Optionally add land and ocean features
        # ax.add_feature(cfeature.LAND, alpha=0.1, color='lightgray')
        # ax.add_feature(cfeature.OCEAN, alpha=0.1, color='lightblue')
    else:
        # Use GeoPandas for matplotlib mode
        try:
            world = gpd.read_file('data/land/ne_10m_admin_0_countries.shp')
            world.boundary.plot(ax=ax, color='gray', linewidth=0.25, alpha=0.7)
        except Exception as e:
            print(f"Could not load country boundaries: {e}")
            # Add faint gridlines if no coastlines available
            ax.grid(True, color='lightgray', alpha=0.3, linewidth=0.5)
    
    # Add faint gridlines for better orientation
    ax.grid(True, color='lightgray', alpha=0.2, linewidth=0.3, linestyle='-')
    
    # Customize the map extent (exclude Antarctica by limiting to -57°S)
    if cartopy_mode:
        # Set custom extent to exclude Antarctica
        ax.set_extent([-180, 180, -57, 85], crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-57, 85)
    
    # Configure axis display based on settings
    if show_axes and not cartopy_mode:
        # Show axis labels and coordinate ticks (only for matplotlib mode)
        ax.set_xlabel('Longitude', fontsize=font_size_label, fontfamily=font_family)
        ax.set_ylabel('Latitude', fontsize=font_size_label, fontfamily=font_family)
        
        # Add coordinate ticks with proper geographic notation
        lon_ticks = np.arange(-180, 181, 60)
        lat_ticks = np.arange(-60, 91, 30)  # Keep -60 for reference even though map shows -57
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
        ax.set_xticklabels(lon_labels, fontsize=font_size_tick, fontfamily=font_family)
        
        # Format latitude labels: 60°S, 30°S, 0°, 30°N, 60°N
        lat_labels = []
        for lat in lat_ticks:
            if lat == 0:
                lat_labels.append('0°')
            elif lat > 0:
                lat_labels.append(f'{int(lat)}°N')
            else:
                lat_labels.append(f'{int(abs(lat))}°S')
        ax.set_yticklabels(lat_labels, fontsize=font_size_tick, fontfamily=font_family)
    elif show_axes and cartopy_mode:
        # For Cartopy, add gridlines with labels
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='lightgray', alpha=0.2)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': font_size_tick, 'family': font_family}
            gl.ylabel_style = {'size': font_size_tick, 'family': font_family}
        except:
            # Fallback for older Cartopy versions
            ax.gridlines(linewidth=0.3, color='lightgray', alpha=0.2)
    else:
        # Remove axis labels and ticks for clean publication look
        if not cartopy_mode:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False)
    
    # Configure frame/box around map
    if not show_frame:
        if cartopy_mode:
            # Remove Cartopy frame/outline - try different methods based on version
            try:
                ax.spines['geo'].set_visible(False)
            except KeyError:
                try:
                    ax.outline_patch.set_visible(False)
                except AttributeError:
                    # For newer Cartopy versions, remove all spines
                    for spine in ax.spines.values():
                        spine.set_visible(False)
        else:
            # Remove matplotlib spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # Minimal title - just the field name
    ax.set_title('', fontsize=12)  # Remove title per journal standards
    
    
    # Add horizontal colorbar beneath the map with appropriate ends
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=0.15)
    
    # Set colorbar label and ticks based on field type
    if is_rmse_field:
        cbar.set_label('RMSE (°C)', fontsize=font_size_label, fontfamily=font_family)
        # Set nice rounded ticks for RMSE colorbar - every 2 degrees
        if rmse_range is not None:
            rmse_ticks = np.arange(rmse_range[0], rmse_range[1] + 0.1, 2)  # Every 2 degrees: 0, 2, 4, 6, 8, 10
            cbar.set_ticks(rmse_ticks)
            cbar.set_ticklabels([f'{int(tick)}°' for tick in rmse_ticks], fontsize=font_size_tick, fontfamily=font_family)
    elif 'mae' in field_name.lower():
        cbar.set_label('MAE (°C)', fontsize=font_size_label, fontfamily=font_family)
    elif 'bias' in field_name.lower():
        cbar.set_label('Bias (°C)', fontsize=font_size_label, fontfamily=font_family)
        # Set nice rounded ticks for bias colorbar - every 2.5 degrees
        if bias_range is not None:
            bias_ticks = np.arange(bias_range[0], bias_range[1] + 0.1, 2.5)  # Every 2.5 degrees: -7.5, -5.0, -2.5, 0, 2.5, 5.0, 7.5
            cbar.set_ticks(bias_ticks)
            cbar.set_ticklabels([f'{tick:g}°' for tick in bias_ticks], fontsize=font_size_tick, fontfamily=font_family)
    elif is_temp_field:
        cbar.set_label('Temperature (°C)', fontsize=font_size_label, fontfamily=font_family)
        # Set nice rounded ticks for temperature colorbar - every 5 degrees
        if temp_range is not None:
            temp_ticks = np.arange(temp_range[0], temp_range[1] + 1, 5)  # Every 5 degrees: -10, -5, 0, 5, ..., 40
            cbar.set_ticks(temp_ticks)
            cbar.set_ticklabels([f'{int(tick)}°' for tick in temp_ticks], fontsize=font_size_tick, fontfamily=font_family)
    else:
        cbar.set_label('Value', fontsize=font_size_label, fontfamily=font_family)
    
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


def create_all_spatial_maps(results_data, resolution, output_dir='png', temp_range=(-13, 40), temp_colormap='thermal', bias_range=(-7, 7), bias_colormap='RdBu_r', rmse_range=(0, 10), rmse_colormap='viridis_r', font_family='Helvetica', font_size_label=10, font_size_tick=8, show_axes=True, show_frame=True, use_cartopy=False, projection='PlateCarree', central_longitude=0, color_levels=20):
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
                rmse_colormap=rmse_colormap,
                font_family=font_family,
                font_size_label=font_size_label,
                font_size_tick=font_size_tick,
                show_axes=show_axes,
                show_frame=show_frame,
                use_cartopy=use_cartopy,
                projection=projection,
                central_longitude=central_longitude,
                color_levels=color_levels
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
    
    # Font configuration for publication quality
    FONT_FAMILY = 'Helvetica'  # Font family for all text elements
    FONT_SIZE_LABEL = 12       # Font size for colorbar labels and axis labels
    FONT_SIZE_TICK = 10         # Font size for tick labels (axis and colorbar)
    
    # Axis display configuration
    SHOW_AXES = False          # Show axis labels and tick marks (True) or clean map (False)
    SHOW_FRAME = False         # Show frame/box around the map (True) or remove it (False)
    
    # Cartopy projection configuration
    USE_CARTOPY = True         # Use Cartopy for map projections (True) or matplotlib (False)
    PROJECTION = 'Robinson'    # Map projection when using Cartopy
    CENTRAL_LONGITUDE = 0      # Central longitude for projection (0=Greenwich, -30=Atlantic-centered, 180=Pacific-centered)
    
    # Color levels configuration
    COLOR_LEVELS = 21         # Number of color levels for contour plots (more levels = smoother gradients)
    #  Available projections:
    #  - 'Robinson' - Robinson projection (good for global views)
    #  - 'PlateCarree' - Equirectangular projection (simple lat/lon grid)
    #  - 'Mollweide' - Mollweide projection (equal-area)
    #  - 'Orthographic' - Orthographic projection (globe view)
    #  - 'NorthPolarStereo' - North Polar Stereographic
    #  - 'SouthPolarStereo' - South Polar Stereographic
    #  - 'Mercator' - Mercator projection
    #  - 'InterruptedGoodeHomolosine' - Interrupted Goode Homolosine
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
        created_maps = create_all_spatial_maps(results_data, resolution, str(output_dir), temp_range, TEMP_COLORMAP, bias_range, BIAS_COLORMAP, rmse_range, RMSE_COLORMAP, FONT_FAMILY, FONT_SIZE_LABEL, FONT_SIZE_TICK, SHOW_AXES, SHOW_FRAME, USE_CARTOPY, PROJECTION, CENTRAL_LONGITUDE, COLOR_LEVELS)
        
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

### File: plot_temperature_comparison_colored.py
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

### File: plot_temperature_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
from pathlib import Path
from geo_mesh_processor import load_mesh_data

def load_temperature_results(results_file):
    """Load temperature results from benchmark file"""
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract temperature data from results
    temperature_data = {}
    
    for result in results_data['results']:
        point_info = result['point_info']
        lat, lon = point_info['lat'], point_info['lon']
        
        # Extract temperature values from all LLM responses
        temps = []
        for response in result['llm_responses']:
            if response and 'parsed_data' in response:
                # Find temperature key (e.g., 'july_temp_mean', 'january_temp_mean')
                parsed = response['parsed_data']
                temp_key = next((k for k in parsed.keys() if k.endswith('_temp_mean')), None)
                if temp_key:
                    temps.append(parsed[temp_key])
        
        if temps:
            temperature_data[(lat, lon)] = temps
    
    return temperature_data, results_data['metadata']

def create_temperature_colormap():
    """Create a temperature colormap from cold (blue) to hot (red)"""
    # Custom temperature colormap: blue -> cyan -> green -> yellow -> orange -> red
    colors_list = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#800000']
    n_bins = 256
    cmap = colors.LinearSegmentedColormap.from_list('temperature', colors_list, N=n_bins)
    return cmap

def plot_temperature_map(mesh_data, temperature_data, temp_values, title_suffix="", 
                        land_shapefile_path='./data/land/ne_10m_land.shp',
                        output_file=None, figsize=(15, 10), vmin=None, vmax=None):
    """Plot temperature map with color-coded temperature values"""
    
    # Get all mesh points
    mesh_points = mesh_data['mesh_points']
    
    # Collect only points that have temperature data
    land_coords = []
    temp_colors = []
    
    # Create temperature colormap
    temp_cmap = create_temperature_colormap()
    
    # Set temperature range if not provided
    if vmin is None or vmax is None:
        all_temps = list(temp_values.values())
        if all_temps:
            if vmin is None:
                vmin = min(all_temps)
            if vmax is None:
                vmax = max(all_temps)
        else:
            vmin, vmax = 0, 30  # Default range
    
    # Process all mesh points: only keep land points that have temperature data
    for point in mesh_points:
        lat, lon = point['lat'], point['lon']
        if point['is_land'] and (lat, lon) in temp_values and not np.isnan(temp_values[(lat, lon)]):
            temp = temp_values[(lat, lon)]
            # Normalize temperature to [0, 1] range for colormap
            normalized_temp = (temp - vmin) / (vmax - vmin) if vmax != vmin else 0.5
            normalized_temp = max(0, min(1, normalized_temp))  # Clamp to [0, 1]
            land_coords.append([lon, lat])
            temp_colors.append(temp_cmap(normalized_temp))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot land points with temperature colors
    if land_coords:
        land_coords = np.array(land_coords)
        scatter = ax.scatter(land_coords[:, 0], land_coords[:, 1], c=temp_colors, s=15,
                           alpha=0.9, label='Temperature data points')
    
    # Load and plot land boundaries (on top)
    try:
        land_gdf = gpd.read_file(land_shapefile_path)
        land_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.8, zorder=10)
    except Exception as e:
        print(f"Could not load land shapefile: {e}")
    
    # Customize the plot
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.set_title(f'Temperature Map{title_suffix}\nRange: {vmin:.1f}°C to {vmax:.1f}°C', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=temp_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Add statistics
    temp_points = len([t for t in temp_values.values() if not np.isnan(t)])
    stats_text = f'Temperature data points: {temp_points}'
    if temp_points > 0:
        temps_list = [t for t in temp_values.values() if not np.isnan(t)]
        mean_temp = np.mean(temps_list)
        std_temp = np.std(temps_list)
        stats_text += f'\nMean: {mean_temp:.1f}°C\nStd: {std_temp:.1f}°C'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        # Create png directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Temperature map saved to {output_file}")
    else:
        plt.show()
    
    return fig, ax, vmin, vmax

def create_temperature_maps(mesh_file, results_file, output_prefix="temperature_map"):
    """Create temperature maps from mesh and results data"""
    
    # Load mesh data
    print(f"Loading mesh data from {mesh_file}...")
    mesh_data = load_mesh_data(mesh_file)
    resolution = mesh_data['resolution']
    
    # Load temperature results
    print(f"Loading temperature results from {results_file}...")
    temperature_data, metadata = load_temperature_results(results_file)
    
    print(f"Found temperature data for {len(temperature_data)} points")
    
    if not temperature_data:
        print("No temperature data found in results file")
        return
    
    # Determine the number of requests per point
    max_requests = max(len(temps) for temps in temperature_data.values())
    print(f"Maximum number of requests per point: {max_requests}")
    
    # Get global temperature range for consistent color scaling
    all_temps = [temp for temps in temperature_data.values() for temp in temps]
    global_vmin = min(all_temps)
    global_vmax = max(all_temps)
    print(f"Temperature range: {global_vmin:.1f}°C to {global_vmax:.1f}°C")
    
    # Create individual maps for each request series
    if max_requests > 1:
        for request_idx in range(max_requests):
            print(f"Creating map for request series {request_idx + 1}/{max_requests}...")
            
            # Extract temperatures for this request series
            temp_values = {}
            for (lat, lon), temps in temperature_data.items():
                if len(temps) > request_idx:
                    temp_values[(lat, lon)] = temps[request_idx]
                else:
                    temp_values[(lat, lon)] = np.nan
            
            # Create map
            title_suffix = f" - Request Series {request_idx + 1}"
            output_file = f"png/{output_prefix}_{resolution}deg_series_{request_idx + 1}.png"
            plot_temperature_map(mesh_data, temperature_data, temp_values, title_suffix, 
                               output_file=output_file, vmin=global_vmin, vmax=global_vmax)
    
    # Create mean temperature map
    print("Creating mean temperature map...")
    mean_temps = {}
    std_temps = {}
    
    for (lat, lon), temps in temperature_data.items():
        if temps:
            mean_temps[(lat, lon)] = np.mean(temps)
            std_temps[(lat, lon)] = np.std(temps) if len(temps) > 1 else 0.0
        else:
            mean_temps[(lat, lon)] = np.nan
            std_temps[(lat, lon)] = np.nan
    
    # Mean temperature map
    title_suffix = f" - Mean Temperature"
    if max_requests > 1:
        title_suffix += f" ({max_requests} requests)"
    
    output_file = f"png/{output_prefix}_{resolution}deg_mean.png"
    plot_temperature_map(mesh_data, temperature_data, mean_temps, title_suffix,
                        output_file=output_file, vmin=global_vmin, vmax=global_vmax)
    
    # Standard deviation map (if multiple requests)
    if max_requests > 1:
        print("Creating temperature standard deviation map...")
        
        # Use different color scale for standard deviation (0 to max std)
        std_values = [s for s in std_temps.values() if not np.isnan(s)]
        if std_values:
            std_vmin = 0
            std_vmax = max(std_values)
            
            title_suffix = f" - Temperature Standard Deviation"
            output_file = f"png/{output_prefix}_{resolution}deg_std.png"
            
            # Create plot with different colormap for std dev
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # Get all mesh points
            mesh_points = mesh_data['mesh_points']
            land_coords = []
            std_colors = []
            
            # Use 'viridis' colormap for standard deviation (purple to yellow)
            std_cmap = plt.cm.viridis
            
            for point in mesh_points:
                lat, lon = point['lat'], point['lon']
                if point['is_land'] and (lat, lon) in std_temps and not np.isnan(std_temps[(lat, lon)]):
                    land_coords.append([lon, lat])
                    std_val = std_temps[(lat, lon)]
                    normalized_std = std_val / std_vmax if std_vmax > 0 else 0
                    std_colors.append(std_cmap(normalized_std))
            
            # Plot land points with std colors
            if land_coords:
                land_coords = np.array(land_coords)
                ax.scatter(land_coords[:, 0], land_coords[:, 1], c=std_colors, s=15, alpha=0.9)
            
            # Load land boundaries
            try:
                land_gdf = gpd.read_file('./data/land/ne_10m_land.shp')
                land_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, alpha=0.8, zorder=10)
            except:
                pass
            
            # Customize plot
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 85)
            ax.set_xlabel('Longitude (degrees)', fontsize=12)
            ax.set_ylabel('Latitude (degrees)', fontsize=12)
            ax.set_title(f'Temperature Standard Deviation{title_suffix}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for std dev
            sm = plt.cm.ScalarMappable(cmap=std_cmap, norm=plt.Normalize(vmin=std_vmin, vmax=std_vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label('Temperature Std Dev (°C)', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Standard deviation map saved to {output_file}")

def main():
    """Main function"""
    import sys
    
    # Default files
    mesh_file = 'meshes/mesh_data_10.0deg.json'
    results_file = 'results/climate_results_10.0deg_r10_simple.json'
    output_prefix = 'temperature_map'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    if len(sys.argv) > 2:
        results_file = sys.argv[2] 
    if len(sys.argv) > 3:
        output_prefix = sys.argv[3]
    
    print(f"Temperature Results Plotter")
    print(f"Mesh file: {mesh_file}")
    print(f"Results file: {results_file}")
    print(f"Output prefix: {output_prefix}")
    
    # Check if files exist
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        return
    
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        create_temperature_maps(mesh_file, results_file, output_prefix)
        print("Temperature mapping completed successfully!")
        
    except Exception as e:
        print(f"Error creating temperature maps: {e}")

if __name__ == "__main__":
    main()

### File: point_rmse_analysis.py
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

### File: process_era5_climatology.py
import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

def calculate_period_climatology(ds, start_date, end_date, period_name):
    """
    Calculate monthly climatology for a specific time period
    
    Parameters:
    ds: xarray Dataset with ERA5 data
    start_date: Start date string (YYYY-MM-DD)
    end_date: End date string (YYYY-MM-DD)
    period_name: Name of the period for metadata
    
    Returns:
    xarray Dataset with climatology statistics
    """
    print(f"Processing climatology for period {start_date} to {end_date}...")
    
    # Filter data for the specified period
    ds_filtered = ds.sel(valid_time=slice(start_date, end_date))
    
    print(f"Filtered dataset shape: {ds_filtered.t2m.shape}")
    print(f"Time range: {ds_filtered.valid_time.min().values} to {ds_filtered.valid_time.max().values}")
    
    # Add month coordinate
    ds_filtered = ds_filtered.assign_coords(month=ds_filtered['valid_time'].dt.month)
    
    # Group by month and calculate statistics
    monthly_mean = ds_filtered['t2m'].groupby('month').mean('valid_time')
    monthly_min = ds_filtered['t2m'].groupby('month').min('valid_time')
    monthly_max = ds_filtered['t2m'].groupby('month').max('valid_time')
    monthly_std = ds_filtered['t2m'].groupby('month').std('valid_time')
    
    # Create output dataset with period-specific variable names
    period_suffix = period_name.replace('-', '_')
    climatology_ds = xr.Dataset({
        f't2m_mean_{period_suffix}': monthly_mean,
        f't2m_min_{period_suffix}': monthly_min,
        f't2m_max_{period_suffix}': monthly_max,
        f't2m_std_{period_suffix}': monthly_std
    })
    
    # Add attributes
    for var_type in ['mean', 'min', 'max', 'std']:
        var_name = f't2m_{var_type}_{period_suffix}'
        climatology_ds[var_name].attrs = {
            'long_name': f'2m temperature climatological {var_type} ({period_name})',
            'units': '°C',
            'standard_name': 'air_temperature',
            'period': period_name
        }
    
    return climatology_ds


def process_era5_climatology(input_file, output_dir='data'):
    """
    Process ERA5 temperature data to create monthly climatologies for multiple periods
    and calculate climate change signals
    
    Parameters:
    input_file: Path to NetCDF file with ERA5 data
    output_dir: Directory to save output files
    """
    
    print(f"Loading ERA5 data from {input_file}...")
    
    # Load the dataset
    ds = xr.open_dataset(input_file)
    
    print("Dataset loaded successfully")
    print(f"Dataset dimensions: {dict(ds.dims)}")
    print(f"Temperature variable shape: {ds.t2m.shape}")
    
    # Convert valid_time to datetime
    ds['valid_time'] = pd.to_datetime(ds['valid_time'])
    
    # Convert temperature from Kelvin to Celsius
    print("Converting temperature from Kelvin to Celsius...")
    ds['t2m'] = ds['t2m'] - 273.15
    ds['t2m'].attrs['units'] = '°C'
    ds['t2m'].attrs['long_name'] = '2 metre temperature'
    
    # Define the three time periods
    periods = {
        '1950-1974': ('1950-01-01', '1974-12-31'),
        '1991-2020': ('1991-01-01', '2020-12-31'),
        '2000-2024': ('2000-01-01', '2024-12-31')
    }
    
    print("Calculating climatologies for multiple periods...")
    
    # Calculate climatology for each period
    climatology_datasets = {}
    for period_name, (start_date, end_date) in periods.items():
        climatology_datasets[period_name] = calculate_period_climatology(
            ds, start_date, end_date, period_name
        )
    
    # Merge all climatology datasets
    print("Merging climatology datasets...")
    climatology_ds = xr.merge(list(climatology_datasets.values()))
    
    # Calculate climate change signal (2000-2024 minus 1950-1974)
    print("Calculating climate change signal (2000-2024 minus 1950-1974)...")
    climate_change_mean = (climatology_ds['t2m_mean_2000_2024'] - 
                          climatology_ds['t2m_mean_1950_1974'])
    climate_change_min = (climatology_ds['t2m_min_2000_2024'] - 
                         climatology_ds['t2m_min_1950_1974'])
    climate_change_max = (climatology_ds['t2m_max_2000_2024'] - 
                         climatology_ds['t2m_max_1950_1974'])
    
    # Add climate change variables to dataset
    climatology_ds['t2m_change_mean'] = climate_change_mean
    climatology_ds['t2m_change_min'] = climate_change_min
    climatology_ds['t2m_change_max'] = climate_change_max
    
    # Add attributes for climate change variables
    climatology_ds['t2m_change_mean'].attrs = {
        'long_name': '2m temperature change signal mean (2000-2024 minus 1950-1974)',
        'units': '°C',
        'standard_name': 'air_temperature',
        'description': 'Climate change signal calculated as difference between 2000-2024 and 1950-1974 climatologies'
    }
    
    climatology_ds['t2m_change_min'].attrs = {
        'long_name': '2m temperature change signal minimum (2000-2024 minus 1950-1974)',
        'units': '°C',
        'standard_name': 'air_temperature',
        'description': 'Climate change signal calculated as difference between 2000-2024 and 1950-1974 climatologies'
    }
    
    climatology_ds['t2m_change_max'].attrs = {
        'long_name': '2m temperature change signal maximum (2000-2024 minus 1950-1974)',
        'units': '°C',
        'standard_name': 'air_temperature',
        'description': 'Climate change signal calculated as difference between 2000-2024 and 1950-1974 climatologies'
    }
    
    # Add global attributes
    climatology_ds.attrs = {
        'title': 'ERA5 2m Temperature Monthly Climatologies and Climate Change Signal',
        'source': 'ERA5 reanalysis data',
        'institution': 'European Centre for Medium-Range Weather Forecasts',
        'periods': '1950-1974, 1991-2020, 2000-2024',
        'climate_change_signal': '2000-2024 minus 1950-1974',
        'created_by': 'process_era5_climatology.py',
        'variable': '2m temperature (t2m)',
        'units': 'degrees Celsius',
        'Conventions': 'CF-1.7',
        'description': 'Multi-period climatologies and climate change analysis'
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save to NetCDF file
    output_filename = f"t2m_climatology_multi_period.nc"
    output_filepath = output_path / output_filename
    
    print(f"Saving multi-period climatology to {output_filepath}...")
    
    # Create encoding for all variables
    encoding = {}
    for var_name in climatology_ds.data_vars:
        encoding[var_name] = {'zlib': True, 'complevel': 4}
    
    climatology_ds.to_netcdf(output_filepath, encoding=encoding)
    
    print("Multi-period climatology processing completed successfully!")
    
    # Print summary statistics for all periods
    print("\nSummary Statistics by Period:")
    print("="*80)
    
    for period in ['1950_1974', '1991_2020', '2000_2024']:
        period_display = period.replace('_', '-')
        print(f"\n{period_display} Climatology:")
        print("-" * 40)
        
        for month in range(1, 13):
            month_name = pd.to_datetime(f'2000-{month:02d}-01').strftime('%B')
            mean_global = float(climatology_ds[f't2m_mean_{period}'].sel(month=month).mean())
            min_global = float(climatology_ds[f't2m_min_{period}'].sel(month=month).min())
            max_global = float(climatology_ds[f't2m_max_{period}'].sel(month=month).max())
            
            print(f"{month_name:>9}: Mean={mean_global:6.2f}°C, Min={min_global:7.2f}°C, Max={max_global:6.2f}°C")
    
    # Print climate change signal statistics
    print("\nClimate Change Signal (2000-2024 minus 1950-1974):")
    print("-" * 50)
    
    for month in range(1, 13):
        month_name = pd.to_datetime(f'2000-{month:02d}-01').strftime('%B')
        change_mean = float(climatology_ds['t2m_change_mean'].sel(month=month).mean())
        change_min = float(climatology_ds['t2m_change_min'].sel(month=month).mean())
        change_max = float(climatology_ds['t2m_change_max'].sel(month=month).mean())
        
        print(f"{month_name:>9}: ΔMean={change_mean:+6.2f}°C, ΔMin={change_min:+6.2f}°C, ΔMax={change_max:+6.2f}°C")
    
    # Global annual statistics
    print("\nGlobal Annual Statistics:")
    print("-" * 30)
    
    for period in ['1950_1974', '1991_2020', '2000_2024']:
        period_display = period.replace('_', '-')
        annual_mean = float(climatology_ds[f't2m_mean_{period}'].mean())
        print(f"{period_display}: {annual_mean:6.2f}°C")
    
    annual_change = float(climatology_ds['t2m_change_mean'].mean())
    print(f"Climate Change: {annual_change:+6.2f}°C")
    
    return climatology_ds, output_filepath


def main():
    """Main function"""
    import sys
    
    # Default input file
    input_file = 'data/data_stream-moda_stepType-avgua.nc'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    print("ERA5 Climatology Processor")
    print(f"Input file: {input_file}")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    try:
        # Process ERA5 data
        _, output_file = process_era5_climatology(input_file)
        
        print(f"\nClimatology saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing climatology: {e}")

if __name__ == "__main__":
    main()

### File: run_complete_analysis_pipeline.py
#!/usr/bin/env python3
"""
Complete analysis pipeline for LLM climate benchmark results.

This script runs the complete analysis pipeline:
1. Extends results with spatial RMSE data
2. Adds bathymetry/elevation data  
3. Adds population density data
4. Generates spatial analysis plots
5. Generates temperature comparison plots
6. Generates elevation clustering plots
7. Generates population clustering plots
8. Generates bathymetry maps and comparisons
9. Generates population maps and comparisons
10. Generates filtered spatial analysis plots (pop≥5/km², elev≤2000m)

Usage:
    python run_complete_analysis_pipeline.py [original_results_file]

Default file: climate_results_1.0deg_r10_simple.json
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, args):
    """Run a Python script with arguments"""
    subprocess.run(['python', script_name] + args, check=True)

def main():
    """Main pipeline function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple.json'
    
    # Parse command line arguments
    original_results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    # Define intermediate and final filenames
    original_path = Path(original_results_file)
    base_name = original_path.stem
    parent_dir = original_path.parent
    
    # Generate progressive filenames
    spatial_file = parent_dir / f"{base_name}_spatial_rmse.json"
    bathymetry_file = parent_dir / f"{base_name}_spatial_rmse_bathymetry.json" 
    final_file = parent_dir / f"{base_name}_spatial_rmse_bathymetry_population.json"
    
    # Step 1: Add spatial RMSE data
    run_script('extend_results_with_spatial_rmse.py', [original_results_file])
    
    # Step 2: Add bathymetry data
    run_script('add_bathymetry_to_results.py', [str(spatial_file)])
    
    # Step 3: Add population data
    run_script('add_population_to_results.py', [str(bathymetry_file)])
    
    # Step 4: Generate spatial analysis plots
    run_script('plot_spatial_analysis.py', [str(final_file)])
    
    # Step 5: Generate temperature comparison plots
    run_script('plot_temperature_comparison_colored.py', [str(final_file)])
    
    # Step 6: Generate elevation clustering plots
    run_script('plot_elevation_clusters.py', [str(final_file)])
    
    # Step 7: Generate population clustering plots
    run_script('plot_population_clusters.py', [str(final_file)])
    
    # Step 8: Generate bathymetry maps and comparisons
    run_script('plot_bathymetry_map.py', [str(final_file)])
    
    # Step 9: Generate population maps and comparisons
    run_script('plot_population_map.py', [str(final_file)])
    
    # Step 10: Generate filtered spatial analysis plots
    run_script('plot_spatial_analysis_filtered.py', [str(final_file)])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

### File: spatial_rmse_analysis.py
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

### File: split_mesh.py
#!/usr/bin/env python3
"""
Split mesh file into multiple chunks with equal distribution of land points.

Usage:
    python split_mesh.py mesh_file.json [num_chunks]
    
Default num_chunks is 20.
"""

import json
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict

def load_mesh_data(mesh_file: str) -> Dict:
    """Load mesh data from JSON file"""
    print(f"Loading mesh data from {mesh_file}...")
    
    with open(mesh_file, 'r') as f:
        mesh_data = json.load(f)
    
    return mesh_data

def split_land_points_equally(land_points: List[Dict], num_chunks: int = 20) -> List[List[Dict]]:
    """Split land points into approximately equal chunks"""
    total_points = len(land_points)
    points_per_chunk = total_points // num_chunks
    remainder = total_points % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Distribute remainder across first chunks
        chunk_size = points_per_chunk + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        
        chunk_points = land_points[start_idx:end_idx]
        chunks.append(chunk_points)
        
        print(f"Chunk {i+1:2d}: {len(chunk_points):4d} points (indices {start_idx:4d}-{end_idx-1:4d})")
        start_idx = end_idx
    
    return chunks

def create_chunk_mesh(original_mesh: Dict, land_points_chunk: List[Dict], chunk_id: int) -> Dict:
    """Create a new mesh data structure for a chunk"""
    
    # Create new mesh with only the land points in this chunk
    chunk_mesh = {
        'mesh_info': {
            **original_mesh['mesh_info'],
            'chunk_id': chunk_id,
            'total_chunks': None,  # Will be set when saving
            'land_points_in_chunk': len(land_points_chunk)
        },
        'resolution': original_mesh['resolution'],
        'mesh_points': land_points_chunk
    }
    
    return chunk_mesh

def save_chunk_mesh(chunk_mesh: Dict, output_file: str):
    """Save chunk mesh to JSON file"""
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(chunk_mesh, f, indent=2)
    
    print(f"Saved chunk to {output_file}")

def split_mesh_file(mesh_file: str, num_chunks: int = 20):
    """Split mesh file into multiple chunks"""
    
    # Load original mesh
    mesh_data = load_mesh_data(mesh_file)
    
    # Extract land points
    all_points = mesh_data['mesh_points']
    land_points = [point for point in all_points if point.get('is_land', False)]
    
    print(f"Total mesh points: {len(all_points)}")
    print(f"Land points: {len(land_points)}")
    print(f"Splitting into {num_chunks} chunks...\n")
    
    # Split land points into chunks
    point_chunks = split_land_points_equally(land_points, num_chunks)
    
    # Create output file pattern
    input_path = Path(mesh_file)
    base_name = input_path.stem  # filename without extension
    output_dir = input_path.parent / "chunks"
    
    # Save each chunk
    for i, chunk_points in enumerate(point_chunks):
        chunk_mesh = create_chunk_mesh(mesh_data, chunk_points, i + 1)
        chunk_mesh['mesh_info']['total_chunks'] = num_chunks
        
        output_file = output_dir / f"{base_name}_chunk_{i+1:02d}_of_{num_chunks:02d}.json"
        save_chunk_mesh(chunk_mesh, str(output_file))
    
    print(f"\nSplit complete!")
    print(f"Created {num_chunks} chunk files in {output_dir}/")
    print(f"Each chunk contains land points only")
    
    # Print summary
    total_land_points = sum(len(chunk) for chunk in point_chunks)
    min_points = min(len(chunk) for chunk in point_chunks)
    max_points = max(len(chunk) for chunk in point_chunks)
    
    print(f"\nSummary:")
    print(f"Total land points distributed: {total_land_points}")
    print(f"Points per chunk: {min_points}-{max_points}")
    print(f"Average points per chunk: {total_land_points / num_chunks:.1f}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python split_mesh.py mesh_file.json [num_chunks]")
        print("Default num_chunks is 20")
        return
    
    mesh_file = sys.argv[1]
    num_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    # Check if mesh file exists
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file '{mesh_file}' not found.")
        return
    
    try:
        split_mesh_file(mesh_file, num_chunks)
    except Exception as e:
        print(f"Error splitting mesh: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

### File: test.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma3n:e4b")

chain = prompt | model

request = chain.invoke({"question": "What is LangChain?"})
request = model.invoke("where is Montaldo di mondovi?")
request = model.invoke("где находится  Montaldo di mondovi?")

print(request)

