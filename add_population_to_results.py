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