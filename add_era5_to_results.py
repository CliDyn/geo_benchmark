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