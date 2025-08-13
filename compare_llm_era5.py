import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            # Select nearest grid point
            point_data = ds.sel(latitude=lat, longitude=lon, month=month_num, method='nearest')
            
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
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()