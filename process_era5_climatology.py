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