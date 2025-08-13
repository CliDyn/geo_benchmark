import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

def process_era5_climatology(input_file, output_dir='data'):
    """
    Process ERA5 temperature data to create monthly climatology (1991-2020)
    
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
    
    # Filter data for 1991-2020 period
    start_date = '1991-01-01'
    end_date = '2020-12-31'
    
    print(f"Filtering data for period {start_date} to {end_date}...")
    ds_filtered = ds.sel(valid_time=slice(start_date, end_date))
    
    print(f"Filtered dataset shape: {ds_filtered.t2m.shape}")
    print(f"Time range: {ds_filtered.valid_time.min().values} to {ds_filtered.valid_time.max().values}")
    
    # Convert temperature from Kelvin to Celsius
    print("Converting temperature from Kelvin to Celsius...")
    ds_filtered['t2m'] = ds_filtered['t2m'] - 273.15
    ds_filtered['t2m'].attrs['units'] = '°C'
    ds_filtered['t2m'].attrs['long_name'] = '2 metre temperature'
    
    # Add month coordinate
    ds_filtered = ds_filtered.assign_coords(month=ds_filtered['valid_time'].dt.month)
    
    print("Calculating monthly climatology statistics...")
    
    # Group by month and calculate statistics
    # Calculate mean, min, max for each month
    monthly_mean = ds_filtered['t2m'].groupby('month').mean('valid_time')
    monthly_min = ds_filtered['t2m'].groupby('month').min('valid_time')
    monthly_max = ds_filtered['t2m'].groupby('month').max('valid_time')
    monthly_std = ds_filtered['t2m'].groupby('month').std('valid_time')
    
    # Create output dataset
    climatology_ds = xr.Dataset({
        't2m_mean': monthly_mean,
        't2m_min': monthly_min,
        't2m_max': monthly_max,
        't2m_std': monthly_std
    })
    
    # Add attributes
    climatology_ds['t2m_mean'].attrs = {
        'long_name': '2m temperature climatological mean (1991-2020)',
        'units': '°C',
        'standard_name': 'air_temperature'
    }
    
    climatology_ds['t2m_min'].attrs = {
        'long_name': '2m temperature climatological minimum (1991-2020)',
        'units': '°C',
        'standard_name': 'air_temperature'
    }
    
    climatology_ds['t2m_max'].attrs = {
        'long_name': '2m temperature climatological maximum (1991-2020)',
        'units': '°C',
        'standard_name': 'air_temperature'
    }
    
    climatology_ds['t2m_std'].attrs = {
        'long_name': '2m temperature climatological standard deviation (1991-2020)',
        'units': '°C',
        'standard_name': 'air_temperature'
    }
    
    # Add global attributes
    climatology_ds.attrs = {
        'title': 'ERA5 2m Temperature Monthly Climatology 1991-2020',
        'source': 'ERA5 reanalysis data',
        'institution': 'European Centre for Medium-Range Weather Forecasts',
        'period': '1991-2020',
        'created_by': 'process_era5_climatology.py',
        'variable': '2m temperature (t2m)',
        'units': 'degrees Celsius',
        'Conventions': 'CF-1.7'
    }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save to NetCDF file
    output_filename = f"t2m_climatology_1991-2020.nc"
    output_filepath = output_path / output_filename
    
    print(f"Saving climatology to {output_filepath}...")
    climatology_ds.to_netcdf(output_filepath, 
                            encoding={
                                't2m_mean': {'zlib': True, 'complevel': 4},
                                't2m_min': {'zlib': True, 'complevel': 4},
                                't2m_max': {'zlib': True, 'complevel': 4},
                                't2m_std': {'zlib': True, 'complevel': 4}
                            })
    
    print("Climatology processing completed successfully!")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("="*50)
    
    for month in range(1, 13):
        month_name = pd.to_datetime(f'2000-{month:02d}-01').strftime('%B')
        mean_global = float(climatology_ds['t2m_mean'].sel(month=month).mean())
        min_global = float(climatology_ds['t2m_min'].sel(month=month).min())
        max_global = float(climatology_ds['t2m_max'].sel(month=month).max())
        std_global = float(climatology_ds['t2m_std'].sel(month=month).mean())
        
        print(f"{month_name:>9}: Mean={mean_global:6.2f}°C, Min={min_global:7.2f}°C, "
              f"Max={max_global:6.2f}°C, Std={std_global:5.2f}°C")
    
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