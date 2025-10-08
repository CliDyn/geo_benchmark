#!/usr/bin/env python3
"""
Plot ERA5 climate change data from multi-period climatology NetCDF file.

This script creates a spatial map of temperature change (t2m_change_mean) from
the ERA5 climatology data.

Usage:
    python plot_era5_climate_change.py

Output: png/era5_t2m_change_mean.png
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
import cmocean

# Optional Cartopy import
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Using matplotlib-only plotting.")

# Configuration
DATA_FILE = 'data/t2m_climatology_multi_period.nc'
OUTPUT_FILE = 'png/era5_t2m_change_mean.png'

# Temperature range configuration
TEMP_MIN = 0.0   # Minimum temperature change in °C
TEMP_MAX = 2.0   # Maximum temperature change in °C

# Colormap configuration
TEMP_COLORMAP = 'thermal'  # cmocean thermal colormap

# Font configuration
FONT_FAMILY = 'Helvetica'
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10

# Display configuration
SHOW_AXES = False
SHOW_FRAME = False
USE_CARTOPY = True
PROJECTION = 'Robinson'
CENTRAL_LONGITUDE = 0
COLOR_LEVELS = 20

# Figure size
FIGSIZE = (7.1, 4.0)


def load_climate_change_data(data_file):
    """Load t2m_change_mean from NetCDF file"""
    print(f"Loading data from: {data_file}")

    ds = xr.open_dataset(data_file)

    # Extract t2m_change_mean variable
    if 't2m_change_mean' not in ds:
        raise ValueError(f"Variable 't2m_change_mean' not found in {data_file}")

    t2m_change = ds['t2m_change_mean']

    print(f"Original data shape: {t2m_change.shape}")

    # Average across all months to get annual mean climate change
    t2m_change_annual = t2m_change.mean(dim='month')

    print(f"Annual mean data shape: {t2m_change_annual.shape}")
    print(f"Data range: {float(t2m_change_annual.min()):.2f} to {float(t2m_change_annual.max()):.2f} °C")

    return t2m_change_annual


def create_climate_change_map(t2m_change, output_file):
    """Create spatial map of temperature change"""

    # Get coordinates
    lats = t2m_change.latitude.values
    lons = t2m_change.longitude.values

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Get data values
    data = t2m_change.values

    # Clip data to temperature range
    data_clipped = np.clip(data, TEMP_MIN, TEMP_MAX)

    # Mask invalid values
    data_masked = np.ma.masked_invalid(data_clipped)

    # Create figure
    if USE_CARTOPY and CARTOPY_AVAILABLE:
        if PROJECTION == 'Robinson':
            proj = ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)
        elif PROJECTION == 'PlateCarree':
            proj = ccrs.PlateCarree(central_longitude=CENTRAL_LONGITUDE)
        elif PROJECTION == 'Mollweide':
            proj = ccrs.Mollweide(central_longitude=CENTRAL_LONGITUDE)
        else:
            proj = ccrs.Robinson(central_longitude=CENTRAL_LONGITUDE)

        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw={'projection': proj})
        cartopy_mode = True
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        cartopy_mode = False

    # Set figure background
    ax.set_facecolor('white')
    if hasattr(fig, 'patch'):
        fig.patch.set_facecolor('white')

    # Create colormap
    try:
        cmap = getattr(cmocean.cm, TEMP_COLORMAP)
    except AttributeError:
        cmap = plt.colormaps[TEMP_COLORMAP]

    # Create contour levels
    levels = np.linspace(TEMP_MIN, TEMP_MAX, COLOR_LEVELS)

    # Plot data
    if cartopy_mode:
        contour = ax.pcolormesh(
            lon_grid, lat_grid, data_masked,
            cmap=cmap, vmin=TEMP_MIN, vmax=TEMP_MAX,
            shading='nearest', transform=ccrs.PlateCarree(), zorder=1
        )
    else:
        contour = ax.pcolormesh(
            lon_grid, lat_grid, data_masked,
            cmap=cmap, vmin=TEMP_MIN, vmax=TEMP_MAX,
            shading='nearest', zorder=1
        )

    # Add geographic features
    if cartopy_mode:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, color='gray', alpha=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, color='gray', alpha=0.5)

    # Add gridlines
    ax.grid(True, color='lightgray', alpha=0.2, linewidth=0.3, linestyle='-')

    # Set extent (exclude Antarctica)
    if cartopy_mode:
        ax.set_extent([-180, 180, -57, 85], crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-57, 85)

    # Configure axes
    if SHOW_AXES and not cartopy_mode:
        ax.set_xlabel('Longitude', fontsize=FONT_SIZE_LABEL, fontfamily=FONT_FAMILY)
        ax.set_ylabel('Latitude', fontsize=FONT_SIZE_LABEL, fontfamily=FONT_FAMILY)
    elif SHOW_AXES and cartopy_mode:
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='lightgray', alpha=0.2)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': FONT_SIZE_TICK, 'family': FONT_FAMILY}
            gl.ylabel_style = {'size': FONT_SIZE_TICK, 'family': FONT_FAMILY}
        except:
            ax.gridlines(linewidth=0.3, color='lightgray', alpha=0.2)
    else:
        if not cartopy_mode:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False)

    # Configure frame
    if not SHOW_FRAME:
        if cartopy_mode:
            try:
                ax.spines['geo'].set_visible(False)
            except KeyError:
                try:
                    ax.outline_patch.set_visible(False)
                except AttributeError:
                    for spine in ax.spines.values():
                        spine.set_visible(False)
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Remove title
    ax.set_title('', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', shrink=0.8, aspect=40, pad=0.15, extend='neither')
    cbar.set_label('Temperature Change (°C)', fontsize=FONT_SIZE_LABEL, fontfamily=FONT_FAMILY)

    # Set colorbar ticks
    temp_ticks = np.arange(TEMP_MIN, TEMP_MAX + 0.1, 1)  # Every 1 degree
    cbar.set_ticks(temp_ticks)
    cbar.set_ticklabels([f'{int(tick)}°' for tick in temp_ticks], fontsize=FONT_SIZE_TICK, fontfamily=FONT_FAMILY)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")

    plt.close()


def main():
    """Main function"""

    # Check if data file exists
    if not Path(DATA_FILE).exists():
        print(f"Error: Data file '{DATA_FILE}' not found")
        return

    # Load climate change data
    t2m_change = load_climate_change_data(DATA_FILE)

    # Create map
    create_climate_change_map(t2m_change, OUTPUT_FILE)

    print("Done!")


if __name__ == "__main__":
    main()
