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