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