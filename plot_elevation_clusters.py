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