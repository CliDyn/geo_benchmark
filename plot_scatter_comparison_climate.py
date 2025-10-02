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
            'era5_change_signal' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and
            not np.isnan(point_info.get('era5_change_signal', np.nan))):

            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_change_signal'],
                'llm_std': point_info.get('llm_temp_std', 0.0),
                'era5_std': point_info.get('era5_temp_std', np.nan),
                'temp_diff': point_info.get('temp_difference',
                                          point_info['llm_temp_mean'] - point_info['era5_change_signal']),
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
        ax.set_xlabel('ERA5 Temperature Change (°C)', fontsize=font_size_label, fontfamily=font_family)
        ax.set_ylabel('LLM Temperature Change (°C)', fontsize=font_size_label, fontfamily=font_family)
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