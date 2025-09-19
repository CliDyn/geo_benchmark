#!/usr/bin/env python3
"""
Density plot comparison between LLM and ERA5 temperature predictions.

This script creates density plots comparing LLM predictions vs ERA5 climatology
with color-coded point density and difference histogram analysis.

Usage:
    python plot_density_comparison.py results_file.json

Output: pub_f2_density_comparison_{resolution}deg.png
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
FONT_SIZE_LABEL = 12       # Font size for axis labels
FONT_SIZE_TICK = 10        # Font size for tick labels

# Axis display configuration
SHOW_AXES = True           # Show axis labels and tick marks (True) or clean plot (False)
SHOW_FRAME = True          # Show frame/box around the plot (True) or remove it (False)
SHOW_LEFT_AXIS = True      # Show left Y-axis spine and ticks (True) or hide (False)
SHOW_RIGHT_AXIS = False    # Show right Y-axis spine and ticks (True) or hide (False)
SHOW_TOP_AXIS = False      # Show top X-axis spine and ticks (True) or hide (False)
SHOW_BOTTOM_AXIS = True    # Show bottom X-axis spine and ticks (True) or hide (False)

# Colorbar configuration (for consistency with scatter plot)
COLORBAR_PAD = 0.08        # Distance between plot and colorbar (0.08 = closer, 0.15 = further)

# X-axis limits for density plot
X_AXIS_MIN = -20           # Minimum X-axis value (temperature difference)
X_AXIS_MAX = 20            # Maximum X-axis value (temperature difference)


def extract_comparison_data(results_data):
    """Extract LLM vs ERA5 comparison data from results"""
    comparison_data = []
    
    for result in results_data['results']:
        point_info = result['point_info']
        
        # Only include land points with valid LLM and ERA5 data
        if (point_info.get('is_land', False) and 
            'llm_temp_mean' in point_info and 
            'era5_temp_mean' in point_info and
            not np.isnan(point_info.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point_info.get('era5_temp_mean', np.nan))):
            
            comparison_data.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'],
                'llm_temp': point_info['llm_temp_mean'],
                'era5_temp': point_info['era5_temp_mean'],
                'llm_std': point_info.get('llm_temp_std', 0.0),
                'era5_std': point_info.get('era5_temp_std', np.nan),
                'temp_diff': point_info.get('temp_difference', 
                                          point_info['llm_temp_mean'] - point_info['era5_temp_mean']),
                'llm_count': point_info.get('llm_temp_count', 1),
                'country': point_info.get('country', '')
            })
    
    print(f"Found {len(comparison_data)} valid comparison data points")
    return comparison_data


def create_density_plot(comparison_data, output_file=None, font_family='Helvetica', font_size_label=12, font_size_tick=10, show_axes=True, show_frame=True, show_left_axis=True, show_right_axis=False, show_top_axis=False, show_bottom_axis=True, colorbar_pad=0.08, x_axis_min=-20, x_axis_max=20):
    """Create difference histogram (density distribution) comparing LLM vs ERA5"""
    
    if not comparison_data:
        print("No comparison data available for density plot")
        return None, None
    
    # Extract arrays for plotting
    llm_vals = np.array([d['llm_temp'] for d in comparison_data])
    era5_vals = np.array([d['era5_temp'] for d in comparison_data])
    
    # Set font for matplotlib
    plt.rcParams['font.family'] = font_family
    
    # Create figure with square plot area (width x height)
    # Extra height for potential future colorbar
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 7.0))
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((llm_vals - era5_vals)**2))
    mae = np.mean(np.abs(llm_vals - era5_vals))
    bias = np.mean(llm_vals - era5_vals)
    r_corr = np.corrcoef(llm_vals, era5_vals)[0, 1]
    
    # === Difference histogram ===
    differences = llm_vals - era5_vals
    ax.hist(differences, bins=25, alpha=0.7, edgecolor='black', color='skyblue', density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean diff = {np.mean(differences):+.2f}°C')
    
    # Add normal distribution overlay for reference
    x_norm = np.linspace(differences.min(), differences.max(), 100)
    y_norm = (1/np.sqrt(2*np.pi*np.var(differences))) * np.exp(-0.5*((x_norm - np.mean(differences))**2)/np.var(differences))
    ax.plot(x_norm, y_norm, 'g--', alpha=0.8, linewidth=2, label='Normal fit')
    
    # Configure axes and labels based on options
    if show_axes:
        ax.set_xlabel('Temperature Difference (LLM - ERA5) °C', fontsize=font_size_label, fontfamily=font_family)
        ax.set_ylabel('Probability Density', fontsize=font_size_label, fontfamily=font_family)
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
    
    # Set X-axis limits for temperature difference range
    ax.set_xlim(x_axis_min, x_axis_max)
    
    # Note: Not using square aspect ratio for histogram as X and Y axes have different units
    # X-axis: Temperature difference (°C), Y-axis: Probability density (dimensionless)
    
    # Configure legend font
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontfamily(font_family)
        text.set_fontsize(font_size_tick)
    
    # Statistics calculated for return values only (not displayed on plot)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"LLM vs ERA5 density plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()
    return fig, (rmse, mae, bias, r_corr)


def main():
    """Main function to create density plot comparison"""
    
    if len(sys.argv) < 2:
        print("Usage: python plot_density_comparison.py results_file.json")
        print("Example: python plot_density_comparison.py results/climate_results_1.0deg_r10_gpt-5_simple_spatial_rmse_bathymetry_population.json")
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
    output_file = output_dir / f"pub_f2_density_comparison_{resolution}deg.png"
    
    # Create density plot
    fig, stats = create_density_plot(comparison_data, output_file, FONT_FAMILY, FONT_SIZE_LABEL, FONT_SIZE_TICK, SHOW_AXES, SHOW_FRAME, SHOW_LEFT_AXIS, SHOW_RIGHT_AXIS, SHOW_TOP_AXIS, SHOW_BOTTOM_AXIS, COLORBAR_PAD, X_AXIS_MIN, X_AXIS_MAX)
    
    if fig is not None:
        rmse, mae, bias, r_corr = stats
        print(f"Density plot statistics - RMSE: {rmse:.2f}°C, MAE: {mae:.2f}°C, Bias: {bias:+.2f}°C, Correlation: {r_corr:.3f}")


if __name__ == "__main__":
    main()