import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import json
from pathlib import Path
from collections import defaultdict
import sys
import os

# Add the current directory to path to import other modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_llm_results_and_add_era5(results_file, climatology_file='data/t2m_climatology_1991-2020.nc'):
    """Load LLM results and add ERA5 comparison on the fly"""
    print(f"Loading LLM results from {results_file}...")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is already an ERA5 comparison file
    if 'comparison_results' in data:
        print("File already contains ERA5 comparison data")
        return data['comparison_results'], data.get('metadata', {})
    
    # Otherwise, load original LLM results and add ERA5 data
    print("Adding ERA5 climatology data...")
    
    try:
        # Import ERA5 comparison functions
        from compare_llm_era5 import load_llm_results, extract_era5_climatology, combine_llm_era5_data
        
        # Load LLM results using the existing function
        llm_results, metadata = load_llm_results(results_file)
        
        # Extract coordinates and month
        coordinates = [(r['lat'], r['lon']) for r in llm_results]
        month = llm_results[0]['month'] if llm_results else 'july'
        
        print(f"Processing {len(coordinates)} points for month: {month}")
        
        # Load ERA5 climatology
        era5_data = extract_era5_climatology(climatology_file, coordinates, month)
        
        # Combine data
        comparison_results = combine_llm_era5_data(llm_results, era5_data)
        
        print(f"Successfully created comparison data for {len(comparison_results)} points")
        return comparison_results, metadata
        
    except ImportError as e:
        print(f"Error importing ERA5 functions: {e}")
        print("Please ensure compare_llm_era5.py is in the same directory")
        return None, None
    except FileNotFoundError:
        print(f"ERA5 climatology file not found: {climatology_file}")
        print("Please ensure ERA5 climatology file is available")
        return None, None

def extract_valid_country_data(comparison_results):
    """Extract valid data points grouped by country"""
    country_data = defaultdict(list)
    
    for point in comparison_results:
        # Check if both LLM and ERA5 data are valid
        if (not np.isnan(point.get('llm_temp_mean', np.nan)) and 
            not np.isnan(point.get('era5_temp_mean', np.nan))):
            
            country = point.get('country', 'Unknown')
            if country and country != 'N/A':
                country_data[country].append({
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'llm_temp': point['llm_temp_mean'],
                    'era5_temp': point['era5_temp_mean'],
                    'llm_std': point.get('llm_temp_std', np.nan),
                    'era5_std': point.get('era5_temp_std', np.nan),
                    'temp_diff': point.get('temp_difference', np.nan),
                    'llm_count': point.get('llm_temp_count', 1)
                })
    
    print(f"Found data for {len(country_data)} countries")
    
    # Sort countries by number of points (for better visualization)
    country_data = dict(sorted(country_data.items(), key=lambda x: len(x[1]), reverse=True))
    
    return country_data

def create_country_colormap(country_data):
    """Create a color map for top 15 countries by point count, others are gray"""
    # Sort countries by number of points (descending)
    sorted_countries = sorted(country_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Take top 15 countries
    top_countries = [country for country, _ in sorted_countries[:15]]
    
    # Create distinct colors for top 15 countries
    if len(top_countries) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_countries)))
    else:
        # Use tab20 for up to 15 countries (more distinct than other colormaps)
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_countries)))
    
    # Create color map
    color_map = {}
    gray_color = (0.6, 0.6, 0.6)  # Gray for other countries
    
    # Assign colors to top countries
    for i, country in enumerate(top_countries):
        color_map[country] = colors[i]
    
    # Assign gray to all other countries
    for country in country_data.keys():
        if country not in top_countries:
            color_map[country] = gray_color
    
    return color_map, top_countries

def plot_country_analysis(country_data, output_file=None, figsize=(16, 12)):
    """Create comprehensive country-based analysis plots"""
    
    if not country_data:
        print("No valid country data for plotting")
        return
    
    # Create color map for countries (top 15 get colors, others gray)
    color_map, top_countries = create_country_colormap(country_data)
    countries = list(country_data.keys())
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Main scatter plot (larger subplot)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    # Collect all data for statistics
    all_llm_temps = []
    all_era5_temps = []
    all_llm_stds = []
    all_era5_stds = []
    country_labels = []
    
    # Plot data for each country
    for country, points in country_data.items():
        llm_temps = [p['llm_temp'] for p in points]
        era5_temps = [p['era5_temp'] for p in points]
        llm_stds = [p['llm_std'] if not np.isnan(p['llm_std']) else 0 for p in points]
        era5_stds = [p['era5_std'] if not np.isnan(p['era5_std']) else 0 for p in points]
        
        # Scatter plot with error bars - use higher alpha and larger markers for better visibility
        ax1.errorbar(era5_temps, llm_temps, 
                    xerr=era5_stds, yerr=llm_stds,
                    fmt='o', color=color_map[country], 
                    alpha=0.8, markersize=8, capsize=3, capthick=1.5,
                    elinewidth=1.5, markeredgewidth=0.5, markeredgecolor='black',
                    label=f'{country} (n={len(points)})')
        
        all_llm_temps.extend(llm_temps)
        all_era5_temps.extend(era5_temps)
        all_llm_stds.extend(llm_stds)
        all_era5_stds.extend(era5_stds)
        country_labels.extend([country] * len(points))
    
    # Add 1:1 line
    min_temp = min(min(all_llm_temps), min(all_era5_temps))
    max_temp = max(max(all_llm_temps), max(all_era5_temps))
    ax1.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.8, linewidth=2, label='1:1 line')
    
    # Add regression line
    coeffs = np.polyfit(all_era5_temps, all_llm_temps, 1)
    regression_line = np.poly1d(coeffs)
    x_reg = np.linspace(min_temp, max_temp, 100)
    ax1.plot(x_reg, regression_line(x_reg), 'r-', alpha=0.8, linewidth=2, 
             label=f'Regression: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    
    ax1.set_xlabel('ERA5 Temperature (°C)', fontsize=12)
    ax1.set_ylabel('LLM Temperature (°C)', fontsize=12)
    ax1.set_title('LLM vs ERA5 Temperature by Country\n(Top 15 countries by data points colored, others gray)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Calculate overall statistics
    rmse = np.sqrt(np.mean((np.array(all_llm_temps) - np.array(all_era5_temps))**2))
    mae = np.mean(np.abs(np.array(all_llm_temps) - np.array(all_era5_temps)))
    bias = np.mean(np.array(all_llm_temps) - np.array(all_era5_temps))
    r_corr = np.corrcoef(all_llm_temps, all_era5_temps)[0, 1]
    
    # Add statistics text
    stats_text = f'Overall Statistics\\n'
    stats_text += f'N = {len(all_llm_temps)} points\\n'
    stats_text += f'Countries = {len(countries)}\\n'
    stats_text += f'RMSE = {rmse:.2f}°C\\n'
    stats_text += f'MAE = {mae:.2f}°C\\n'
    stats_text += f'Bias = {bias:+.2f}°C\\n'
    stats_text += f'Correlation = {r_corr:.3f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Country legend (separate subplot)
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax2.axis('off')
    
    # Create legend for top countries only
    legend_handles = []
    legend_labels = []
    
    # Add colored countries (top 15)
    for country in top_countries:
        points = country_data[country]
        handle = plt.Line2D([0], [0], marker='o', color=color_map[country], 
                           linestyle='', markersize=10, alpha=0.8,
                           markeredgewidth=0.5, markeredgecolor='black')
        legend_handles.append(handle)
        legend_labels.append(f'{country} (n={len(points)})')
    
    # Add gray countries entry
    gray_countries_count = len(countries) - len(top_countries)
    if gray_countries_count > 0:
        gray_handle = plt.Line2D([0], [0], marker='o', color=(0.6, 0.6, 0.6), 
                                linestyle='', markersize=10, alpha=0.8,
                                markeredgewidth=0.5, markeredgecolor='black')
        legend_handles.append(gray_handle)
        legend_labels.append(f'Other countries (n={gray_countries_count})')
    
    ax2.legend(legend_handles, legend_labels, loc='center', fontsize=9, 
               title='Countries', title_fontsize=11)
    
    # Country performance summary (bottom left)
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    
    # Calculate RMSE for each country (countries with >2 points)
    country_rmse = {}
    for country, points in country_data.items():
        if len(points) >= 3:  # Only countries with 3+ points
            llm_temps = [p['llm_temp'] for p in points]
            era5_temps = [p['era5_temp'] for p in points]
            rmse_country = np.sqrt(np.mean((np.array(llm_temps) - np.array(era5_temps))**2))
            country_rmse[country] = rmse_country
    
    # Plot top 10 countries by RMSE
    if country_rmse:
        sorted_countries = sorted(country_rmse.items(), key=lambda x: x[1])[:10]
        countries_plot = [x[0] for x in sorted_countries]
        rmse_values = [x[1] for x in sorted_countries]
        colors_plot = [color_map[c] for c in countries_plot]
        
        bars = ax3.barh(range(len(countries_plot)), rmse_values, color=colors_plot, alpha=0.7)
        ax3.set_yticks(range(len(countries_plot)))
        ax3.set_yticklabels([c[:12] + ('...' if len(c) > 12 else '') for c in countries_plot], fontsize=9)
        ax3.set_xlabel('RMSE (°C)', fontsize=10)
        ax3.set_title('Best Countries\\n(Lowest RMSE)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add RMSE values on bars
        for i, (bar, rmse_val) in enumerate(zip(bars, rmse_values)):
            ax3.text(rmse_val + 0.1, i, f'{rmse_val:.1f}', 
                    va='center', ha='left', fontsize=8)
    
    # Difference histogram (bottom center)
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    
    differences = np.array(all_llm_temps) - np.array(all_era5_temps)
    ax4.hist(differences, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax4.axvline(np.mean(differences), color='blue', linestyle='-', linewidth=2, 
                label=f'Mean = {np.mean(differences):+.2f}°C')
    
    ax4.set_xlabel('Temperature Difference (LLM - ERA5) °C', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Difference Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # Country point counts (bottom right)
    ax5 = plt.subplot2grid((3, 3), (2, 2))
    
    # Plot distribution of points per country
    point_counts = [len(points) for points in country_data.values()]
    ax5.hist(point_counts, bins=min(15, len(countries)), alpha=0.7, edgecolor='black', color='lightgreen')
    ax5.set_xlabel('Points per Country', fontsize=10)
    ax5.set_ylabel('Number of Countries', fontsize=10)
    ax5.set_title('Point Distribution', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Country analysis plot saved to: {output_file}")
    else:
        plt.show()
    
    return fig, country_data

def calculate_country_rmse(country_data):
    """Calculate RMSE for each country with sufficient data points"""
    country_rmse = {}
    
    for country, points in country_data.items():
        if len(points) >= 2:  # Only countries with 2+ points
            llm_temps = [p['llm_temp'] for p in points]
            era5_temps = [p['era5_temp'] for p in points]
            rmse = np.sqrt(np.mean((np.array(llm_temps) - np.array(era5_temps))**2))
            country_rmse[country] = rmse
    
    return country_rmse

def get_country_name_variants(country_name):
    """Get possible name variations for country matching"""
    # Create comprehensive mapping from shapefile names to analysis data names
    name_mapping = {
        'United States of America': 'United States',
        'Russian Federation': 'Russia', 
        'Bosnia and Herz.': 'Bosnia and Herzegovina',
        'Central African Rep.': 'Central African Republic',
        'Congo': 'Congo-Brazzaville',
        'Dem. Rep. Congo': 'Democratic Republic of the Congo',
        'Dominican Rep.': 'Dominican Republic',
        'Eq. Guinea': 'Equatorial Guinea',
        'S. Sudan': 'South Sudan',
        'Solomon Is.': 'Solomon Islands',
        'Czech Republic': 'Czechia',
        'Macedonia': 'North Macedonia',
        'eSwatini': 'Eswatini',
        'Côte d\'Ivoire': 'Côte d\'Ivoire',
        'São Tomé and Principe': 'Sao Tome and Principe',
        'Timor-Leste': 'East Timor',
        'W. Sahara': 'Sahrawi Arab Democratic Republic'
    }
    
    # Return list of possible name variations
    return [
        country_name,
        name_mapping.get(country_name, country_name),
        country_name.replace(' ', ''),
        country_name.replace('United States of America', 'United States'),
        country_name.replace('Russian Federation', 'Russia'),
        country_name.replace('United Kingdom', 'United Kingdom')
    ]

def check_country_matching(country_data, countries_shapefile='data/land/ne_10m_admin_0_countries.shp'):
    """Check which countries from shapefile don't have matching data"""
    try:
        world = gpd.read_file(countries_shapefile)
        
        countries_without_data = []
        countries_with_data = []
        
        for _, country_row in world.iterrows():
            country_name = country_row['NAME']
            
            # Get possible name variations for matching
            possible_names = get_country_name_variants(country_name)
            
            found_data = False
            for name_variant in possible_names:
                if name_variant in country_data:
                    found_data = True
                    countries_with_data.append(country_name)
                    break
            
            if not found_data:
                countries_without_data.append(country_name)
        
        print(f"\nCountry Matching Summary:")
        print(f"Countries with data: {len(countries_with_data)}")
        print(f"Countries without data: {len(countries_without_data)}")
        
        if countries_without_data:
            print(f"\nCountries from shapefile that appear in GRAY (no matching data):")
            for i, country in enumerate(sorted(countries_without_data), 1):
                print(f"{i:3d}. {country}")
        
        return countries_with_data, countries_without_data
        
    except Exception as e:
        print(f"Error checking country matching: {e}")
        return [], []

def plot_country_rmse_map(country_data, countries_shapefile='data/land/ne_10m_admin_0_countries.shp', output_file=None, figsize=(16, 10)):
    """Plot world map with countries colored by RMSE values"""
    
    # Calculate RMSE for each country
    country_rmse = calculate_country_rmse(country_data)
    
    if not country_rmse:
        print("No countries with sufficient data for RMSE mapping")
        return
    
    try:
        # Load world countries shapefile
        world = gpd.read_file(countries_shapefile)
        
        # Create a colormap for RMSE values
        rmse_values = list(country_rmse.values())
        vmin = min(rmse_values)
        vmax = max(rmse_values)
        
        # Use a reversed RdYlBu colormap (blue=low RMSE/good, red=high RMSE/bad)
        cmap = plt.cm.RdYlBu_r
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all countries with light gray background
        world.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
        
        # Plot countries with RMSE data
        countries_with_data = []
        rmse_colors = []
        
        for _, country_row in world.iterrows():
            country_name = country_row['NAME']
            
            # Get possible name variations for matching
            possible_names = get_country_name_variants(country_name)
            
            rmse_val = None
            for name_variant in possible_names:
                if name_variant in country_rmse:
                    rmse_val = country_rmse[name_variant]
                    break
            
            if rmse_val is not None:
                countries_with_data.append(country_row.geometry)
                rmse_colors.append(cmap(norm(rmse_val)))
        
        # Plot countries with RMSE data using colors
        if countries_with_data:
            for geom, color in zip(countries_with_data, rmse_colors):
                gpd.GeoSeries([geom]).plot(ax=ax, color=color, edgecolor='white', linewidth=0.5)
        
        # Customize the map
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xlabel('Longitude (degrees)', fontsize=12)
        ax.set_ylabel('Latitude (degrees)', fontsize=12)
        ax.set_title(f'Country Performance: RMSE (LLM vs ERA5)\nRange: {vmin:.2f}°C to {vmax:.2f}°C', fontsize=14, fontweight='bold')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-60, 91, 30))
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=30)
        cbar.set_label('RMSE (°C)', fontsize=12)
        
        # Add statistics text
        n_countries = len(country_rmse)
        mean_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        
        stats_text = f'Countries with data: {n_countries}\n'
        stats_text += f'Mean RMSE: {mean_rmse:.2f}°C\n'
        stats_text += f'Std RMSE: {std_rmse:.2f}°C\n'
        stats_text += f'Best: {min(rmse_values):.2f}°C\n'
        stats_text += f'Worst: {max(rmse_values):.2f}°C'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save or show plot
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Country RMSE map saved to: {output_file}")
        else:
            plt.show()
        
        return fig, country_rmse
        
    except FileNotFoundError:
        print(f"Countries shapefile not found: {countries_shapefile}")
        print("Please ensure Natural Earth countries shapefile is available")
        return None, country_rmse
    except Exception as e:
        print(f"Error creating country map: {e}")
        return None, country_rmse

def print_country_summary(country_data):
    """Print detailed country performance summary"""
    print("\\nCountry Performance Summary:")
    print("=" * 80)
    
    # Calculate statistics for each country
    country_stats = []
    
    for country, points in country_data.items():
        if len(points) >= 2:  # Only countries with 2+ points
            llm_temps = [p['llm_temp'] for p in points]
            era5_temps = [p['era5_temp'] for p in points]
            differences = [p['temp_diff'] for p in points]
            
            rmse = np.sqrt(np.mean((np.array(llm_temps) - np.array(era5_temps))**2))
            mae = np.mean(np.abs(differences))
            bias = np.mean(differences)
            correlation = np.corrcoef(llm_temps, era5_temps)[0, 1] if len(points) > 2 else np.nan
            
            country_stats.append({
                'country': country,
                'points': len(points),
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'correlation': correlation,
                'mean_llm': np.mean(llm_temps),
                'mean_era5': np.mean(era5_temps)
            })
    
    # Sort by RMSE
    country_stats.sort(key=lambda x: x['rmse'])
    
    print(f"{'Country':<20} {'Points':<6} {'RMSE':<6} {'MAE':<6} {'Bias':<7} {'Corr':<6} {'LLM':<6} {'ERA5':<6}")
    print("-" * 80)
    
    for stats in country_stats:
        corr_str = f"{stats['correlation']:.3f}" if not np.isnan(stats['correlation']) else "N/A"
        print(f"{stats['country']:<20} {stats['points']:<6} {stats['rmse']:<6.2f} "
              f"{stats['mae']:<6.2f} {stats['bias']:<+7.2f} {corr_str:<6} "
              f"{stats['mean_llm']:<6.1f} {stats['mean_era5']:<6.1f}")
    
    # Overall summary
    total_points = sum(len(points) for points in country_data.values())
    all_rmse = [s['rmse'] for s in country_stats]
    
    print("\\nSummary:")
    print(f"Total countries: {len(country_data)}")
    print(f"Countries with 2+ points: {len(country_stats)}")
    print(f"Total points: {total_points}")
    print(f"Best performing country (lowest RMSE): {country_stats[0]['country']} ({country_stats[0]['rmse']:.2f}°C)")
    print(f"Worst performing country (highest RMSE): {country_stats[-1]['country']} ({country_stats[-1]['rmse']:.2f}°C)")
    print(f"Mean RMSE across countries: {np.mean(all_rmse):.2f}°C")

def main():
    """Main function"""
    
    # Default file
    results_file = 'results/climate_results_1.0deg_r10_simple.json'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    
    print(f"Country Performance Analysis")
    print(f"Results file: {results_file}")
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        print("Please run climate_llm_benchmark.py first to generate results.")
        return
    
    try:
        # Load results and add ERA5 comparison if needed
        comparison_results, _ = load_llm_results_and_add_era5(results_file)
        
        if comparison_results is None:
            print("Failed to load or process results file.")
            return
        
        # Extract country data
        country_data = extract_valid_country_data(comparison_results)
        
        if not country_data:
            print("No valid country data found in results file.")
            return
        
        # Generate output filename
        results_path = Path(results_file)
        output_file = f"png/country_analysis_{results_path.stem.replace('_era5', '')}.png"
        
        # Create country analysis plot
        plot_country_analysis(country_data, output_file)
        
        # Check country matching between data and shapefile
        check_country_matching(country_data)
        
        # Create world map with RMSE coloring
        map_output_file = f"png/country_rmse_map_{results_path.stem.replace('_era5', '')}.png"
        plot_country_rmse_map(country_data, output_file=map_output_file)
        
        # Print detailed summary
        print_country_summary(country_data)
        
        print("\\nCountry analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()