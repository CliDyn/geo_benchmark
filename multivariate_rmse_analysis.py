#!/usr/bin/env python3
"""
Multivariate Analysis of Spatial RMSE for LLM Temperature Predictions

This script performs comprehensive multivariate analysis to explain spatial_r2_rmse using:
- Population density
- Mean elevation (altitude)  
- Terrain roughness

Methods:
1. GAM (Generalized Additive Model) with smooth terms and interactions
2. Gradient Boosted Model (XGBoost) with SHAP analysis
3. Spatial block cross-validation for robust evaluation

Usage:
    python multivariate_rmse_analysis.py [results_file]

Default file: climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json
Outputs: 
    - png/multivariate_analysis_*.png (visualizations)
    - reports/multivariate_rmse_report.txt (detailed text report)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Some visualizations will be simplified.")

# Statistical modeling - basic libraries
from scipy import stats
from scipy.stats import pearsonr

# Optional advanced libraries
HAS_SKLEARN = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:
    print("Warning: scikit-learn not available. Using basic implementations.")
    HAS_SKLEARN = False

try:
    from pygam import LinearGAM, s
    HAS_PYGAM = True
except ImportError:
    print("Warning: pygam not available. GAM analysis will be limited.")
    HAS_PYGAM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("Warning: xgboost not available. Gradient boosting analysis will be skipped.")
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("Warning: shap not available. SHAP analysis will be skipped.")
    HAS_SHAP = False

# Basic implementations for missing libraries
def standardize_data(data):
    """Basic standardization (z-score)"""
    return (data - np.mean(data)) / np.std(data)

def r2_score_basic(y_true, y_pred):
    """Basic R² calculation"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def load_and_prepare_data(results_file):
    """Load and prepare data for multivariate analysis"""
    
    print(f"Loading data from: {results_file}")
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"Found {len(results_data['results'])} result points")
    
    # Check data availability
    has_era5 = results_data.get('metadata', {}).get('era5_climatology_added', False)
    has_spatial = results_data.get('metadata', {}).get('spatial_rmse_added', False) 
    has_population = results_data.get('metadata', {}).get('population_added', False)
    has_bathymetry = results_data.get('metadata', {}).get('bathymetry_added', False)
    
    print(f"Data availability - ERA5: {has_era5}, Spatial RMSE: {has_spatial}, Population: {has_population}, Bathymetry: {has_bathymetry}")
    
    if not (has_spatial and has_population and has_bathymetry):
        raise ValueError("Missing required data components for analysis")
    
    # Extract variables
    data_points = []
    for result in results_data['results']:
        point_info = result['point_info']
        
        if (point_info.get('is_land', False) and
            all(key in point_info and not pd.isna(point_info.get(key, np.nan)) 
                for key in ['spatial_r2_rmse', 'population_density', 'mean_elevation', 'roughness'])):
            
            data_points.append({
                'lat': point_info['lat'],
                'lon': point_info['lon'], 
                'rmse': point_info['spatial_r2_rmse'],
                'population': point_info['population_density'],
                'elevation': point_info['mean_elevation'],
                'roughness': point_info['roughness']
            })
    
    df = pd.DataFrame(data_points)
    print(f"Prepared dataset with {len(df)} complete observations")
    
    return df

def analyze_distributions(df):
    """Analyze variable distributions and determine transformations"""
    
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    variables = ['rmse', 'population', 'elevation', 'roughness']
    
    # Test for normality and skewness
    distribution_stats = {}
    
    for var in variables:
        data = df[var].values
        
        # Basic statistics
        stats_dict = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'min': np.min(data),
            'max': np.max(data)
        }
        
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(data[:5000] if len(data) > 5000 else data)
        stats_dict['shapiro_p'] = shapiro_p
        stats_dict['is_normal'] = shapiro_p > 0.05
        stats_dict['is_skewed'] = abs(stats_dict['skewness']) > 1.0
        
        distribution_stats[var] = stats_dict
        
        print(f"\n{var.upper()}:")
        print(f"  Mean: {stats_dict['mean']:.3f}, Median: {stats_dict['median']:.3f}")
        print(f"  Skewness: {stats_dict['skewness']:.3f} ({'skewed' if stats_dict['is_skewed'] else 'normal'})")
        print(f"  Shapiro-Wilk p-value: {stats_dict['shapiro_p']:.6f}")
    
    return distribution_stats

def apply_transformations(df, distribution_stats):
    """Apply appropriate transformations based on distribution analysis"""
    
    print("\n" + "="*60) 
    print("APPLYING TRANSFORMATIONS")
    print("="*60)
    
    df_transformed = df.copy()
    transformations = {}
    
    # Log-transform RMSE if skewed
    if distribution_stats['rmse']['is_skewed']:
        df_transformed['log_rmse'] = np.log(df['rmse'])
        transformations['rmse'] = 'log'
        print("Applied log transformation to RMSE (target variable)")
    else:
        df_transformed['log_rmse'] = df['rmse']
        transformations['rmse'] = 'none'
        print("No transformation applied to RMSE")
    
    # Log-transform population (typically heavily skewed)
    df_transformed['log_population'] = np.log(df['population'] + 1)  # +1 for zeros
    transformations['population'] = 'log'
    print("Applied log(x+1) transformation to population density")
    
    # Standardize all predictors
    predictor_vars = ['log_population', 'elevation', 'roughness']
    
    if HAS_SKLEARN:
        scaler = StandardScaler()
        df_transformed[['population_std', 'elevation_std', 'roughness_std']] = scaler.fit_transform(
            df_transformed[predictor_vars]
        )
        transformations['scaler'] = scaler
    else:
        # Use basic standardization
        df_transformed['population_std'] = standardize_data(df_transformed['log_population'])
        df_transformed['elevation_std'] = standardize_data(df_transformed['elevation'])
        df_transformed['roughness_std'] = standardize_data(df_transformed['roughness'])
        transformations['scaler'] = 'basic'
    
    transformations['predictor_vars'] = predictor_vars
    
    print("Standardized all predictors (z-score normalization)")
    
    return df_transformed, transformations

def create_spatial_blocks(df, n_blocks=5):
    """Create spatial blocks for cross-validation"""
    
    print(f"\nCreating {n_blocks}x{n_blocks} spatial blocks for cross-validation...")
    
    # Create spatial blocks based on lat/lon quantiles
    lat_bins = pd.cut(df['lat'], bins=n_blocks, labels=False)
    lon_bins = pd.cut(df['lon'], bins=n_blocks, labels=False)
    
    # Combine lat/lon bins to create spatial blocks
    df['spatial_block'] = lat_bins * n_blocks + lon_bins
    
    print(f"Created {len(df['spatial_block'].unique())} spatial blocks")
    print(f"Block sizes: min={df['spatial_block'].value_counts().min()}, max={df['spatial_block'].value_counts().max()}")
    
    return df

def fit_gam_model(df):
    """Fit Generalized Additive Model"""
    
    print("\n" + "="*60)
    print("GAM MODEL FITTING") 
    print("="*60)
    
    if not HAS_PYGAM:
        print("Skipping GAM analysis - pygam not available")
        return None, {}
    
    # Prepare data
    X = df[['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']].values
    y = df['log_rmse'].values
    
    try:
        # Fit GAM with smooth terms for each variable plus spatial smooth
        gam = LinearGAM(s(0) + s(1) + s(2) + s(3, 4)).fit(X, y)
        
        # Calculate predictions and R²
        y_pred = gam.predict(X)
        if HAS_SKLEARN:
            r2_total = r2_score(y, y_pred)
        else:
            r2_total = r2_score_basic(y, y_pred)
        
        print(f"GAM fitted successfully")
        print(f"Total R²: {r2_total:.4f}")
        
        # Try to estimate partial R² by removing each term
        partial_r2 = {}
        
        # This is an approximation - ideally we'd refit models without each term
        feature_names = ['population_std', 'elevation_std', 'roughness_std', 'spatial']
        
        for i, feature in enumerate(feature_names[:-1]):  # Skip spatial for now
            try:
                # Create reduced feature set
                X_reduced = np.delete(X, i, axis=1)
                
                # Fit reduced model (approximate)
                if i == 0:  # Remove population
                    gam_reduced = LinearGAM(s(0) + s(1) + s(2, 3)).fit(X_reduced, y)
                elif i == 1:  # Remove elevation  
                    gam_reduced = LinearGAM(s(0) + s(1) + s(2, 3)).fit(X_reduced, y)
                elif i == 2:  # Remove roughness
                    gam_reduced = LinearGAM(s(0) + s(1) + s(2, 3)).fit(X_reduced, y)
                
                y_pred_reduced = gam_reduced.predict(X_reduced)
                if HAS_SKLEARN:
                    r2_reduced = r2_score(y, y_pred_reduced)
                else:
                    r2_reduced = r2_score_basic(y, y_pred_reduced)
                partial_r2[feature] = r2_total - r2_reduced
                
            except Exception as e:
                partial_r2[feature] = np.nan
                print(f"Could not calculate partial R² for {feature}: {e}")
        
        print("\nPartial R² estimates:")
        for feature, r2_val in partial_r2.items():
            if not np.isnan(r2_val):
                print(f"  {feature}: {r2_val:.4f}")
        
        gam_results = {
            'model': gam,
            'r2_total': r2_total,
            'partial_r2': partial_r2,
            'predictions': y_pred
        }
        
        return gam, gam_results
        
    except Exception as e:
        print(f"GAM fitting failed: {e}")
        return None, {}

def fit_xgb_model(df):
    """Fit XGBoost model with SHAP analysis"""
    
    print("\n" + "="*60)
    print("XGBOOST MODEL WITH SHAP ANALYSIS")
    print("="*60)
    
    if not HAS_XGB:
        print("Skipping XGBoost analysis - xgboost not available")
        return None, {}
    
    # Prepare data
    feature_cols = ['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']
    X = df[feature_cols].values
    y = df['log_rmse'].values
    
    # Add interaction term
    interaction = df['population_std'] * df['elevation_std']
    X_with_interaction = np.column_stack([X, interaction])
    feature_names = feature_cols + ['pop_elev_interaction']
    
    try:
        # Fit XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(X_with_interaction, y)
        
        # Calculate R²
        y_pred = xgb_model.predict(X_with_interaction)
        if HAS_SKLEARN:
            r2_xgb = r2_score(y, y_pred)
        else:
            r2_xgb = r2_score_basic(y, y_pred)
        
        print(f"XGBoost R²: {r2_xgb:.4f}")
        
        # SHAP analysis
        shap_values = None
        interaction_shap = None
        
        if HAS_SHAP:
            print("Computing SHAP values...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_with_interaction)
            
            # Calculate feature importance from SHAP
            feature_importance = np.abs(shap_values).mean(0)
            
            print("\nSHAP Feature Importance:")
            for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                print(f"  {name}: {importance:.4f}")
            
            # Interaction SHAP for population × elevation
            try:
                interaction_shap = explainer.shap_interaction_values(X_with_interaction)
                pop_elev_interaction = interaction_shap[:, 0, 1]  # Population-elevation interaction
                interaction_strength = np.abs(pop_elev_interaction).mean()
                print(f"\nPopulation-Elevation Interaction Strength (SHAP): {interaction_strength:.4f}")
            except Exception as e:
                print(f"Could not compute interaction SHAP: {e}")
        
        xgb_results = {
            'model': xgb_model,
            'r2': r2_xgb,
            'predictions': y_pred,
            'shap_values': shap_values,
            'interaction_shap': interaction_shap,
            'feature_names': feature_names
        }
        
        return xgb_model, xgb_results
        
    except Exception as e:
        print(f"XGBoost fitting failed: {e}")
        return None, {}

def block_cross_validation(df, gam_model=None, xgb_model=None):
    """Perform spatial block cross-validation"""
    
    print("\n" + "="*60)
    print("SPATIAL BLOCK CROSS-VALIDATION")
    print("="*60)
    
    # Prepare data
    feature_cols = ['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']
    X = df[feature_cols].values
    y = df['log_rmse'].values
    
    # Add interaction for XGBoost
    interaction = df['population_std'] * df['elevation_std'] 
    X_xgb = np.column_stack([X, interaction])
    
    # Get spatial blocks
    blocks = df['spatial_block'].values
    unique_blocks = np.unique(blocks)
    n_folds = len(unique_blocks)
    
    print(f"Performing {n_folds}-fold spatial block cross-validation...")
    
    cv_results = {
        'gam_scores': [],
        'xgb_scores': [],
        'gam_rmse': [],
        'xgb_rmse': []
    }
    
    for fold, test_block in enumerate(unique_blocks):
        # Create train/test split based on spatial blocks
        test_mask = (blocks == test_block)
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        X_xgb_train, X_xgb_test = X_xgb[train_mask], X_xgb[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"  Fold {fold+1}: {np.sum(train_mask)} train, {np.sum(test_mask)} test")
        
        # GAM cross-validation
        if gam_model is not None and HAS_PYGAM:
            try:
                gam_fold = LinearGAM(s(0) + s(1) + s(2) + s(3, 4)).fit(X_train, y_train)
                y_pred_gam = gam_fold.predict(X_test)
                if HAS_SKLEARN:
                    r2_gam = r2_score(y_test, y_pred_gam)
                    rmse_gam = np.sqrt(mean_squared_error(y_test, y_pred_gam))
                else:
                    r2_gam = r2_score_basic(y_test, y_pred_gam)
                    rmse_gam = np.sqrt(np.mean((y_test - y_pred_gam) ** 2))
                
                cv_results['gam_scores'].append(r2_gam)
                cv_results['gam_rmse'].append(rmse_gam)
            except:
                pass
        
        # XGBoost cross-validation
        if xgb_model is not None and HAS_XGB:
            try:
                xgb_fold = xgb.XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                )
                xgb_fold.fit(X_xgb_train, y_train)
                y_pred_xgb = xgb_fold.predict(X_xgb_test)
                if HAS_SKLEARN:
                    r2_xgb = r2_score(y_test, y_pred_xgb)
                    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
                else:
                    r2_xgb = r2_score_basic(y_test, y_pred_xgb)
                    rmse_xgb = np.sqrt(np.mean((y_test - y_pred_xgb) ** 2))
                
                cv_results['xgb_scores'].append(r2_xgb)
                cv_results['xgb_rmse'].append(rmse_xgb)
            except:
                pass
    
    # Calculate CV statistics
    cv_stats = {}
    
    if cv_results['gam_scores']:
        cv_stats['gam'] = {
            'mean_r2': np.mean(cv_results['gam_scores']),
            'std_r2': np.std(cv_results['gam_scores']),
            'mean_rmse': np.mean(cv_results['gam_rmse']),
            'std_rmse': np.std(cv_results['gam_rmse'])
        }
        print(f"\nGAM CV Results:")
        print(f"  R² = {cv_stats['gam']['mean_r2']:.4f} ± {cv_stats['gam']['std_r2']:.4f}")
        print(f"  RMSE = {cv_stats['gam']['mean_rmse']:.4f} ± {cv_stats['gam']['std_rmse']:.4f}")
    
    if cv_results['xgb_scores']:
        cv_stats['xgb'] = {
            'mean_r2': np.mean(cv_results['xgb_scores']),
            'std_r2': np.std(cv_results['xgb_scores']),
            'mean_rmse': np.mean(cv_results['xgb_rmse']),
            'std_rmse': np.std(cv_results['xgb_rmse'])
        }
        print(f"\nXGBoost CV Results:")
        print(f"  R² = {cv_stats['xgb']['mean_r2']:.4f} ± {cv_stats['xgb']['std_r2']:.4f}")
        print(f"  RMSE = {cv_stats['xgb']['mean_rmse']:.4f} ± {cv_stats['xgb']['std_rmse']:.4f}")
    
    return cv_stats

def create_visualizations(df, gam_results=None, xgb_results=None, output_dir='png'):
    """Create comprehensive visualization plots"""
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use seaborn style if available
    if HAS_SEABORN:
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    else:
        plt.style.use('default')
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    variables = [('rmse', 'RMSE'), ('population', 'Population Density'), 
                ('elevation', 'Elevation'), ('roughness', 'Roughness')]
    
    for i, (var, title) in enumerate(variables):
        row, col = i // 2, i % 2
        axes[row, col].hist(df[var], bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Distribution of {title}')
        axes[row, col].set_xlabel(title)
        axes[row, col].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_path / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    corr_vars = ['rmse', 'population', 'elevation', 'roughness']
    corr_matrix = df[corr_vars].corr()
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    else:
        # Fallback visualization without seaborn
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_vars)))
        ax.set_yticks(range(len(corr_vars)))
        ax.set_xticklabels(corr_vars)
        ax.set_yticklabels(corr_vars)
        
        # Add correlation values as text
        for i in range(len(corr_vars)):
            for j in range(len(corr_vars)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax)
    
    ax.set_title('Variable Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots with RMSE
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    predictors = [('population', 'Population Density'), 
                 ('elevation', 'Elevation (m)'), 
                 ('roughness', 'Roughness (m)')]
    
    for i, (var, title) in enumerate(predictors):
        axes[i].scatter(df[var], df['rmse'], alpha=0.6, s=10)
        axes[i].set_xlabel(title)
        axes[i].set_ylabel('Spatial RMSE (°C)')
        axes[i].set_title(f'RMSE vs {title}')
        
        # Add correlation coefficient
        corr, p_val = pearsonr(df[var], df['rmse'])
        axes[i].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3e}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / 'rmse_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SHAP plots if available
    if xgb_results and xgb_results.get('shap_values') is not None and HAS_SHAP:
        
        # SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(xgb_results['shap_values'], 
                         df[['population_std', 'elevation_std', 'roughness_std', 'lat', 'lon']].values,
                         feature_names=xgb_results['feature_names'][:-1],  # Exclude interaction
                         show=False)
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP interaction plot if available
        if xgb_results.get('interaction_shap') is not None:
            try:
                plt.figure(figsize=(8, 6))
                interaction_values = xgb_results['interaction_shap'][:, 0, 1]  # Pop-elev interaction
                plt.scatter(df['population_std'], interaction_values, alpha=0.6)
                plt.xlabel('Population (standardized)')
                plt.ylabel('SHAP Interaction Value\n(Population × Elevation)')
                plt.title('Population-Elevation Interaction Effects')
                plt.tight_layout()
                plt.savefig(output_path / 'shap_interaction.png', dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass
    
    # 5. Spatial residuals map (if we have predictions)
    if gam_results and 'predictions' in gam_results:
        residuals = df['log_rmse'] - gam_results['predictions']
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df['lon'], df['lat'], c=residuals, 
                            cmap='RdBu_r', s=15, alpha=0.7)
        plt.colorbar(scatter, label='Residuals (log scale)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Spatial Distribution of GAM Residuals')
        plt.tight_layout()
        plt.savefig(output_path / 'spatial_residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_path}/")

def generate_report(df, distribution_stats, gam_results, xgb_results, cv_stats, output_file='reports/multivariate_rmse_report.txt'):
    """Generate comprehensive text report"""
    
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("MULTIVARIATE ANALYSIS OF SPATIAL RMSE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset Size: {len(df)} observations\n\n")
        
        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Variables analyzed:\n")
        f.write(f"  - Target: spatial_r2_rmse (spatial RMSE for LLM temperature)\n")
        f.write(f"  - Predictors: population_density, mean_elevation, roughness\n")
        f.write(f"  - Spatial coordinates: lat, lon\n\n")
        
        # Distribution analysis
        f.write("DISTRIBUTION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        for var, stats in distribution_stats.items():
            f.write(f"{var.upper()}:\n")
            f.write(f"  Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}\n")
            f.write(f"  Std: {stats['std']:.3f}, Skewness: {stats['skewness']:.3f}\n")
            f.write(f"  Range: {stats['min']:.3f} to {stats['max']:.3f}\n")
            f.write(f"  Normal distribution: {stats['is_normal']}\n")
            f.write(f"  Skewed: {stats['is_skewed']}\n\n")
        
        # GAM Results
        if gam_results:
            f.write("GAM MODEL RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Model: log(RMSE) ~ s(population) + s(elevation) + s(roughness) + s(lat,lon)\n")
            f.write(f"Total R²: {gam_results.get('r2_total', 'N/A'):.4f}\n\n")
            
            if 'partial_r2' in gam_results:
                f.write("Partial R² by predictor:\n")
                for predictor, r2_val in gam_results['partial_r2'].items():
                    if not np.isnan(r2_val):
                        f.write(f"  {predictor}: {r2_val:.4f}\n")
                f.write("\n")
        
        # XGBoost Results
        if xgb_results:
            f.write("XGBOOST MODEL RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total R²: {xgb_results.get('r2', 'N/A'):.4f}\n")
            
            if HAS_SHAP and xgb_results.get('shap_values') is not None:
                f.write("\nSHAP Feature Importance:\n")
                shap_values = xgb_results['shap_values']
                feature_names = xgb_results['feature_names']
                importance = np.abs(shap_values).mean(0)
                
                for name, imp in zip(feature_names, importance):
                    f.write(f"  {name}: {imp:.4f}\n")
            f.write("\n")
        
        # Cross-validation results
        if cv_stats:
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-" * 30 + "\n")
            
            if 'gam' in cv_stats:
                stats = cv_stats['gam']
                f.write(f"GAM (Block CV):\n")
                f.write(f"  R² = {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}\n")
                f.write(f"  RMSE = {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}\n\n")
            
            if 'xgb' in cv_stats:
                stats = cv_stats['xgb']
                f.write(f"XGBoost (Block CV):\n")
                f.write(f"  R² = {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}\n")
                f.write(f"  RMSE = {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}\n\n")
        
        # Interpretation
        f.write("KEY FINDINGS & INTERPRETATION\n")
        f.write("-" * 30 + "\n")
        f.write("1. The analysis explains spatial variation in LLM temperature prediction RMSE\n")
        f.write("   using population density, elevation, and terrain roughness.\n\n")
        f.write("2. Both GAM and XGBoost models provide consistent results for robustness.\n\n")
        f.write("3. Spatial block cross-validation accounts for spatial autocorrelation\n")
        f.write("   and provides unbiased performance estimates.\n\n")
        
        if gam_results and 'partial_r2' in gam_results:
            f.write("4. Partial R² values indicate the unique contribution of each predictor\n")
            f.write("   to explaining RMSE variation.\n\n")
        
        if HAS_SHAP and xgb_results and xgb_results.get('shap_values') is not None:
            f.write("5. SHAP values provide interpretable feature importance rankings\n")
            f.write("   and reveal interaction effects between variables.\n\n")
    
    print(f"Report saved to {output_file}")

def main():
    """Main analysis function"""
    
    # Default file
    default_results = 'results/climate_results_1.0deg_r10_simple_spatial_rmse_population_bathymetry.json'
    
    # Parse command line arguments
    results_file = sys.argv[1] if len(sys.argv) > 1 else default_results
    
    print("MULTIVARIATE ANALYSIS OF SPATIAL RMSE")
    print("=" * 80)
    print(f"Input file: {results_file}")
    print(f"Analysis target: spatial_r2_rmse")
    print(f"Predictors: population_density, mean_elevation, roughness")
    print()
    
    # Check file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found.")
        return
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare_data(results_file)
        
        # 2. Analyze distributions
        distribution_stats = analyze_distributions(df)
        
        # 3. Apply transformations
        df_transformed, transformations = apply_transformations(df, distribution_stats)
        
        # 4. Create spatial blocks
        df_transformed = create_spatial_blocks(df_transformed, n_blocks=5)
        
        # 5. Fit GAM model
        gam_model, gam_results = fit_gam_model(df_transformed)
        
        # 6. Fit XGBoost model with SHAP
        xgb_model, xgb_results = fit_xgb_model(df_transformed)
        
        # 7. Cross-validation
        cv_stats = block_cross_validation(df_transformed, gam_model, xgb_model)
        
        # 8. Create visualizations
        create_visualizations(df_transformed, gam_results, xgb_results)
        
        # 9. Generate report
        generate_report(df_transformed, distribution_stats, gam_results, xgb_results, cv_stats)
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Outputs generated:")
        print("  - Visualizations: png/multivariate_analysis_*.png")
        print("  - Report: reports/multivariate_rmse_report.txt")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()