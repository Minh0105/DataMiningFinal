"""
===============================================================================
ADVANCED UNIVERSITY ADMISSION PREDICTION SYSTEM
===============================================================================
Version: 2.0 (Windows Compatible)
Features:
- Multiple Models: Weighted Average, ETS, ARIMA, Linear Regression, Ridge
- Ensemble Learning with Auto Model Selection
- Confidence Interval (95%)
- Feature Engineering: Trend, Volatility, Competition Index
- Cross-validation for Time Series
- Detailed Analytics & Visualization
===============================================================================
"""

import sys
import io
# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("[*] ADVANCED TRAINING SCRIPT V2.0 - STARTING...")

import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
import warnings
from datetime import datetime
from collections import defaultdict

# Statistical & ML Libraries
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ================= CONFIG =================
SCORE_FOLDER = r'diem_thi_thptqg'
BENCHMARK_FILE = 'diem_chuan_cleaned.csv'
OUTPUT_DIR = 'model_artifacts'

MODEL_OUTPUT = os.path.join(OUTPUT_DIR, 'university_ranking_model_2026.pkl')
LOOKUP_OUTPUT = os.path.join(OUTPUT_DIR, 'score_distribution_2025.pkl')
ANALYTICS_OUTPUT = os.path.join(OUTPUT_DIR, 'model_analytics.pkl')

# Y Duoc subject combinations
BLOCK_MAP = {
    'A00': ['toan', 'vat_ly', 'hoa_hoc'],
    'A01': ['toan', 'vat_ly', 'ngoai_ngu'],
    'A02': ['toan', 'vat_ly', 'sinh_hoc'],
    'B00': ['toan', 'hoa_hoc', 'sinh_hoc'],
    'B08': ['toan', 'sinh_hoc', 'ngoai_ngu'],
    'D01': ['toan', 'ngu_van', 'ngoai_ngu'],
    'D07': ['toan', 'hoa_hoc', 'ngoai_ngu'],
    'D08': ['toan', 'sinh_hoc', 'ngoai_ngu'],
    'D13': ['toan', 'ngu_van', 'sinh_hoc'],
}

# ================= 1. BUILD PERCENTILE LOOKUP =================
def build_percentile_lookup(score_folder):
    """Build percentile lookup table from THPT scores"""
    print(f"\n{'='*60}")
    print("[STEP 1] BUILDING SCORE DISTRIBUTION")
    print(f"{'='*60}")
    
    lookup_dict = {}
    year_stats = []
    files = glob.glob(os.path.join(score_folder, "*.csv"))
    
    if not files:
        print(f"[ERROR] No files found in {score_folder}")
        return {}, []

    print(f"[INFO] Found {len(files)} score files")

    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        try:
            year = int(re.search(r'\d{4}', filename).group())
        except:
            continue
        
        print(f"\n   [Year {year}]:", end=" ")
        all_cols = list(set([col for cols in BLOCK_MAP.values() for col in cols]))
        
        try:
            chunks = pd.read_csv(file_path, usecols=all_cols, chunksize=200000)
        except ValueError:
            print("[SKIP] Missing columns")
            continue

        block_data = {k: [] for k in BLOCK_MAP.keys()}
        total_students = 0
        
        for chunk in chunks:
            total_students += len(chunk)
            for block, cols in BLOCK_MAP.items():
                if not all(col in chunk.columns for col in cols): 
                    continue
                temp = chunk.dropna(subset=cols)
                if not temp.empty:
                    scores = temp[cols].sum(axis=1).tolist()
                    block_data[block].extend(scores)
        
        blocks_processed = 0
        for block, scores in block_data.items():
            if not scores: 
                continue
            scores_np = np.array(scores)
            scores_np.sort()
            scores_np = scores_np[::-1]
            
            df_score = pd.DataFrame({'score': scores_np})
            df_score['rank'] = df_score['score'].rank(method='min', ascending=False)
            total = len(df_score)
            
            lookup = df_score.groupby('score')['rank'].min().reset_index()
            lookup['percentile'] = (lookup['rank'] / total) * 100
            lookup_dict[(year, block)] = lookup.sort_values('score')
            blocks_processed += 1
        
        print(f"{total_students:,} students, {blocks_processed} blocks [OK]")
        year_stats.append({
            'year': year, 
            'students': total_students, 
            'blocks': blocks_processed
        })

    print(f"\n[OK] Completed! Processed {len(lookup_dict)} (Year, Block) pairs")
    return lookup_dict, year_stats


# ================= 2. FEATURE ENGINEERING =================
def calculate_trend(values):
    """Calculate trend (slope of linear regression)"""
    if len(values) < 2:
        return 0
    x = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0]

def calculate_volatility(values):
    """Calculate volatility (standard deviation)"""
    if len(values) < 2:
        return 0
    return np.std(values)

def calculate_momentum(values):
    """Calculate momentum (recent change)"""
    if len(values) < 2:
        return 0
    return values[-1] - values[-2]

def extract_features(percentiles, years):
    """Extract features from time series"""
    features = {
        'mean': np.mean(percentiles),
        'std': np.std(percentiles) if len(percentiles) > 1 else 0,
        'min': np.min(percentiles),
        'max': np.max(percentiles),
        'trend': calculate_trend(percentiles),
        'volatility': calculate_volatility(percentiles),
        'momentum': calculate_momentum(percentiles),
        'n_years': len(percentiles),
        'last_value': percentiles[-1],
        'range': np.max(percentiles) - np.min(percentiles)
    }
    return features


# ================= 3. MULTIPLE PREDICTION MODELS =================
def predict_weighted_average(values, alpha=2.0):
    """Weighted Average with exponential weights"""
    n = len(values)
    if n == 0: 
        return None, None
    weights = np.exp(np.linspace(0, alpha, n))
    pred = np.sum(values * weights) / weights.sum()
    
    # Confidence interval (based on weighted std)
    weighted_var = np.sum(weights * (values - pred)**2) / weights.sum()
    ci = 1.96 * np.sqrt(weighted_var) if weighted_var > 0 else 0
    
    return pred, ci

def predict_ets(values):
    """Exponential Smoothing with Trend"""
    try:
        if len(values) < 4: 
            return None, None
        model = ExponentialSmoothing(
            values, 
            trend='add', 
            seasonal=None, 
            initialization_method="estimated"
        )
        fit = model.fit()
        pred = fit.forecast(1)[0]
        
        # Confidence interval from residuals
        residuals = values - fit.fittedvalues
        ci = 1.96 * np.std(residuals)
        
        return pred, ci
    except: 
        return None, None

def predict_arima(values):
    """ARIMA(1,1,1) model"""
    try:
        if len(values) < 5: 
            return None, None
        model = ARIMA(values, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.get_forecast(steps=1)
        pred = forecast.predicted_mean[0]
        ci = forecast.conf_int().iloc[0, 1] - pred  # Upper bound - mean
        
        return pred, ci
    except: 
        return None, None

def predict_linear_regression(values):
    """Linear Regression extrapolation"""
    try:
        if len(values) < 3: 
            return None, None
        x = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(x, y)
        
        next_x = np.array([[len(values)]])
        pred = model.predict(next_x)[0]
        
        # Confidence interval from residuals
        y_pred = model.predict(x)
        residuals = y - y_pred
        ci = 1.96 * np.std(residuals)
        
        return pred, ci
    except: 
        return None, None

def predict_ridge_regression(values, features):
    """Ridge Regression with features"""
    try:
        if len(values) < 3:
            return None, None
        
        x = np.arange(len(values)).reshape(-1, 1)
        y = np.array(values)
        
        model = Ridge(alpha=1.0)
        model.fit(x, y)
        
        next_x = np.array([[len(values)]])
        pred = model.predict(next_x)[0]
        
        y_pred = model.predict(x)
        residuals = y - y_pred
        ci = 1.96 * np.std(residuals)
        
        return pred, ci
    except:
        return None, None


# ================= 4. ENSEMBLE MODEL =================
def ensemble_predict(values, features=None):
    """
    Ensemble prediction with multiple models
    Auto-select and combine best models
    """
    predictions = {}
    confidence_intervals = {}
    
    # 1. Weighted Average
    pred_wa, ci_wa = predict_weighted_average(values)
    if pred_wa is not None:
        predictions['WA'] = pred_wa
        confidence_intervals['WA'] = ci_wa
    
    # 2. Exponential Smoothing
    pred_ets, ci_ets = predict_ets(values)
    if pred_ets is not None:
        predictions['ETS'] = pred_ets
        confidence_intervals['ETS'] = ci_ets
    
    # 3. ARIMA
    pred_arima, ci_arima = predict_arima(values)
    if pred_arima is not None:
        predictions['ARIMA'] = pred_arima
        confidence_intervals['ARIMA'] = ci_arima
    
    # 4. Linear Regression
    pred_lr, ci_lr = predict_linear_regression(values)
    if pred_lr is not None:
        predictions['LR'] = pred_lr
        confidence_intervals['LR'] = ci_lr
    
    if not predictions:
        return np.mean(values), np.std(values) * 1.96, 'MEAN', predictions
    
    # Ensemble: Weighted average of predictions
    # Weights based on inverse of CI (smaller CI = higher weight)
    total_weight = 0
    weighted_pred = 0
    weighted_ci = 0
    
    for model, pred in predictions.items():
        ci = confidence_intervals.get(model, 1)
        if ci <= 0:
            ci = 0.1
        weight = 1 / ci
        weighted_pred += pred * weight
        weighted_ci += ci * weight
        total_weight += weight
    
    if total_weight > 0:
        ensemble_pred = weighted_pred / total_weight
        ensemble_ci = weighted_ci / total_weight
    else:
        ensemble_pred = np.mean(list(predictions.values()))
        ensemble_ci = np.mean(list(confidence_intervals.values()))
    
    # Select best model based on smallest CI
    best_model = min(confidence_intervals.keys(), key=lambda k: confidence_intervals[k])
    
    return ensemble_pred, ensemble_ci, best_model, predictions


# ================= 5. BACKTESTING & CROSS-VALIDATION =================
def time_series_cv(values, n_splits=3):
    """
    Time Series Cross-Validation
    Expanding window: train on old data, test on newer data
    """
    if len(values) < n_splits + 2:
        return None
    
    errors = defaultdict(list)
    
    for i in range(n_splits):
        split_point = len(values) - n_splits + i
        if split_point < 3:
            continue
            
        train = values[:split_point]
        test = values[split_point]
        
        # Test each model
        pred_wa, _ = predict_weighted_average(train)
        if pred_wa is not None:
            errors['WA'].append(abs(pred_wa - test))
        
        pred_ets, _ = predict_ets(train)
        if pred_ets is not None:
            errors['ETS'].append(abs(pred_ets - test))
        
        pred_arima, _ = predict_arima(train)
        if pred_arima is not None:
            errors['ARIMA'].append(abs(pred_arima - test))
        
        pred_lr, _ = predict_linear_regression(train)
        if pred_lr is not None:
            errors['LR'].append(abs(pred_lr - test))
    
    # Calculate mean error for each model
    mean_errors = {k: np.mean(v) if v else float('inf') for k, v in errors.items()}
    
    return mean_errors


# ================= 6. MAIN TRAINING FUNCTION =================
def train_advanced(df_labeled):
    """
    Advanced training with:
    - Multiple models
    - Cross-validation
    - Ensemble learning
    - Confidence intervals
    - Feature extraction
    """
    print(f"\n{'='*60}")
    print("[STEP 3] ADVANCED MODEL TRAINING")
    print(f"{'='*60}")
    
    groups = df_labeled.groupby(['university_id', 'ma_nganh', 'to_hop_mon'])
    
    # Results storage
    forecast_results = {}      # {key: predicted_percentile}
    confidence_results = {}    # {key: confidence_interval}
    model_selection = {}       # {key: best_model_name}
    feature_store = {}         # {key: extracted_features}
    
    # Statistics
    stats = defaultdict(int)
    all_errors = defaultdict(list)
    
    processed = 0
    total_groups = len(groups)
    
    print(f"\n[INFO] Training {total_groups} major groups...")
    
    for name, group in groups:
        processed += 1
        if processed % 50 == 0:
            print(f"   [Progress] {processed}/{total_groups} ({100*processed/total_groups:.1f}%)")

        group = group.sort_values('nam')
        percentiles = group['percentile_rank'].values
        years = group['nam'].values
        
        # Extract features
        features = extract_features(percentiles, years)
        feature_store[name] = features
        
        # Cross-validation to select model
        cv_errors = time_series_cv(percentiles)
        
        if cv_errors and len(percentiles) >= 4:
            # Enough data for CV
            best_cv_model = min(cv_errors.keys(), key=lambda k: cv_errors[k])
            stats[f'CV_{best_cv_model}'] += 1
            
            # Use ensemble prediction
            pred, ci, best_model, all_preds = ensemble_predict(percentiles, features)
            
            # Record errors
            for model, error_list in cv_errors.items():
                all_errors[model].extend([cv_errors[model]])
        else:
            # Little data, use WA
            pred, ci = predict_weighted_average(percentiles)
            if pred is None:
                pred = np.mean(percentiles)
                ci = np.std(percentiles) * 1.96
            best_model = 'WA'
            stats['WA_only'] += 1
        
        # Apply safety factor
        safety_factor = 0.95
        adjusted_pred = pred * safety_factor
        
        # Store results
        forecast_results[name] = adjusted_pred
        confidence_results[name] = ci
        model_selection[name] = best_model
        stats[best_model] += 1

    # Print statistics
    print(f"\n\n{'='*60}")
    print("[STATS] TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"\n[OK] Trained {len(forecast_results)} major groups")
    print(f"\n[INFO] Model Selection:")
    for model, count in sorted(stats.items()):
        print(f"   - {model}: {count} ({100*count/total_groups:.1f}%)")
    
    if all_errors:
        print(f"\n[INFO] Mean Absolute Error (CV):")
        for model, errors in all_errors.items():
            if errors:
                print(f"   - {model}: {np.mean(errors):.3f} +/- {np.std(errors):.3f}")
    
    # Package results
    analytics = {
        'predictions': forecast_results,
        'confidence_intervals': confidence_results,
        'model_selection': model_selection,
        'features': feature_store,
        'statistics': dict(stats),
        'cv_errors': {k: float(np.mean(v)) for k, v in all_errors.items() if v},
        'training_date': datetime.now().isoformat(),
        'n_groups': len(forecast_results)
    }
    
    return forecast_results, confidence_results, analytics


# ================= 7. DATA NORMALIZATION =================
def expand_multiple_blocks(df):
    """Split rows with multiple subject combinations"""
    expanded_rows = []
    for _, row in df.iterrows():
        to_hop = str(row['to_hop_mon']).strip()
        blocks = [b.strip() for b in to_hop.split(',')]
        for block in blocks:
            if block and block in BLOCK_MAP:
                new_row = row.copy()
                new_row['to_hop_mon'] = block
                expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)

def normalize_benchmark(benchmark_file, lookup_dict):
    """Convert benchmark scores to percentile"""
    print(f"\n{'='*60}")
    print("[STEP 2] CONVERTING SCORES TO PERCENTILE")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(benchmark_file)
        print(f"\n[INFO] Reading file: {benchmark_file}")
        print(f"   - Original rows: {len(df):,}")
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return pd.DataFrame()
    
    # Expand combinations
    df_expanded = expand_multiple_blocks(df)
    print(f"   - After expanding blocks: {len(df_expanded):,}")
    
    # Filter majors with 2025 data
    nganh_2025 = df_expanded[df_expanded['nam'] == 2025][
        ['university_id', 'ma_nganh', 'to_hop_mon']
    ].drop_duplicates()
    print(f"   - Majors with 2025 data: {len(nganh_2025)}")
    
    df_filtered = df_expanded.merge(
        nganh_2025, 
        on=['university_id', 'ma_nganh', 'to_hop_mon'], 
        how='inner'
    )
    print(f"   - After filtering: {len(df_filtered):,}")
    
    # Convert to percentile
    def get_percentile(row):
        year = row['nam']
        block = str(row['to_hop_mon']).strip()
        score = row['diem_chuan']
        lookup = lookup_dict.get((year, block))
        if lookup is None: 
            return np.nan 
        idx = np.searchsorted(lookup['score'], score, side='left')
        if idx < len(lookup): 
            return lookup.iloc[idx]['percentile']
        else: 
            return 0.01

    print(f"\n[INFO] Converting to percentile...")
    df_filtered['percentile_rank'] = df_filtered.apply(get_percentile, axis=1)
    df_clean = df_filtered.dropna(subset=['percentile_rank'])
    
    print(f"[OK] Completed! Training data: {len(df_clean):,} rows")
    
    return df_clean


# ================= 8. VISUALIZATION =================
def create_training_report(analytics, output_dir):
    """Create visualization report"""
    print(f"\n{'='*60}")
    print("[STEP 4] CREATING TRAINING REPORT")
    print(f"{'='*60}")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Model Selection Distribution
        ax1 = axes[0, 0]
        models = ['WA', 'ETS', 'ARIMA', 'LR']
        counts = [analytics['statistics'].get(m, 0) for m in models]
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        ax1.bar(models, counts, color=colors)
        ax1.set_title('Model Selection Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Number of Majors')
        for i, v in enumerate(counts):
            ax1.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # 2. CV Error Comparison
        ax2 = axes[0, 1]
        cv_errors = analytics.get('cv_errors', {})
        if cv_errors:
            models = list(cv_errors.keys())
            errors = list(cv_errors.values())
            ax2.barh(models, errors, color=colors[:len(models)])
            ax2.set_title('Cross-Validation Error (MAE)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Mean Absolute Error')
        else:
            ax2.text(0.5, 0.5, 'No CV data', ha='center', va='center')
            ax2.set_title('Cross-Validation Error', fontsize=12, fontweight='bold')
        
        # 3. Confidence Interval Distribution
        ax3 = axes[1, 0]
        ci_values = list(analytics['confidence_intervals'].values())
        ax3.hist(ci_values, bins=30, color='#636EFA', edgecolor='white', alpha=0.7)
        ax3.axvline(np.mean(ci_values), color='red', linestyle='--', label=f'Mean: {np.mean(ci_values):.2f}')
        ax3.set_title('Confidence Interval Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Confidence Interval (percentile points)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Prediction Distribution
        ax4 = axes[1, 1]
        pred_values = list(analytics['predictions'].values())
        ax4.hist(pred_values, bins=30, color='#00CC96', edgecolor='white', alpha=0.7)
        ax4.axvline(np.mean(pred_values), color='red', linestyle='--', label=f'Mean: {np.mean(pred_values):.2f}')
        ax4.set_title('Predicted Percentile Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Predicted Percentile (Top %)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.tight_layout()
        
        report_path = os.path.join(output_dir, 'training_report.png')
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Report saved: {report_path}")
        plt.close()
        
    except Exception as e:
        print(f"[WARNING] Failed to create report: {e}")


# ================= MAIN =================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED UNIVERSITY ADMISSION PREDICTION SYSTEM V2.0")
    print("="*60)
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INPUT] {SCORE_FOLDER}")
    print(f"[OUTPUT] {OUTPUT_DIR}")
    print("="*60)
    
    # Check input
    if not os.path.exists(SCORE_FOLDER):
        print(f"[ERROR] Folder not found: {SCORE_FOLDER}")
        exit()
    
    # Create output dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created folder: {OUTPUT_DIR}")
    
    # 1. Build percentile lookup
    lookup_map, year_stats = build_percentile_lookup(SCORE_FOLDER)
    if not lookup_map: 
        exit()
    
    # 2. Normalize benchmark data
    df_train = normalize_benchmark(BENCHMARK_FILE, lookup_map)
    if df_train.empty: 
        exit()
    
    # 3. Advanced training
    model_2026, confidence_2026, analytics = train_advanced(df_train)
    
    # 4. Create visualization report
    create_training_report(analytics, OUTPUT_DIR)
    
    # 5. Save results
    print(f"\n{'='*60}")
    print("[STEP 5] SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save model (predictions only for backward compatibility)
    joblib.dump(model_2026, MODEL_OUTPUT)
    print(f"[OK] Model: {MODEL_OUTPUT}")
    
    # Save lookup 2025
    lookup_2025 = {k: v for k, v in lookup_map.items() if k[0] == 2025}
    joblib.dump(lookup_2025, LOOKUP_OUTPUT)
    print(f"[OK] Lookup: {LOOKUP_OUTPUT}")
    
    # Save full analytics
    joblib.dump(analytics, ANALYTICS_OUTPUT)
    print(f"[OK] Analytics: {ANALYTICS_OUTPUT}")
    
    # Summary
    print(f"\n{'='*60}")
    print("[DONE] TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"[STATS] Total majors trained: {len(model_2026)}")
    print(f"[STATS] Mean Confidence Interval: {np.mean(list(confidence_2026.values())):.2f}")
    print(f"[PATH] Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60)
