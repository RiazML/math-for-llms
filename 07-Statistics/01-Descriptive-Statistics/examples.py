"""
Descriptive Statistics - Examples
================================
Practical demonstrations of descriptive statistics.
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def example_central_tendency():
    """Measures of central tendency."""
    print("=" * 60)
    print("EXAMPLE 1: Measures of Central Tendency")
    print("=" * 60)
    
    # Symmetric data
    np.random.seed(42)
    symmetric = np.random.normal(50, 10, 1000)
    
    print("Symmetric data (Normal):")
    print(f"  Mean:   {symmetric.mean():.2f}")
    print(f"  Median: {np.median(symmetric):.2f}")
    print(f"  Mode:   {stats.mode(symmetric.round(), keepdims=True)[0][0]:.2f}")
    print("  (Mean ≈ Median for symmetric data)")
    
    # Right-skewed data
    skewed = np.random.exponential(10, 1000)
    
    print("\nRight-skewed data (Exponential):")
    print(f"  Mean:   {skewed.mean():.2f}")
    print(f"  Median: {np.median(skewed):.2f}")
    print("  (Mean > Median for right-skewed)")
    
    # Data with outliers
    with_outliers = np.concatenate([np.random.normal(50, 5, 95), [200, 250, 300, 350, 400]])
    
    print("\nData with outliers:")
    print(f"  Mean:   {with_outliers.mean():.2f}")
    print(f"  Median: {np.median(with_outliers):.2f}")
    print("  (Median is robust to outliers)")


def example_weighted_geometric_harmonic():
    """Weighted, geometric, and harmonic means."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Special Means")
    print("=" * 60)
    
    # Weighted mean
    grades = np.array([85, 90, 78, 92])
    weights = np.array([0.1, 0.3, 0.2, 0.4])  # Exam weights
    
    weighted_mean = np.average(grades, weights=weights)
    simple_mean = grades.mean()
    
    print("Weighted Mean (Exam Scores):")
    print(f"  Grades: {grades}")
    print(f"  Weights: {weights}")
    print(f"  Simple mean: {simple_mean:.2f}")
    print(f"  Weighted mean: {weighted_mean:.2f}")
    
    # Geometric mean (growth rates)
    growth_rates = np.array([1.10, 0.95, 1.20, 1.05])  # 10%, -5%, 20%, 5%
    
    arithmetic_avg = growth_rates.mean()
    geometric_avg = stats.gmean(growth_rates)
    
    print("\nGeometric Mean (Growth Rates):")
    print(f"  Annual multipliers: {growth_rates}")
    print(f"  Arithmetic mean: {arithmetic_avg:.4f}")
    print(f"  Geometric mean: {geometric_avg:.4f}")
    print(f"  After 4 years: ${100 * np.prod(growth_rates):.2f}")
    print(f"  Using geo mean: ${100 * geometric_avg**4:.2f}")
    
    # Harmonic mean (rates, F1-score)
    precision = 0.9
    recall = 0.6
    
    arithmetic_avg = (precision + recall) / 2
    harmonic_avg = stats.hmean([precision, recall])
    
    print("\nHarmonic Mean (F1 Score):")
    print(f"  Precision: {precision}, Recall: {recall}")
    print(f"  Arithmetic mean: {arithmetic_avg:.4f}")
    print(f"  F1 (harmonic mean): {harmonic_avg:.4f}")
    print("  (Harmonic mean penalizes imbalance)")


def example_spread_measures():
    """Measures of spread/dispersion."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Measures of Spread")
    print("=" * 60)
    
    np.random.seed(42)
    data = np.random.normal(100, 15, 200)
    
    print(f"Data: n={len(data)}, Normal(100, 15²)")
    
    # Range
    print(f"\nRange: {data.max() - data.min():.2f}")
    print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
    
    # Variance and Std
    print(f"\nVariance (sample): {data.var(ddof=1):.2f}")
    print(f"Std Dev (sample): {data.std(ddof=1):.2f}")
    print(f"  (True σ = 15)")
    
    # IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    print(f"\nIQR: {iqr:.2f}")
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}")
    
    # MAD
    mad = np.mean(np.abs(data - data.mean()))
    print(f"\nMAD: {mad:.2f}")
    
    # CV
    cv = data.std() / data.mean()
    print(f"\nCoefficient of Variation: {cv:.4f} ({cv*100:.2f}%)")


def example_percentiles():
    """Percentiles and quantiles."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Percentiles and Quantiles")
    print("=" * 60)
    
    np.random.seed(42)
    data = np.random.exponential(10, 1000)
    
    print("Exponential data (1000 samples):")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(data, p)
        print(f"  {p}th percentile: {val:.2f}")
    
    # Five-number summary
    five_num = np.percentile(data, [0, 25, 50, 75, 100])
    print(f"\nFive-number summary:")
    print(f"  Min: {five_num[0]:.2f}")
    print(f"  Q1:  {five_num[1]:.2f}")
    print(f"  Med: {five_num[2]:.2f}")
    print(f"  Q3:  {five_num[3]:.2f}")
    print(f"  Max: {five_num[4]:.2f}")


def example_outlier_detection():
    """Outlier detection using IQR."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Outlier Detection (IQR Method)")
    print("=" * 60)
    
    np.random.seed(42)
    clean_data = np.random.normal(50, 5, 100)
    outliers = np.array([10, 15, 85, 90, 95])
    data = np.concatenate([clean_data, outliers])
    
    print(f"Data: {len(clean_data)} normal points + {len(outliers)} outliers")
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    print(f"\nQ1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"Lower fence: {lower_fence:.2f}")
    print(f"Upper fence: {upper_fence:.2f}")
    
    outlier_mask = (data < lower_fence) | (data > upper_fence)
    detected_outliers = data[outlier_mask]
    
    print(f"\nDetected outliers: {detected_outliers}")
    print(f"Number of outliers: {len(detected_outliers)}")
    
    # Z-score method
    z_scores = np.abs((data - data.mean()) / data.std())
    z_outliers = data[z_scores > 3]
    print(f"\nZ-score method (|z| > 3): {z_outliers}")


def example_skewness_kurtosis():
    """Skewness and kurtosis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Skewness and Kurtosis")
    print("=" * 60)
    
    np.random.seed(42)
    n = 10000
    
    # Different distributions
    normal = np.random.normal(0, 1, n)
    right_skew = np.random.exponential(1, n)
    left_skew = -np.random.exponential(1, n)
    heavy_tails = np.random.standard_t(3, n)
    light_tails = np.random.uniform(-1.73, 1.73, n)  # Same variance as N(0,1)
    
    def describe(data, name):
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)  # Excess kurtosis (Normal = 0)
        print(f"\n{name}:")
        print(f"  Skewness: {skew:.4f}")
        print(f"  Excess Kurtosis: {kurt:.4f}")
    
    describe(normal, "Normal")
    describe(right_skew, "Exponential (right-skewed)")
    describe(left_skew, "Negative Exponential (left-skewed)")
    describe(heavy_tails, "Student's t(3) (heavy tails)")
    describe(light_tails, "Uniform (light tails)")
    
    print("\nInterpretation:")
    print("  Skewness: 0=symmetric, >0=right tail, <0=left tail")
    print("  Excess Kurtosis: 0=normal, >0=heavy tails, <0=light tails")


def example_correlation():
    """Correlation measures."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Correlation")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    
    # Linear relationship
    x = np.random.normal(0, 1, n)
    y_linear = 2*x + 1 + np.random.normal(0, 0.5, n)
    
    # Nonlinear relationship
    y_nonlinear = x**2 + np.random.normal(0, 0.3, n)
    
    # Monotonic but nonlinear
    y_monotonic = np.exp(x) + np.random.normal(0, 0.5, n)
    
    print("Pearson vs Spearman Correlation:")
    
    # Linear
    pearson = np.corrcoef(x, y_linear)[0, 1]
    spearman = stats.spearmanr(x, y_linear)[0]
    print(f"\nLinear (y = 2x + noise):")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    
    # Nonlinear (quadratic)
    pearson = np.corrcoef(x, y_nonlinear)[0, 1]
    spearman = stats.spearmanr(x, y_nonlinear)[0]
    print(f"\nNonlinear (y = x²):")
    print(f"  Pearson:  {pearson:.4f} (misses the relationship!)")
    print(f"  Spearman: {spearman:.4f}")
    
    # Monotonic nonlinear
    pearson = np.corrcoef(x, y_monotonic)[0, 1]
    spearman = stats.spearmanr(x, y_monotonic)[0]
    print(f"\nMonotonic (y = exp(x)):")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f} (captures monotonic relationship)")


def example_covariance_matrix():
    """Covariance matrix for multivariate data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Covariance Matrix")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate correlated data
    n = 500
    mean = [0, 0, 0]
    cov = [[1.0, 0.8, 0.2],
           [0.8, 1.5, 0.5],
           [0.2, 0.5, 2.0]]
    
    data = np.random.multivariate_normal(mean, cov, n)
    
    print("True covariance matrix:")
    print(np.array(cov))
    
    # Sample covariance
    sample_cov = np.cov(data.T)
    print("\nSample covariance matrix:")
    print(sample_cov.round(3))
    
    # Correlation matrix
    sample_corr = np.corrcoef(data.T)
    print("\nCorrelation matrix:")
    print(sample_corr.round(3))


def example_normalization():
    """Data normalization techniques."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Data Normalization")
    print("=" * 60)
    
    np.random.seed(42)
    data = np.random.exponential(10, 100) + 50
    
    print(f"Original data:")
    print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
    
    # Z-score standardization
    z_score = (data - data.mean()) / data.std()
    print(f"\nZ-score standardization:")
    print(f"  Mean: {z_score.mean():.4f}, Std: {z_score.std():.4f}")
    
    # Min-max scaling
    min_max = (data - data.min()) / (data.max() - data.min())
    print(f"\nMin-max scaling [0, 1]:")
    print(f"  Min: {min_max.min():.4f}, Max: {min_max.max():.4f}")
    
    # Robust scaling
    median = np.median(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    robust = (data - median) / iqr
    print(f"\nRobust scaling (IQR):")
    print(f"  Median: {np.median(robust):.4f}")
    print(f"  IQR: {np.percentile(robust, 75) - np.percentile(robust, 25):.4f}")


def example_ml_preprocessing():
    """Descriptive statistics for ML preprocessing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: ML Preprocessing with Descriptive Stats")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate feature matrix
    n_samples = 1000
    
    # Different feature types
    feature1 = np.random.normal(100, 20, n_samples)  # Normal
    feature2 = np.random.exponential(5, n_samples)   # Skewed
    feature3 = np.random.uniform(0, 1, n_samples)    # Uniform
    
    # Add some outliers
    outlier_idx = np.random.choice(n_samples, 20)
    feature1[outlier_idx] *= 3
    
    X = np.column_stack([feature1, feature2, feature3])
    
    print("Feature Analysis:")
    print("-" * 50)
    
    for i in range(3):
        feat = X[:, i]
        print(f"\nFeature {i+1}:")
        print(f"  Mean: {feat.mean():.2f}, Median: {np.median(feat):.2f}")
        print(f"  Std: {feat.std():.2f}")
        print(f"  Skewness: {stats.skew(feat):.2f}")
        
        # Recommend transformation
        if abs(stats.skew(feat)) > 1:
            print("  → Consider log transform (skewed)")
        
        # Check for outliers
        q1, q3 = np.percentile(feat, [25, 75])
        iqr = q3 - q1
        n_outliers = np.sum((feat < q1 - 1.5*iqr) | (feat > q3 + 1.5*iqr))
        if n_outliers > 0:
            print(f"  → {n_outliers} potential outliers detected")
    
    print("\n" + "-" * 50)
    print("Preprocessing recommendations applied!")
    
    # Apply transformations
    X_processed = X.copy()
    
    # Log transform skewed feature
    X_processed[:, 1] = np.log1p(X_processed[:, 1])
    
    # Standardize all
    X_processed = (X_processed - X_processed.mean(axis=0)) / X_processed.std(axis=0)
    
    print(f"\nAfter preprocessing:")
    print(f"  All features: mean ≈ 0, std ≈ 1")
    print(f"  Feature 2 skewness: {stats.skew(X_processed[:, 1]):.2f} (was {stats.skew(X[:, 1]):.2f})")


if __name__ == "__main__":
    example_central_tendency()
    example_weighted_geometric_harmonic()
    example_spread_measures()
    example_percentiles()
    example_outlier_detection()
    example_skewness_kurtosis()
    example_correlation()
    example_covariance_matrix()
    example_normalization()
    example_ml_preprocessing()
