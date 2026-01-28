"""
Descriptive Statistics - Exercises
=================================
Practice problems for descriptive statistics.
"""

import numpy as np
from scipy import stats


class DescriptiveStatsExercises:
    """Exercises for descriptive statistics."""
    
    def exercise_1_mean_median_mode(self):
        """
        Exercise 1: Mean, Median, Mode
        
        Compute all three measures and discuss when each is appropriate.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Mean, Median, Mode")
        
        # Dataset: Income data (typically right-skewed)
        np.random.seed(42)
        incomes = np.concatenate([
            np.random.normal(50000, 15000, 90),
            np.random.normal(200000, 50000, 10)  # High earners
        ])
        incomes = np.maximum(incomes, 20000)  # Min wage
        
        mean_income = incomes.mean()
        median_income = np.median(incomes)
        mode_result = stats.mode(incomes.round(-3), keepdims=True)
        mode_income = mode_result[0][0]
        
        print(f"\nIncome data (n={len(incomes)}):")
        print(f"  Mean:   ${mean_income:,.0f}")
        print(f"  Median: ${median_income:,.0f}")
        print(f"  Mode:   ${mode_income:,.0f}")
        
        print("\nWhen to use each:")
        print("  • Mean: Symmetric data, need mathematical properties")
        print("  • Median: Skewed data, outliers present (BEST for income!)")
        print("  • Mode: Categorical data, finding most common value")
        
        print(f"\nHere, mean (${mean_income:,.0f}) is pulled up by high earners")
        print(f"Median (${median_income:,.0f}) better represents 'typical' income")
    
    def exercise_2_variance_std(self):
        """
        Exercise 2: Variance and Standard Deviation
        
        Calculate sample variance with Bessel's correction.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Variance and Standard Deviation")
        
        data = np.array([12, 15, 18, 22, 25, 28, 30])
        n = len(data)
        
        # Mean
        mean = data.mean()
        print(f"\nData: {data}")
        print(f"n = {n}")
        print(f"Mean = {mean:.2f}")
        
        # Variance (step by step)
        deviations = data - mean
        squared_dev = deviations ** 2
        
        print(f"\nDeviations (x - x̄): {deviations.round(2)}")
        print(f"Squared deviations: {squared_dev.round(2)}")
        print(f"Sum of squared deviations: {squared_dev.sum():.2f}")
        
        # Population variance (divide by n)
        var_pop = squared_dev.sum() / n
        
        # Sample variance (divide by n-1)
        var_sample = squared_dev.sum() / (n - 1)
        
        print(f"\nPopulation variance (÷n): {var_pop:.4f}")
        print(f"Sample variance (÷(n-1)): {var_sample:.4f}")
        print(f"  NumPy check: {data.var(ddof=1):.4f}")
        
        print(f"\nStandard deviation: {np.sqrt(var_sample):.4f}")
        
        print("\nWhy n-1 (Bessel's correction)?")
        print("  Sample mean is closer to sample points than true μ")
        print("  Using n underestimates variance; n-1 corrects for this")
    
    def exercise_3_iqr_outliers(self):
        """
        Exercise 3: IQR and Outlier Detection
        
        Use IQR method to identify outliers.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: IQR and Outlier Detection")
        
        data = np.array([10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 45, 48, 50])
        
        print(f"Data: {data}")
        
        q1 = np.percentile(data, 25)
        q2 = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        print(f"\nQuartiles:")
        print(f"  Q1 (25th percentile): {q1}")
        print(f"  Q2 (median): {q2}")
        print(f"  Q3 (75th percentile): {q3}")
        print(f"  IQR = Q3 - Q1 = {iqr}")
        
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        
        print(f"\nFences:")
        print(f"  Lower: Q1 - 1.5×IQR = {q1} - {1.5*iqr} = {lower_fence}")
        print(f"  Upper: Q3 + 1.5×IQR = {q3} + {1.5*iqr} = {upper_fence}")
        
        outliers = data[(data < lower_fence) | (data > upper_fence)]
        print(f"\nOutliers (outside fences): {outliers}")
        
        print("\nBox plot would show:")
        print(f"  Box: [{q1}, {q3}]")
        print(f"  Whiskers extend to [{max(data.min(), lower_fence)}, {min(data.max(), upper_fence)}]")
        print(f"  Points beyond: outliers")
    
    def exercise_4_skewness(self):
        """
        Exercise 4: Skewness Analysis
        
        Calculate skewness and determine transformation need.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Skewness Analysis")
        
        np.random.seed(42)
        
        # Right-skewed data (common in ML: income, prices, counts)
        original = np.random.exponential(10, 1000)
        
        print("Original data (exponential):")
        print(f"  Mean: {original.mean():.2f}")
        print(f"  Median: {np.median(original):.2f}")
        print(f"  Skewness: {stats.skew(original):.4f}")
        
        print("\nSkewness interpretation:")
        print("  |skew| < 0.5: approximately symmetric")
        print("  0.5 ≤ |skew| < 1: moderately skewed")
        print("  |skew| ≥ 1: highly skewed")
        
        skew = stats.skew(original)
        if abs(skew) >= 1:
            print(f"\n  Current skewness {skew:.2f} is HIGH")
            print("  → Recommend log or Box-Cox transformation")
        
        # Apply log transform
        log_transformed = np.log1p(original)
        
        print(f"\nAfter log(1+x) transformation:")
        print(f"  Mean: {log_transformed.mean():.2f}")
        print(f"  Median: {np.median(log_transformed):.2f}")
        print(f"  Skewness: {stats.skew(log_transformed):.4f}")
        
        print("\nWhy transform?")
        print("  • Many ML algorithms assume roughly normal features")
        print("  • Reduces impact of extreme values")
        print("  • Can improve model performance")
    
    def exercise_5_correlation_types(self):
        """
        Exercise 5: Pearson vs Spearman Correlation
        
        Compare correlations for different relationships.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Pearson vs Spearman Correlation")
        
        np.random.seed(42)
        n = 100
        
        x = np.linspace(0, 5, n)
        noise = np.random.normal(0, 0.3, n)
        
        print("Comparing correlations for different relationships:\n")
        
        # 1. Linear relationship
        y_linear = 2*x + 1 + noise
        r_pearson = np.corrcoef(x, y_linear)[0, 1]
        r_spearman = stats.spearmanr(x, y_linear)[0]
        
        print("1. Linear (y = 2x + 1 + noise):")
        print(f"   Pearson:  {r_pearson:.4f}")
        print(f"   Spearman: {r_spearman:.4f}")
        print("   Both high - linear is both linear and monotonic")
        
        # 2. Quadratic (non-monotonic)
        y_quad = (x - 2.5)**2 + noise
        r_pearson = np.corrcoef(x, y_quad)[0, 1]
        r_spearman = stats.spearmanr(x, y_quad)[0]
        
        print("\n2. Quadratic (y = (x-2.5)² + noise):")
        print(f"   Pearson:  {r_pearson:.4f}")
        print(f"   Spearman: {r_spearman:.4f}")
        print("   Both low - relationship is non-monotonic")
        
        # 3. Exponential (monotonic, nonlinear)
        y_exp = np.exp(x/2) + noise * 5
        r_pearson = np.corrcoef(x, y_exp)[0, 1]
        r_spearman = stats.spearmanr(x, y_exp)[0]
        
        print("\n3. Exponential (y = exp(x/2) + noise):")
        print(f"   Pearson:  {r_pearson:.4f}")
        print(f"   Spearman: {r_spearman:.4f}")
        print("   Spearman higher - captures monotonic relationship better")
        
        print("\nGuidelines:")
        print("  • Pearson: Linear relationships, normally distributed")
        print("  • Spearman: Monotonic relationships, ordinal data, outliers")
    
    def exercise_6_standardization(self):
        """
        Exercise 6: Feature Standardization
        
        Compare different scaling methods.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Feature Standardization")
        
        np.random.seed(42)
        
        # Original data with outliers
        data = np.concatenate([
            np.random.normal(50, 10, 95),
            np.array([150, 160, 170, 180, 190])  # Outliers
        ])
        
        print("Original data with outliers:")
        print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        print(f"  Median: {np.median(data):.2f}")
        print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
        
        # Z-score standardization
        z_score = (data - data.mean()) / data.std()
        print(f"\n1. Z-score (x - μ) / σ:")
        print(f"   Mean: {z_score.mean():.4f}, Std: {z_score.std():.4f}")
        print(f"   Min: {z_score.min():.2f}, Max: {z_score.max():.2f}")
        print("   Outliers still far from center!")
        
        # Min-max scaling
        min_max = (data - data.min()) / (data.max() - data.min())
        print(f"\n2. Min-max [0, 1]:")
        print(f"   Min: {min_max.min():.4f}, Max: {min_max.max():.4f}")
        print(f"   Most values compressed near 0 due to outliers!")
        
        # Robust scaling
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        robust = (data - median) / iqr
        print(f"\n3. Robust (x - median) / IQR:")
        print(f"   Median: {np.median(robust):.4f}")
        print(f"   IQR: {np.percentile(robust, 75) - np.percentile(robust, 25):.4f}")
        print("   Outliers have less influence!")
        
        print("\nRecommendations:")
        print("  • Z-score: No outliers, Gaussian-like features")
        print("  • Min-max: Need bounded [0,1], no outliers")
        print("  • Robust: Outliers present, need stable scaling")
    
    def exercise_7_covariance_matrix(self):
        """
        Exercise 7: Covariance Matrix Properties
        
        Compute and interpret covariance matrix.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Covariance Matrix Properties")
        
        np.random.seed(42)
        
        # Generate correlated features
        n = 500
        x1 = np.random.normal(0, 1, n)
        x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n)  # Correlated with x1
        x3 = np.random.normal(0, 1, n)  # Independent
        
        X = np.column_stack([x1, x2, x3])
        
        print("Data matrix X: 500 samples × 3 features")
        
        # Covariance matrix
        cov_matrix = np.cov(X.T)
        print("\nCovariance matrix:")
        print(cov_matrix.round(3))
        
        print("\nInterpretation:")
        print("  • Diagonal: variances of each feature")
        print(f"    Var(X1) = {cov_matrix[0,0]:.3f}")
        print(f"    Var(X2) = {cov_matrix[1,1]:.3f}")
        print(f"    Var(X3) = {cov_matrix[2,2]:.3f}")
        
        print("  • Off-diagonal: covariances between features")
        print(f"    Cov(X1, X2) = {cov_matrix[0,1]:.3f} (high: correlated)")
        print(f"    Cov(X1, X3) = {cov_matrix[0,2]:.3f} (low: independent)")
        
        # Correlation matrix
        corr_matrix = np.corrcoef(X.T)
        print("\nCorrelation matrix (standardized covariance):")
        print(corr_matrix.round(3))
        
        print("\nProperties of covariance matrix:")
        print("  1. Symmetric: Σᵢⱼ = Σⱼᵢ")
        print(f"     Check: {np.allclose(cov_matrix, cov_matrix.T)}")
        print("  2. Positive semi-definite: eigenvalues ≥ 0")
        eigenvalues = np.linalg.eigvals(cov_matrix)
        print(f"     Eigenvalues: {eigenvalues.round(3)}")
    
    def exercise_8_five_number_summary(self):
        """
        Exercise 8: Five-Number Summary
        
        Compute and interpret the five-number summary.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Five-Number Summary")
        
        # Test scores
        scores = np.array([55, 60, 62, 65, 68, 70, 72, 75, 78, 80,
                          82, 85, 88, 90, 92, 95, 98, 45, 100, 58])
        scores = np.sort(scores)
        
        print(f"Test scores (sorted):")
        print(f"  {scores}")
        
        five_num = {
            'Min': np.min(scores),
            'Q1': np.percentile(scores, 25),
            'Median': np.median(scores),
            'Q3': np.percentile(scores, 75),
            'Max': np.max(scores)
        }
        
        print("\nFive-number summary:")
        for name, value in five_num.items():
            print(f"  {name}: {value}")
        
        iqr = five_num['Q3'] - five_num['Q1']
        print(f"\nIQR: {iqr}")
        
        print("\nBox plot interpretation:")
        print(f"  • 25% of scores below {five_num['Q1']}")
        print(f"  • 50% of scores below {five_num['Median']}")
        print(f"  • 75% of scores below {five_num['Q3']}")
        print(f"  • Middle 50% of scores: [{five_num['Q1']}, {five_num['Q3']}]")
        print(f"  • Range: {five_num['Max'] - five_num['Min']}")
    
    def exercise_9_cv_comparison(self):
        """
        Exercise 9: Coefficient of Variation
        
        Compare variability across different scales.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Coefficient of Variation")
        
        np.random.seed(42)
        
        # Two features on different scales
        height_cm = np.random.normal(170, 10, 100)  # Height in cm
        weight_kg = np.random.normal(70, 15, 100)   # Weight in kg
        
        print("Compare variability of features on different scales:")
        
        print("\nHeight (cm):")
        print(f"  Mean: {height_cm.mean():.2f}")
        print(f"  Std:  {height_cm.std():.2f}")
        cv_height = height_cm.std() / height_cm.mean()
        print(f"  CV:   {cv_height:.4f} ({cv_height*100:.2f}%)")
        
        print("\nWeight (kg):")
        print(f"  Mean: {weight_kg.mean():.2f}")
        print(f"  Std:  {weight_kg.std():.2f}")
        cv_weight = weight_kg.std() / weight_kg.mean()
        print(f"  CV:   {cv_weight:.4f} ({cv_weight*100:.2f}%)")
        
        print("\nInterpretation:")
        print(f"  Weight CV ({cv_weight*100:.1f}%) > Height CV ({cv_height*100:.1f}%)")
        print("  Weight has more relative variability than height")
        print("\nCV is dimensionless - allows comparing different scales!")
        
        print("\nUse cases:")
        print("  • Feature importance comparison")
        print("  • Model stability analysis")
        print("  • Quality control (low CV = consistent)")
    
    def exercise_10_data_quality(self):
        """
        Exercise 10: Data Quality Assessment
        
        Use descriptive statistics to assess data quality.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Data Quality Assessment")
        
        np.random.seed(42)
        
        # Simulate messy dataset
        n = 1000
        
        # Age feature (with some issues)
        age = np.random.normal(35, 12, n)
        age[np.random.choice(n, 10)] = -5  # Invalid negative ages
        age[np.random.choice(n, 5)] = 150  # Unrealistic ages
        age[np.random.choice(n, 50)] = np.nan  # Missing values
        
        # Income feature (skewed with outliers)
        income = np.random.exponential(50000, n)
        income[np.random.choice(n, 3)] = 10000000  # Extreme outliers
        
        print("Data Quality Report")
        print("=" * 50)
        
        # Age analysis
        print("\nFeature: Age")
        valid_age = age[~np.isnan(age)]
        print(f"  Missing values: {np.isnan(age).sum()} ({np.isnan(age).mean()*100:.1f}%)")
        print(f"  Mean: {np.nanmean(age):.1f}, Median: {np.nanmedian(age):.1f}")
        print(f"  Min: {np.nanmin(age):.1f}, Max: {np.nanmax(age):.1f}")
        
        # Check validity
        invalid_low = (valid_age < 0).sum()
        invalid_high = (valid_age > 120).sum()
        print(f"  Invalid (< 0): {invalid_low}")
        print(f"  Invalid (> 120): {invalid_high}")
        print("  → ACTION: Clip to [0, 120] or impute")
        
        # Income analysis
        print("\nFeature: Income")
        print(f"  Mean: ${income.mean():,.0f}")
        print(f"  Median: ${np.median(income):,.0f}")
        print(f"  Skewness: {stats.skew(income):.2f}")
        
        q1, q3 = np.percentile(income, [25, 75])
        iqr = q3 - q1
        n_outliers = ((income < q1 - 1.5*iqr) | (income > q3 + 1.5*iqr)).sum()
        print(f"  Outliers (IQR method): {n_outliers}")
        print("  → ACTION: Log transform, consider capping outliers")
        
        print("\n" + "=" * 50)
        print("Recommendations:")
        print("  1. Handle missing values (imputation/removal)")
        print("  2. Fix invalid age values")
        print("  3. Transform skewed income")
        print("  4. Consider outlier treatment")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = DescriptiveStatsExercises()
    
    print("DESCRIPTIVE STATISTICS EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    print("\n" + "=" * 70)
    
    exercises.solution_2()
    print("\n" + "=" * 70)
    
    exercises.solution_3()
    print("\n" + "=" * 70)
    
    exercises.solution_4()
    print("\n" + "=" * 70)
    
    exercises.solution_5()
    print("\n" + "=" * 70)
    
    exercises.solution_6()
    print("\n" + "=" * 70)
    
    exercises.solution_7()
    print("\n" + "=" * 70)
    
    exercises.solution_8()
    print("\n" + "=" * 70)
    
    exercises.solution_9()
    print("\n" + "=" * 70)
    
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
