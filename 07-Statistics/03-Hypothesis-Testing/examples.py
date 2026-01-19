"""
Hypothesis Testing - Examples
=============================
Practical demonstrations of statistical hypothesis testing.
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def example_one_sample_t():
    """One-sample t-test."""
    print("=" * 60)
    print("EXAMPLE 1: One-Sample t-Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Manufacturing: Testing if mean weight = 500g
    true_mean = 502  # Slightly over
    sample = np.random.normal(true_mean, 10, 30)
    
    print("Manufacturing Quality Control")
    print("H₀: μ = 500g (target weight)")
    print("H₁: μ ≠ 500g (two-tailed)")
    
    mu_0 = 500
    
    # Manual calculation
    x_bar = sample.mean()
    s = sample.std(ddof=1)
    n = len(sample)
    
    t_stat = (x_bar - mu_0) / (s / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    print(f"\nSample: n={n}, x̄={x_bar:.2f}, s={s:.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Using scipy
    t_scipy, p_scipy = stats.ttest_1samp(sample, mu_0)
    print(f"\nSciPy verification: t={t_scipy:.4f}, p={p_scipy:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: Reject H₀ at α={alpha}")
        print("  Evidence suggests mean weight ≠ 500g")
    else:
        print(f"\nConclusion: Fail to reject H₀ at α={alpha}")


def example_two_sample_t():
    """Two-sample t-test."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Two-Sample t-Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # A/B test: comparing conversion rates
    control = np.random.normal(100, 15, 50)    # Baseline
    treatment = np.random.normal(108, 15, 50)  # New feature
    
    print("A/B Test: New Feature vs Control")
    print("H₀: μ_treatment = μ_control")
    print("H₁: μ_treatment > μ_control (one-tailed)")
    
    print(f"\nControl: n={len(control)}, mean={control.mean():.2f}, std={control.std():.2f}")
    print(f"Treatment: n={len(treatment)}, mean={treatment.mean():.2f}, std={treatment.std():.2f}")
    
    # Two-sample t-test (assuming equal variance)
    t_stat, p_value_two = stats.ttest_ind(treatment, control)
    p_value_one = p_value_two / 2  # One-tailed
    
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value (two-tailed): {p_value_two:.4f}")
    print(f"p-value (one-tailed): {p_value_one:.4f}")
    
    # Welch's t-test (unequal variance)
    t_welch, p_welch = stats.ttest_ind(treatment, control, equal_var=False)
    print(f"\nWelch's t-test: t={t_welch:.4f}, p={p_welch:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((control.var() + treatment.var()) / 2)
    cohens_d = (treatment.mean() - control.mean()) / pooled_std
    print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
    
    alpha = 0.05
    if p_value_one < alpha:
        print(f"\nConclusion: Reject H₀ at α={alpha}")
        print("  New feature significantly improves performance")
    else:
        print(f"\nConclusion: Fail to reject H₀")


def example_paired_t():
    """Paired t-test."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Paired t-Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Before/after training scores
    n = 20
    before = np.random.normal(70, 10, n)
    improvement = np.random.normal(5, 3, n)  # Average 5 point improvement
    after = before + improvement
    
    print("Training Program Effectiveness")
    print("H₀: μ_diff = 0 (no improvement)")
    print("H₁: μ_diff > 0 (improvement)")
    
    differences = after - before
    
    print(f"\nBefore: mean={before.mean():.2f}, std={before.std():.2f}")
    print(f"After: mean={after.mean():.2f}, std={after.std():.2f}")
    print(f"Difference: mean={differences.mean():.2f}, std={differences.std():.2f}")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(after, before)
    
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value (two-tailed): {p_value:.4f}")
    print(f"p-value (one-tailed): {p_value/2:.4f}")
    
    # Confidence interval for mean difference
    se = differences.std(ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n-1, loc=differences.mean(), scale=se)
    print(f"\n95% CI for improvement: [{ci[0]:.2f}, {ci[1]:.2f}]")


def example_chi_square_independence():
    """Chi-square test for independence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Chi-Square Test for Independence")
    print("=" * 60)
    
    # Contingency table: User type vs Feature usage
    observed = np.array([
        [50, 30],   # Free users: [Use feature, Don't use]
        [80, 40]    # Premium users
    ])
    
    print("Feature Usage by User Type")
    print("H₀: User type and feature usage are independent")
    print("H₁: They are not independent")
    
    print("\nObserved frequencies:")
    print("              | Use Feature | Don't Use |")
    print(f"Free users    |     {observed[0,0]}      |    {observed[0,1]}     |")
    print(f"Premium users |     {observed[1,0]}      |    {observed[1,1]}     |")
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    print(f"\nExpected frequencies (if independent):")
    print(f"              | Use Feature | Don't Use |")
    print(f"Free users    |    {expected[0,0]:.1f}     |   {expected[0,1]:.1f}    |")
    print(f"Premium users |    {expected[1,0]:.1f}     |   {expected[1,1]:.1f}    |")
    
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: Reject H₀ at α={alpha}")
        print("  User type and feature usage are associated")
    else:
        print(f"\nConclusion: Fail to reject H₀")


def example_anova():
    """One-way ANOVA."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: One-Way ANOVA")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Comparing 3 different models
    model_a = np.random.normal(85, 5, 30)
    model_b = np.random.normal(88, 5, 30)
    model_c = np.random.normal(87, 5, 30)
    
    print("Comparing 3 ML Models")
    print("H₀: μ_A = μ_B = μ_C")
    print("H₁: At least one mean is different")
    
    print(f"\nModel A: mean={model_a.mean():.2f}, std={model_a.std():.2f}")
    print(f"Model B: mean={model_b.mean():.2f}, std={model_b.std():.2f}")
    print(f"Model C: mean={model_c.mean():.2f}, std={model_c.std():.2f}")
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(model_a, model_b, model_c)
    
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: Reject H₀ at α={alpha}")
        print("  At least one model performs differently")
        
        # Post-hoc pairwise comparisons
        print("\nPost-hoc pairwise t-tests:")
        pairs = [('A', 'B', model_a, model_b),
                 ('A', 'C', model_a, model_c),
                 ('B', 'C', model_b, model_c)]
        
        for name1, name2, data1, data2 in pairs:
            _, p = stats.ttest_ind(data1, data2)
            print(f"  {name1} vs {name2}: p = {p:.4f}")
    else:
        print(f"\nConclusion: Fail to reject H₀")


def example_power_analysis():
    """Statistical power analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Power Analysis")
    print("=" * 60)
    
    print("How many samples needed to detect an effect?")
    
    # Parameters
    effect_size = 0.5  # Cohen's d (medium)
    alpha = 0.05
    
    print(f"\nEffect size (Cohen's d): {effect_size}")
    print(f"Significance level α: {alpha}")
    
    # Simulate power for different sample sizes
    n_values = [10, 20, 30, 50, 100, 200]
    n_simulations = 5000
    
    print(f"\n{'n':>6} {'Power':>10}")
    print("-" * 20)
    
    for n in n_values:
        rejections = 0
        for _ in range(n_simulations):
            # Generate data under H1 (true effect exists)
            control = np.random.normal(0, 1, n)
            treatment = np.random.normal(effect_size, 1, n)
            
            _, p = stats.ttest_ind(treatment, control)
            if p < alpha:
                rejections += 1
        
        power = rejections / n_simulations
        print(f"{n:>6} {power:>10.3f}")
    
    print("\nRule of thumb: 80% power is standard")
    print("With d=0.5, need n≈64 per group")


def example_multiple_testing():
    """Multiple testing correction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Multiple Testing Correction")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Testing 20 features, only 2 are truly significant
    m = 20
    n_true_effects = 2
    
    # Simulate p-values
    # True nulls: p ~ Uniform(0,1)
    # True effects: p concentrated near 0
    p_values = np.concatenate([
        np.random.uniform(0, 1, m - n_true_effects),
        np.array([0.01, 0.003])  # True effects
    ])
    np.random.shuffle(p_values)
    
    alpha = 0.05
    
    print(f"Testing {m} hypotheses, {n_true_effects} true effects")
    print(f"α = {alpha}")
    
    # No correction
    significant_uncorrected = np.sum(p_values < alpha)
    print(f"\nWithout correction: {significant_uncorrected} significant")
    print(f"  Expected false positives: {(m - n_true_effects) * alpha:.1f}")
    
    # Bonferroni correction
    alpha_bonf = alpha / m
    significant_bonf = np.sum(p_values < alpha_bonf)
    print(f"\nBonferroni (α/m = {alpha_bonf:.4f}): {significant_bonf} significant")
    
    # Benjamini-Hochberg (FDR)
    from scipy.stats import false_discovery_control
    reject_bh = p_values < false_discovery_control(p_values, alpha=alpha)
    significant_bh = np.sum(reject_bh)
    print(f"\nBenjamini-Hochberg (FDR): {significant_bh} significant")
    
    print("\nNote: Bonferroni is conservative, BH is more powerful")


def example_permutation_test():
    """Permutation test."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Permutation Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Non-normal data
    group1 = np.random.exponential(10, 30)
    group2 = np.random.exponential(15, 30)
    
    print("Comparing two groups (non-normal data)")
    print("Permutation test makes no distributional assumptions")
    
    observed_diff = group2.mean() - group1.mean()
    print(f"\nObserved mean difference: {observed_diff:.4f}")
    
    # Permutation test
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_permutations = 10000
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = combined[n1:].mean() - combined[:n1].mean()
        perm_diffs.append(perm_diff)
    
    perm_diffs = np.array(perm_diffs)
    
    # p-value (two-tailed)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    print(f"Permutation p-value: {p_value:.4f}")
    
    # Compare with parametric t-test
    _, p_t = stats.ttest_ind(group2, group1)
    print(f"t-test p-value: {p_t:.4f}")
    
    # Mann-Whitney U test (rank-based)
    _, p_mw = stats.mannwhitneyu(group2, group1, alternative='two-sided')
    print(f"Mann-Whitney p-value: {p_mw:.4f}")


def example_effect_size():
    """Effect size vs statistical significance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Effect Size vs Statistical Significance")
    print("=" * 60)
    
    np.random.seed(42)
    
    print("Large n can make tiny effects 'significant'\n")
    
    effect_sizes = [0.01, 0.1, 0.5]  # Tiny, small, medium
    sample_sizes = [50, 500, 5000]
    
    print(f"{'Effect':>8} {'n':>8} {'Cohen d':>10} {'p-value':>12} {'Significant':>12}")
    print("-" * 55)
    
    for d in effect_sizes:
        for n in sample_sizes:
            control = np.random.normal(0, 1, n)
            treatment = np.random.normal(d, 1, n)
            
            _, p = stats.ttest_ind(treatment, control)
            significant = "Yes" if p < 0.05 else "No"
            
            print(f"{d:>8.2f} {n:>8} {d:>10.2f} {p:>12.4f} {significant:>12}")
    
    print("\nLesson: With n=5000, even d=0.01 is 'significant'")
    print("Always report effect size alongside p-value!")


def example_ml_model_comparison():
    """Statistical comparison of ML models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: ML Model Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Cross-validation scores for two models
    n_folds = 10
    
    # Model A: slightly worse but consistent
    model_a_scores = np.random.normal(0.85, 0.02, n_folds)
    
    # Model B: slightly better on average
    model_b_scores = np.random.normal(0.87, 0.02, n_folds)
    
    print("Comparing two models using 10-fold CV scores")
    print(f"\nModel A: mean={model_a_scores.mean():.4f}, std={model_a_scores.std():.4f}")
    print(f"Model B: mean={model_b_scores.mean():.4f}, std={model_b_scores.std():.4f}")
    
    # Paired t-test (same folds)
    t_stat, p_value = stats.ttest_rel(model_b_scores, model_a_scores)
    
    print(f"\nPaired t-test (same folds):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Effect size
    diff = model_b_scores - model_a_scores
    cohens_d = diff.mean() / diff.std()
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    # 95% CI for improvement
    se = diff.std() / np.sqrt(n_folds)
    ci = stats.t.interval(0.95, df=n_folds-1, loc=diff.mean(), scale=se)
    print(f"\n95% CI for improvement: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    if ci[0] > 0:
        print("Entire CI > 0: Model B is significantly better")
    elif ci[1] < 0:
        print("Entire CI < 0: Model A is significantly better")
    else:
        print("CI contains 0: No significant difference")


def example_confidence_intervals():
    """Confidence intervals."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Confidence Intervals")
    print("=" * 60)
    
    np.random.seed(42)
    
    true_mu = 100
    true_sigma = 15
    n = 50
    
    sample = np.random.normal(true_mu, true_sigma, n)
    
    x_bar = sample.mean()
    s = sample.std(ddof=1)
    se = s / np.sqrt(n)
    
    print(f"True μ = {true_mu}")
    print(f"Sample: n={n}, x̄={x_bar:.2f}, s={s:.2f}")
    
    # Different confidence levels
    conf_levels = [0.90, 0.95, 0.99]
    
    print("\nConfidence Intervals:")
    for level in conf_levels:
        ci = stats.t.interval(level, df=n-1, loc=x_bar, scale=se)
        width = ci[1] - ci[0]
        contains = ci[0] <= true_mu <= ci[1]
        print(f"  {level*100:.0f}% CI: [{ci[0]:.2f}, {ci[1]:.2f}] "
              f"(width={width:.2f}, contains μ: {contains})")
    
    # Simulation: coverage probability
    n_simulations = 10000
    covers = 0
    
    for _ in range(n_simulations):
        sample = np.random.normal(true_mu, true_sigma, n)
        x_bar = sample.mean()
        se = sample.std(ddof=1) / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n-1, loc=x_bar, scale=se)
        if ci[0] <= true_mu <= ci[1]:
            covers += 1
    
    coverage = covers / n_simulations
    print(f"\nSimulated 95% CI coverage: {coverage*100:.1f}%")


def example_ab_testing():
    """A/B testing for conversion rates."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: A/B Testing (Conversion Rates)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Website A/B test
    n_a, conversions_a = 1000, 50   # 5% conversion
    n_b, conversions_b = 1000, 65   # 6.5% conversion
    
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    
    print("A/B Test: Website Conversion Rate")
    print(f"Control (A): {conversions_a}/{n_a} = {p_a:.2%}")
    print(f"Treatment (B): {conversions_b}/{n_b} = {p_b:.2%}")
    
    # Two-proportion z-test
    p_pooled = (conversions_a + conversions_b) / (n_a + n_b)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\nPooled proportion: {p_pooled:.4f}")
    print(f"z-statistic: {z:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    # Relative lift
    lift = (p_b - p_a) / p_a
    print(f"\nRelative lift: {lift:.1%}")
    
    # Chi-square test (equivalent)
    contingency = [[conversions_a, n_a - conversions_a],
                   [conversions_b, n_b - conversions_b]]
    chi2, p_chi = stats.chi2_contingency(contingency)[:2]
    print(f"\nChi-square test: χ²={chi2:.4f}, p={p_chi:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: Treatment B is significantly better (p < {alpha})")
    else:
        print(f"\nConclusion: No significant difference")


if __name__ == "__main__":
    example_one_sample_t()
    example_two_sample_t()
    example_paired_t()
    example_chi_square_independence()
    example_anova()
    example_power_analysis()
    example_multiple_testing()
    example_permutation_test()
    example_effect_size()
    example_ml_model_comparison()
    example_confidence_intervals()
    example_ab_testing()
