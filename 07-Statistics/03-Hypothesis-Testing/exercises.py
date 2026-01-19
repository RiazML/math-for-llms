"""
Hypothesis Testing - Exercises
==============================
Practice problems for hypothesis testing.
"""

import numpy as np
from scipy import stats


class HypothesisTestingExercises:
    """Exercises for hypothesis testing."""
    
    def exercise_1_one_sample_t(self):
        """
        Exercise 1: One-Sample t-Test
        
        A factory claims their batteries last 500 hours on average.
        Test this claim with the given sample.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: One-Sample t-Test")
        
        np.random.seed(42)
        
        # Sample of battery lifetimes
        sample = np.array([485, 510, 495, 502, 490, 505, 498, 492, 508, 
                          487, 515, 480, 510, 495, 490, 505, 488, 512, 
                          493, 507])
        mu_0 = 500
        
        print(f"\nClaim: μ = {mu_0} hours")
        print(f"Sample: n = {len(sample)}")
        print(f"Sample mean: {sample.mean():.2f}")
        print(f"Sample std: {sample.std(ddof=1):.2f}")
        
        print("\nStep 1: State hypotheses")
        print(f"  H₀: μ = {mu_0}")
        print(f"  H₁: μ ≠ {mu_0} (two-tailed)")
        
        print("\nStep 2: Calculate test statistic")
        x_bar = sample.mean()
        s = sample.std(ddof=1)
        n = len(sample)
        t_stat = (x_bar - mu_0) / (s / np.sqrt(n))
        print(f"  t = (x̄ - μ₀)/(s/√n) = ({x_bar:.2f} - {mu_0})/({s:.2f}/√{n})")
        print(f"  t = {t_stat:.4f}")
        
        print("\nStep 3: Calculate p-value")
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        print(f"  df = n - 1 = {n-1}")
        print(f"  p-value = {p_value:.4f}")
        
        print("\nStep 4: Make decision")
        alpha = 0.05
        if p_value < alpha:
            print(f"  p = {p_value:.4f} < α = {alpha}")
            print("  Reject H₀: Battery life differs from 500 hours")
        else:
            print(f"  p = {p_value:.4f} ≥ α = {alpha}")
            print("  Fail to reject H₀: Cannot conclude life differs from 500")
        
        # Confidence interval
        se = s / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n-1, loc=x_bar, scale=se)
        print(f"\n95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    def exercise_2_two_sample_t(self):
        """
        Exercise 2: Two-Sample t-Test
        
        Compare accuracy of two ML models.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Two-Sample t-Test")
        
        np.random.seed(42)
        
        # Model accuracies (percentage)
        model_a = np.array([85.2, 84.8, 86.1, 83.9, 85.5, 84.7, 85.8, 84.2, 86.0, 85.3])
        model_b = np.array([87.1, 86.5, 88.2, 86.8, 87.5, 86.3, 87.9, 86.7, 88.0, 87.2])
        
        print("Model comparison test")
        print(f"Model A: mean = {model_a.mean():.2f}%, std = {model_a.std(ddof=1):.2f}")
        print(f"Model B: mean = {model_b.mean():.2f}%, std = {model_b.std(ddof=1):.2f}")
        
        print("\nHypotheses:")
        print("  H₀: μ_A = μ_B")
        print("  H₁: μ_A ≠ μ_B")
        
        # Independent two-sample t-test
        t_stat, p_value = stats.ttest_ind(model_a, model_b)
        
        print(f"\nTest statistic: t = {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        # Effect size
        pooled_std = np.sqrt((model_a.var(ddof=1) + model_b.var(ddof=1)) / 2)
        cohens_d = (model_b.mean() - model_a.mean()) / pooled_std
        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        
        alpha = 0.05
        print(f"\nDecision at α = {alpha}:")
        if p_value < alpha:
            print(f"  p = {p_value:.4f} < {alpha}, reject H₀")
            print("  Models have significantly different accuracy")
        else:
            print(f"  p = {p_value:.4f} ≥ {alpha}, fail to reject H₀")
    
    def exercise_3_paired_t(self):
        """
        Exercise 3: Paired t-Test
        
        Test if a preprocessing step improves model performance.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Paired t-Test")
        
        np.random.seed(42)
        
        # Same datasets, with and without preprocessing
        without_preprocess = np.array([78, 82, 75, 80, 77, 81, 79, 76, 83, 80])
        with_preprocess = np.array([82, 85, 79, 84, 81, 85, 83, 80, 87, 84])
        
        print("Testing preprocessing effect on accuracy")
        print(f"Without: mean = {without_preprocess.mean():.2f}")
        print(f"With: mean = {with_preprocess.mean():.2f}")
        
        differences = with_preprocess - without_preprocess
        print(f"\nDifferences: {differences}")
        print(f"Mean difference: {differences.mean():.2f}")
        
        print("\nHypotheses:")
        print("  H₀: μ_diff = 0 (no improvement)")
        print("  H₁: μ_diff > 0 (improvement)")
        
        # Paired t-test
        t_stat, p_value_two = stats.ttest_rel(with_preprocess, without_preprocess)
        p_value_one = p_value_two / 2  # One-tailed
        
        print(f"\nTest statistic: t = {t_stat:.4f}")
        print(f"p-value (one-tailed): {p_value_one:.6f}")
        
        alpha = 0.05
        if p_value_one < alpha:
            print(f"\nReject H₀: Preprocessing significantly improves accuracy")
        else:
            print(f"\nFail to reject H₀")
        
        # CI for improvement
        n = len(differences)
        se = differences.std(ddof=1) / np.sqrt(n)
        ci = stats.t.interval(0.95, df=n-1, loc=differences.mean(), scale=se)
        print(f"\n95% CI for improvement: [{ci[0]:.2f}, {ci[1]:.2f}] percentage points")
    
    def exercise_4_chi_square(self):
        """
        Exercise 4: Chi-Square Test
        
        Test if user satisfaction is independent of subscription type.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Chi-Square Test for Independence")
        
        # Observed frequencies
        # Rows: Basic, Premium
        # Cols: Unsatisfied, Neutral, Satisfied
        observed = np.array([
            [30, 50, 20],   # Basic
            [15, 35, 50]    # Premium
        ])
        
        print("Contingency Table:")
        print("           | Unsatisfied | Neutral | Satisfied |")
        print(f"Basic      |     {observed[0,0]:3d}     |   {observed[0,1]:3d}   |    {observed[0,2]:3d}    |")
        print(f"Premium    |     {observed[1,0]:3d}     |   {observed[1,1]:3d}   |    {observed[1,2]:3d}    |")
        
        print("\nHypotheses:")
        print("  H₀: Satisfaction is independent of subscription type")
        print("  H₁: They are not independent")
        
        # Chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        
        print(f"\nExpected frequencies (under H₀):")
        print("           | Unsatisfied | Neutral | Satisfied |")
        print(f"Basic      |    {expected[0,0]:5.1f}   |  {expected[0,1]:5.1f}  |   {expected[0,2]:5.1f}   |")
        print(f"Premium    |    {expected[1,0]:5.1f}   |  {expected[1,1]:5.1f}  |   {expected[1,2]:5.1f}   |")
        
        print(f"\nχ² statistic: {chi2:.4f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_value:.4f}")
        
        alpha = 0.05
        print(f"\nDecision at α = {alpha}:")
        if p_value < alpha:
            print("  Reject H₀: Satisfaction depends on subscription type")
        else:
            print("  Fail to reject H₀: No evidence of association")
    
    def exercise_5_type_errors(self):
        """
        Exercise 5: Understanding Type I and Type II Errors
        
        Simulate and count error rates.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Type I and Type II Errors")
        
        np.random.seed(42)
        
        n = 30
        alpha = 0.05
        n_simulations = 10000
        
        print(f"Sample size: n = {n}")
        print(f"Significance level: α = {alpha}")
        print(f"Simulations: {n_simulations}")
        
        # Scenario 1: H₀ is true (μ = 0)
        print("\n--- Scenario 1: H₀ TRUE (μ = 0) ---")
        type_1_errors = 0
        
        for _ in range(n_simulations):
            sample = np.random.normal(0, 1, n)
            _, p = stats.ttest_1samp(sample, 0)
            if p < alpha:
                type_1_errors += 1
        
        type_1_rate = type_1_errors / n_simulations
        print(f"Type I errors: {type_1_errors}/{n_simulations}")
        print(f"Type I error rate: {type_1_rate:.4f}")
        print(f"Expected (α): {alpha:.4f}")
        
        # Scenario 2: H₀ is false (μ = 0.5)
        print("\n--- Scenario 2: H₀ FALSE (μ = 0.5, effect size d=0.5) ---")
        type_2_errors = 0
        
        for _ in range(n_simulations):
            sample = np.random.normal(0.5, 1, n)
            _, p = stats.ttest_1samp(sample, 0)
            if p >= alpha:
                type_2_errors += 1
        
        type_2_rate = type_2_errors / n_simulations
        power = 1 - type_2_rate
        
        print(f"Type II errors: {type_2_errors}/{n_simulations}")
        print(f"Type II error rate (β): {type_2_rate:.4f}")
        print(f"Power (1 - β): {power:.4f}")
        
        print("\nSummary:")
        print("  Type I (False Positive): Reject H₀ when it's true")
        print("  Type II (False Negative): Fail to reject H₀ when it's false")
    
    def exercise_6_power_calculation(self):
        """
        Exercise 6: Power Calculation
        
        Determine sample size needed for 80% power.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Power Calculation")
        
        np.random.seed(42)
        
        effect_size = 0.3  # Small effect
        alpha = 0.05
        target_power = 0.80
        
        print(f"Goal: Find n for {target_power*100:.0f}% power")
        print(f"Effect size (Cohen's d): {effect_size}")
        print(f"Significance level: α = {alpha}")
        
        # Simulate power for different n
        n_values = [50, 100, 150, 200, 250, 300, 350, 400]
        n_sims = 3000
        
        print(f"\n{'n':>6} {'Power':>10}")
        print("-" * 18)
        
        sufficient_n = None
        for n in n_values:
            rejections = 0
            for _ in range(n_sims):
                sample = np.random.normal(effect_size, 1, n)
                _, p = stats.ttest_1samp(sample, 0)
                if p < alpha:
                    rejections += 1
            
            power = rejections / n_sims
            print(f"{n:>6} {power:>10.3f}")
            
            if power >= target_power and sufficient_n is None:
                sufficient_n = n
        
        print(f"\nNeed n ≥ {sufficient_n} for {target_power*100:.0f}% power")
        
        # Analytical formula (approximate)
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(target_power)
        n_formula = ((z_alpha + z_beta) / effect_size) ** 2
        print(f"\nAnalytical approximation: n ≈ {n_formula:.0f}")
    
    def exercise_7_bonferroni(self):
        """
        Exercise 7: Multiple Testing Correction
        
        Apply Bonferroni correction to feature selection p-values.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Bonferroni Correction")
        
        # P-values from testing 10 features
        p_values = np.array([0.001, 0.012, 0.023, 0.045, 0.067,
                            0.089, 0.102, 0.234, 0.456, 0.789])
        feature_names = [f"Feature {i+1}" for i in range(len(p_values))]
        
        m = len(p_values)
        alpha = 0.05
        alpha_bonf = alpha / m
        
        print(f"Testing {m} features")
        print(f"α = {alpha}")
        print(f"Bonferroni adjusted α = {alpha_bonf:.4f}")
        
        print(f"\n{'Feature':>12} {'p-value':>10} {'Sig@0.05':>10} {'Bonferroni':>12}")
        print("-" * 48)
        
        sig_original = 0
        sig_bonferroni = 0
        
        for name, p in zip(feature_names, p_values):
            is_sig = "Yes" if p < alpha else "No"
            is_sig_bonf = "Yes" if p < alpha_bonf else "No"
            
            if p < alpha:
                sig_original += 1
            if p < alpha_bonf:
                sig_bonferroni += 1
            
            print(f"{name:>12} {p:>10.4f} {is_sig:>10} {is_sig_bonf:>12}")
        
        print(f"\nSignificant without correction: {sig_original}")
        print(f"Significant with Bonferroni: {sig_bonferroni}")
        print("\nBonferroni controls family-wise error rate (FWER)")
    
    def exercise_8_permutation(self):
        """
        Exercise 8: Permutation Test
        
        Non-parametric test for non-normal data.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Permutation Test")
        
        np.random.seed(42)
        
        # Skewed data (response times in ms)
        group1 = np.array([120, 135, 150, 145, 180, 160, 140, 155, 190, 170])
        group2 = np.array([200, 210, 195, 220, 240, 230, 215, 205, 250, 235])
        
        print("Testing difference in response times (skewed data)")
        print(f"Group 1: mean = {group1.mean():.1f}, median = {np.median(group1):.1f}")
        print(f"Group 2: mean = {group2.mean():.1f}, median = {np.median(group2):.1f}")
        
        observed_diff = group2.mean() - group1.mean()
        print(f"\nObserved mean difference: {observed_diff:.2f}")
        
        # Permutation test
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        n_permutations = 10000
        
        perm_diffs = np.zeros(n_permutations)
        for i in range(n_permutations):
            np.random.shuffle(combined)
            perm_diffs[i] = combined[n1:].mean() - combined[:n1].mean()
        
        # Two-tailed p-value
        p_perm = np.mean(np.abs(perm_diffs) >= abs(observed_diff))
        
        print(f"\nPermutation test (10000 permutations):")
        print(f"  p-value: {p_perm:.4f}")
        
        # Compare with t-test
        _, p_t = stats.ttest_ind(group2, group1)
        print(f"\nParametric t-test p-value: {p_t:.4f}")
        
        # Mann-Whitney U
        _, p_mw = stats.mannwhitneyu(group2, group1, alternative='two-sided')
        print(f"Mann-Whitney U p-value: {p_mw:.4f}")
        
        print("\nPermutation test makes no distributional assumptions!")
    
    def exercise_9_ab_test(self):
        """
        Exercise 9: A/B Test
        
        Test if new website design improves conversion rate.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: A/B Test")
        
        # Website conversion data
        n_control, conversions_control = 5000, 200   # 4%
        n_treatment, conversions_treatment = 5000, 250  # 5%
        
        p_control = conversions_control / n_control
        p_treatment = conversions_treatment / n_treatment
        
        print("A/B Test: New Website Design")
        print(f"Control: {conversions_control}/{n_control} = {p_control:.2%}")
        print(f"Treatment: {conversions_treatment}/{n_treatment} = {p_treatment:.2%}")
        
        print("\nHypotheses:")
        print("  H₀: p_treatment = p_control")
        print("  H₁: p_treatment > p_control (one-tailed)")
        
        # Two-proportion z-test
        p_pooled = (conversions_control + conversions_treatment) / (n_control + n_treatment)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        z = (p_treatment - p_control) / se
        p_value = 1 - stats.norm.cdf(z)  # One-tailed
        
        print(f"\nPooled proportion: {p_pooled:.4f}")
        print(f"Standard error: {se:.4f}")
        print(f"z-statistic: {z:.4f}")
        print(f"p-value (one-tailed): {p_value:.4f}")
        
        # Relative lift and CI
        lift = (p_treatment - p_control) / p_control
        
        # CI for difference in proportions
        se_diff = np.sqrt(p_control*(1-p_control)/n_control + 
                         p_treatment*(1-p_treatment)/n_treatment)
        diff_ci = (p_treatment - p_control - 1.96*se_diff,
                   p_treatment - p_control + 1.96*se_diff)
        
        print(f"\nRelative lift: {lift:.1%}")
        print(f"95% CI for difference: [{diff_ci[0]:.3%}, {diff_ci[1]:.3%}]")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"\nConclusion: New design significantly improves conversion")
        else:
            print(f"\nConclusion: No significant improvement")
    
    def exercise_10_effect_size(self):
        """
        Exercise 10: Effect Size Interpretation
        
        Calculate and interpret effect sizes.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Effect Size Interpretation")
        
        np.random.seed(42)
        
        # Three scenarios with same effect size but different n
        effect_size = 0.3  # Small effect
        
        print(f"True effect size (Cohen's d): {effect_size}")
        print("\nDemonstrating how p-value depends on sample size:\n")
        
        n_values = [30, 100, 500, 2000]
        
        print(f"{'n':>6} {'Mean diff':>12} {'Cohen d':>10} {'p-value':>12} {'Significant':>12}")
        print("-" * 56)
        
        for n in n_values:
            control = np.random.normal(0, 1, n)
            treatment = np.random.normal(effect_size, 1, n)
            
            mean_diff = treatment.mean() - control.mean()
            pooled_std = np.sqrt((control.var() + treatment.var()) / 2)
            d = mean_diff / pooled_std
            
            _, p = stats.ttest_ind(treatment, control)
            sig = "Yes" if p < 0.05 else "No"
            
            print(f"{n:>6} {mean_diff:>12.4f} {d:>10.4f} {p:>12.4f} {sig:>12}")
        
        print("\nInterpretation:")
        print("  • Same underlying effect produces different p-values")
        print("  • Large n makes small effects 'significant'")
        print("  • Always report effect size alongside p-value!")
        
        print("\nCohen's d interpretation:")
        print("  d = 0.2: Small effect")
        print("  d = 0.5: Medium effect")
        print("  d = 0.8: Large effect")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = HypothesisTestingExercises()
    
    print("HYPOTHESIS TESTING EXERCISES")
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
