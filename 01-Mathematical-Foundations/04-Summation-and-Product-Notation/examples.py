"""
Summation and Product Notation - Examples
=========================================
Practical demonstrations of sigma and pi notation.
"""

import numpy as np
from math import factorial, log, exp


def example_basic_summation():
    """Demonstrate basic summation notation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Summation")
    print("=" * 60)
    
    # Sum of first n integers
    n = 10
    manual_sum = sum(range(1, n + 1))
    formula_sum = n * (n + 1) // 2
    
    print(f"Σᵢ₌₁ⁿ i for n = {n}")
    print(f"Expansion: 1 + 2 + 3 + ... + {n}")
    print(f"Manual sum: {manual_sum}")
    print(f"Formula n(n+1)/2: {formula_sum}")
    
    # Sum of squares
    print(f"\nΣᵢ₌₁ⁿ i² for n = {n}")
    manual_sum_sq = sum(i**2 for i in range(1, n + 1))
    formula_sum_sq = n * (n + 1) * (2*n + 1) // 6
    print(f"Expansion: 1² + 2² + 3² + ... + {n}² = 1 + 4 + 9 + ... + {n**2}")
    print(f"Manual sum: {manual_sum_sq}")
    print(f"Formula n(n+1)(2n+1)/6: {formula_sum_sq}")
    
    # Custom sum
    print(f"\nΣᵢ₌₁⁵ (2i - 1)")
    terms = [2*i - 1 for i in range(1, 6)]
    print(f"Terms: {terms}")
    print(f"Sum: {sum(terms)}")
    print(f"Note: This is the sum of first 5 odd numbers = 5² = 25")


def example_summation_properties():
    """Demonstrate summation properties."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Summation Properties")
    print("=" * 60)
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    c = 3
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"c = {c}")
    
    # Linearity: sum of sums
    print("\n1. Linearity: Σ(xᵢ + yᵢ) = Σxᵢ + Σyᵢ")
    left = np.sum(x + y)
    right = np.sum(x) + np.sum(y)
    print(f"   Σ(xᵢ + yᵢ) = {left}")
    print(f"   Σxᵢ + Σyᵢ = {np.sum(x)} + {np.sum(y)} = {right}")
    
    # Constant factor
    print(f"\n2. Constant factor: Σ(c·xᵢ) = c·Σxᵢ")
    left = np.sum(c * x)
    right = c * np.sum(x)
    print(f"   Σ({c}·xᵢ) = {left}")
    print(f"   {c}·Σxᵢ = {c}·{np.sum(x)} = {right}")
    
    # Constant sum
    n = len(x)
    print(f"\n3. Constant sum: Σᵢ₌₁ⁿ c = n·c")
    left = sum(c for _ in range(n))
    right = n * c
    print(f"   Σᵢ₌₁⁵ {c} = {left}")
    print(f"   5·{c} = {right}")


def example_common_formulas():
    """Demonstrate common summation formulas."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Common Summation Formulas")
    print("=" * 60)
    
    n = 10
    
    # Arithmetic sum
    print(f"1. Arithmetic Sum: Σᵢ₌₁ⁿ i = n(n+1)/2")
    computed = sum(range(1, n + 1))
    formula = n * (n + 1) // 2
    print(f"   For n = {n}: computed = {computed}, formula = {formula}")
    
    # Sum of squares
    print(f"\n2. Sum of Squares: Σᵢ₌₁ⁿ i² = n(n+1)(2n+1)/6")
    computed = sum(i**2 for i in range(1, n + 1))
    formula = n * (n + 1) * (2*n + 1) // 6
    print(f"   For n = {n}: computed = {computed}, formula = {formula}")
    
    # Sum of cubes
    print(f"\n3. Sum of Cubes: Σᵢ₌₁ⁿ i³ = [n(n+1)/2]²")
    computed = sum(i**3 for i in range(1, n + 1))
    formula = (n * (n + 1) // 2) ** 2
    print(f"   For n = {n}: computed = {computed}, formula = {formula}")
    
    # Geometric sum
    r = 2
    print(f"\n4. Geometric Sum: Σᵢ₌₀ⁿ rⁱ = (1 - r^(n+1))/(1 - r)")
    computed = sum(r**i for i in range(n + 1))
    formula = (1 - r**(n + 1)) // (1 - r)
    print(f"   For r = {r}, n = {n}: computed = {computed}, formula = {formula}")
    
    # Infinite geometric (convergent)
    r = 0.5
    print(f"\n5. Infinite Geometric: Σᵢ₌₀^∞ rⁱ = 1/(1-r) for |r| < 1")
    partial_sum = sum(r**i for i in range(100))
    formula = 1 / (1 - r)
    print(f"   For r = {r}: partial sum (100 terms) = {partial_sum:.6f}")
    print(f"   Formula: {formula}")


def example_basic_product():
    """Demonstrate basic product notation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Basic Product Notation")
    print("=" * 60)
    
    # Factorial
    n = 5
    print(f"Πᵢ₌₁ⁿ i = n! (factorial)")
    computed = 1
    for i in range(1, n + 1):
        computed *= i
    print(f"Πᵢ₌₁⁵ i = 1 × 2 × 3 × 4 × 5 = {computed}")
    print(f"5! = {factorial(5)}")
    
    # Powers of 2
    print(f"\nΠᵢ₌₁⁴ 2ⁱ")
    computed = 1
    for i in range(1, 5):
        computed *= 2**i
    print(f"= 2¹ × 2² × 2³ × 2⁴ = 2 × 4 × 8 × 16 = {computed}")
    print(f"= 2^(1+2+3+4) = 2^10 = {2**10}")
    
    # Repeated multiplication
    x = 3
    n = 4
    print(f"\nΠᵢ₌₁ⁿ x = xⁿ")
    computed = 1
    for _ in range(n):
        computed *= x
    print(f"Πᵢ₌₁⁴ {x} = {x}⁴ = {computed}")


def example_product_log_conversion():
    """Demonstrate the crucial product-to-sum conversion via logarithm."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Product to Sum via Logarithm")
    print("=" * 60)
    
    values = np.array([2.0, 3.0, 4.0, 5.0])
    
    print("This is CRITICAL for numerical stability in ML!")
    print(f"\nValues: {values}")
    
    # Direct product
    product = np.prod(values)
    print(f"\nDirect product: Π values = {product}")
    
    # Log-sum method
    log_sum = np.sum(np.log(values))
    product_via_log = np.exp(log_sum)
    print(f"Log-sum method: Σ log(values) = {log_sum:.4f}")
    print(f"exp(Σ log(values)) = {product_via_log:.4f}")
    
    # Why this matters: very small probabilities
    print("\nWhy this matters (numerical stability):")
    small_probs = np.array([0.001, 0.002, 0.003, 0.001, 0.002])
    print(f"Small probabilities: {small_probs}")
    
    # Direct product might underflow
    direct_prod = np.prod(small_probs)
    print(f"Direct product: {direct_prod:.2e} (risk of underflow)")
    
    # Log-sum is stable
    log_likelihood = np.sum(np.log(small_probs))
    print(f"Log-likelihood: {log_likelihood:.4f} (numerically stable)")


def example_double_summation():
    """Demonstrate double summation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Double Summation")
    print("=" * 60)
    
    # Simple double sum
    print("Σᵢ₌₁² Σⱼ₌₁³ ij")
    print("\nExpansion:")
    total = 0
    for i in range(1, 3):
        row_sum = 0
        terms = []
        for j in range(1, 4):
            terms.append(f"{i}×{j}")
            row_sum += i * j
        print(f"  i={i}: {' + '.join(terms)} = {row_sum}")
        total += row_sum
    print(f"Total: {total}")
    
    # Matrix interpretation
    print("\n\nMatrix interpretation:")
    A = np.array([[1, 2, 3],
                  [2, 4, 6]])
    print(f"A = \n{A}")
    print(f"Sum of all elements: {A.sum()}")
    
    # Order doesn't matter for independent limits
    print("\nOrder independence (independent limits):")
    sum1 = sum(sum(i*j for j in range(1, 4)) for i in range(1, 3))
    sum2 = sum(sum(i*j for i in range(1, 3)) for j in range(1, 4))
    print(f"Σᵢ₌₁² Σⱼ₌₁³ ij = {sum1}")
    print(f"Σⱼ₌₁³ Σᵢ₌₁² ij = {sum2}")


def example_triangular_sum():
    """Demonstrate triangular double summation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Triangular Summation")
    print("=" * 60)
    
    n = 4
    print(f"Σᵢ₌₁ⁿ Σⱼ₌ᵢⁿ aᵢⱼ for n = {n}")
    print("\nThis sums over upper triangular indices:")
    
    print("\nIndices (i, j) included:")
    for i in range(1, n + 1):
        indices = [(i, j) for j in range(i, n + 1)]
        print(f"  i={i}: {indices}")
    
    # Example with A[i][j] = i + j
    print("\nWith aᵢⱼ = i + j:")
    total = 0
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            total += i + j
            print(f"  a[{i},{j}] = {i + j}")
    print(f"Total: {total}")
    
    # Count of terms
    num_terms = n * (n + 1) // 2
    print(f"\nNumber of terms: n(n+1)/2 = {num_terms}")


def example_ml_mean_variance():
    """Demonstrate mean and variance using summation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: ML Statistics - Mean and Variance")
    print("=" * 60)
    
    x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    n = len(x)
    
    print(f"Data: x = {x}")
    print(f"n = {n}")
    
    # Mean
    mean = sum(x) / n
    print(f"\nMean: x̄ = (1/n) Σxᵢ")
    print(f"     = (1/{n}) × {sum(x)}")
    print(f"     = {mean}")
    print(f"NumPy: {np.mean(x)}")
    
    # Variance
    variance = sum((xi - mean)**2 for xi in x) / n
    print(f"\nVariance: σ² = (1/n) Σ(xᵢ - x̄)²")
    print(f"Deviations from mean: {[xi - mean for xi in x]}")
    print(f"Squared deviations: {[(xi - mean)**2 for xi in x]}")
    print(f"σ² = {variance}")
    print(f"NumPy: {np.var(x)}")
    
    # Standard deviation
    std = variance ** 0.5
    print(f"\nStandard deviation: σ = {std:.4f}")
    print(f"NumPy: {np.std(x):.4f}")


def example_ml_mse():
    """Demonstrate MSE loss function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: ML Loss - Mean Squared Error")
    print("=" * 60)
    
    y_true = np.array([3, 5, 2, 7, 4])
    y_pred = np.array([2.5, 4.5, 3, 6, 4.5])
    n = len(y_true)
    
    print(f"True values:      y = {y_true}")
    print(f"Predictions:      ŷ = {y_pred}")
    
    # MSE calculation
    errors = y_true - y_pred
    squared_errors = errors ** 2
    mse = sum(squared_errors) / n
    
    print(f"\nErrors (y - ŷ):   {errors}")
    print(f"Squared errors:   {squared_errors}")
    print(f"\nMSE = (1/n) Σ(yᵢ - ŷᵢ)²")
    print(f"    = (1/{n}) × {sum(squared_errors)}")
    print(f"    = {mse}")
    
    # Using NumPy
    print(f"\nNumPy: {np.mean((y_true - y_pred)**2)}")


def example_ml_cross_entropy():
    """Demonstrate cross-entropy loss."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: ML Loss - Cross-Entropy")
    print("=" * 60)
    
    # True labels (one-hot encoded) and predictions
    y_true = np.array([1, 0, 0])  # Class 0
    y_pred = np.array([0.7, 0.2, 0.1])  # Predicted probabilities
    
    print("Classification with 3 classes")
    print(f"True label (one-hot):  {y_true}")
    print(f"Predicted probs:       {y_pred}")
    
    # Cross-entropy
    print("\nCross-Entropy: L = -Σ yᵢ log(ŷᵢ)")
    terms = []
    for i, (y, yhat) in enumerate(zip(y_true, y_pred)):
        term = -y * np.log(yhat) if y > 0 else 0
        terms.append(term)
        if y > 0:
            print(f"  i={i}: -{y} × log({yhat}) = {term:.4f}")
        else:
            print(f"  i={i}: 0 (y=0, term vanishes)")
    
    ce_loss = sum(terms)
    print(f"\nCross-Entropy Loss = {ce_loss:.4f}")
    
    # Perfect prediction comparison
    print("\nComparison with perfect prediction:")
    y_perfect = np.array([1.0, 0.0, 0.0])
    ce_perfect = -np.sum(y_true * np.log(np.clip(y_perfect, 1e-15, 1)))
    print(f"Perfect: y_pred = {y_perfect}, CE = {ce_perfect:.4f}")


def example_ml_softmax():
    """Demonstrate softmax using summation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Softmax Function")
    print("=" * 60)
    
    z = np.array([2.0, 1.0, 0.1])
    
    print(f"Logits: z = {z}")
    print("\nSoftmax: σ(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)")
    
    # Compute exponentials
    exp_z = np.exp(z)
    print(f"\nexp(z) = {exp_z}")
    
    # Sum of exponentials
    sum_exp = np.sum(exp_z)
    print(f"Σⱼ exp(zⱼ) = {sum_exp:.4f}")
    
    # Softmax
    softmax = exp_z / sum_exp
    print(f"\nSoftmax:")
    for i, (zi, si) in enumerate(zip(z, softmax)):
        print(f"  σ(z_{i}) = exp({zi}) / {sum_exp:.4f} = {si:.4f}")
    
    print(f"\nσ(z) = {np.round(softmax, 4)}")
    print(f"Sum of softmax: {softmax.sum():.4f} (should be 1.0)")
    
    # Numerical stability
    print("\nNumerical stability trick:")
    z_large = np.array([1000, 1001, 1002])
    print(f"Large logits: z = {z_large}")
    print("Direct exp would overflow!")
    
    z_stable = z_large - np.max(z_large)
    print(f"Subtract max: z - max(z) = {z_stable}")
    softmax_stable = np.exp(z_stable) / np.sum(np.exp(z_stable))
    print(f"Stable softmax: {np.round(softmax_stable, 4)}")


def example_ml_likelihood():
    """Demonstrate likelihood and log-likelihood."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Likelihood and Log-Likelihood")
    print("=" * 60)
    
    # Bernoulli example: coin flips
    data = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]  # 1=heads, 0=tails
    p = 0.7  # Probability of heads
    
    print(f"Coin flip data: {data}")
    print(f"Assumed P(heads) = p = {p}")
    
    # Likelihood
    print("\nLikelihood: L(p) = Πᵢ P(xᵢ|p)")
    print("            = Πᵢ p^xᵢ (1-p)^(1-xᵢ)")
    
    likelihood = 1
    for x in data:
        prob = p if x == 1 else (1 - p)
        likelihood *= prob
        print(f"  x={x}: P = {prob}")
    
    print(f"\nL({p}) = {likelihood:.6e}")
    
    # Log-likelihood
    print("\nLog-Likelihood: ℓ(p) = Σᵢ log P(xᵢ|p)")
    print("               = Σᵢ [xᵢ log(p) + (1-xᵢ) log(1-p)]")
    
    log_likelihood = 0
    for x in data:
        log_prob = np.log(p) if x == 1 else np.log(1 - p)
        log_likelihood += log_prob
    
    print(f"ℓ({p}) = {log_likelihood:.4f}")
    print(f"Verify: exp(ℓ) = {np.exp(log_likelihood):.6e}")
    
    # MLE
    print("\nMaximum Likelihood Estimate:")
    n_heads = sum(data)
    n_total = len(data)
    mle = n_heads / n_total
    print(f"p_MLE = #heads / #total = {n_heads}/{n_total} = {mle}")


if __name__ == "__main__":
    example_basic_summation()
    example_summation_properties()
    example_common_formulas()
    example_basic_product()
    example_product_log_conversion()
    example_double_summation()
    example_triangular_sum()
    example_ml_mean_variance()
    example_ml_mse()
    example_ml_cross_entropy()
    example_ml_softmax()
    example_ml_likelihood()
