"""
Floating-Point Arithmetic - Examples
====================================
Demonstrating floating-point behavior and numerical stability.
"""

import numpy as np
import struct


def example_ieee754_representation():
    """Examine IEEE 754 floating-point representation."""
    print("=" * 60)
    print("EXAMPLE 1: IEEE 754 Representation")
    print("=" * 60)
    
    def float_to_binary(f):
        """Convert float to binary representation."""
        # Pack as float, unpack as int
        packed = struct.pack('>f', f)
        integer = struct.unpack('>I', packed)[0]
        return format(integer, '032b')
    
    def double_to_binary(f):
        """Convert double to binary representation."""
        packed = struct.pack('>d', f)
        integer = struct.unpack('>Q', packed)[0]
        return format(integer, '064b')
    
    def parse_float32(binary):
        """Parse float32 binary string."""
        sign = int(binary[0])
        exponent = int(binary[1:9], 2)
        mantissa = binary[9:]
        return sign, exponent, mantissa
    
    numbers = [1.0, -1.0, 0.5, 0.1, 3.14159, 0.0, float('inf')]
    
    print("Float32 (Single Precision) Representation:")
    print(f"{'Value':>12} {'Sign':>6} {'Exp':>10} {'Mantissa':>25}")
    print("-" * 60)
    
    for num in numbers:
        if not np.isinf(num):
            binary = float_to_binary(num)
            sign, exp, mantissa = parse_float32(binary)
            print(f"{num:>12.6f} {sign:>6} {exp:>10} {mantissa[:15]}...")
    
    print("\nNote: 0.1 cannot be exactly represented in binary!")
    print(f"0.1 in float64: {0.1:.20f}")


def example_machine_epsilon():
    """Demonstrate machine epsilon."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Machine Epsilon")
    print("=" * 60)
    
    def compute_machine_epsilon(dtype):
        """Compute machine epsilon for a given dtype."""
        eps = dtype(1.0)
        while dtype(1.0) + eps != dtype(1.0):
            eps_prev = eps
            eps = eps / dtype(2.0)
        return eps_prev
    
    print("Machine epsilon by data type:")
    print(f"{'Type':>12} {'Computed':>20} {'numpy.finfo':>20}")
    print("-" * 55)
    
    for dtype in [np.float16, np.float32, np.float64]:
        computed = compute_machine_epsilon(dtype)
        info = np.finfo(dtype)
        print(f"{dtype.__name__:>12} {float(computed):>20.2e} {info.eps:>20.2e}")
    
    print("\nDemonstrating epsilon effect:")
    eps32 = np.finfo(np.float32).eps
    print(f"float32 epsilon: {eps32}")
    print(f"1.0 + eps/2 == 1.0: {np.float32(1.0) + np.float32(eps32/2) == np.float32(1.0)}")
    print(f"1.0 + eps == 1.0:   {np.float32(1.0) + np.float32(eps32) == np.float32(1.0)}")


def example_rounding_errors():
    """Demonstrate rounding errors."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Rounding Errors")
    print("=" * 60)
    
    print("Classic example: 0.1 + 0.2 ≠ 0.3")
    a = 0.1
    b = 0.2
    c = 0.3
    
    print(f"0.1 = {a:.20f}")
    print(f"0.2 = {b:.20f}")
    print(f"0.3 = {c:.20f}")
    print(f"0.1 + 0.2 = {a + b:.20f}")
    print(f"0.1 + 0.2 == 0.3: {a + b == c}")
    print(f"Difference: {(a + b) - c:.2e}")
    
    print("\nAccumulation error:")
    sum_naive = 0.0
    for _ in range(1000000):
        sum_naive += 0.1
    
    print(f"Sum of 0.1 × 1,000,000:")
    print(f"  Expected: 100000.0")
    print(f"  Actual:   {sum_naive}")
    print(f"  Error:    {abs(sum_naive - 100000.0):.10f}")


def example_catastrophic_cancellation():
    """Demonstrate catastrophic cancellation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Catastrophic Cancellation")
    print("=" * 60)
    
    # Quadratic formula: ax² + bx + c = 0
    # Standard: x = (-b ± sqrt(b² - 4ac)) / 2a
    
    print("Solving x² - 1000000.001x + 1 = 0")
    a, b, c = 1, -1000000.001, 1
    
    # Standard formula
    disc = np.sqrt(b**2 - 4*a*c)
    x1_standard = (-b + disc) / (2*a)
    x2_standard = (-b - disc) / (2*a)
    
    # For x2, b and disc are nearly equal → cancellation!
    # Alternative: x1 * x2 = c/a (Vieta's formula)
    x1_stable = (-b + disc) / (2*a)
    x2_stable = c / (a * x1_stable)  # Using Vieta's formula
    
    print("\nStandard formula:")
    print(f"  x1 = {x1_standard}")
    print(f"  x2 = {x2_standard}")
    
    print("\nStable formula (using Vieta's):")
    print(f"  x1 = {x1_stable}")
    print(f"  x2 = {x2_stable}")
    
    # Verify
    print("\nVerification (should be 0):")
    print(f"  Standard x2: {a*x2_standard**2 + b*x2_standard + c:.2e}")
    print(f"  Stable x2:   {a*x2_stable**2 + b*x2_stable + c:.2e}")


def example_overflow_underflow():
    """Demonstrate overflow and underflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Overflow and Underflow")
    print("=" * 60)
    
    print("Float32 limits:")
    info = np.finfo(np.float32)
    print(f"  Max: {info.max:.2e}")
    print(f"  Min normal: {info.tiny:.2e}")
    print(f"  Min subnormal: {info.smallest_subnormal:.2e}")
    
    print("\nOverflow example:")
    x = np.float32(1e38)
    print(f"  1e38 * 10 = {x * 10}")  # Inf
    
    print("\nUnderflow example:")
    x = np.float32(1e-40)
    print(f"  1e-40 / 1e5 = {x / 1e5}")  # 0 or subnormal
    
    print("\nSoftmax overflow issue:")
    x = np.array([1000, 1001, 1002])
    
    print(f"  Input: {x}")
    with np.errstate(over='ignore', invalid='ignore'):
        naive = np.exp(x) / np.sum(np.exp(x))
    print(f"  Naive softmax: {naive}")  # NaN
    
    # Stable version
    x_stable = x - np.max(x)
    stable = np.exp(x_stable) / np.sum(np.exp(x_stable))
    print(f"  Stable softmax: {stable}")


def example_logsumexp():
    """Demonstrate log-sum-exp trick."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Log-Sum-Exp Trick")
    print("=" * 60)
    
    def logsumexp_naive(x):
        """Naive implementation - prone to overflow."""
        return np.log(np.sum(np.exp(x)))
    
    def logsumexp_stable(x):
        """Numerically stable implementation."""
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x - m)))
    
    # Test with moderate values
    x_safe = np.array([1.0, 2.0, 3.0])
    print("Safe input: [1, 2, 3]")
    print(f"  Naive:  {logsumexp_naive(x_safe):.6f}")
    print(f"  Stable: {logsumexp_stable(x_safe):.6f}")
    print(f"  NumPy:  {np.logaddexp.reduce(x_safe):.6f}")
    
    # Test with large values
    x_large = np.array([1000.0, 1001.0, 1002.0])
    print("\nLarge input: [1000, 1001, 1002]")
    with np.errstate(over='ignore'):
        naive_result = logsumexp_naive(x_large)
    print(f"  Naive:  {naive_result}")  # inf or nan
    print(f"  Stable: {logsumexp_stable(x_large):.6f}")
    
    # Test with small values
    x_small = np.array([-1000.0, -999.0, -998.0])
    print("\nSmall input: [-1000, -999, -998]")
    with np.errstate(divide='ignore'):
        naive_result = logsumexp_naive(x_small)
    print(f"  Naive:  {naive_result}")  # -inf
    print(f"  Stable: {logsumexp_stable(x_small):.6f}")


def example_stable_softmax():
    """Numerically stable softmax."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Stable Softmax")
    print("=" * 60)
    
    def softmax_naive(x):
        """Naive softmax - overflow issues."""
        e_x = np.exp(x)
        return e_x / np.sum(e_x)
    
    def softmax_stable(x):
        """Numerically stable softmax."""
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x)
    
    # Normal case
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print("Normal input: [1, 2, 3, 4, 5]")
    print(f"  Naive:  {softmax_naive(x)}")
    print(f"  Stable: {softmax_stable(x)}")
    
    # Large values
    x_large = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    print("\nLarge input: [100, 200, 300, 400, 500]")
    with np.errstate(over='ignore', invalid='ignore'):
        print(f"  Naive:  {softmax_naive(x_large)}")
    print(f"  Stable: {softmax_stable(x_large)}")
    
    # Very negative values
    x_neg = np.array([-500.0, -400.0, -300.0, -200.0, -100.0])
    print("\nNegative input: [-500, -400, -300, -200, -100]")
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        print(f"  Naive:  {softmax_naive(x_neg)}")
    print(f"  Stable: {softmax_stable(x_neg)}")


def example_stable_sigmoid():
    """Numerically stable sigmoid."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Stable Sigmoid")
    print("=" * 60)
    
    def sigmoid_naive(x):
        """Naive sigmoid - overflow for negative x."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_stable(x):
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    x = np.array([-1000, -100, -10, 0, 10, 100, 1000])
    
    print(f"{'x':>8} {'Naive':>15} {'Stable':>15}")
    print("-" * 40)
    
    with np.errstate(over='ignore'):
        for xi in x:
            naive = sigmoid_naive(xi)
            stable = sigmoid_stable(xi)
            print(f"{xi:>8} {naive:>15.6f} {stable:>15.6f}")


def example_kahan_summation():
    """Kahan summation for accurate accumulation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Kahan Summation")
    print("=" * 60)
    
    def naive_sum(arr):
        """Simple summation."""
        total = 0.0
        for x in arr:
            total += x
        return total
    
    def kahan_sum(arr):
        """Kahan compensated summation."""
        total = 0.0
        c = 0.0  # Compensation for lost low-order bits
        
        for x in arr:
            y = x - c
            t = total + y
            c = (t - total) - y
            total = t
        
        return total
    
    # Test with many small values
    n = 10000000
    values = np.ones(n, dtype=np.float32) * 0.1
    
    expected = n * 0.1
    naive_result = naive_sum(values)
    kahan_result = kahan_sum(values)
    numpy_result = np.sum(values)
    
    print(f"Summing 0.1 × {n:,}:")
    print(f"  Expected: {expected:.6f}")
    print(f"  Naive:    {naive_result:.6f} (error: {abs(naive_result - expected):.6f})")
    print(f"  Kahan:    {kahan_result:.6f} (error: {abs(kahan_result - expected):.6f})")
    print(f"  NumPy:    {numpy_result:.6f} (error: {abs(numpy_result - expected):.6f})")
    
    # Test with mixed large and small values
    large_small = np.array([1e10, 1.0, -1e10, 1.0, 1.0], dtype=np.float64)
    print(f"\nSumming [1e10, 1, -1e10, 1, 1]:")
    print(f"  Expected: 3.0")
    print(f"  Naive:    {naive_sum(large_small)}")
    print(f"  Kahan:    {kahan_sum(large_small)}")


def example_condition_number():
    """Demonstrate condition number effects."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Condition Number")
    print("=" * 60)
    
    def solve_and_check(A, b, name):
        """Solve Ax = b and check sensitivity."""
        cond = np.linalg.cond(A)
        x = np.linalg.solve(A, b)
        
        # Perturb b slightly
        delta_b = np.random.randn(len(b)) * 1e-10
        x_perturbed = np.linalg.solve(A, b + delta_b)
        
        relative_change_b = np.linalg.norm(delta_b) / np.linalg.norm(b)
        relative_change_x = np.linalg.norm(x - x_perturbed) / np.linalg.norm(x)
        amplification = relative_change_x / relative_change_b
        
        print(f"\n{name}:")
        print(f"  Condition number: {cond:.2e}")
        print(f"  Relative change in b: {relative_change_b:.2e}")
        print(f"  Relative change in x: {relative_change_x:.2e}")
        print(f"  Amplification factor: {amplification:.2f}")
    
    np.random.seed(42)
    
    # Well-conditioned matrix
    A_good = np.array([[2.0, 1.0],
                       [1.0, 3.0]])
    b_good = np.array([1.0, 1.0])
    solve_and_check(A_good, b_good, "Well-conditioned")
    
    # Ill-conditioned matrix
    epsilon = 1e-10
    A_bad = np.array([[1.0, 1.0],
                      [1.0, 1.0 + epsilon]])
    b_bad = np.array([2.0, 2.0])
    solve_and_check(A_bad, b_bad, "Ill-conditioned")
    
    # Hilbert matrix (famously ill-conditioned)
    n = 5
    A_hilbert = 1.0 / (np.arange(n)[:, None] + np.arange(n) + 1)
    b_hilbert = np.ones(n)
    solve_and_check(A_hilbert, b_hilbert, f"Hilbert matrix ({n}×{n})")


def example_mixed_precision():
    """Demonstrate mixed precision concepts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Mixed Precision")
    print("=" * 60)
    
    # Simulate forward pass in different precisions
    np.random.seed(42)
    
    x = np.random.randn(100, 100).astype(np.float32)
    w = np.random.randn(100, 100).astype(np.float32)
    
    # float32 computation
    y_fp32 = x @ w
    
    # float16 computation
    x_fp16 = x.astype(np.float16)
    w_fp16 = w.astype(np.float16)
    y_fp16 = (x_fp16 @ w_fp16).astype(np.float32)
    
    # Compare
    diff = np.abs(y_fp32 - y_fp16)
    
    print("Matrix multiplication: 100×100 @ 100×100")
    print(f"\nFloat32 result range: [{y_fp32.min():.4f}, {y_fp32.max():.4f}]")
    print(f"Float16 result range: [{y_fp16.min():.4f}, {y_fp16.max():.4f}]")
    print(f"\nMax absolute difference: {diff.max():.6f}")
    print(f"Mean absolute difference: {diff.mean():.6f}")
    print(f"Relative error: {(diff / (np.abs(y_fp32) + 1e-10)).mean():.6f}")
    
    # Memory comparison
    print(f"\nMemory usage:")
    print(f"  Float32: {x.nbytes + w.nbytes} bytes")
    print(f"  Float16: {x_fp16.nbytes + w_fp16.nbytes} bytes")
    print(f"  Savings: {100 * (1 - (x_fp16.nbytes + w_fp16.nbytes) / (x.nbytes + w.nbytes)):.0f}%")


def example_numerical_stability_in_ml():
    """Common numerical stability issues in ML."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: ML Numerical Stability")
    print("=" * 60)
    
    # Cross-entropy loss
    print("1. Cross-entropy loss:")
    
    def cross_entropy_naive(p, y):
        """Naive cross-entropy - log(0) issue."""
        return -np.sum(y * np.log(p))
    
    def cross_entropy_stable(logits, y, eps=1e-10):
        """Stable cross-entropy from logits."""
        p = np.exp(logits - np.max(logits))
        p = p / np.sum(p)
        return -np.sum(y * np.log(p + eps))
    
    # Perfect prediction (probability = 1 for correct class)
    p_perfect = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    
    print(f"  Perfect prediction p=[0, 1, 0]:")
    print(f"    Naive loss: {cross_entropy_naive(p_perfect, y)}")  # 0
    
    # Near-zero probability
    p_bad = np.array([0.0, 0.0, 1.0])
    print(f"  Wrong prediction p=[0, 0, 1]:")
    with np.errstate(divide='ignore'):
        print(f"    Naive loss: {cross_entropy_naive(p_bad, y)}")  # inf
    
    # Variance computation
    print("\n2. Variance computation:")
    
    def variance_one_pass(x):
        """One-pass formula - numerically unstable."""
        n = len(x)
        mean_x2 = np.mean(x ** 2)
        mean_x = np.mean(x)
        return mean_x2 - mean_x ** 2
    
    def variance_two_pass(x):
        """Two-pass formula - stable."""
        mean_x = np.mean(x)
        return np.mean((x - mean_x) ** 2)
    
    # Large values with small variance
    x = np.array([1e8, 1e8 + 1, 1e8 + 2])
    print(f"  Data: [1e8, 1e8+1, 1e8+2]")
    print(f"    One-pass variance: {variance_one_pass(x)}")
    print(f"    Two-pass variance: {variance_two_pass(x)}")
    print(f"    True variance: {np.var(x):.6f}")
    
    # Batch normalization
    print("\n3. Batch normalization:")
    
    def batch_norm_naive(x, eps=1e-5):
        """Naive batch norm."""
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        return (x - mean) / np.sqrt(var + eps)
    
    x = np.random.randn(32, 100)
    x_norm = batch_norm_naive(x)
    print(f"  Input mean: {x.mean(axis=0)[:5].round(3)}")
    print(f"  Output mean: {x_norm.mean(axis=0)[:5].round(3)}")
    print(f"  Output std: {x_norm.std(axis=0)[:5].round(3)}")


if __name__ == "__main__":
    example_ieee754_representation()
    example_machine_epsilon()
    example_rounding_errors()
    example_catastrophic_cancellation()
    example_overflow_underflow()
    example_logsumexp()
    example_stable_softmax()
    example_stable_sigmoid()
    example_kahan_summation()
    example_condition_number()
    example_mixed_precision()
    example_numerical_stability_in_ml()
