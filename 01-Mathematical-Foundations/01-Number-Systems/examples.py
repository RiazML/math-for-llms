"""
Number Systems - Examples
=========================
Python implementations demonstrating number system concepts.

Requirements: numpy, matplotlib
Run: python examples.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple


# =============================================================================
# EXAMPLE 1: Number Types in Python and NumPy
# =============================================================================

def example_number_types():
    """
    Demonstrate different number types in Python and NumPy.
    
    Mathematical background:
    - Python supports integers of arbitrary precision
    - NumPy has fixed-size number types for performance
    - Understanding types is crucial for memory management in ML
    """
    print("=" * 60)
    print("EXAMPLE 1: Number Types in Python and NumPy")
    print("=" * 60)
    
    # Python native types
    print("\n1. Python Native Types:")
    print("-" * 40)
    
    # Integer (arbitrary precision in Python 3)
    big_int = 10 ** 100
    print(f"Big integer (10^100): {str(big_int)[:50]}...")
    print(f"Type: {type(big_int)}")
    
    # Float (double precision by default)
    pi = 3.141592653589793
    print(f"\nPi: {pi}")
    print(f"Type: {type(pi)}")
    
    # Complex
    z = 3 + 4j
    print(f"\nComplex number: {z}")
    print(f"Real part: {z.real}, Imaginary part: {z.imag}")
    print(f"Magnitude: {abs(z)}")
    
    # NumPy types
    print("\n2. NumPy Number Types:")
    print("-" * 40)
    
    dtypes = [np.int8, np.int16, np.int32, np.int64,
              np.float16, np.float32, np.float64,
              np.complex64, np.complex128]
    
    print(f"{'Type':<15} {'Size (bytes)':<15} {'Min':<25} {'Max':<25}")
    print("-" * 80)
    
    for dtype in dtypes:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            print(f"{dtype.__name__:<15} {dtype().nbytes:<15} {info.min:<25} {info.max:<25}")
        elif np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
            print(f"{dtype.__name__:<15} {dtype().nbytes:<15} {info.min:<25.2e} {info.max:<25.2e}")
        else:
            # Complex
            size = dtype().nbytes
            print(f"{dtype.__name__:<15} {size:<15} {'N/A':<25} {'N/A':<25}")


# =============================================================================
# EXAMPLE 2: Floating Point Precision Issues
# =============================================================================

def example_floating_point_precision():
    """
    Demonstrate floating point precision issues relevant to ML.
    
    Mathematical background:
    - Floating point numbers are approximations
    - This affects gradient computations and comparisons
    - Understanding these issues prevents subtle bugs
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Floating Point Precision Issues")
    print("=" * 60)
    
    # Classic example
    print("\n1. Classic Precision Issue:")
    print("-" * 40)
    result = 0.1 + 0.2
    print(f"0.1 + 0.2 = {result}")
    print(f"0.1 + 0.2 == 0.3? {result == 0.3}")
    print(f"Using np.isclose: {np.isclose(result, 0.3)}")
    
    # Accumulation errors
    print("\n2. Accumulation Errors:")
    print("-" * 40)
    
    # Naive summation
    values = [0.1] * 10
    naive_sum = sum(values)
    print(f"Sum of 10 x 0.1 (naive): {naive_sum}")
    print(f"Expected: 1.0, Error: {abs(naive_sum - 1.0):.2e}")
    
    # Using NumPy (more accurate)
    numpy_sum = np.sum(np.array(values, dtype=np.float64))
    print(f"Sum of 10 x 0.1 (numpy): {numpy_sum}")
    
    # Machine epsilon
    print("\n3. Machine Epsilon:")
    print("-" * 40)
    
    for dtype in [np.float16, np.float32, np.float64]:
        eps = np.finfo(dtype).eps
        print(f"{dtype.__name__}: epsilon = {eps:.2e}")
        
        # Verify
        one = dtype(1.0)
        one_plus_eps = one + dtype(eps)
        one_plus_half_eps = one + dtype(eps / 2)
        print(f"  1.0 + eps != 1.0? {one_plus_eps != one}")
        print(f"  1.0 + eps/2 != 1.0? {one_plus_half_eps != one}")


# =============================================================================
# EXAMPLE 3: Numerical Overflow and Underflow
# =============================================================================

def example_overflow_underflow():
    """
    Demonstrate overflow and underflow in ML contexts.
    
    Mathematical background:
    - Overflow: number too large to represent
    - Underflow: number too small (becomes zero)
    - Both are common issues in neural network training
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Numerical Overflow and Underflow")
    print("=" * 60)
    
    # Overflow in softmax
    print("\n1. Overflow in Softmax:")
    print("-" * 40)
    
    def naive_softmax(x):
        """Naive softmax - prone to overflow."""
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    def stable_softmax(x):
        """Numerically stable softmax."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    # Normal case
    x_normal = np.array([1.0, 2.0, 3.0])
    print(f"Input (normal): {x_normal}")
    print(f"Naive softmax: {naive_softmax(x_normal)}")
    print(f"Stable softmax: {stable_softmax(x_normal)}")
    
    # Overflow case
    x_large = np.array([1000.0, 1001.0, 1002.0])
    print(f"\nInput (large): {x_large}")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        naive_result = naive_softmax(x_large)
    print(f"Naive softmax: {naive_result}")  # Will have NaN or inf
    print(f"Stable softmax: {stable_softmax(x_large)}")  # Works correctly
    
    # Underflow in log probabilities
    print("\n2. Underflow in Log Probabilities:")
    print("-" * 40)
    
    # Product of small probabilities
    small_probs = np.array([0.001] * 100)
    
    # Direct product (underflows)
    direct_product = np.prod(small_probs)
    print(f"Product of 100 x 0.001: {direct_product}")
    
    # Log-sum approach (stable)
    log_sum = np.sum(np.log(small_probs))
    print(f"Sum of logs: {log_sum}")
    print(f"This represents: e^{log_sum:.2f} (very small but tracked)")
    
    # Log-sum-exp trick
    print("\n3. Log-Sum-Exp Trick:")
    print("-" * 40)
    
    def logsumexp_naive(x):
        """Naive log-sum-exp."""
        return np.log(np.sum(np.exp(x)))
    
    def logsumexp_stable(x):
        """Numerically stable log-sum-exp."""
        max_x = np.max(x)
        return max_x + np.log(np.sum(np.exp(x - max_x)))
    
    x = np.array([1000, 1001, 1002])
    print(f"Input: {x}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        naive = logsumexp_naive(x)
    print(f"Naive logsumexp: {naive}")
    print(f"Stable logsumexp: {logsumexp_stable(x)}")
    print(f"scipy.special.logsumexp: {np.log(np.sum(np.exp(x - np.max(x)))) + np.max(x)}")


# =============================================================================
# EXAMPLE 4: Complex Numbers in ML
# =============================================================================

def example_complex_numbers():
    """
    Demonstrate complex numbers and their applications in ML.
    
    Mathematical background:
    - Complex numbers: z = a + bi where i² = -1
    - Used in Fourier transforms, eigenvalue decomposition
    - Essential for signal processing in ML
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Complex Numbers in ML")
    print("=" * 60)
    
    # Basic operations
    print("\n1. Basic Complex Operations:")
    print("-" * 40)
    
    z1 = 3 + 4j
    z2 = 1 - 2j
    
    print(f"z1 = {z1}")
    print(f"z2 = {z2}")
    print(f"z1 + z2 = {z1 + z2}")
    print(f"z1 * z2 = {z1 * z2}")
    print(f"|z1| (magnitude) = {abs(z1)}")
    print(f"conjugate(z1) = {z1.conjugate()}")
    print(f"z1 * conjugate(z1) = {z1 * z1.conjugate()} (always real!)")
    
    # Euler's formula
    print("\n2. Euler's Formula: e^(iθ) = cos(θ) + i·sin(θ)")
    print("-" * 40)
    
    theta = np.pi / 4  # 45 degrees
    euler = np.exp(1j * theta)
    trig = np.cos(theta) + 1j * np.sin(theta)
    
    print(f"θ = π/4")
    print(f"e^(iθ) = {euler}")
    print(f"cos(θ) + i·sin(θ) = {trig}")
    print(f"Equal? {np.isclose(euler, trig)}")
    
    # Fourier Transform
    print("\n3. Fourier Transform (FFT):")
    print("-" * 40)
    
    # Create a simple signal
    t = np.linspace(0, 1, 100)
    freq1, freq2 = 5, 12  # Hz
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])
    
    print(f"Signal: combination of {freq1}Hz and {freq2}Hz sine waves")
    print(f"FFT output type: {fft_result.dtype}")
    print(f"FFT gives complex values representing amplitude and phase")
    
    # Find dominant frequencies
    magnitude = np.abs(fft_result)
    dominant_idx = np.argsort(magnitude[frequencies > 0])[-2:]
    dominant_freqs = frequencies[frequencies > 0][dominant_idx]
    print(f"Detected dominant frequencies: {dominant_freqs} Hz")
    
    # Eigenvalues can be complex
    print("\n4. Complex Eigenvalues:")
    print("-" * 40)
    
    # Rotation matrix (has complex eigenvalues)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    eigenvalues, eigenvectors = np.linalg.eig(R)
    
    print(f"Rotation matrix (θ=π/4):\n{R}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvalues are complex! This indicates rotation.")
    print(f"|eigenvalues| = {np.abs(eigenvalues)} (magnitude 1 = pure rotation)")


# =============================================================================
# EXAMPLE 5: Visualization of Number Sets
# =============================================================================

def example_visualize_number_systems():
    """
    Visualize the relationships between number systems.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Visualization of Number Systems")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Number line showing different number types
    ax1 = axes[0]
    
    # Natural numbers
    naturals = np.arange(0, 5)
    ax1.scatter(naturals, [0] * len(naturals), s=100, c='green', 
                label='Natural (ℕ)', zorder=5)
    
    # Integers (negative)
    negatives = np.arange(-4, 0)
    ax1.scatter(negatives, [0] * len(negatives), s=100, c='blue', 
                label='Integers (ℤ)', zorder=5)
    
    # Rationals (some fractions)
    rationals = [-0.5, 0.25, 1.5, 2.75, 3.33]
    ax1.scatter(rationals, [0] * len(rationals), s=50, c='orange', 
                label='Rationals (ℚ)', zorder=4, marker='s')
    
    # Irrationals
    irrationals = [np.sqrt(2), np.pi, np.e]
    ax1.scatter(irrationals, [0] * len(irrationals), s=50, c='red', 
                label='Irrationals', zorder=4, marker='^')
    
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Real Number Line')
    ax1.set_title('Real Numbers (ℝ)')
    ax1.legend(loc='upper left')
    ax1.set_yticks([])
    
    # Annotate special irrationals
    ax1.annotate('√2', (np.sqrt(2), 0.1), ha='center')
    ax1.annotate('π', (np.pi, 0.1), ha='center')
    ax1.annotate('e', (np.e, 0.1), ha='center')
    
    # Right: Complex plane
    ax2 = axes[1]
    
    # Plot unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    
    # Some complex numbers
    complex_nums = [1+0j, 0+1j, -1+0j, 0-1j,  # On unit circle
                    2+1j, -1+2j, 1-1.5j,  # Other points
                    np.exp(1j * np.pi/4),  # e^(iπ/4)
                    np.exp(1j * np.pi/3)]  # e^(iπ/3)
    
    for z in complex_nums:
        ax2.scatter(z.real, z.imag, s=100, zorder=5)
        ax2.annotate(f'{z:.2f}' if abs(z.imag) > 0.01 else f'{z.real:.1f}',
                    (z.real + 0.1, z.imag + 0.1))
    
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_title('Complex Plane (ℂ)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('number_systems.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved as 'number_systems.png'")


# =============================================================================
# EXAMPLE 6: Memory Usage in ML
# =============================================================================

def example_memory_usage():
    """
    Compare memory usage of different number types in ML contexts.
    
    This is crucial for:
    - Large-scale training
    - Model deployment
    - GPU memory management
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Memory Usage in ML")
    print("=" * 60)
    
    # Simulate a neural network layer
    n_inputs = 1024
    n_outputs = 512
    batch_size = 64
    
    print(f"\nNeural Network Layer Simulation:")
    print(f"- Input size: {n_inputs}")
    print(f"- Output size: {n_outputs}")
    print(f"- Batch size: {batch_size}")
    print("-" * 40)
    
    dtypes = [np.float16, np.float32, np.float64]
    
    for dtype in dtypes:
        # Weight matrix
        W = np.random.randn(n_inputs, n_outputs).astype(dtype)
        
        # Input batch
        X = np.random.randn(batch_size, n_inputs).astype(dtype)
        
        # Bias
        b = np.random.randn(n_outputs).astype(dtype)
        
        # Forward pass
        output = X @ W + b
        
        # Memory calculation
        weight_mem = W.nbytes / 1024  # KB
        input_mem = X.nbytes / 1024   # KB
        output_mem = output.nbytes / 1024  # KB
        total_mem = weight_mem + input_mem + output_mem
        
        print(f"\n{dtype.__name__}:")
        print(f"  Weights: {weight_mem:.2f} KB")
        print(f"  Input batch: {input_mem:.2f} KB")
        print(f"  Output: {output_mem:.2f} KB")
        print(f"  Total: {total_mem:.2f} KB")
    
    # Real-world scale
    print("\n" + "=" * 40)
    print("Real-World Scale (GPT-2 sized model):")
    print("=" * 40)
    
    # GPT-2 small: ~124M parameters
    n_params = 124_000_000
    
    for dtype, name in [(np.float16, 'float16'), 
                        (np.float32, 'float32'),
                        (np.float64, 'float64')]:
        mem_bytes = n_params * np.dtype(dtype).itemsize
        mem_mb = mem_bytes / (1024 ** 2)
        mem_gb = mem_bytes / (1024 ** 3)
        
        print(f"{name}: {mem_mb:.1f} MB ({mem_gb:.2f} GB)")


# =============================================================================
# EXAMPLE 7: Quantization for ML Deployment
# =============================================================================

def example_quantization():
    """
    Demonstrate quantization techniques used in ML model deployment.
    
    Quantization reduces model size and speeds up inference by using
    lower-precision numbers (e.g., int8 instead of float32).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Quantization for ML Deployment")
    print("=" * 60)
    
    # 1. Basic INT8 Quantization
    print("\n1. Basic INT8 Quantization:")
    print("-" * 40)
    
    def quantize_symmetric(weights: np.ndarray, bits: int = 8) -> tuple:
        """Symmetric quantization: maps [-max, +max] to [-127, +127]."""
        max_val = np.max(np.abs(weights))
        scale = max_val / (2**(bits-1) - 1)  # 127 for int8
        quantized = np.round(weights / scale).astype(np.int8)
        return quantized, scale
    
    def dequantize_symmetric(quantized: np.ndarray, scale: float) -> np.ndarray:
        """Convert quantized values back to float."""
        return quantized.astype(np.float32) * scale
    
    # Simulate neural network weights
    np.random.seed(42)
    original_weights = np.random.randn(4, 4).astype(np.float32) * 0.5
    
    print("Original weights (float32):")
    print(original_weights)
    print(f"Memory: {original_weights.nbytes} bytes")
    
    # Quantize to int8
    quantized, scale = quantize_symmetric(original_weights)
    print(f"\nQuantized weights (int8):")
    print(quantized)
    print(f"Memory: {quantized.nbytes} bytes (4x compression!)")
    print(f"Scale factor: {scale:.6f}")
    
    # Dequantize and check error
    reconstructed = dequantize_symmetric(quantized, scale)
    error = np.abs(original_weights - reconstructed)
    print(f"\nReconstruction error:")
    print(f"  Mean error: {error.mean():.6f}")
    print(f"  Max error: {error.max():.6f}")
    
    # 2. Asymmetric Quantization (for ReLU outputs)
    print("\n2. Asymmetric Quantization:")
    print("-" * 40)
    
    def quantize_asymmetric(values: np.ndarray, bits: int = 8) -> tuple:
        """Asymmetric quantization: maps [min, max] to [0, 255]."""
        min_val = values.min()
        max_val = values.max()
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = int(np.round(-min_val / scale))
        quantized = np.round(values / scale + zero_point).astype(np.uint8)
        return quantized, scale, zero_point
    
    def dequantize_asymmetric(quantized: np.ndarray, scale: float, 
                               zero_point: int) -> np.ndarray:
        """Convert asymmetrically quantized values back to float."""
        return (quantized.astype(np.float32) - zero_point) * scale
    
    # Simulate ReLU activations (all non-negative)
    relu_activations = np.maximum(0, np.random.randn(8) * 2 + 1)
    print(f"ReLU activations: {relu_activations}")
    
    q_asym, scale_a, zp = quantize_asymmetric(relu_activations)
    print(f"Quantized (uint8): {q_asym}")
    print(f"Scale: {scale_a:.6f}, Zero point: {zp}")
    
    reconstructed_a = dequantize_asymmetric(q_asym, scale_a, zp)
    error_a = np.abs(relu_activations - reconstructed_a)
    print(f"Max reconstruction error: {error_a.max():.6f}")
    
    # 3. INT4 Quantization (for LLMs)
    print("\n3. INT4 Quantization (LLM-style):")
    print("-" * 40)
    
    def quantize_int4(weights: np.ndarray) -> tuple:
        """4-bit quantization (range: -8 to 7 or 0 to 15)."""
        max_val = np.max(np.abs(weights))
        scale = max_val / 7  # 4-bit signed: -8 to 7
        quantized = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
        return quantized, scale
    
    # Simulate a larger weight matrix
    large_weights = np.random.randn(16, 16).astype(np.float32) * 0.3
    
    q_int8, s8 = quantize_symmetric(large_weights, bits=8)
    q_int4, s4 = quantize_int4(large_weights)
    
    r_int8 = dequantize_symmetric(q_int8, s8)
    r_int4 = q_int4.astype(np.float32) * s4
    
    print(f"Original shape: {large_weights.shape}")
    print(f"Original memory: {large_weights.nbytes} bytes")
    print(f"\nINT8 quantization:")
    print(f"  Memory: {q_int8.nbytes} bytes")
    print(f"  Mean error: {np.abs(large_weights - r_int8).mean():.6f}")
    print(f"\nINT4 quantization:")
    print(f"  Memory: {q_int4.nbytes} bytes (would be {q_int4.nbytes//2} with packing)")
    print(f"  Mean error: {np.abs(large_weights - r_int4).mean():.6f}")
    
    # 4. Quantization-induced accuracy analysis
    print("\n4. Impact on Matrix Multiplication:")
    print("-" * 40)
    
    # Simulate a linear layer: y = Wx
    W = np.random.randn(64, 32).astype(np.float32) * 0.1
    x = np.random.randn(32).astype(np.float32)
    
    # Full precision
    y_fp32 = W @ x
    
    # Quantized computation
    W_q, W_scale = quantize_symmetric(W)
    W_deq = dequantize_symmetric(W_q, W_scale)
    y_quantized = W_deq @ x
    
    # Compare
    mse = np.mean((y_fp32 - y_quantized) ** 2)
    relative_error = np.mean(np.abs(y_fp32 - y_quantized) / (np.abs(y_fp32) + 1e-8))
    
    print(f"Output vector size: {len(y_fp32)}")
    print(f"MSE (FP32 vs INT8): {mse:.8f}")
    print(f"Mean relative error: {relative_error*100:.4f}%")
    print(f"\nConclusion: INT8 quantization preserves accuracy remarkably well!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("       NUMBER SYSTEMS - EXAMPLES")
    print("=" * 60)
    
    example_number_types()
    example_floating_point_precision()
    example_overflow_underflow()
    example_complex_numbers()
    example_visualize_number_systems()
    example_memory_usage()
    example_quantization()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
