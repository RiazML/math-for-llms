"""
Number Systems - Exercises
==========================
Practice problems to solidify understanding of number systems.

Instructions:
1. Implement each function
2. Run tests to verify
3. Solutions at bottom (try first!)
"""

import numpy as np
from typing import Union, List, Tuple


# =============================================================================
# EXERCISE 1: Safe Division (Easy)
# =============================================================================

def safe_divide(a: float, b: float, epsilon: float = 1e-10) -> float:
    """
    Perform division with numerical stability.
    
    In ML, we often need to divide by values that might be zero or very small
    (e.g., in normalization). This function should handle those cases safely.
    
    Parameters
    ----------
    a : float
        Numerator
    b : float
        Denominator
    epsilon : float
        Small value to prevent division by zero
    
    Returns
    -------
    float
        Result of a / (b + epsilon) if b >= 0, or a / (b - epsilon) if b < 0
    
    Example
    -------
    >>> safe_divide(1.0, 0.0)
    1e10  # approximately
    >>> safe_divide(1.0, 2.0)
    0.5
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Stable Softmax (Easy)
# =============================================================================

def stable_softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax in a numerically stable way.
    
    The naive softmax exp(x_i) / sum(exp(x_j)) can overflow for large values.
    The trick is to subtract max(x) before exponentiating.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of any shape
    
    Returns
    -------
    np.ndarray
        Softmax probabilities (same shape as input)
    
    Example
    -------
    >>> stable_softmax(np.array([1000, 1001, 1002]))
    array([0.09003057, 0.24472847, 0.66524096])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Log-Sum-Exp (Medium)
# =============================================================================

def stable_logsumexp(x: np.ndarray) -> float:
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    
    This is useful for computing log-probabilities.
    
    The trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    
    Returns
    -------
    float
        The log-sum-exp value
    
    Example
    -------
    >>> stable_logsumexp(np.array([1000, 1001, 1002]))
    1002.407606  # approximately
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Floating Point Comparison (Easy)
# =============================================================================

def almost_equal(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two floats are approximately equal.
    
    Two values are considered equal if:
    |a - b| <= atol + rtol * |b|
    
    Parameters
    ----------
    a, b : float
        Values to compare
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    
    Returns
    -------
    bool
        True if values are approximately equal
    
    Example
    -------
    >>> almost_equal(0.1 + 0.2, 0.3)
    True
    >>> almost_equal(1.0, 1.0001, rtol=1e-5)
    False
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: Complex Magnitude and Phase (Medium)
# =============================================================================

def complex_to_polar(z: complex) -> Tuple[float, float]:
    """
    Convert a complex number to polar form (magnitude, phase).
    
    For z = a + bi:
    - magnitude (r) = sqrt(a² + b²)
    - phase (θ) = atan2(b, a)
    
    Note: Use atan2 for correct quadrant handling.
    
    Parameters
    ----------
    z : complex
        Complex number
    
    Returns
    -------
    Tuple[float, float]
        (magnitude, phase) where phase is in radians
    
    Example
    -------
    >>> complex_to_polar(1 + 1j)
    (1.414..., 0.785...)  # (√2, π/4)
    """
    # YOUR CODE HERE
    pass


def polar_to_complex(r: float, theta: float) -> complex:
    """
    Convert polar form to complex number.
    
    Using Euler's formula: r * e^(iθ) = r * (cos(θ) + i*sin(θ))
    
    Parameters
    ----------
    r : float
        Magnitude
    theta : float
        Phase in radians
    
    Returns
    -------
    complex
        Complex number
    
    Example
    -------
    >>> polar_to_complex(2, np.pi/2)
    (0 + 2j)  # approximately
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 6: Numerical Gradient Check (Medium)
# =============================================================================

def numerical_gradient(f, x: float, epsilon: float = 1e-5) -> float:
    """
    Compute numerical gradient using central difference.
    
    gradient ≈ (f(x + ε) - f(x - ε)) / (2ε)
    
    This is used in ML to verify analytical gradients (gradient checking).
    
    Parameters
    ----------
    f : callable
        Function to differentiate
    x : float
        Point at which to compute gradient
    epsilon : float
        Small perturbation
    
    Returns
    -------
    float
        Numerical estimate of f'(x)
    
    Example
    -------
    >>> numerical_gradient(lambda x: x**2, 3.0)
    6.0  # approximately, since d/dx(x²) = 2x
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 7: Mixed Precision Simulation (Hard)
# =============================================================================

def mixed_precision_update(
    weight: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate mixed precision training update.
    
    In mixed precision training:
    1. Keep master weights in float32
    2. Compute gradients in float16
    3. Apply update to float32 master weights
    4. Cast back to float16 for next forward pass
    
    Parameters
    ----------
    weight : np.ndarray
        Current weight (float32 master weight)
    gradient : np.ndarray
        Gradient (float16)
    learning_rate : float
        Learning rate
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (updated_master_weight_fp32, weight_for_forward_fp16)
    
    Example
    -------
    >>> w = np.array([1.0, 2.0], dtype=np.float32)
    >>> g = np.array([0.1, 0.2], dtype=np.float16)
    >>> w_new, w_fp16 = mixed_precision_update(w, g)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 8: Detect Overflow Risk (Medium)
# =============================================================================

def check_overflow_risk(values: np.ndarray, dtype: type = np.float32) -> dict:
    """
    Check if an array has values that might cause overflow/underflow.
    
    Parameters
    ----------
    values : np.ndarray
        Array to check
    dtype : type
        Target data type
    
    Returns
    -------
    dict
        Dictionary with:
        - 'max_value': maximum value in array
        - 'min_value': minimum non-zero absolute value
        - 'overflow_risk': bool, True if max_value might overflow
        - 'underflow_risk': bool, True if min_value might underflow
        - 'safe': bool, True if no risks detected
    
    Example
    -------
    >>> check_overflow_risk(np.array([1e30, 1e-30]))
    {'max_value': 1e30, 'min_value': 1e-30, 
     'overflow_risk': False, 'underflow_risk': False, 'safe': True}
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 9: INT8 Quantization (Medium)
# =============================================================================

def quantize_weights(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Implement symmetric INT8 quantization for neural network weights.
    
    Symmetric quantization maps the range [-max_abs, +max_abs] to [-127, +127].
    
    Formula:
        scale = max(|w|) / 127
        quantized = round(w / scale)
    
    Parameters
    ----------
    weights : np.ndarray
        Float32 weight tensor of any shape
    
    Returns
    -------
    Tuple[np.ndarray, float]
        (quantized_weights as int8, scale factor)
    
    Example
    -------
    >>> w = np.array([0.5, -0.3, 0.127])
    >>> q, s = quantize_weights(w)
    >>> q
    array([127, -76, 32], dtype=int8)  # approximately
    """
    # YOUR CODE HERE
    pass


def dequantize_weights(quantized: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert quantized INT8 weights back to float32.
    
    Formula:
        weights = quantized * scale
    
    Parameters
    ----------
    quantized : np.ndarray
        INT8 quantized weights
    scale : float
        Scale factor from quantization
    
    Returns
    -------
    np.ndarray
        Dequantized float32 weights
    
    Example
    -------
    >>> q = np.array([127, -76, 32], dtype=np.int8)
    >>> dequantize_weights(q, 0.00394)
    array([0.5, -0.3, 0.126], dtype=float32)  # approximately
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run all tests."""
    print("Running tests...\n")
    
    # Test 1: safe_divide
    print("Test 1: safe_divide")
    try:
        assert np.isclose(safe_divide(1.0, 2.0), 0.5)
        assert safe_divide(1.0, 0.0) > 1e9  # Should be large, not inf
        assert not np.isinf(safe_divide(1.0, 0.0))
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 2: stable_softmax
    print("Test 2: stable_softmax")
    try:
        result = stable_softmax(np.array([1000, 1001, 1002]))
        assert np.allclose(result.sum(), 1.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 3: stable_logsumexp
    print("Test 3: stable_logsumexp")
    try:
        result = stable_logsumexp(np.array([1000, 1001, 1002]))
        assert not np.isnan(result)
        assert not np.isinf(result)
        assert np.isclose(result, 1002.407, rtol=1e-3)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 4: almost_equal
    print("Test 4: almost_equal")
    try:
        assert almost_equal(0.1 + 0.2, 0.3)
        assert almost_equal(1.0, 1.000001)
        assert not almost_equal(1.0, 1.1)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 5: complex_to_polar and polar_to_complex
    print("Test 5: complex_to_polar / polar_to_complex")
    try:
        r, theta = complex_to_polar(1 + 1j)
        assert np.isclose(r, np.sqrt(2))
        assert np.isclose(theta, np.pi/4)
        
        z = polar_to_complex(2, np.pi/2)
        assert np.isclose(z.real, 0, atol=1e-10)
        assert np.isclose(z.imag, 2)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 6: numerical_gradient
    print("Test 6: numerical_gradient")
    try:
        grad = numerical_gradient(lambda x: x**2, 3.0)
        assert np.isclose(grad, 6.0, rtol=1e-4)
        
        grad = numerical_gradient(np.sin, 0.0)
        assert np.isclose(grad, 1.0, rtol=1e-4)  # d/dx sin(x) at x=0 is cos(0)=1
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 7: mixed_precision_update
    print("Test 7: mixed_precision_update")
    try:
        w = np.array([1.0, 2.0], dtype=np.float32)
        g = np.array([0.1, 0.2], dtype=np.float16)
        w_new, w_fp16 = mixed_precision_update(w, g, learning_rate=0.1)
        
        assert w_new.dtype == np.float32
        assert w_fp16.dtype == np.float16
        assert np.allclose(w_new, [0.99, 1.98], rtol=1e-2)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 8: check_overflow_risk
    print("Test 8: check_overflow_risk")
    try:
        result = check_overflow_risk(np.array([1e30, 1e-30]))
        assert 'overflow_risk' in result
        assert 'underflow_risk' in result
        assert 'safe' in result
        
        result = check_overflow_risk(np.array([1e40]))
        assert result['overflow_risk'] == True
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 9: quantize_weights and dequantize_weights
    print("Test 9: quantize_weights / dequantize_weights")
    try:
        weights = np.array([0.5, -0.3, 0.127, -0.5], dtype=np.float32)
        q, scale = quantize_weights(weights)
        
        assert q.dtype == np.int8
        assert scale > 0
        assert q[0] == 127  # Max value maps to 127
        assert q[3] == -127  # Min value maps to -127
        
        reconstructed = dequantize_weights(q, scale)
        assert reconstructed.dtype == np.float32
        assert np.allclose(weights, reconstructed, rtol=0.01)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    print("\nTests complete!")


# =============================================================================
# SOLUTIONS (Don't look until you've tried!)
# =============================================================================
"""
Scroll down for solutions...
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
"""

def safe_divide_solution(a: float, b: float, epsilon: float = 1e-10) -> float:
    """Solution for Exercise 1."""
    if b >= 0:
        return a / (b + epsilon)
    else:
        return a / (b - epsilon)


def stable_softmax_solution(x: np.ndarray) -> np.ndarray:
    """Solution for Exercise 2."""
    # Subtract max for numerical stability
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def stable_logsumexp_solution(x: np.ndarray) -> float:
    """Solution for Exercise 3."""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def almost_equal_solution(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Solution for Exercise 4."""
    return abs(a - b) <= atol + rtol * abs(b)


def complex_to_polar_solution(z: complex) -> Tuple[float, float]:
    """Solution for Exercise 5."""
    magnitude = abs(z)  # or np.sqrt(z.real**2 + z.imag**2)
    phase = np.arctan2(z.imag, z.real)
    return magnitude, phase


def polar_to_complex_solution(r: float, theta: float) -> complex:
    """Solution for Exercise 5."""
    return r * np.exp(1j * theta)
    # Or equivalently: r * (np.cos(theta) + 1j * np.sin(theta))


def numerical_gradient_solution(f, x: float, epsilon: float = 1e-5) -> float:
    """Solution for Exercise 6."""
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)


def mixed_precision_update_solution(
    weight: np.ndarray,
    gradient: np.ndarray,
    learning_rate: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """Solution for Exercise 7."""
    # Convert gradient to float32 for accurate update
    gradient_fp32 = gradient.astype(np.float32)
    
    # Update master weights in float32
    updated_weight = weight - learning_rate * gradient_fp32
    
    # Cast back to float16 for forward pass
    weight_fp16 = updated_weight.astype(np.float16)
    
    return updated_weight, weight_fp16


def check_overflow_risk_solution(values: np.ndarray, dtype: type = np.float32) -> dict:
    """Solution for Exercise 8."""
    finfo = np.finfo(dtype)
    
    max_val = np.max(np.abs(values))
    non_zero = values[values != 0]
    min_val = np.min(np.abs(non_zero)) if len(non_zero) > 0 else 0
    
    overflow_risk = max_val > finfo.max * 0.9  # 90% of max
    underflow_risk = min_val < finfo.tiny * 10 if min_val > 0 else False
    
    return {
        'max_value': max_val,
        'min_value': min_val,
        'overflow_risk': overflow_risk,
        'underflow_risk': underflow_risk,
        'safe': not (overflow_risk or underflow_risk)
    }


def quantize_weights_solution(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solution for Exercise 9."""
    max_abs = np.max(np.abs(weights))
    scale = max_abs / 127
    quantized = np.round(weights / scale).astype(np.int8)
    return quantized, scale


def dequantize_weights_solution(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Solution for Exercise 9."""
    return quantized.astype(np.float32) * scale


# Uncomment to use solutions:
# safe_divide = safe_divide_solution
# stable_softmax = stable_softmax_solution
# stable_logsumexp = stable_logsumexp_solution
# almost_equal = almost_equal_solution
# complex_to_polar = complex_to_polar_solution
# polar_to_complex = polar_to_complex_solution
# numerical_gradient = numerical_gradient_solution
# mixed_precision_update = mixed_precision_update_solution
# check_overflow_risk = check_overflow_risk_solution
# quantize_weights = quantize_weights_solution
# dequantize_weights = dequantize_weights_solution


if __name__ == "__main__":
    run_tests()
