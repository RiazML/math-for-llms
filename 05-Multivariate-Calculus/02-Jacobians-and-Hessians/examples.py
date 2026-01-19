"""
Jacobians and Hessians - Examples
=================================
Practical demonstrations of Jacobian and Hessian matrices.
"""

import numpy as np


def example_jacobian_basic():
    """Demonstrate basic Jacobian computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Jacobian Computation")
    print("=" * 60)
    
    print("For f(x, y) = [x² + y, xy, x - y²]ᵀ")
    print("\nPartial derivatives:")
    print("∂f₁/∂x = 2x,  ∂f₁/∂y = 1")
    print("∂f₂/∂x = y,   ∂f₂/∂y = x")
    print("∂f₃/∂x = 1,   ∂f₃/∂y = -2y")
    
    print("\nJacobian matrix (3×2):")
    print("J = [ 2x    1  ]")
    print("    [  y    x  ]")
    print("    [  1   -2y ]")
    
    def f(v):
        x, y = v
        return np.array([x**2 + y, x*y, x - y**2])
    
    def jacobian_analytic(x, y):
        return np.array([
            [2*x, 1],
            [y, x],
            [1, -2*y]
        ])
    
    point = (2.0, 3.0)
    J = jacobian_analytic(*point)
    
    print(f"\nAt point {point}:")
    print(f"J = \n{J}")
    
    # Numerical verification
    h = 1e-7
    J_numerical = np.zeros((3, 2))
    v = np.array(point)
    for j in range(2):
        v_plus = v.copy()
        v_plus[j] += h
        J_numerical[:, j] = (f(v_plus) - f(v)) / h
    
    print(f"\nNumerical Jacobian:\n{J_numerical.round(4)}")


def example_jacobian_chain_rule():
    """Demonstrate chain rule with Jacobians."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Jacobian Chain Rule")
    print("=" * 60)
    
    print("g: ℝ² → ℝ²: g(x, y) = [x + y, xy]")
    print("f: ℝ² → ℝ²: f(u, v) = [u², uv]")
    print("Composition: (f ∘ g)(x, y)")
    
    print("\nJacobian of g:")
    print("J_g = [ 1   1 ]")
    print("      [ y   x ]")
    
    print("\nJacobian of f (at g(x,y)):")
    print("J_f = [ 2u    0 ]")
    print("      [  v    u ]")
    
    print("\nChain rule: J_{f∘g} = J_f × J_g")
    
    x, y = 2.0, 3.0
    
    # g and its Jacobian
    u, v = x + y, x * y  # g(x, y)
    J_g = np.array([[1, 1], [y, x]])
    
    # f and its Jacobian at g(x,y)
    J_f = np.array([[2*u, 0], [v, u]])
    
    # Chain rule
    J_composition = J_f @ J_g
    
    print(f"\nAt (x, y) = ({x}, {y}):")
    print(f"g(x, y) = ({u}, {v})")
    print(f"J_g = \n{J_g}")
    print(f"J_f = \n{J_f}")
    print(f"J_f × J_g = \n{J_composition}")
    
    # Verify by direct computation
    def composition(v):
        x, y = v
        u = x + y
        w = x * y
        return np.array([u**2, u*w])
    
    print("\n--- Verification ---")
    v_in = np.array([x, y])
    h = 1e-7
    J_direct = np.zeros((2, 2))
    for j in range(2):
        v_plus = v_in.copy()
        v_plus[j] += h
        J_direct[:, j] = (composition(v_plus) - composition(v_in)) / h
    
    print(f"Direct Jacobian (numerical):\n{J_direct.round(4)}")


def example_hessian_basic():
    """Demonstrate basic Hessian computation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Basic Hessian Computation")
    print("=" * 60)
    
    print("f(x, y) = x³ - 2xy + y²")
    print("\nFirst derivatives:")
    print("∂f/∂x = 3x² - 2y")
    print("∂f/∂y = -2x + 2y")
    
    print("\nSecond derivatives:")
    print("∂²f/∂x² = 6x")
    print("∂²f/∂y² = 2")
    print("∂²f/∂x∂y = -2")
    
    print("\nHessian:")
    print("H = [ 6x   -2 ]")
    print("    [ -2    2 ]")
    
    def f(v):
        x, y = v
        return x**3 - 2*x*y + y**2
    
    def hessian_analytic(x, y):
        return np.array([
            [6*x, -2],
            [-2, 2]
        ])
    
    point = (1.0, 2.0)
    H = hessian_analytic(*point)
    
    print(f"\nAt point {point}:")
    print(f"H = \n{H}")
    
    # Check symmetry
    print(f"\nSymmetric? {np.allclose(H, H.T)}")
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvals(H)
    print(f"Eigenvalues: {eigenvalues}")


def example_critical_point_classification():
    """Classify critical points using Hessian."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Critical Point Classification")
    print("=" * 60)
    
    print("f(x, y) = x² - y²")
    print("\n∇f = [2x, -2y]ᵀ = 0 at (0, 0)")
    print("\nH = [ 2   0 ]")
    print("    [ 0  -2 ]")
    
    H = np.array([[2, 0], [0, -2]])
    eigenvalues = np.linalg.eigvals(H)
    
    print(f"\nEigenvalues: {eigenvalues}")
    print("One positive, one negative → SADDLE POINT")
    
    print("\n--- Another example ---")
    print("f(x, y) = x² + y²")
    print("H = [ 2   0 ]")
    print("    [ 0   2 ]")
    
    H2 = np.array([[2, 0], [0, 2]])
    eigenvalues2 = np.linalg.eigvals(H2)
    print(f"Eigenvalues: {eigenvalues2}")
    print("All positive → LOCAL MINIMUM")
    
    print("\n--- Third example ---")
    print("f(x, y) = -x² - y²")
    print("H = [ -2   0 ]")
    print("    [  0  -2 ]")
    
    H3 = np.array([[-2, 0], [0, -2]])
    eigenvalues3 = np.linalg.eigvals(H3)
    print(f"Eigenvalues: {eigenvalues3}")
    print("All negative → LOCAL MAXIMUM")


def example_newton_method():
    """Demonstrate Newton's method for optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Newton's Method")
    print("=" * 60)
    
    print("Minimize f(x, y) = (x - 1)² + 2(y - 2)²")
    print("∇f = [2(x-1), 4(y-2)]ᵀ")
    print("H = [ 2   0 ]  (constant)")
    print("    [ 0   4 ]")
    
    def f(v):
        x, y = v
        return (x - 1)**2 + 2*(y - 2)**2
    
    def grad_f(v):
        x, y = v
        return np.array([2*(x - 1), 4*(y - 2)])
    
    def hessian_f(v):
        return np.array([[2, 0], [0, 4]])
    
    x = np.array([0.0, 0.0])
    
    print(f"\nStarting point: {x}")
    
    for i in range(5):
        g = grad_f(x)
        H = hessian_f(x)
        
        # Newton step: x_new = x - H^(-1) * g
        delta = np.linalg.solve(H, g)
        x_new = x - delta
        
        print(f"Iter {i}: x = {x.round(4)}, f(x) = {f(x):.6f}, ||∇f|| = {np.linalg.norm(g):.6f}")
        x = x_new
    
    print(f"\nFinal: {x} (should be [1, 2])")
    print("\nNote: For quadratic functions, Newton converges in 1 step!")


def example_newton_rosenbrock():
    """Newton's method on Rosenbrock function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Newton's Method on Rosenbrock")
    print("=" * 60)
    
    print("Rosenbrock: f(x, y) = (1-x)² + 100(y-x²)²")
    
    def rosenbrock(v):
        x, y = v
        return (1 - x)**2 + 100*(y - x**2)**2
    
    def grad_rosenbrock(v):
        x, y = v
        dx = -2*(1 - x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return np.array([dx, dy])
    
    def hessian_rosenbrock(v):
        x, y = v
        h11 = 2 - 400*(y - x**2) + 800*x**2
        h12 = -400*x
        h22 = 200
        return np.array([[h11, h12], [h12, h22]])
    
    x = np.array([0.0, 0.0])
    
    print(f"\nStarting: x = {x}, f(x) = {rosenbrock(x)}")
    
    for i in range(20):
        g = grad_rosenbrock(x)
        H = hessian_rosenbrock(x)
        
        # Add regularization if Hessian is nearly singular
        H = H + 0.1 * np.eye(2)
        
        delta = np.linalg.solve(H, g)
        
        # Line search (damped Newton)
        alpha = 1.0
        while rosenbrock(x - alpha*delta) > rosenbrock(x) and alpha > 1e-10:
            alpha *= 0.5
        
        x = x - alpha * delta
        
        if i < 5 or i % 5 == 0:
            print(f"Iter {i}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {rosenbrock(x):.6f}")
    
    print(f"\nFinal: [{x[0]:.6f}, {x[1]:.6f}]")
    print("Minimum at: [1, 1]")


def example_softmax_jacobian():
    """Compute the softmax Jacobian."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Softmax Jacobian")
    print("=" * 60)
    
    print("Softmax: p_i = exp(z_i) / Σ exp(z_j)")
    print("\nJacobian: J_ij = ∂p_i/∂z_j = p_i(δ_ij - p_j)")
    print("         = diag(p) - p·pᵀ")
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
    def softmax_jacobian(p):
        """Compute Jacobian from softmax output p."""
        n = len(p)
        return np.diag(p) - np.outer(p, p)
    
    z = np.array([1.0, 2.0, 3.0])
    p = softmax(z)
    J = softmax_jacobian(p)
    
    print(f"\nInput z = {z}")
    print(f"Softmax p = {p.round(4)}")
    print(f"\nJacobian (3×3):\n{J.round(4)}")
    
    # Properties
    print("\nProperties:")
    print(f"  Symmetric: {np.allclose(J, J.T)}")
    print(f"  Row sums: {J.sum(axis=1).round(10)}")  # Should be 0
    print(f"  (Rows sum to 0 because Σp_i = 1, so Σ ∂p_i/∂z_j = 0)")
    
    # Numerical verification
    h = 1e-7
    J_numerical = np.zeros((3, 3))
    for j in range(3):
        z_plus = z.copy()
        z_plus[j] += h
        J_numerical[:, j] = (softmax(z_plus) - softmax(z)) / h
    
    print(f"\nNumerical Jacobian:\n{J_numerical.round(4)}")


def example_hessian_eigenvalues():
    """Analyze loss landscape via Hessian eigenvalues."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Loss Landscape Analysis via Hessian")
    print("=" * 60)
    
    print("f(x, y) = x⁴ + y⁴ - 4xy + 1")
    print("(This has interesting critical points)")
    
    def f(v):
        x, y = v
        return x**4 + y**4 - 4*x*y + 1
    
    def grad_f(v):
        x, y = v
        return np.array([4*x**3 - 4*y, 4*y**3 - 4*x])
    
    def hessian_f(v):
        x, y = v
        return np.array([
            [12*x**2, -4],
            [-4, 12*y**2]
        ])
    
    # Find and classify critical points
    critical_points = [(0, 0), (1, 1), (-1, -1)]
    
    print("\nCritical Point Analysis:")
    for point in critical_points:
        x = np.array(point)
        f_val = f(x)
        g = grad_f(x)
        H = hessian_f(x)
        eigenvalues = np.linalg.eigvals(H)
        
        # Classification
        if all(eigenvalues > 0):
            point_type = "Local Minimum"
        elif all(eigenvalues < 0):
            point_type = "Local Maximum"
        else:
            point_type = "Saddle Point"
        
        print(f"\n  Point {point}:")
        print(f"    f(x) = {f_val:.4f}")
        print(f"    ||∇f|| = {np.linalg.norm(g):.6f}")
        print(f"    Eigenvalues: {eigenvalues.round(4)}")
        print(f"    Type: {point_type}")


def example_quadratic_form_hessian():
    """Hessian of quadratic form."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Hessian of Quadratic Form")
    print("=" * 60)
    
    print("For f(x) = xᵀAx + bᵀx + c")
    print("\n∇f = (A + Aᵀ)x + b")
    print("H = A + Aᵀ")
    
    print("\nIf A is symmetric:")
    print("∇f = 2Ax + b")
    print("H = 2A")
    
    # Example
    A = np.array([[2, 1], [1, 3]])
    b = np.array([1, 2])
    c = 5
    
    print(f"\nA = \n{A}")
    print(f"b = {b}")
    
    def f(x):
        return x @ A @ x + b @ x + c
    
    # Analytical Hessian
    H_analytic = A + A.T
    
    print(f"\nAnalytical H = A + Aᵀ = \n{H_analytic}")
    
    # Numerical Hessian
    x = np.array([1.0, 2.0])
    h = 1e-5
    H_numerical = np.zeros((2, 2))
    
    for i in range(2):
        for j in range(2):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            H_numerical[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h*h)
    
    print(f"\nNumerical H = \n{H_numerical.round(4)}")


def example_backprop_jacobian():
    """Backpropagation as Jacobian multiplication."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Backpropagation as Jacobian Products")
    print("=" * 60)
    
    print("Two-layer network: x → h → y → L")
    print("h = σ(W₁x + b₁)")
    print("y = W₂h + b₂")
    print("L = ||y - t||²")
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Dimensions
    n_input = 3
    n_hidden = 4
    n_output = 2
    
    # Initialize
    np.random.seed(42)
    x = np.random.randn(n_input)
    t = np.random.randn(n_output)
    W1 = np.random.randn(n_hidden, n_input) * 0.1
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_output, n_hidden) * 0.1
    b2 = np.zeros(n_output)
    
    # Forward pass
    z1 = W1 @ x + b1
    h = sigmoid(z1)
    z2 = W2 @ h + b2
    y = z2  # Linear output
    L = np.sum((y - t)**2)
    
    print(f"\nForward pass:")
    print(f"  x shape: {x.shape}")
    print(f"  h shape: {h.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Loss L = {L:.4f}")
    
    # Backward pass (using Jacobians)
    print("\nBackward pass using Jacobians:")
    
    # ∂L/∂y
    dL_dy = 2 * (y - t)  # Shape: (n_output,)
    print(f"  ∂L/∂y shape: {dL_dy.shape}")
    
    # Jacobian of y = W₂h w.r.t. h is W₂ (n_output × n_hidden)
    # ∂L/∂h = (∂y/∂h)ᵀ × ∂L/∂y = W₂ᵀ × ∂L/∂y
    J_y_h = W2  # Shape: (n_output, n_hidden)
    dL_dh = J_y_h.T @ dL_dy  # Shape: (n_hidden,)
    print(f"  J_y_h shape: {J_y_h.shape}")
    print(f"  ∂L/∂h shape: {dL_dh.shape}")
    
    # Jacobian of h = σ(z₁) is diag(σ'(z₁)) (n_hidden × n_hidden)
    sigma_prime = h * (1 - h)  # Diagonal elements
    dL_dz1 = dL_dh * sigma_prime  # Element-wise (implicit diag mult)
    print(f"  ∂L/∂z₁ shape: {dL_dz1.shape}")
    
    # Gradients for weights
    dL_dW2 = np.outer(dL_dy, h)  # Shape: (n_output, n_hidden)
    dL_db2 = dL_dy
    dL_dW1 = np.outer(dL_dz1, x)  # Shape: (n_hidden, n_input)
    dL_db1 = dL_dz1
    
    print(f"\nGradient shapes:")
    print(f"  ∂L/∂W₂: {dL_dW2.shape}")
    print(f"  ∂L/∂W₁: {dL_dW1.shape}")


def example_numerical_hessian():
    """Numerical Hessian computation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Numerical Hessian Computation")
    print("=" * 60)
    
    def f(v):
        x, y = v
        return np.sin(x*y) + x**2 + y**3
    
    def numerical_hessian(f, x, h=1e-5):
        n = len(x)
        H = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
                
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
        
        return H
    
    x = np.array([1.0, 2.0])
    H = numerical_hessian(f, x)
    
    print(f"f(x, y) = sin(xy) + x² + y³")
    print(f"\nAt point {x}:")
    print(f"Numerical Hessian:\n{H.round(4)}")
    
    # Analytical for comparison
    # ∂f/∂x = y·cos(xy) + 2x
    # ∂f/∂y = x·cos(xy) + 3y²
    # ∂²f/∂x² = -y²·sin(xy) + 2
    # ∂²f/∂y² = -x²·sin(xy) + 6y
    # ∂²f/∂x∂y = cos(xy) - xy·sin(xy)
    
    x_val, y_val = x
    H_analytic = np.array([
        [-y_val**2 * np.sin(x_val*y_val) + 2, 
         np.cos(x_val*y_val) - x_val*y_val*np.sin(x_val*y_val)],
        [np.cos(x_val*y_val) - x_val*y_val*np.sin(x_val*y_val), 
         -x_val**2 * np.sin(x_val*y_val) + 6*y_val]
    ])
    
    print(f"\nAnalytical Hessian:\n{H_analytic.round(4)}")
    print(f"\nMatch: {np.allclose(H, H_analytic, atol=1e-4)}")


if __name__ == "__main__":
    example_jacobian_basic()
    example_jacobian_chain_rule()
    example_hessian_basic()
    example_critical_point_classification()
    example_newton_method()
    example_newton_rosenbrock()
    example_softmax_jacobian()
    example_hessian_eigenvalues()
    example_quadratic_form_hessian()
    example_backprop_jacobian()
    example_numerical_hessian()
