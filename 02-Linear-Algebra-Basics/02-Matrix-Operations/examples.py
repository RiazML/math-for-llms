"""
Matrix Operations - Examples
============================
Practical demonstrations of matrix operations using NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt


def example_matrix_creation():
    """Different ways to create matrices in NumPy."""
    print("=" * 60)
    print("EXAMPLE 1: Matrix Creation")
    print("=" * 60)
    
    # From lists
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"From list:\n{A}")
    print(f"Shape: {A.shape}")  # (2, 3)
    
    # Special matrices
    zeros = np.zeros((3, 4))
    ones = np.ones((2, 3))
    identity = np.eye(4)
    diagonal = np.diag([1, 2, 3, 4])
    
    print(f"\nZeros (3×4):\n{zeros}")
    print(f"\nOnes (2×3):\n{ones}")
    print(f"\nIdentity (4×4):\n{identity}")
    print(f"\nDiagonal:\n{diagonal}")
    
    # Random matrices
    random_uniform = np.random.rand(2, 3)  # Uniform [0, 1)
    random_normal = np.random.randn(2, 3)  # Standard normal
    random_int = np.random.randint(0, 10, (2, 3))  # Integers
    
    print(f"\nRandom uniform:\n{random_uniform}")
    print(f"\nRandom normal:\n{random_normal}")
    print(f"\nRandom integers:\n{random_int}")
    
    # Useful initializations for ML
    xavier = np.random.randn(3, 4) * np.sqrt(2 / (3 + 4))  # Xavier initialization
    print(f"\nXavier initialization:\n{xavier}")


def example_basic_operations():
    """Matrix addition, subtraction, and scalar multiplication."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Basic Operations")
    print("=" * 60)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"A:\n{A}")
    print(f"\nB:\n{B}")
    
    # Addition
    print(f"\nA + B:\n{A + B}")
    
    # Subtraction
    print(f"\nA - B:\n{A - B}")
    
    # Scalar multiplication
    print(f"\n3 * A:\n{3 * A}")
    
    # Scalar addition (broadcasts)
    print(f"\nA + 10:\n{A + 10}")
    
    # Verify properties
    C = np.array([[1, 1], [1, 1]])
    print(f"\nCommutative: A + B == B + A: {np.allclose(A + B, B + A)}")
    print(f"Associative: (A + B) + C == A + (B + C): {np.allclose((A + B) + C, A + (B + C))}")


def example_matrix_multiplication():
    """Matrix multiplication vs element-wise multiplication."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Matrix Multiplication")
    print("=" * 60)
    
    A = np.array([[1, 2],
                  [3, 4]])  # 2×2
    
    B = np.array([[5, 6],
                  [7, 8]])  # 2×2
    
    print(f"A (2×2):\n{A}")
    print(f"\nB (2×2):\n{B}")
    
    # Matrix multiplication (3 equivalent ways)
    C1 = A @ B
    C2 = np.matmul(A, B)
    C3 = np.dot(A, B)
    
    print(f"\nMatrix multiplication (A @ B):\n{C1}")
    print(f"All methods equal: {np.allclose(C1, C2) and np.allclose(C2, C3)}")
    
    # Element-wise multiplication (Hadamard product)
    H = A * B
    print(f"\nElement-wise multiplication (A * B):\n{H}")
    
    # Demonstrate non-commutativity
    print(f"\nA @ B:\n{A @ B}")
    print(f"\nB @ A:\n{B @ A}")
    print(f"\nA @ B == B @ A: {np.allclose(A @ B, B @ A)}")
    
    # Different dimensions
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2×3
    
    W = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])  # 3×2
    
    result = X @ W  # (2×3) @ (3×2) = (2×2)
    print(f"\nX (2×3) @ W (3×2) = (2×2):\n{result}")


def example_transpose():
    """Matrix transpose operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Transpose")
    print("=" * 60)
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])  # 2×3
    
    print(f"A (2×3):\n{A}")
    print(f"\nAᵀ (3×2):\n{A.T}")
    
    # Different ways to transpose
    print(f"\nnp.transpose(A):\n{np.transpose(A)}")
    print(f"\nA.T:\n{A.T}")
    
    # Verify (Aᵀ)ᵀ = A
    print(f"\n(Aᵀ)ᵀ = A: {np.allclose(A.T.T, A)}")
    
    # (AB)ᵀ = BᵀAᵀ
    B = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])  # 3×2
    
    print(f"\n(A @ B)ᵀ:\n{(A @ B).T}")
    print(f"\nBᵀ @ Aᵀ:\n{B.T @ A.T}")
    print(f"\n(AB)ᵀ == BᵀAᵀ: {np.allclose((A @ B).T, B.T @ A.T)}")
    
    # Symmetric matrix
    S = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])
    print(f"\nSymmetric matrix S:\n{S}")
    print(f"S == Sᵀ: {np.allclose(S, S.T)}")


def example_special_matrices():
    """Working with special types of matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Special Matrices")
    print("=" * 60)
    
    # Symmetric matrix from any matrix
    A = np.random.randn(3, 3)
    symmetric = (A + A.T) / 2
    print(f"Creating symmetric matrix from random:\n{symmetric}")
    print(f"Is symmetric: {np.allclose(symmetric, symmetric.T)}")
    
    # Diagonal matrix operations
    diag_matrix = np.diag([1, 2, 3])
    print(f"\nDiagonal matrix:\n{diag_matrix}")
    
    # Extract diagonal
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(f"\nExtract diagonal of A: {np.diag(A)}")
    
    # Upper and lower triangular
    print(f"\nUpper triangular:\n{np.triu(A)}")
    print(f"\nLower triangular:\n{np.tril(A)}")
    
    # Trace (sum of diagonal)
    print(f"\nTrace of A: {np.trace(A)}")
    
    # Positive semi-definite matrix (XᵀX is always PSD)
    X = np.random.randn(5, 3)
    psd = X.T @ X
    eigenvalues = np.linalg.eigvals(psd)
    print(f"\nPSD matrix (XᵀX):\n{psd}")
    print(f"Eigenvalues (all ≥ 0): {eigenvalues}")


def example_broadcasting():
    """Matrix broadcasting operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Broadcasting")
    print("=" * 60)
    
    # Matrix + scalar
    A = np.array([[1, 2], [3, 4]])
    print(f"A:\n{A}")
    print(f"\nA + 10:\n{A + 10}")
    
    # Matrix + row vector
    row_vec = np.array([10, 20])
    print(f"\nRow vector: {row_vec}")
    print(f"A + row_vec:\n{A + row_vec}")  # Adds to each row
    
    # Matrix + column vector
    col_vec = np.array([[100], [200]])
    print(f"\nColumn vector:\n{col_vec}")
    print(f"A + col_vec:\n{A + col_vec}")  # Adds to each column
    
    # Practical: Mean normalization
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    row_means = X.mean(axis=1, keepdims=True)
    col_means = X.mean(axis=0, keepdims=True)
    
    print(f"\nOriginal X:\n{X}")
    print(f"\nRow means: {row_means.flatten()}")
    print(f"X - row_means (center each row):\n{X - row_means}")
    
    print(f"\nColumn means: {col_means.flatten()}")
    print(f"X - col_means (center each column):\n{X - col_means}")


def example_indexing_slicing():
    """Matrix indexing and slicing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Indexing and Slicing")
    print("=" * 60)
    
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    
    print(f"Matrix A:\n{A}")
    
    # Single element
    print(f"\nElement [1, 2]: {A[1, 2]}")
    
    # Row
    print(f"Row 0: {A[0, :]}")
    print(f"Row 1: {A[1]}")  # Shorthand
    
    # Column
    print(f"Column 2: {A[:, 2]}")
    
    # Submatrix
    print(f"\nSubmatrix [0:2, 1:3]:\n{A[0:2, 1:3]}")
    
    # Boolean indexing
    print(f"\nElements > 5: {A[A > 5]}")
    
    # Fancy indexing
    rows = [0, 2]
    cols = [1, 3]
    print(f"\nElements at (0,1) and (2,3): {A[rows, cols]}")
    
    # Reshaping
    B = A.reshape(2, 6)
    print(f"\nReshaped to (2, 6):\n{B}")
    
    C = A.flatten()
    print(f"\nFlattened: {C}")


def example_ml_linear_layer():
    """Simulating a neural network linear layer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: ML Application - Linear Layer")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Batch of 4 samples, each with 3 features
    X = np.random.randn(4, 3)
    print(f"Input X (batch_size=4, features=3):\n{X}")
    
    # Weight matrix: 3 input features → 2 output features
    W = np.random.randn(3, 2) * 0.1
    b = np.zeros(2)
    
    print(f"\nWeight W (3×2):\n{W}")
    print(f"Bias b: {b}")
    
    # Forward pass: Y = XW + b
    Y = X @ W + b
    print(f"\nOutput Y = XW + b (4×2):\n{Y}")
    
    # With activation (ReLU)
    Y_relu = np.maximum(0, Y)
    print(f"\nAfter ReLU:\n{Y_relu}")
    
    # Multiple layers
    W1 = np.random.randn(3, 4) * 0.1
    W2 = np.random.randn(4, 2) * 0.1
    
    # Layer 1
    H = np.maximum(0, X @ W1)
    # Layer 2
    output = H @ W2
    
    print(f"\nTwo-layer network output:\n{output}")


def example_covariance_matrix():
    """Computing covariance matrix from data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Covariance Matrix")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate correlated data
    # True covariance structure
    true_cov = np.array([[1.0, 0.8],
                         [0.8, 1.0]])
    
    # Generate samples using Cholesky decomposition
    n_samples = 1000
    L = np.linalg.cholesky(true_cov)
    Z = np.random.randn(n_samples, 2)
    X = Z @ L.T
    
    print(f"Data shape: {X.shape}")
    print(f"True covariance:\n{true_cov}")
    
    # Compute sample covariance
    # Method 1: Manual
    X_centered = X - X.mean(axis=0)
    cov_manual = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Method 2: NumPy
    cov_numpy = np.cov(X.T)
    
    print(f"\nSample covariance (manual):\n{cov_manual}")
    print(f"\nSample covariance (NumPy):\n{cov_numpy}")
    print(f"\nMethods equal: {np.allclose(cov_manual, cov_numpy)}")
    
    # Covariance matrix properties
    print(f"\nIs symmetric: {np.allclose(cov_numpy, cov_numpy.T)}")
    eigenvalues = np.linalg.eigvals(cov_numpy)
    print(f"Eigenvalues (all ≥ 0): {eigenvalues}")


def example_gram_matrix():
    """Computing and using Gram matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Gram Matrix")
    print("=" * 60)
    
    # Feature vectors
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])  # 3 samples, 2 features
    
    print(f"Feature matrix X:\n{X}")
    
    # Gram matrix: XᵀX (feature-feature similarity)
    gram_features = X.T @ X
    print(f"\nGram matrix XᵀX (feature similarities):\n{gram_features}")
    
    # Gram matrix: XXᵀ (sample-sample similarity)
    gram_samples = X @ X.T
    print(f"\nGram matrix XXᵀ (sample similarities):\n{gram_samples}")
    
    # This is related to the kernel matrix
    # gram_samples[i,j] = x_i · x_j (linear kernel)
    
    # Verify
    print(f"\nXXᵀ[0,1] = {gram_samples[0, 1]}")
    print(f"X[0] · X[1] = {np.dot(X[0], X[1])}")
    
    # Application: Style Transfer (simplified)
    # In style transfer, Gram matrix captures texture information
    activation = np.random.randn(4, 16)  # 4 channels, 16 spatial locations
    style_matrix = activation @ activation.T / 16
    print(f"\nStyle matrix shape: {style_matrix.shape}")


def visualize_matrix_multiplication():
    """Visualize how matrix multiplication transforms space."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Matrix Transformation")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original points (unit square)
    square = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1],
                       [0, 0]]).T
    
    # Grid points
    x = np.linspace(-0.5, 1.5, 20)
    y = np.linspace(-0.5, 1.5, 20)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    
    transformations = [
        (np.array([[2, 0], [0, 2]]), "Scaling"),
        (np.array([[1, 0.5], [0, 1]]), "Shear"),
        (np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                   [np.sin(np.pi/4), np.cos(np.pi/4)]]), "Rotation (45°)")
    ]
    
    for ax, (T, title) in zip(axes, transformations):
        # Transform
        transformed_square = T @ square
        transformed_grid = T @ grid
        
        # Plot
        ax.scatter(grid[0], grid[1], c='lightblue', s=5, alpha=0.3)
        ax.scatter(transformed_grid[0], transformed_grid[1], c='lightcoral', s=5, alpha=0.3)
        
        ax.plot(square[0], square[1], 'b-', linewidth=2, label='Original')
        ax.plot(transformed_square[0], transformed_square[1], 'r-', linewidth=2, label='Transformed')
        
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title}\nT = {T.tolist()}")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('matrix_transformations.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: matrix_transformations.png")


if __name__ == "__main__":
    example_matrix_creation()
    example_basic_operations()
    example_matrix_multiplication()
    example_transpose()
    example_special_matrices()
    example_broadcasting()
    example_indexing_slicing()
    example_ml_linear_layer()
    example_covariance_matrix()
    example_gram_matrix()
    
    # Uncomment to generate visualization
    # visualize_matrix_multiplication()
