"""
Principal Component Analysis (PCA) - Exercises
===============================================
Practice problems for PCA concepts.
"""

import numpy as np
from numpy.linalg import eig, svd, norm


class PCAExercises:
    """Exercises for Principal Component Analysis."""
    
    def exercise_1_compute_pca_manual(self):
        """
        Exercise 1: Compute PCA Manually
        
        For data X = [[1, 2], [3, 4], [5, 6]]:
        a) Center the data
        b) Compute the covariance matrix
        c) Find eigenvalues and eigenvectors
        d) Identify the principal components
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Compute PCA Manually")
        
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=float)
        
        print(f"Original data X:\n{X}")
        
        print("\na) Center the data:")
        mean = X.mean(axis=0)
        X_centered = X - mean
        print(f"   Mean: {mean}")
        print(f"   Centered X:\n{X_centered}")
        
        print("\nb) Covariance matrix:")
        n = X.shape[0]
        cov = (X_centered.T @ X_centered) / (n - 1)
        print(f"   C = XᵀX / (n-1) =\n{cov}")
        
        print("\nc) Eigenvalues and eigenvectors:")
        eigenvalues, eigenvectors = eig(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        
        print(f"   Eigenvalues: {eigenvalues}")
        print(f"   Eigenvectors:\n{eigenvectors}")
        
        print("\nd) Principal components:")
        print(f"   PC1 (direction): {eigenvectors[:, 0]} (variance: {eigenvalues[0]:.4f})")
        print(f"   PC2 (direction): {eigenvectors[:, 1]} (variance: {eigenvalues[1]:.4f})")
        
        # Note: This data lies on a line, so PC2 has ~0 variance
        print("\n   Note: Data lies on a line, so PC2 has ~0 variance")
    
    def exercise_2_pca_via_svd(self):
        """
        Exercise 2: PCA via SVD
        
        For the same data X = [[1, 2], [3, 4], [5, 6]]:
        a) Compute SVD of centered data
        b) Extract principal components from V
        c) Compute variances from singular values
        d) Verify they match eigenvalues
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: PCA via SVD")
        
        X = np.array([[1, 2],
                      [3, 4],
                      [5, 6]], dtype=float)
        
        X_centered = X - X.mean(axis=0)
        n = X.shape[0]
        
        print("a) SVD of centered data:")
        U, s, Vt = svd(X_centered, full_matrices=False)
        print(f"   Singular values: {s}")
        print(f"   V (right singular vectors):\n{Vt.T}")
        
        print("\nb) Principal components from V:")
        print(f"   PC1: {Vt[0, :]}")
        print(f"   PC2: {Vt[1, :]}")
        
        print("\nc) Variances from singular values:")
        variances_svd = s**2 / (n - 1)
        print(f"   σ²/(n-1) = {variances_svd}")
        
        print("\nd) Verification with eigenvalues:")
        cov = (X_centered.T @ X_centered) / (n - 1)
        eigenvalues, _ = eig(cov)
        eigenvalues = np.sort(eigenvalues.real)[::-1]
        print(f"   Eigenvalues: {eigenvalues}")
        print(f"   Match: {np.allclose(variances_svd, eigenvalues)}")
    
    def exercise_3_variance_explained(self):
        """
        Exercise 3: Variance Explained
        
        Given eigenvalues λ = [5.0, 2.0, 1.0, 0.5, 0.3]:
        a) Compute total variance
        b) Compute variance ratio for each component
        c) Compute cumulative variance
        d) How many components for 90% variance?
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Variance Explained")
        
        eigenvalues = np.array([5.0, 2.0, 1.0, 0.5, 0.3])
        
        print(f"Eigenvalues: {eigenvalues}")
        
        print("\na) Total variance:")
        total_var = np.sum(eigenvalues)
        print(f"   Σλᵢ = {total_var}")
        
        print("\nb) Variance ratio for each component:")
        var_ratio = eigenvalues / total_var
        for i, (lam, ratio) in enumerate(zip(eigenvalues, var_ratio)):
            print(f"   PC{i+1}: λ={lam}, ratio={100*ratio:.1f}%")
        
        print("\nc) Cumulative variance:")
        cumsum = np.cumsum(var_ratio)
        for i, cum in enumerate(cumsum):
            print(f"   First {i+1} components: {100*cum:.1f}%")
        
        print("\nd) Components for 90% variance:")
        k = np.searchsorted(cumsum, 0.90) + 1
        print(f"   k = {k} components (cumulative: {100*cumsum[k-1]:.1f}%)")
    
    def exercise_4_projection_reconstruction(self):
        """
        Exercise 4: Projection and Reconstruction
        
        For X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]:
        a) Center the data
        b) Compute PCA
        c) Project to 2D
        d) Reconstruct and compute error
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Projection and Reconstruction")
        
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12]], dtype=float)
        
        print(f"Original X ({X.shape}):\n{X}")
        
        print("\na) Center the data:")
        mean = X.mean(axis=0)
        X_centered = X - mean
        print(f"   Mean: {mean}")
        
        print("\nb) PCA via SVD:")
        U, s, Vt = svd(X_centered, full_matrices=False)
        print(f"   Singular values: {s}")
        var_ratio = s**2 / np.sum(s**2)
        print(f"   Variance ratios: {np.round(100*var_ratio, 2)}%")
        
        print("\nc) Project to 2D (k=2):")
        k = 2
        V_k = Vt[:k, :].T
        Z = X_centered @ V_k
        print(f"   Z (projected, shape {Z.shape}):\n{np.round(Z, 4)}")
        
        print("\nd) Reconstruct:")
        X_reconstructed = Z @ V_k.T + mean
        print(f"   Reconstructed:\n{np.round(X_reconstructed, 4)}")
        
        error = norm(X - X_reconstructed, 'fro')
        rel_error = error / norm(X, 'fro')
        print(f"\n   Reconstruction error: {error:.6f}")
        print(f"   Relative error: {100*rel_error:.4f}%")
        print(f"   Variance captured: {100*np.sum(var_ratio[:k]):.2f}%")
    
    def exercise_5_standardization(self):
        """
        Exercise 5: Effect of Standardization
        
        For X with features on different scales:
        Feature 1: mean=100, std=20
        Feature 2: mean=5, std=0.5
        
        a) Generate such data (n=100)
        b) Apply PCA without standardization
        c) Apply PCA with standardization
        d) Compare the results
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Effect of Standardization")
        
        np.random.seed(42)
        n = 100
        
        print("a) Generate data with different scales:")
        X1 = 100 + 20 * np.random.randn(n)  # Large scale
        X2 = 5 + 0.5 * np.random.randn(n)   # Small scale
        X = np.column_stack([X1, X2])
        
        print(f"   Feature 1: mean={X[:, 0].mean():.1f}, std={X[:, 0].std():.1f}")
        print(f"   Feature 2: mean={X[:, 1].mean():.2f}, std={X[:, 1].std():.2f}")
        
        print("\nb) PCA without standardization:")
        X_centered = X - X.mean(axis=0)
        U, s, Vt = svd(X_centered, full_matrices=False)
        var_ratio = s**2 / np.sum(s**2)
        
        print(f"   Variance ratios: {np.round(100*var_ratio, 2)}%")
        print(f"   PC1 direction: {np.round(Vt[0, :], 4)}")
        print("   (PC1 mostly aligned with feature 1 due to scale)")
        
        print("\nc) PCA with standardization:")
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        U_std, s_std, Vt_std = svd(X_std, full_matrices=False)
        var_ratio_std = s_std**2 / np.sum(s_std**2)
        
        print(f"   Variance ratios: {np.round(100*var_ratio_std, 2)}%")
        print(f"   PC1 direction: {np.round(Vt_std[0, :], 4)}")
        print("   (More balanced contribution from both features)")
        
        print("\nd) Comparison:")
        print("   Without standardization: Large-scale feature dominates")
        print("   With standardization: Equal importance to both features")
        print("   Choose based on whether scale differences are meaningful")
    
    def exercise_6_choosing_k(self):
        """
        Exercise 6: Choosing Number of Components
        
        Given data with singular values s = [10, 8, 5, 2, 1, 0.5, 0.2, 0.1]:
        
        a) Plot scree diagram (describe)
        b) Apply 90% variance rule
        c) Apply 95% variance rule
        d) Apply Kaiser criterion (keep λ > 1)
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Choosing Number of Components")
        
        s = np.array([10, 8, 5, 2, 1, 0.5, 0.2, 0.1])
        n = 100  # Assume 100 samples
        
        # Convert to eigenvalues (variances)
        eigenvalues = s**2 / (n - 1)
        var_ratio = eigenvalues / np.sum(eigenvalues)
        cumsum = np.cumsum(var_ratio)
        
        print(f"Singular values: {s}")
        print(f"Eigenvalues (λ = σ²/(n-1)): {np.round(eigenvalues, 4)}")
        print(f"Variance ratios: {np.round(100*var_ratio, 2)}%")
        print(f"Cumulative: {np.round(100*cumsum, 2)}%")
        
        print("\na) Scree diagram (values):")
        print("   Component | Eigenvalue | Cum. Var%")
        for i, (lam, cum) in enumerate(zip(eigenvalues, cumsum)):
            bar = "█" * int(lam * 5)
            print(f"   {i+1:9d} | {lam:10.4f} | {100*cum:6.2f}% {bar}")
        print("   → Elbow appears around component 3-4")
        
        print("\nb) 90% variance rule:")
        k_90 = np.searchsorted(cumsum, 0.90) + 1
        print(f"   k = {k_90} (cumulative: {100*cumsum[k_90-1]:.2f}%)")
        
        print("\nc) 95% variance rule:")
        k_95 = np.searchsorted(cumsum, 0.95) + 1
        print(f"   k = {k_95} (cumulative: {100*cumsum[k_95-1]:.2f}%)")
        
        print("\nd) Kaiser criterion (λ > 1):")
        k_kaiser = np.sum(eigenvalues > 1)
        print(f"   Components with λ > 1: {np.where(eigenvalues > 1)[0] + 1}")
        print(f"   k = {k_kaiser}")
    
    def exercise_7_pca_classification(self):
        """
        Exercise 7: PCA for Classification Preprocessing
        
        Simulate a 2-class dataset with 20 features.
        a) Generate data where only 3 features are informative
        b) Apply PCA
        c) Check if top components capture class separation
        d) Compare class means in original vs PC space
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: PCA for Classification")
        
        np.random.seed(42)
        n_per_class = 50
        d = 20
        
        print("a) Generate 2-class data (3 informative features):")
        # Class 0: centered at origin
        X0 = np.random.randn(n_per_class, d)
        
        # Class 1: shifted in first 3 dimensions
        X1 = np.random.randn(n_per_class, d)
        X1[:, 0] += 2  # Informative
        X1[:, 1] += 1.5  # Informative
        X1[:, 2] += 1  # Informative
        
        X = np.vstack([X0, X1])
        y = np.array([0]*n_per_class + [1]*n_per_class)
        
        print(f"   Data shape: {X.shape}")
        print(f"   Class 0: {n_per_class} samples")
        print(f"   Class 1: {n_per_class} samples")
        
        print("\nb) Apply PCA:")
        X_centered = X - X.mean(axis=0)
        U, s, Vt = svd(X_centered, full_matrices=False)
        var_ratio = s**2 / np.sum(s**2)
        
        print(f"   Top 5 variance ratios: {np.round(100*var_ratio[:5], 2)}%")
        
        print("\nc) Check class separation in PC space:")
        Z = X_centered @ Vt.T
        
        for k in [1, 2, 3]:
            Z_k = Z[:, :k]
            
            # Compute class means
            mean_0 = Z_k[y == 0].mean(axis=0)
            mean_1 = Z_k[y == 1].mean(axis=0)
            
            # Distance between class means
            dist = norm(mean_1 - mean_0)
            print(f"\n   Using {k} PC(s):")
            print(f"   Class 0 mean: {np.round(mean_0, 3)}")
            print(f"   Class 1 mean: {np.round(mean_1, 3)}")
            print(f"   Distance: {dist:.4f}")
        
        print("\nd) Comparison:")
        # Original space - just first 3 features
        mean_0_orig = X[y == 0, :3].mean(axis=0)
        mean_1_orig = X[y == 1, :3].mean(axis=0)
        dist_orig = norm(mean_1_orig - mean_0_orig)
        print(f"\n   Original (first 3 features) distance: {dist_orig:.4f}")
        print("   PCA automatically finds directions with class separation")
    
    def exercise_8_reconstruction_error(self):
        """
        Exercise 8: Reconstruction Error Analysis
        
        For a 100×10 data matrix:
        a) Compute reconstruction error for k = 1, 2, ..., 10
        b) Verify error = sum of discarded eigenvalues
        c) Plot error vs k
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Reconstruction Error")
        
        np.random.seed(42)
        n, d = 100, 10
        
        # Generate structured data
        X = np.random.randn(n, d)
        X[:, 0] *= 5  # High variance
        X[:, 1] *= 3
        X[:, 2] *= 2
        
        X_centered = X - X.mean(axis=0)
        U, s, Vt = svd(X_centered, full_matrices=False)
        
        eigenvalues = s**2 / (n - 1)
        
        print("a) Reconstruction error for each k:")
        print(f"{'k':>3} {'Error':>12} {'Σλ_discarded':>14} {'Match':>8}")
        
        errors = []
        for k in range(1, d + 1):
            # Reconstruct with k components
            X_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            
            # Compute error
            error = norm(X_centered - X_k, 'fro')**2
            errors.append(error)
            
            # Theoretical error
            discarded = np.sum(s[k:]**2)
            
            print(f"{k:3d} {error:12.4f} {discarded:14.4f} {np.isclose(error, discarded):>8}")
        
        print("\nb) Verification:")
        print("   Reconstruction error = Σσᵢ² for i > k")
        print("   (Sum of squared singular values of discarded components)")
        
        print("\nc) Error vs k (decreasing):")
        for k, err in enumerate(errors, 1):
            bar = "█" * int(err / max(errors) * 20)
            print(f"   k={k:2d}: {err:10.2f} {bar}")
    
    def exercise_9_pca_vs_random(self):
        """
        Exercise 9: PCA vs Random Projection
        
        Compare dimensionality reduction using:
        a) PCA (optimal)
        b) Random projection
        
        For the same target dimension k, compare:
        - Reconstruction error
        - Distance preservation
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: PCA vs Random Projection")
        
        np.random.seed(42)
        n, d = 100, 50
        k = 5  # Target dimension
        
        # Generate data
        X = np.random.randn(n, d)
        X[:, :5] *= 3  # Some features more important
        
        X_centered = X - X.mean(axis=0)
        
        print(f"Data: {n} samples × {d} features → {k} dimensions")
        
        print("\na) PCA projection:")
        U, s, Vt = svd(X_centered, full_matrices=False)
        V_pca = Vt[:k, :].T
        Z_pca = X_centered @ V_pca
        X_pca_reconstructed = Z_pca @ V_pca.T
        error_pca = norm(X_centered - X_pca_reconstructed, 'fro')
        print(f"   Reconstruction error: {error_pca:.4f}")
        
        print("\nb) Random projection:")
        # Random projection matrix
        R = np.random.randn(d, k) / np.sqrt(k)  # Normalized
        Z_random = X_centered @ R
        X_random_reconstructed = Z_random @ R.T  # Approximate inverse
        error_random = norm(X_centered - X_random_reconstructed, 'fro')
        print(f"   Reconstruction error: {error_random:.4f}")
        
        print(f"\n   PCA error is {error_random/error_pca:.1f}× smaller")
        
        print("\nc) Distance preservation:")
        # Sample pairs for distance comparison
        n_pairs = 100
        idx1 = np.random.randint(0, n, n_pairs)
        idx2 = np.random.randint(0, n, n_pairs)
        
        # Original distances
        orig_dists = np.array([norm(X_centered[i] - X_centered[j]) 
                               for i, j in zip(idx1, idx2)])
        
        # PCA distances
        pca_dists = np.array([norm(Z_pca[i] - Z_pca[j]) 
                              for i, j in zip(idx1, idx2)])
        
        # Random projection distances
        random_dists = np.array([norm(Z_random[i] - Z_random[j]) 
                                 for i, j in zip(idx1, idx2)])
        
        # Correlation between original and reduced distances
        corr_pca = np.corrcoef(orig_dists, pca_dists)[0, 1]
        corr_random = np.corrcoef(orig_dists, random_dists)[0, 1]
        
        print(f"   Distance correlation (PCA): {corr_pca:.4f}")
        print(f"   Distance correlation (Random): {corr_random:.4f}")
        print("\n   PCA preserves distances better due to optimal projection")
    
    def exercise_10_interpret_components(self):
        """
        Exercise 10: Interpreting Principal Components
        
        For a dataset with known feature meanings:
        Features: [height, weight, age, income, education_years]
        
        a) Generate synthetic data
        b) Compute PCA
        c) Interpret the loadings (what does each PC represent?)
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Interpreting Components")
        
        np.random.seed(42)
        n = 200
        
        features = ['height', 'weight', 'age', 'income', 'education']
        
        print("a) Generate synthetic data:")
        # Height (cm): correlated with weight
        height = 170 + 10 * np.random.randn(n)
        
        # Weight (kg): correlated with height
        weight = 0.5 * height - 15 + 5 * np.random.randn(n)
        
        # Age: independent
        age = 35 + 15 * np.random.randn(n)
        
        # Income: correlated with education and age
        education = 14 + 3 * np.random.randn(n)
        income = 3000 * education + 500 * age + 5000 * np.random.randn(n)
        
        X = np.column_stack([height, weight, age, income, education])
        
        print(f"   Shape: {X.shape}")
        for i, feat in enumerate(features):
            print(f"   {feat}: mean={X[:, i].mean():.1f}, std={X[:, i].std():.1f}")
        
        print("\nb) PCA (standardized):")
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        U, s, Vt = svd(X_std, full_matrices=False)
        
        var_ratio = s**2 / np.sum(s**2)
        print(f"   Variance explained: {np.round(100*var_ratio, 1)}%")
        
        print("\nc) Interpreting loadings (V matrix):")
        print("\n   Feature loadings on each PC:")
        print(f"   {'Feature':<12} {'PC1':>8} {'PC2':>8} {'PC3':>8}")
        print("   " + "-" * 40)
        
        for i, feat in enumerate(features):
            print(f"   {feat:<12} {Vt[0, i]:8.3f} {Vt[1, i]:8.3f} {Vt[2, i]:8.3f}")
        
        print("\n   Interpretation:")
        print("   PC1: Strongly loads on income and education")
        print("        → 'Socioeconomic status' component")
        print("   PC2: Loads on height and weight together")
        print("        → 'Body size' component")
        print("   PC3: Loads mainly on age")
        print("        → 'Age' component")
        
        # Highlight dominant loadings
        print("\n   Dominant features per PC (|loading| > 0.4):")
        for pc in range(3):
            dominant = [features[i] for i in range(5) if abs(Vt[pc, i]) > 0.4]
            print(f"   PC{pc+1}: {', '.join(dominant)}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = PCAExercises()
    
    print("PRINCIPAL COMPONENT ANALYSIS EXERCISES")
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
