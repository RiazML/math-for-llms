"""
Mutual Information - Exercises
==============================
Practice problems for mutual information.
"""

import numpy as np
from scipy import stats
from collections import Counter


class MutualInformationExercises:
    """Exercises for mutual information."""
    
    def exercise_1_basic_computation(self):
        """
        Exercise 1: Compute Mutual Information
        
        Given joint distribution, compute MI different ways.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic MI Computation")
        print("=" * 60)
        
        # Joint distribution
        p_xy = np.array([
            [0.2, 0.1],
            [0.1, 0.6]
        ])
        
        print("Joint distribution P(X, Y):")
        print(p_xy)
        
        # Marginals
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        print(f"\nP(X) = {p_x}")
        print(f"P(Y) = {p_y}")
        
        # Method 1: Definition
        mi_1 = 0
        for i in range(2):
            for j in range(2):
                if p_xy[i, j] > 0:
                    mi_1 += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        # Method 2: H(X) + H(Y) - H(X,Y)
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        
        H_X = entropy(p_x)
        H_Y = entropy(p_y)
        H_XY = entropy(p_xy.flatten())
        mi_2 = H_X + H_Y - H_XY
        
        # Method 3: H(X) - H(X|Y)
        H_X_given_Y = 0
        for j in range(2):
            if p_y[j] > 0:
                p_x_given_y = p_xy[:, j] / p_y[j]
                H_X_given_Y += p_y[j] * entropy(p_x_given_y)
        mi_3 = H_X - H_X_given_Y
        
        print(f"\nMethod 1 (definition): I(X;Y) = {mi_1:.4f}")
        print(f"Method 2 (H(X)+H(Y)-H(X,Y)): I(X;Y) = {mi_2:.4f}")
        print(f"Method 3 (H(X)-H(X|Y)): I(X;Y) = {mi_3:.4f}")
    
    def exercise_2_independence(self):
        """
        Exercise 2: Independence and MI
        
        Show that I(X;Y) = 0 iff independent.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Independence and MI")
        print("=" * 60)
        
        def mutual_info(p_xy):
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            mi = 0
            for i in range(p_xy.shape[0]):
                for j in range(p_xy.shape[1]):
                    if p_xy[i, j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            return mi
        
        # Independent case
        p_x = np.array([0.3, 0.7])
        p_y = np.array([0.4, 0.6])
        p_xy_indep = np.outer(p_x, p_y)
        
        # Dependent case
        p_xy_dep = np.array([
            [0.3, 0.1],
            [0.1, 0.5]
        ])
        
        mi_indep = mutual_info(p_xy_indep)
        mi_dep = mutual_info(p_xy_dep)
        
        print("Independent: P(X,Y) = P(X)P(Y)")
        print(f"P(X) = {p_x}, P(Y) = {p_y}")
        print(f"P(X,Y) = \n{p_xy_indep}")
        print(f"I(X;Y) = {mi_indep:.6f} ≈ 0 ✓")
        
        print(f"\nDependent:")
        print(f"P(X,Y) = \n{p_xy_dep}")
        print(f"I(X;Y) = {mi_dep:.4f} > 0 ✓")
        
        print("\nTheorem: I(X;Y) = 0 ⟺ X and Y are independent")
    
    def exercise_3_gaussian_mi(self):
        """
        Exercise 3: Gaussian Mutual Information
        
        Derive and compute MI for Gaussians.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Gaussian MI")
        print("=" * 60)
        
        print("For jointly Gaussian (X, Y) with correlation ρ:")
        print("\nDerivation:")
        print("H(X) = ½ log(2πe σ_X²)")
        print("H(Y) = ½ log(2πe σ_Y²)")
        print("H(X,Y) = ½ log((2πe)² |Σ|)")
        print("       = ½ log((2πe)² σ_X² σ_Y² (1-ρ²))")
        print("\nI(X;Y) = H(X) + H(Y) - H(X,Y)")
        print("       = -½ log(1 - ρ²)")
        
        def gaussian_mi(rho):
            return -0.5 * np.log(1 - rho**2)
        
        print(f"\n{'ρ':>8} {'I(X;Y)':>12}")
        print("-" * 25)
        
        for rho in [0.0, 0.5, 0.8, 0.9, 0.99]:
            mi = gaussian_mi(rho)
            print(f"{rho:>8.2f} {mi:>12.4f}")
        
        print("\nNote: MI → ∞ as ρ → ±1")
    
    def exercise_4_chain_rule(self):
        """
        Exercise 4: Chain Rule for MI
        
        Verify I(X; Y, Z) = I(X; Y) + I(X; Z | Y).
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: MI Chain Rule")
        print("=" * 60)
        
        # P(X, Y, Z) - all binary
        p_xyz = np.zeros((2, 2, 2))
        p_xyz[0, 0, 0] = 0.15
        p_xyz[0, 0, 1] = 0.05
        p_xyz[0, 1, 0] = 0.10
        p_xyz[0, 1, 1] = 0.20
        p_xyz[1, 0, 0] = 0.05
        p_xyz[1, 0, 1] = 0.15
        p_xyz[1, 1, 0] = 0.10
        p_xyz[1, 1, 1] = 0.20
        
        def entropy(p):
            p = p.flatten()
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        
        # Marginals
        p_x = p_xyz.sum(axis=(1, 2))
        p_y = p_xyz.sum(axis=(0, 2))
        p_z = p_xyz.sum(axis=(0, 1))
        p_xy = p_xyz.sum(axis=2)
        p_xz = p_xyz.sum(axis=1)
        p_yz = p_xyz.sum(axis=0)
        
        # I(X; Y, Z)
        H_X = entropy(p_x)
        H_YZ = entropy(p_xyz.sum(axis=0))
        H_XYZ = entropy(p_xyz)
        I_X_YZ = H_X + H_YZ - H_XYZ
        
        # I(X; Y)
        H_Y = entropy(p_y)
        H_XY = entropy(p_xy)
        I_X_Y = H_X + H_Y - H_XY
        
        # I(X; Z | Y) = H(X|Y) + H(Z|Y) - H(X,Z|Y)
        # Compute conditional entropies
        I_X_Z_given_Y = 0
        for y in range(2):
            py = p_xyz.sum(axis=(0, 2))[y]
            if py > 0:
                p_xz_given_y = p_xyz[:, y, :] / py
                p_x_given_y = p_xz_given_y.sum(axis=1)
                p_z_given_y = p_xz_given_y.sum(axis=0)
                
                H_X_given_y = entropy(p_x_given_y)
                H_Z_given_y = entropy(p_z_given_y)
                H_XZ_given_y = entropy(p_xz_given_y)
                
                I_X_Z_given_Y += py * (H_X_given_y + H_Z_given_y - H_XZ_given_y)
        
        print("Chain Rule: I(X; Y, Z) = I(X; Y) + I(X; Z | Y)")
        print(f"\nI(X; Y, Z) = {I_X_YZ:.4f}")
        print(f"I(X; Y) = {I_X_Y:.4f}")
        print(f"I(X; Z | Y) = {I_X_Z_given_Y:.4f}")
        print(f"I(X; Y) + I(X; Z | Y) = {I_X_Y + I_X_Z_given_Y:.4f}")
    
    def exercise_5_data_processing(self):
        """
        Exercise 5: Data Processing Inequality
        
        Demonstrate that processing can't create information.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Data Processing Inequality")
        print("=" * 60)
        
        print("For Markov chain X → Y → Z:")
        print("I(X; Z) ≤ I(X; Y)")
        print("I(X; Z) ≤ I(Y; Z)")
        
        np.random.seed(42)
        n = 2000
        
        # X: source
        X = np.random.randn(n)
        
        # Y: lossy encoding of X
        Y = np.sign(X)  # Binary quantization
        
        # Z: noisy version of Y
        noise = 0.3 * np.random.randn(n)
        Z = Y + noise
        
        def estimate_mi(A, B, bins=20):
            hist, _, _ = np.histogram2d(A, B, bins=bins)
            p_xy = hist / hist.sum()
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            
            mi = 0
            for i in range(bins):
                for j in range(bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            return mi
        
        I_XY = estimate_mi(X, Y)
        I_YZ = estimate_mi(Y, Z)
        I_XZ = estimate_mi(X, Z)
        
        print(f"\nX → Y (quantize) → Z (add noise)")
        print(f"\nI(X; Y) = {I_XY:.4f}")
        print(f"I(Y; Z) = {I_YZ:.4f}")
        print(f"I(X; Z) = {I_XZ:.4f}")
        
        print(f"\nCheck: I(X;Z) ≤ I(X;Y)? {I_XZ:.4f} ≤ {I_XY:.4f}: {I_XZ <= I_XY + 0.01}")
        print(f"Check: I(X;Z) ≤ I(Y;Z)? {I_XZ:.4f} ≤ {I_YZ:.4f}: {I_XZ <= I_YZ + 0.01}")
    
    def exercise_6_feature_selection(self):
        """
        Exercise 6: MI for Feature Selection
        
        Use MI to rank features by relevance.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Feature Selection")
        print("=" * 60)
        
        np.random.seed(42)
        n = 1000
        
        # Generate features
        X1 = np.random.randn(n)  # Very relevant
        X2 = np.random.randn(n)  # Somewhat relevant
        X3 = np.random.randn(n)  # Irrelevant
        X4 = X1 + 0.5 * np.random.randn(n)  # Redundant with X1
        
        # Target
        Y = (2*X1 + X2 + np.random.randn(n) > 1).astype(int)
        
        def mi_discrete_continuous(X, Y_disc, bins=10):
            X_bins = np.digitize(X, np.linspace(X.min(), X.max(), bins))
            
            joint = Counter(zip(X_bins, Y_disc))
            n_total = len(X)
            
            p_x = Counter(X_bins)
            p_y = Counter(Y_disc)
            
            mi = 0
            for (x, y), count in joint.items():
                p_joint = count / n_total
                px = p_x[x] / n_total
                py = p_y[y] / n_total
                if p_joint > 0:
                    mi += p_joint * np.log(p_joint / (px * py))
            return mi
        
        scores = {
            'X1 (relevant)': mi_discrete_continuous(X1, Y),
            'X2 (somewhat)': mi_discrete_continuous(X2, Y),
            'X3 (irrelevant)': mi_discrete_continuous(X3, Y),
            'X4 (redundant)': mi_discrete_continuous(X4, Y),
        }
        
        print("Feature relevance via MI:")
        print(f"Y = sign(2*X1 + X2 + noise)")
        
        print(f"\n{'Feature':>20} {'I(Xi; Y)':>12}")
        print("-" * 35)
        
        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"{name:>20} {score:>12.4f}")
        
        print("\nX1 should rank highest, X3 lowest")
        print("X4 high MI but redundant with X1")
    
    def exercise_7_nmi_clustering(self):
        """
        Exercise 7: NMI for Clustering Evaluation
        
        Compare clusterings using normalized MI.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: NMI for Clustering")
        print("=" * 60)
        
        def entropy_labels(labels):
            counts = Counter(labels)
            n = len(labels)
            probs = np.array(list(counts.values())) / n
            return -np.sum(probs * np.log(probs + 1e-10))
        
        def mi_labels(l1, l2):
            joint = Counter(zip(l1, l2))
            n = len(l1)
            c1, c2 = Counter(l1), Counter(l2)
            
            mi = 0
            for (a, b), count in joint.items():
                p_joint = count / n
                p1 = c1[a] / n
                p2 = c2[b] / n
                if p_joint > 0:
                    mi += p_joint * np.log(p_joint / (p1 * p2))
            return mi
        
        def nmi(l1, l2):
            mi = mi_labels(l1, l2)
            h1 = entropy_labels(l1)
            h2 = entropy_labels(l2)
            return 2 * mi / (h1 + h2 + 1e-10)
        
        # Ground truth
        true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        
        # Different clusterings
        pred_perfect = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        pred_permuted = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
        pred_merged = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        pred_split = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5]
        pred_random = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        
        print("Ground truth: 3 clusters of 5 elements each")
        
        clusterings = [
            ('Perfect', pred_perfect),
            ('Permuted labels', pred_permuted),
            ('Two merged', pred_merged),
            ('Over-split', pred_split),
            ('Random', pred_random),
        ]
        
        print(f"\n{'Clustering':>20} {'NMI':>10}")
        print("-" * 35)
        
        for name, pred in clusterings:
            score = nmi(true, pred)
            print(f"{name:>20} {score:>10.4f}")
        
        print("\nNMI is invariant to label permutation")
    
    def exercise_8_conditional_mi(self):
        """
        Exercise 8: Conditional Mutual Information
        
        Compute I(X; Y | Z).
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Conditional MI")
        print("=" * 60)
        
        # Example: X=Exam score, Y=Study hours, Z=Intelligence
        # Given intelligence, study hours tells less about score
        
        # P(X, Y, Z) where all are binary
        p_xyz = np.zeros((2, 2, 2))
        
        # Z=0 (low intelligence)
        p_xyz[0, 0, 0] = 0.20  # low score, few hours
        p_xyz[0, 1, 0] = 0.10  # low score, many hours
        p_xyz[1, 0, 0] = 0.05  # high score, few hours
        p_xyz[1, 1, 0] = 0.15  # high score, many hours
        
        # Z=1 (high intelligence)
        p_xyz[0, 0, 1] = 0.05  # low score, few hours
        p_xyz[0, 1, 1] = 0.05  # low score, many hours
        p_xyz[1, 0, 1] = 0.15  # high score, few hours
        p_xyz[1, 1, 1] = 0.25  # high score, many hours
        
        def entropy(p):
            p = p.flatten()
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        
        def mi(p_xy):
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            m = 0
            for i in range(p_xy.shape[0]):
                for j in range(p_xy.shape[1]):
                    if p_xy[i, j] > 0:
                        m += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            return m
        
        # Unconditional I(X; Y)
        p_xy = p_xyz.sum(axis=2)
        i_xy = mi(p_xy)
        
        # Conditional I(X; Y | Z)
        p_z = p_xyz.sum(axis=(0, 1))
        i_xy_given_z = 0
        
        for z in range(2):
            if p_z[z] > 0:
                p_xy_given_z = p_xyz[:, :, z] / p_z[z]
                i_xy_given_z += p_z[z] * mi(p_xy_given_z)
        
        print("X: Exam score, Y: Study hours, Z: Intelligence")
        print(f"\nI(Score; StudyHours) = {i_xy:.4f}")
        print(f"I(Score; StudyHours | Intelligence) = {i_xy_given_z:.4f}")
        print(f"\nDrop in MI: {i_xy - i_xy_given_z:.4f}")
        print("\nIntelligence 'explains away' some of the")
        print("association between study hours and score")
    
    def exercise_9_symmetry(self):
        """
        Exercise 9: MI Symmetry
        
        Prove and verify I(X;Y) = I(Y;X).
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: MI Symmetry")
        print("=" * 60)
        
        print("Proof that I(X;Y) = I(Y;X):")
        print("\nI(X;Y) = H(X) - H(X|Y)")
        print("I(Y;X) = H(Y) - H(Y|X)")
        print("\nAlso:")
        print("I(X;Y) = H(X) + H(Y) - H(X,Y)")
        print("I(Y;X) = H(Y) + H(X) - H(Y,X)")
        print("\nSince H(X,Y) = H(Y,X):")
        print("I(X;Y) = I(Y;X) ✓")
        
        # Numerical verification
        p_xy = np.array([
            [0.15, 0.25],
            [0.35, 0.25]
        ])
        
        def entropy(p):
            p = p.flatten()
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        H_X = entropy(p_x)
        H_Y = entropy(p_y)
        H_XY = entropy(p_xy)
        
        # H(X|Y)
        H_X_given_Y = 0
        for j in range(2):
            if p_y[j] > 0:
                H_X_given_Y += p_y[j] * entropy(p_xy[:, j] / p_y[j])
        
        # H(Y|X)
        H_Y_given_X = 0
        for i in range(2):
            if p_x[i] > 0:
                H_Y_given_X += p_x[i] * entropy(p_xy[i, :] / p_x[i])
        
        I_XY = H_X - H_X_given_Y
        I_YX = H_Y - H_Y_given_X
        
        print(f"\nNumerical verification:")
        print(f"I(X;Y) = H(X) - H(X|Y) = {H_X:.4f} - {H_X_given_Y:.4f} = {I_XY:.4f}")
        print(f"I(Y;X) = H(Y) - H(Y|X) = {H_Y:.4f} - {H_Y_given_X:.4f} = {I_YX:.4f}")
    
    def exercise_10_estimation(self):
        """
        Exercise 10: MI Estimation
        
        Estimate MI from samples using different methods.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: MI Estimation")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Correlated Gaussians
        rho = 0.6
        n = 1000
        
        mean = [0, 0]
        cov = [[1, rho], [rho, 1]]
        samples = np.random.multivariate_normal(mean, cov, n)
        X, Y = samples[:, 0], samples[:, 1]
        
        true_mi = -0.5 * np.log(1 - rho**2)
        
        # Method 1: Histogram
        def histogram_mi(X, Y, bins=10):
            hist, _, _ = np.histogram2d(X, Y, bins=bins)
            p_xy = hist / hist.sum()
            p_x = p_xy.sum(axis=1)
            p_y = p_xy.sum(axis=0)
            
            mi = 0
            for i in range(bins):
                for j in range(bins):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            return mi
        
        # Method 2: Gaussian assumption
        sample_cov = np.cov(X, Y)
        rho_est = sample_cov[0, 1] / np.sqrt(sample_cov[0, 0] * sample_cov[1, 1])
        gaussian_mi = -0.5 * np.log(1 - rho_est**2)
        
        print(f"True MI (ρ={rho}): {true_mi:.4f}")
        
        print(f"\nHistogram estimator (different bins):")
        print(f"{'Bins':>8} {'Estimate':>12}")
        print("-" * 25)
        
        for bins in [5, 10, 20, 50]:
            est = histogram_mi(X, Y, bins)
            print(f"{bins:>8} {est:>12.4f}")
        
        print(f"\nGaussian assumption: {gaussian_mi:.4f}")
        print(f"  (Estimated ρ = {rho_est:.4f})")
        
        print("\nNote: Histogram tends to overestimate")
        print("Gaussian assumption works when data is Gaussian")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = MutualInformationExercises()
    
    print("MUTUAL INFORMATION EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    exercises.solution_2()
    exercises.solution_3()
    exercises.solution_4()
    exercises.solution_5()
    exercises.solution_6()
    exercises.solution_7()
    exercises.solution_8()
    exercises.solution_9()
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
