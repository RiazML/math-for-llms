"""
Spectral Graph Theory - Exercises
=================================

Hands-on exercises implementing spectral methods for graphs,
building understanding from theory to GNN applications.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict


class SpectralGraphExercises:
    """Exercises for spectral graph theory."""
    
    # =========================================================================
    # Exercise 1: Laplacian Properties
    # =========================================================================
    
    @staticmethod
    def exercise_1_laplacian_properties():
        """
        Verify and explore properties of the graph Laplacian.
        
        Properties to verify:
        1. L is symmetric
        2. L is positive semi-definite
        3. Row sums are zero
        4. Smallest eigenvalue is 0
        5. Number of zero eigenvalues = connected components
        """
        
        def verify_laplacian_properties(A: np.ndarray) -> Dict[str, bool]:
            """
            Verify all standard Laplacian properties.
            
            Returns dict of property name -> True/False
            """
            # TODO: Implement
            # 1. Compute L = D - A
            # 2. Check symmetry: L == L.T
            # 3. Check PSD: all eigenvalues >= 0
            # 4. Check row sums: L @ 1 = 0
            # 5. Count zero eigenvalues
            pass
        
        def quadratic_form_interpretation(A: np.ndarray, 
                                          x: np.ndarray) -> float:
            """
            Compute x^T L x and show it equals Σ (x_i - x_j)² for edges.
            
            This is the Dirichlet energy measuring signal smoothness.
            """
            # TODO: Implement both ways and verify they match
            pass
        
        def prove_psd(A: np.ndarray) -> bool:
            """
            Prove L is PSD by showing L = B B^T where B is incidence matrix.
            """
            # TODO: Implement
            # 1. Build incidence matrix B
            # 2. Verify L = B @ B.T
            # 3. This proves PSD since x^T L x = x^T B B^T x = ||B^T x||² >= 0
            pass
        
        # Test
        print("Exercise 1: Laplacian Properties")
        print("-" * 40)
        
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Uncomment to test:
        # props = verify_laplacian_properties(A)
        # for name, valid in props.items():
        #     print(f"  {name}: {valid}")
        
        return verify_laplacian_properties
    
    # =========================================================================
    # Exercise 2: Spectrum of Special Graphs
    # =========================================================================
    
    @staticmethod
    def exercise_2_special_graph_spectra():
        """
        Compute and understand spectra of special graphs.
        
        1. Complete graph K_n
        2. Cycle graph C_n
        3. Path graph P_n
        4. Star graph S_n
        """
        
        def complete_graph_spectrum(n: int) -> np.ndarray:
            """
            Laplacian spectrum of complete graph K_n.
            
            K_n: Every vertex connected to every other.
            Spectrum: 0 (multiplicity 1), n (multiplicity n-1)
            """
            # TODO: Build adjacency and compute spectrum
            pass
        
        def cycle_graph_spectrum(n: int) -> np.ndarray:
            """
            Laplacian spectrum of cycle graph C_n.
            
            C_n: Vertices form a ring.
            Spectrum: λ_k = 2 - 2cos(2πk/n), k = 0, 1, ..., n-1
            """
            # TODO: Implement and verify analytical formula
            pass
        
        def path_graph_spectrum(n: int) -> np.ndarray:
            """
            Laplacian spectrum of path graph P_n.
            
            P_n: n vertices in a line.
            Spectrum: λ_k = 2 - 2cos(πk/n), k = 0, 1, ..., n-1
            """
            # TODO: Implement
            pass
        
        def star_graph_spectrum(n: int) -> np.ndarray:
            """
            Laplacian spectrum of star graph S_n.
            
            S_n: One central vertex connected to n-1 leaves.
            Spectrum: 0, 1 (multiplicity n-2), n
            """
            # TODO: Implement
            pass
        
        def compare_algebraic_connectivity():
            """
            Compare λ₂ for different graph families.
            
            λ₂ measures connectivity/robustness.
            """
            # TODO: Implement comparison
            pass
        
        # Test
        print("\nExercise 2: Special Graph Spectra")
        print("-" * 40)
        
        # Uncomment to test:
        # for n in [5, 10]:
        #     print(f"\nn = {n}:")
        #     print(f"  K_{n} spectrum: {complete_graph_spectrum(n)}")
        #     print(f"  C_{n} spectrum: {np.round(cycle_graph_spectrum(n), 4)}")
        #     print(f"  P_{n} spectrum: {np.round(path_graph_spectrum(n), 4)}")
        
        return cycle_graph_spectrum, path_graph_spectrum
    
    # =========================================================================
    # Exercise 3: Normalized Laplacians Comparison
    # =========================================================================
    
    @staticmethod
    def exercise_3_normalized_laplacians():
        """
        Compare different Laplacian normalizations.
        
        1. Unnormalized: L = D - A
        2. Symmetric: L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2)
        3. Random walk: L_rw = D^(-1) L = I - D^(-1) A
        
        Understand when to use each.
        """
        
        def compute_all_laplacians(A: np.ndarray) -> Dict[str, np.ndarray]:
            """Compute all three Laplacian variants."""
            # TODO: Implement
            pass
        
        def compare_spectra(A: np.ndarray) -> Dict[str, np.ndarray]:
            """
            Compare eigenvalue ranges of different Laplacians.
            
            - L: [0, 2*d_max]
            - L_sym: [0, 2]
            - L_rw: [0, 2]
            """
            # TODO: Implement
            pass
        
        def eigenvector_interpretation(A: np.ndarray):
            """
            Show relationship between eigenvectors:
            
            If (λ, u) is eigenpair of L_rw, then:
            - (λ, D^(1/2) u) is eigenpair of L_sym
            """
            # TODO: Implement and verify
            pass
        
        def random_walk_connection(A: np.ndarray):
            """
            Show L_rw connection to random walk:
            
            P = D^(-1) A is random walk transition matrix.
            L_rw = I - P
            
            Eigenvalues of P are 1 - λ(L_rw).
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 3: Normalized Laplacians")
        print("-" * 40)
        
        # Graph with varying degrees
        A = np.array([
            [0, 1, 1, 1, 1],  # High degree
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0]
        ])
        
        # Uncomment to test:
        # spectra = compare_spectra(A)
        # for name, eigenvalues in spectra.items():
        #     print(f"  {name}: {np.round(eigenvalues, 4)}")
        
        return compute_all_laplacians
    
    # =========================================================================
    # Exercise 4: Spectral Clustering from Scratch
    # =========================================================================
    
    @staticmethod
    def exercise_4_spectral_clustering_scratch():
        """
        Implement spectral clustering completely from scratch.
        
        Steps:
        1. Compute similarity graph from data points
        2. Compute Laplacian
        3. Find eigenvectors
        4. Cluster in embedding space
        """
        
        def similarity_graph(X: np.ndarray, 
                           method: str = 'knn',
                           k: int = 5,
                           sigma: float = 1.0) -> np.ndarray:
            """
            Build similarity graph from data points.
            
            Methods:
            - 'full': Gaussian kernel between all pairs
            - 'knn': k-nearest neighbors
            - 'epsilon': ε-neighborhood
            """
            # TODO: Implement
            pass
        
        def spectral_embedding(A: np.ndarray, k: int) -> np.ndarray:
            """
            Get k-dimensional spectral embedding.
            
            Use k smallest eigenvectors of L_sym.
            """
            # TODO: Implement
            pass
        
        def kmeans_cluster(X: np.ndarray, k: int, 
                          max_iter: int = 100) -> np.ndarray:
            """Simple k-means clustering."""
            # TODO: Implement
            pass
        
        def spectral_cluster_data(X: np.ndarray, k: int) -> np.ndarray:
            """
            Full spectral clustering pipeline for data points.
            """
            # TODO: Implement full pipeline
            # 1. Build similarity graph
            # 2. Compute spectral embedding
            # 3. Run k-means
            pass
        
        # Test
        print("\nExercise 4: Spectral Clustering from Scratch")
        print("-" * 40)
        
        # Generate clustered data
        np.random.seed(42)
        n_per_cluster = 20
        
        # Two clusters
        cluster1 = np.random.randn(n_per_cluster, 2) + np.array([0, 0])
        cluster2 = np.random.randn(n_per_cluster, 2) + np.array([5, 5])
        X = np.vstack([cluster1, cluster2])
        true_labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)
        
        # Uncomment to test:
        # labels = spectral_cluster_data(X, k=2)
        # accuracy = np.mean(labels == true_labels)
        # print(f"  Clustering accuracy: {accuracy:.2%}")
        
        return spectral_cluster_data
    
    # =========================================================================
    # Exercise 5: Chebyshev Polynomial Filters
    # =========================================================================
    
    @staticmethod
    def exercise_5_chebyshev_filters():
        """
        Implement Chebyshev polynomial filters for GNNs.
        
        ChebNet: g(L) = Σ θ_k T_k(L̃) where L̃ = 2L/λ_max - I
        """
        
        def chebyshev_basis(L: np.ndarray, K: int) -> List[np.ndarray]:
            """
            Compute Chebyshev polynomial basis {T_0(L̃), T_1(L̃), ..., T_K(L̃)}.
            
            Recurrence: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            """
            # TODO: Implement
            pass
        
        def chebyshev_filter(X: np.ndarray, L: np.ndarray,
                           theta: np.ndarray) -> np.ndarray:
            """
            Apply Chebyshev filter to features.
            
            Y = Σ θ_k T_k(L̃) X
            """
            # TODO: Implement
            pass
        
        def approximate_filter_function(target_func: Callable,
                                       lambda_max: float,
                                       K: int) -> np.ndarray:
            """
            Find Chebyshev coefficients to approximate target filter.
            
            Uses Chebyshev approximation theory.
            """
            # TODO: Implement
            # Hint: Use discrete cosine transform approach
            pass
        
        def localization_analysis(L: np.ndarray, K: int):
            """
            Analyze spatial localization of K-th order filter.
            
            Show that filter only depends on K-hop neighborhood.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 5: Chebyshev Filters")
        print("-" * 40)
        
        # Cycle graph
        n = 10
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i + 1) % n] = 1
            A[(i + 1) % n, i] = 1
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Uncomment to test:
        # basis = chebyshev_basis(L, K=3)
        # print(f"  Chebyshev basis shapes: {[b.shape for b in basis]}")
        
        return chebyshev_filter
    
    # =========================================================================
    # Exercise 6: Graph Wavelet Transform
    # =========================================================================
    
    @staticmethod
    def exercise_6_graph_wavelets():
        """
        Implement spectral graph wavelets.
        
        Wavelet at vertex i, scale s:
        ψ_{s,i}(j) = g(sL)[i, j]
        
        where g(λ) is wavelet generating kernel.
        """
        
        def mexican_hat_wavelet(lam: float, s: float) -> float:
            """
            Mexican hat wavelet kernel.
            
            g(λ) = λ² exp(-λ²)
            Scaled: g_s(λ) = s² λ² exp(-s² λ²)
            """
            # TODO: Implement
            pass
        
        def compute_wavelets(L: np.ndarray, 
                           scales: List[float]) -> List[np.ndarray]:
            """
            Compute wavelet transform at multiple scales.
            
            Returns list of wavelet matrices (one per scale).
            """
            # TODO: Implement
            pass
        
        def wavelet_coefficients(L: np.ndarray, 
                                signal: np.ndarray,
                                scales: List[float]) -> np.ndarray:
            """
            Compute wavelet coefficients for signal.
            
            W[i, s] = <ψ_{s,i}, signal>
            """
            # TODO: Implement
            pass
        
        def reconstruct_from_wavelets(L: np.ndarray,
                                     coefficients: np.ndarray,
                                     scales: List[float]) -> np.ndarray:
            """
            Reconstruct signal from wavelet coefficients.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 6: Graph Wavelets")
        print("-" * 40)
        
        # Create graph and signal
        n = 20
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = A[i + 1, i] = 1
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Uncomment to test:
        # scales = [0.5, 1.0, 2.0, 4.0]
        # wavelets = compute_wavelets(L, scales)
        # print(f"  Wavelet matrices computed for scales: {scales}")
        
        return compute_wavelets
    
    # =========================================================================
    # Exercise 7: Over-Smoothing Analysis
    # =========================================================================
    
    @staticmethod
    def exercise_7_oversmoothing():
        """
        Analyze over-smoothing in deep GNNs.
        
        As layers increase, node representations converge.
        Related to repeated application of low-pass filter.
        """
        
        def measure_smoothness(X: np.ndarray, L: np.ndarray) -> float:
            """
            Measure feature smoothness: Σ ||x_i - x_j||² for edges.
            
            Or equivalently: trace(X^T L X)
            """
            # TODO: Implement
            pass
        
        def simulate_gcn_layers(A: np.ndarray, X: np.ndarray,
                               num_layers: int) -> List[np.ndarray]:
            """
            Simulate GCN propagation for multiple layers.
            
            Returns features after each layer.
            """
            # TODO: Implement
            pass
        
        def dirichlet_energy_by_layer(A: np.ndarray, 
                                     X: np.ndarray,
                                     num_layers: int) -> List[float]:
            """
            Track Dirichlet energy through layers.
            
            Should decrease (features become smoother).
            """
            # TODO: Implement
            pass
        
        def spectral_analysis_oversmoothing(A: np.ndarray, 
                                           X: np.ndarray,
                                           num_layers: int):
            """
            Analyze over-smoothing through spectral lens.
            
            After k layers, signal is approximately:
            U diag(λ_1^k, ..., λ_n^k) U^T X
            
            Since λ_1 (largest) dominates, features converge.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 7: Over-Smoothing Analysis")
        print("-" * 40)
        
        # Create graph
        A = np.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0]
        ], dtype=float)
        
        # Random features
        np.random.seed(42)
        X = np.random.randn(5, 3)
        
        # Uncomment to test:
        # energies = dirichlet_energy_by_layer(A, X, num_layers=20)
        # print(f"  Dirichlet energies: {np.round(energies[:10], 4)}")
        # print("  Energy decreases -> over-smoothing")
        
        return measure_smoothness, dirichlet_energy_by_layer
    
    # =========================================================================
    # Exercise 8: Spectral Graph Convolution Layer
    # =========================================================================
    
    @staticmethod
    def exercise_8_spectral_conv_layer():
        """
        Implement a full spectral convolution layer.
        
        Including:
        - Forward pass
        - Backward pass (gradient computation)
        - Parameter initialization
        """
        
        class SpectralConvLayer:
            """Spectral graph convolution layer."""
            
            def __init__(self, in_features: int, out_features: int,
                        K: int = 3):
                """
                Initialize layer.
                
                Args:
                    in_features: Input feature dimension
                    out_features: Output feature dimension
                    K: Chebyshev polynomial order
                """
                # TODO: Initialize weights
                self.K = K
                self.in_features = in_features
                self.out_features = out_features
                self.weights = None  # Shape: (K, in_features, out_features)
            
            def forward(self, X: np.ndarray, L: np.ndarray) -> np.ndarray:
                """
                Forward pass.
                
                Y = Σ_k T_k(L̃) X W_k
                """
                # TODO: Implement
                pass
            
            def backward(self, grad_output: np.ndarray, 
                        X: np.ndarray, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                """
                Backward pass.
                
                Returns:
                    grad_input: Gradient w.r.t. input
                    grad_weights: Gradient w.r.t. weights
                """
                # TODO: Implement
                pass
        
        def verify_gradient(layer, X, L, eps=1e-5):
            """Verify gradient using finite differences."""
            # TODO: Implement gradient checking
            pass
        
        # Test
        print("\nExercise 8: Spectral Convolution Layer")
        print("-" * 40)
        
        # Uncomment to test:
        # layer = SpectralConvLayer(in_features=4, out_features=8, K=3)
        # Y = layer.forward(X, L)
        # print(f"  Output shape: {Y.shape}")
        
        return SpectralConvLayer
    
    # =========================================================================
    # Exercise 9: Multi-Scale Spectral Features
    # =========================================================================
    
    @staticmethod
    def exercise_9_multiscale_features():
        """
        Extract multi-scale spectral features from graphs.
        
        Features at different scales capture different structural properties.
        """
        
        def heat_kernel_signature(L: np.ndarray, 
                                 times: List[float]) -> np.ndarray:
            """
            Heat Kernel Signature: Multi-scale node descriptor.
            
            HKS(i, t) = Σ exp(-λ_k t) u_k(i)²
            """
            # TODO: Implement
            pass
        
        def wave_kernel_signature(L: np.ndarray, 
                                 energies: List[float]) -> np.ndarray:
            """
            Wave Kernel Signature: Energy-based descriptor.
            
            WKS(i, e) = Σ exp(-(λ_k - e)² / δ²) u_k(i)²
            """
            # TODO: Implement
            pass
        
        def spectral_clustering_features(L: np.ndarray, 
                                        k: int) -> np.ndarray:
            """
            Features from first k Laplacian eigenvectors.
            
            Standard embedding for spectral clustering.
            """
            # TODO: Implement
            pass
        
        def graph_spectrum_histogram(L: np.ndarray, 
                                    bins: int = 10) -> np.ndarray:
            """
            Histogram of Laplacian eigenvalues.
            
            Global graph descriptor.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 9: Multi-Scale Spectral Features")
        print("-" * 40)
        
        # Create graph
        n = 15
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        A = (A > 0.7).astype(float)
        np.fill_diagonal(A, 0)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Uncomment to test:
        # hks = heat_kernel_signature(L, times=[0.1, 1.0, 10.0])
        # print(f"  HKS shape: {hks.shape}")
        
        return heat_kernel_signature, wave_kernel_signature
    
    # =========================================================================
    # Exercise 10: Spectral Perturbation Analysis
    # =========================================================================
    
    @staticmethod
    def exercise_10_perturbation():
        """
        Analyze how eigenvalues change with graph perturbations.
        
        Important for understanding:
        - Graph robustness
        - Attack vulnerability
        - Stability of spectral methods
        """
        
        def eigenvalue_sensitivity(A: np.ndarray, 
                                  i: int, j: int) -> np.ndarray:
            """
            Compute sensitivity of eigenvalues to edge (i, j).
            
            ∂λ_k/∂A_ij = u_k(i) u_k(j) for Laplacian
            """
            # TODO: Implement
            pass
        
        def most_influential_edge(A: np.ndarray, k: int) -> Tuple[int, int]:
            """
            Find edge with largest effect on k-th eigenvalue.
            """
            # TODO: Implement
            pass
        
        def spectral_attack(A: np.ndarray, 
                          target: str = 'connectivity',
                          budget: int = 1) -> np.ndarray:
            """
            Find optimal edge modifications to affect spectrum.
            
            Targets:
            - 'connectivity': Minimize λ₂
            - 'diameter': Change spectral gap
            """
            # TODO: Implement
            pass
        
        def verify_perturbation_bound(A: np.ndarray):
            """
            Verify Weyl's inequality for eigenvalue perturbation.
            
            |λ_k(A+E) - λ_k(A)| ≤ ||E||
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 10: Spectral Perturbation Analysis")
        print("-" * 40)
        
        # Create graph
        A = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)
        
        # Uncomment to test:
        # sens = eigenvalue_sensitivity(A, 0, 3)
        # print(f"  Sensitivity to edge (0,3): {np.round(sens, 4)}")
        
        return eigenvalue_sensitivity, spectral_attack


def verify_implementations():
    """Verify all exercise implementations."""
    print("=" * 60)
    print("Spectral Graph Theory - Exercise Verification")
    print("=" * 60)
    
    exercises = SpectralGraphExercises()
    
    exercises.exercise_1_laplacian_properties()
    exercises.exercise_2_special_graph_spectra()
    exercises.exercise_3_normalized_laplacians()
    exercises.exercise_4_spectral_clustering_scratch()
    exercises.exercise_5_chebyshev_filters()
    exercises.exercise_6_graph_wavelets()
    exercises.exercise_7_oversmoothing()
    exercises.exercise_8_spectral_conv_layer()
    exercises.exercise_9_multiscale_features()
    exercises.exercise_10_perturbation()
    
    print("\n" + "=" * 60)
    print("Complete the TODO sections in each exercise!")
    print("=" * 60)


# =============================================================================
# Solutions (Reference Implementation)
# =============================================================================

class Solutions:
    """Reference solutions for exercises."""
    
    @staticmethod
    def solution_1_laplacian_properties():
        """Solution for Exercise 1."""
        
        def verify_laplacian_properties(A: np.ndarray) -> Dict[str, bool]:
            n = len(A)
            D = np.diag(A.sum(axis=1))
            L = D - A
            
            eigenvalues = np.linalg.eigvalsh(L)
            
            return {
                'symmetric': np.allclose(L, L.T),
                'psd': np.all(eigenvalues >= -1e-10),
                'zero_row_sum': np.allclose(L.sum(axis=1), 0),
                'min_eigenvalue_zero': np.abs(eigenvalues.min()) < 1e-10,
                'num_zero_eigenvalues': int(np.sum(np.abs(eigenvalues) < 1e-10))
            }
        
        return verify_laplacian_properties
    
    @staticmethod
    def solution_2_cycle_spectrum():
        """Solution for Exercise 2."""
        
        def cycle_graph_spectrum(n: int) -> np.ndarray:
            # Build cycle graph
            A = np.zeros((n, n))
            for i in range(n):
                A[i, (i + 1) % n] = 1
                A[(i + 1) % n, i] = 1
            
            D = np.diag(A.sum(axis=1))
            L = D - A
            
            eigenvalues = np.sort(np.linalg.eigvalsh(L))
            
            # Verify analytical: λ_k = 2 - 2cos(2πk/n)
            analytical = np.sort([2 - 2 * np.cos(2 * np.pi * k / n) 
                                 for k in range(n)])
            assert np.allclose(eigenvalues, analytical)
            
            return eigenvalues
        
        return cycle_graph_spectrum
    
    @staticmethod
    def solution_5_chebyshev():
        """Solution for Exercise 5."""
        
        def chebyshev_filter(X: np.ndarray, L: np.ndarray,
                           theta: np.ndarray) -> np.ndarray:
            n = len(L)
            K = len(theta)
            
            # Scale L
            lambda_max = np.max(np.linalg.eigvalsh(L))
            L_scaled = 2 * L / lambda_max - np.eye(n)
            
            # T_0 X
            T_0_X = X.copy()
            result = theta[0] * T_0_X
            
            if K == 1:
                return result
            
            # T_1 X
            T_1_X = L_scaled @ X
            result += theta[1] * T_1_X
            
            # Higher order
            T_prev, T_curr = T_0_X, T_1_X
            for k in range(2, K):
                T_next = 2 * L_scaled @ T_curr - T_prev
                result += theta[k] * T_next
                T_prev, T_curr = T_curr, T_next
            
            return result
        
        return chebyshev_filter
    
    @staticmethod
    def solution_7_oversmoothing():
        """Solution for Exercise 7."""
        
        def measure_smoothness(X: np.ndarray, L: np.ndarray) -> float:
            return float(np.trace(X.T @ L @ X))
        
        def dirichlet_energy_by_layer(A: np.ndarray, 
                                     X: np.ndarray,
                                     num_layers: int) -> List[float]:
            n = len(A)
            A_tilde = A + np.eye(n)
            D_tilde = np.diag(A_tilde.sum(axis=1))
            D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
            A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
            
            D = np.diag(A.sum(axis=1))
            L = D - A
            
            energies = [measure_smoothness(X, L)]
            
            Y = X.copy()
            for _ in range(num_layers):
                Y = A_norm @ Y
                energies.append(measure_smoothness(Y, L))
            
            return energies
        
        return measure_smoothness, dirichlet_energy_by_layer


if __name__ == "__main__":
    verify_implementations()
