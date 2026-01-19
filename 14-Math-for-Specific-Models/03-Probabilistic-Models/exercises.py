"""
Probabilistic Models: Exercises
==============================

Practice implementing probabilistic models from scratch.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional


# =============================================================================
# Exercise 1: Bayesian Inference with Conjugate Priors
# =============================================================================

def exercise1_bayesian_inference():
    """
    Implement Bayesian inference for Normal-Inverse-Gamma conjugate pair.
    
    Model:
        x_i | mu, sigma^2 ~ N(mu, sigma^2)
        mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0)
        sigma^2 ~ Inv-Gamma(alpha_0, beta_0)
    
    Posterior:
        mu | sigma^2, x ~ N(mu_n, sigma^2 / kappa_n)
        sigma^2 | x ~ Inv-Gamma(alpha_n, beta_n)
    
    Tasks:
    1. Implement posterior parameter updates
    2. Compute posterior predictive distribution
    3. Calculate marginal likelihood
    """
    
    def posterior_update(
        data: np.ndarray,
        mu_0: float, kappa_0: float,
        alpha_0: float, beta_0: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute posterior parameters.
        
        Returns:
            mu_n, kappa_n, alpha_n, beta_n
        """
        # YOUR CODE HERE
        pass
    
    def posterior_predictive(
        x_new: np.ndarray,
        mu_n: float, kappa_n: float,
        alpha_n: float, beta_n: float
    ) -> np.ndarray:
        """
        Compute posterior predictive probability.
        
        p(x_new | data) = Student-t(2*alpha_n, mu_n, beta_n*(kappa_n+1)/(alpha_n*kappa_n))
        
        Returns:
            Log probability densities
        """
        # YOUR CODE HERE
        pass
    
    def marginal_likelihood(
        data: np.ndarray,
        mu_0: float, kappa_0: float,
        alpha_0: float, beta_0: float
    ) -> float:
        """
        Compute log marginal likelihood p(data).
        
        Returns:
            Log marginal likelihood
        """
        # YOUR CODE HERE
        pass
    
    # Test
    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 50)
    
    # Prior parameters (weakly informative)
    mu_0, kappa_0 = 0.0, 0.1
    alpha_0, beta_0 = 1.0, 1.0
    
    # Test your implementations
    params = posterior_update(data, mu_0, kappa_0, alpha_0, beta_0)
    print(f"Posterior parameters: mu_n={params[0]:.3f}, kappa_n={params[1]:.3f}")
    print(f"Sample mean: {np.mean(data):.3f}, sample std: {np.std(data):.3f}")


def solution1_bayesian_inference():
    """Solution for Exercise 1."""
    
    def posterior_update(
        data: np.ndarray,
        mu_0: float, kappa_0: float,
        alpha_0: float, beta_0: float
    ) -> Tuple[float, float, float, float]:
        """Compute posterior parameters."""
        n = len(data)
        x_bar = np.mean(data)
        
        # Update parameters
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        alpha_n = alpha_0 + n / 2
        
        # Sum of squared deviations
        ss = np.sum((data - x_bar)**2)
        
        beta_n = beta_0 + 0.5 * ss + \
                 0.5 * kappa_0 * n * (x_bar - mu_0)**2 / kappa_n
        
        return mu_n, kappa_n, alpha_n, beta_n
    
    def posterior_predictive(
        x_new: np.ndarray,
        mu_n: float, kappa_n: float,
        alpha_n: float, beta_n: float
    ) -> np.ndarray:
        """Compute posterior predictive log probability."""
        # Degrees of freedom
        nu = 2 * alpha_n
        
        # Scale
        sigma_sq = beta_n * (kappa_n + 1) / (alpha_n * kappa_n)
        
        # Standardize
        t = (x_new - mu_n) / np.sqrt(sigma_sq)
        
        # Log Student-t density
        from scipy.special import gammaln
        log_prob = gammaln((nu + 1) / 2) - gammaln(nu / 2)
        log_prob -= 0.5 * np.log(nu * np.pi * sigma_sq)
        log_prob -= (nu + 1) / 2 * np.log(1 + t**2 / nu)
        
        return log_prob
    
    def marginal_likelihood(
        data: np.ndarray,
        mu_0: float, kappa_0: float,
        alpha_0: float, beta_0: float
    ) -> float:
        """Compute log marginal likelihood."""
        from scipy.special import gammaln
        
        n = len(data)
        mu_n, kappa_n, alpha_n, beta_n = posterior_update(
            data, mu_0, kappa_0, alpha_0, beta_0
        )
        
        log_ml = -n / 2 * np.log(2 * np.pi)
        log_ml += gammaln(alpha_n) - gammaln(alpha_0)
        log_ml += alpha_0 * np.log(beta_0) - alpha_n * np.log(beta_n)
        log_ml += 0.5 * np.log(kappa_0) - 0.5 * np.log(kappa_n)
        
        return log_ml
    
    # Test
    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 50)
    
    mu_0, kappa_0 = 0.0, 0.1
    alpha_0, beta_0 = 1.0, 1.0
    
    params = posterior_update(data, mu_0, kappa_0, alpha_0, beta_0)
    print(f"Posterior mu_n: {params[0]:.3f} (sample mean: {np.mean(data):.3f})")
    print(f"Posterior alpha_n: {params[2]:.1f}, beta_n: {params[3]:.1f}")
    
    # Posterior predictive
    x_test = np.array([3.0, 5.0, 7.0])
    log_probs = posterior_predictive(x_test, *params)
    print(f"Predictive log-probs at {x_test}: {log_probs}")
    
    # Marginal likelihood
    log_ml = marginal_likelihood(data, mu_0, kappa_0, alpha_0, beta_0)
    print(f"Log marginal likelihood: {log_ml:.2f}")


# =============================================================================
# Exercise 2: Expectation-Maximization for GMM
# =============================================================================

def exercise2_em_gmm():
    """
    Implement EM algorithm for Gaussian Mixture Model with diagonal covariances.
    
    Model:
        p(x) = sum_k pi_k * N(x | mu_k, diag(sigma_k^2))
    
    Tasks:
    1. Implement E-step (compute responsibilities)
    2. Implement M-step (update parameters)
    3. Implement log-likelihood computation
    4. Add early stopping based on convergence
    """
    
    def e_step(
        X: np.ndarray,
        pi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        E-step: compute responsibilities.
        
        Args:
            X: Data (n_samples, n_features)
            pi: Mixture weights (K,)
            mu: Means (K, n_features)
            sigma: Diagonal std devs (K, n_features)
        
        Returns:
            responsibilities: (n_samples, K)
        """
        # YOUR CODE HERE
        pass
    
    def m_step(
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        M-step: update parameters.
        
        Returns:
            pi, mu, sigma (updated parameters)
        """
        # YOUR CODE HERE
        pass
    
    def log_likelihood(
        X: np.ndarray,
        pi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> float:
        """Compute log-likelihood."""
        # YOUR CODE HERE
        pass
    
    def fit_gmm(
        X: np.ndarray,
        K: int,
        n_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Fit GMM using EM.
        
        Returns:
            pi, mu, sigma, ll_history
        """
        # YOUR CODE HERE
        pass
    
    # Test
    np.random.seed(42)
    
    # Generate mixture data
    n_samples = 300
    X = np.vstack([
        np.random.randn(100, 2) + [0, 0],
        np.random.randn(100, 2) + [5, 0],
        np.random.randn(100, 2) + [2.5, 4]
    ])
    
    # Test your implementation
    # pi, mu, sigma, ll_history = fit_gmm(X, K=3, n_iter=50)
    # print(f"Converged in {len(ll_history)} iterations")
    # print(f"Final log-likelihood: {ll_history[-1]:.2f}")


def solution2_em_gmm():
    """Solution for Exercise 2."""
    
    def gaussian_log_pdf(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Compute log PDF of diagonal Gaussian."""
        d = X.shape[1]
        diff = X - mu
        log_det = np.sum(np.log(sigma**2))
        mahal = np.sum(diff**2 / sigma**2, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
    
    def e_step(
        X: np.ndarray,
        pi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """E-step: compute responsibilities."""
        K = len(pi)
        n_samples = X.shape[0]
        
        log_resp = np.zeros((n_samples, K))
        for k in range(K):
            log_resp[:, k] = np.log(pi[k]) + gaussian_log_pdf(X, mu[k], sigma[k])
        
        # Log-sum-exp normalization
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        log_resp_normalized = log_resp - log_resp_max - np.log(
            np.sum(np.exp(log_resp - log_resp_max), axis=1, keepdims=True)
        )
        
        return np.exp(log_resp_normalized)
    
    def m_step(
        X: np.ndarray,
        responsibilities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: update parameters."""
        n_samples, n_features = X.shape
        K = responsibilities.shape[1]
        
        # Effective counts
        N_k = np.sum(responsibilities, axis=0)
        
        # Update mixture weights
        pi = N_k / n_samples
        
        # Update means
        mu = np.zeros((K, n_features))
        for k in range(K):
            mu[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
        
        # Update diagonal covariances
        sigma = np.zeros((K, n_features))
        for k in range(K):
            diff = X - mu[k]
            sigma[k] = np.sqrt(
                np.sum(responsibilities[:, k:k+1] * diff**2, axis=0) / N_k[k]
            )
            sigma[k] = np.maximum(sigma[k], 1e-6)  # Prevent collapse
        
        return pi, mu, sigma
    
    def log_likelihood(
        X: np.ndarray,
        pi: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> float:
        """Compute log-likelihood."""
        K = len(pi)
        n_samples = X.shape[0]
        
        log_probs = np.zeros((n_samples, K))
        for k in range(K):
            log_probs[:, k] = np.log(pi[k]) + gaussian_log_pdf(X, mu[k], sigma[k])
        
        # Log-sum-exp
        log_probs_max = np.max(log_probs, axis=1)
        return np.sum(log_probs_max + np.log(np.sum(np.exp(log_probs - log_probs_max[:, None]), axis=1)))
    
    def fit_gmm(
        X: np.ndarray,
        K: int,
        n_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """Fit GMM using EM."""
        n_samples, n_features = X.shape
        
        # Initialize with k-means++
        mu = np.zeros((K, n_features))
        idx = np.random.randint(n_samples)
        mu[0] = X[idx]
        
        for k in range(1, K):
            dists = np.min([np.sum((X - mu[j])**2, axis=1) for j in range(k)], axis=0)
            probs = dists / np.sum(dists)
            idx = np.random.choice(n_samples, p=probs)
            mu[k] = X[idx]
        
        pi = np.ones(K) / K
        sigma = np.ones((K, n_features)) * np.std(X, axis=0)
        
        ll_history = []
        prev_ll = -np.inf
        
        for _ in range(n_iter):
            # E-step
            responsibilities = e_step(X, pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = m_step(X, responsibilities)
            
            # Compute log-likelihood
            ll = log_likelihood(X, pi, mu, sigma)
            ll_history.append(ll)
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        
        return pi, mu, sigma, ll_history
    
    # Test
    np.random.seed(42)
    
    n_samples = 300
    X = np.vstack([
        np.random.randn(100, 2) + [0, 0],
        np.random.randn(100, 2) + [5, 0],
        np.random.randn(100, 2) + [2.5, 4]
    ])
    
    pi, mu, sigma, ll_history = fit_gmm(X, K=3, n_iter=50)
    
    print("Fitted GMM:")
    print(f"  Converged in {len(ll_history)} iterations")
    print(f"  Final log-likelihood: {ll_history[-1]:.2f}")
    print(f"  Mixture weights: {pi}")
    print(f"  Means:\n{mu}")


# =============================================================================
# Exercise 3: Variational Inference for Factor Analysis
# =============================================================================

def exercise3_variational_factor_analysis():
    """
    Implement variational EM for factor analysis.
    
    Model:
        z ~ N(0, I)           (q-dimensional latent)
        x | z ~ N(W @ z + mu, Psi)   (d-dimensional observed, Psi diagonal)
    
    Variational posterior:
        q(z | x) = N(m, S)
    
    Tasks:
    1. Implement E-step (compute posterior over z)
    2. Implement M-step (update W, mu, Psi)
    3. Compute ELBO
    """
    
    def e_step(
        X: np.ndarray,
        W: np.ndarray,
        mu: np.ndarray,
        Psi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: compute q(z | x).
        
        Returns:
            E_z: Expected latent (n_samples, q)
            E_zz: Expected outer product (n_samples, q, q)
        """
        # YOUR CODE HERE
        pass
    
    def m_step(
        X: np.ndarray,
        E_z: np.ndarray,
        E_zz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        M-step: update W, mu, Psi.
        
        Returns:
            W, mu, Psi
        """
        # YOUR CODE HERE
        pass
    
    def compute_elbo(
        X: np.ndarray,
        W: np.ndarray,
        mu: np.ndarray,
        Psi: np.ndarray,
        E_z: np.ndarray,
        E_zz: np.ndarray
    ) -> float:
        """Compute Evidence Lower Bound."""
        # YOUR CODE HERE
        pass
    
    # Test
    np.random.seed(42)
    
    # Generate factor analysis data
    n, d, q = 200, 10, 3
    true_W = np.random.randn(d, q)
    true_mu = np.random.randn(d)
    true_Psi = 0.5 * np.ones(d)
    
    z = np.random.randn(n, q)
    X = z @ true_W.T + true_mu + np.sqrt(true_Psi) * np.random.randn(n, d)
    
    # Test your implementation
    print("Test factor analysis implementation...")


def solution3_variational_factor_analysis():
    """Solution for Exercise 3."""
    
    def e_step(
        X: np.ndarray,
        W: np.ndarray,
        mu: np.ndarray,
        Psi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """E-step: compute q(z | x)."""
        n_samples = X.shape[0]
        q = W.shape[1]
        
        # Precision-weighted W
        Psi_inv = 1.0 / Psi
        
        # Posterior covariance: S = (I + W^T Psi^-1 W)^-1
        S = np.linalg.inv(np.eye(q) + W.T @ (Psi_inv[:, None] * W))
        
        # Posterior mean: m = S @ W^T @ Psi^-1 @ (x - mu)
        X_centered = X - mu
        E_z = X_centered @ (Psi_inv[:, None] * W) @ S.T  # (n, q)
        
        # Expected outer product: E[zz^T] = S + m m^T (per sample)
        # Sum over samples for M-step efficiency
        E_zz = np.zeros((n_samples, q, q))
        for i in range(n_samples):
            E_zz[i] = S + np.outer(E_z[i], E_z[i])
        
        return E_z, E_zz
    
    def m_step(
        X: np.ndarray,
        E_z: np.ndarray,
        E_zz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M-step: update W, mu, Psi."""
        n_samples, d = X.shape
        q = E_z.shape[1]
        
        # Update mu
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        
        # Update W: W_new = (sum_n x_n E[z_n]^T) @ (sum_n E[z_n z_n^T])^-1
        sum_E_zz = np.sum(E_zz, axis=0)
        W = X_centered.T @ E_z @ np.linalg.inv(sum_E_zz)
        
        # Update Psi: Psi_jj = (1/n) sum_n (x_nj - mu_j)^2 - 2 W_j E[z_n] (x_n - mu)_j + W_j E[z_n z_n^T] W_j^T
        Psi = np.zeros(d)
        for j in range(d):
            # Variance term
            Psi[j] = np.mean(X_centered[:, j]**2)
            
            # Cross term
            Psi[j] -= 2 * np.mean(E_z @ W[j, :] * X_centered[:, j])
            
            # Quadratic term
            for i in range(n_samples):
                Psi[j] += W[j, :] @ E_zz[i] @ W[j, :] / n_samples
        
        Psi = np.maximum(Psi, 1e-6)
        
        return W, mu, Psi
    
    def compute_elbo(
        X: np.ndarray,
        W: np.ndarray,
        mu: np.ndarray,
        Psi: np.ndarray,
        E_z: np.ndarray,
        E_zz: np.ndarray
    ) -> float:
        """Compute Evidence Lower Bound."""
        n_samples, d = X.shape
        q = E_z.shape[1]
        
        X_centered = X - mu
        Psi_inv = 1.0 / Psi
        
        # Expected log likelihood
        E_log_lik = -0.5 * n_samples * d * np.log(2 * np.pi)
        E_log_lik -= 0.5 * n_samples * np.sum(np.log(Psi))
        
        for i in range(n_samples):
            # (x - mu - Wz)^T Psi^-1 (x - mu - Wz)
            recon = W @ E_z[i]
            resid = X_centered[i] - recon
            E_log_lik -= 0.5 * np.sum(Psi_inv * resid**2)
            # Trace term from E[z z^T]
            E_log_lik -= 0.5 * np.trace(W.T @ (Psi_inv[:, None] * W) @ E_zz[i])
            E_log_lik += 0.5 * np.sum((W @ E_z[i])**2 * Psi_inv)
        
        # KL divergence (assuming standard normal prior)
        # Approximation: sum over samples
        S = np.linalg.inv(np.eye(q) + W.T @ (Psi_inv[:, None] * W))
        KL = 0.5 * n_samples * (-np.linalg.slogdet(S)[1] - q + np.trace(S))
        KL += 0.5 * np.sum(E_z**2)
        
        return E_log_lik - KL
    
    # Test
    np.random.seed(42)
    
    n, d, q = 200, 10, 3
    true_W = np.random.randn(d, q)
    true_mu = np.random.randn(d)
    true_Psi = 0.5 * np.ones(d)
    
    z = np.random.randn(n, q)
    X = z @ true_W.T + true_mu + np.sqrt(true_Psi) * np.random.randn(n, d)
    
    # Initialize
    W = 0.1 * np.random.randn(d, q)
    mu = np.mean(X, axis=0)
    Psi = np.var(X, axis=0)
    
    # Run EM
    for iteration in range(50):
        E_z, E_zz = e_step(X, W, mu, Psi)
        W, mu, Psi = m_step(X, E_z, E_zz)
        elbo = compute_elbo(X, W, mu, Psi, E_z, E_zz)
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: ELBO = {elbo:.2f}")
    
    print(f"\nTrue Psi: {true_Psi[0]:.3f}")
    print(f"Estimated Psi mean: {np.mean(Psi):.3f}")


# =============================================================================
# Exercise 4: Hidden Markov Model - Complete Implementation
# =============================================================================

def exercise4_hmm():
    """
    Implement complete HMM with continuous Gaussian emissions.
    
    Model:
        z_t | z_{t-1} ~ Categorical(A[z_{t-1}, :])
        x_t | z_t ~ N(mu[z_t], sigma[z_t]^2)
    
    Tasks:
    1. Forward algorithm (with scaling)
    2. Backward algorithm
    3. Viterbi decoding
    4. Baum-Welch learning
    """
    
    class GaussianHMM:
        def __init__(self, n_states: int):
            self.K = n_states
            self.pi = None
            self.A = None
            self.mu = None
            self.sigma = None
        
        def _emission_prob(self, x: float, k: int) -> float:
            """Compute emission probability p(x | z=k)."""
            # YOUR CODE HERE
            pass
        
        def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Forward algorithm with scaling.
            
            Returns:
                alpha: Scaled forward probabilities (T, K)
                scale: Scaling factors (T,)
            """
            # YOUR CODE HERE
            pass
        
        def backward(self, X: np.ndarray, scale: np.ndarray) -> np.ndarray:
            """
            Backward algorithm.
            
            Returns:
                beta: Scaled backward probabilities (T, K)
            """
            # YOUR CODE HERE
            pass
        
        def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
            """
            Viterbi decoding.
            
            Returns:
                path: Most likely state sequence
                log_prob: Log probability of path
            """
            # YOUR CODE HERE
            pass
        
        def fit(self, X: np.ndarray, n_iter: int = 100) -> List[float]:
            """
            Baum-Welch algorithm.
            
            Returns:
                ll_history: Log-likelihood history
            """
            # YOUR CODE HERE
            pass
    
    # Test
    np.random.seed(42)
    
    # Generate HMM data
    T = 200
    true_A = np.array([[0.9, 0.1], [0.2, 0.8]])
    true_pi = np.array([0.6, 0.4])
    true_mu = np.array([0.0, 3.0])
    true_sigma = np.array([0.5, 1.0])
    
    # Generate sequence
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T)
    
    states[0] = np.random.choice(2, p=true_pi)
    observations[0] = true_mu[states[0]] + true_sigma[states[0]] * np.random.randn()
    
    for t in range(1, T):
        states[t] = np.random.choice(2, p=true_A[states[t-1]])
        observations[t] = true_mu[states[t]] + true_sigma[states[t]] * np.random.randn()
    
    print("Test Gaussian HMM implementation...")


def solution4_hmm():
    """Solution for Exercise 4."""
    
    class GaussianHMM:
        def __init__(self, n_states: int):
            self.K = n_states
            self.pi = None
            self.A = None
            self.mu = None
            self.sigma = None
        
        def _emission_prob(self, x: float, k: int) -> float:
            """Compute emission probability p(x | z=k)."""
            return np.exp(-0.5 * ((x - self.mu[k]) / self.sigma[k])**2) / (
                np.sqrt(2 * np.pi) * self.sigma[k]
            )
        
        def _emission_probs(self, X: np.ndarray) -> np.ndarray:
            """Compute all emission probabilities."""
            T = len(X)
            B = np.zeros((T, self.K))
            for k in range(self.K):
                B[:, k] = np.exp(-0.5 * ((X - self.mu[k]) / self.sigma[k])**2) / (
                    np.sqrt(2 * np.pi) * self.sigma[k]
                )
            return B
        
        def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Forward algorithm with scaling."""
            T = len(X)
            alpha = np.zeros((T, self.K))
            scale = np.zeros(T)
            
            B = self._emission_probs(X)
            
            # Initialize
            alpha[0] = self.pi * B[0]
            scale[0] = np.sum(alpha[0])
            alpha[0] /= scale[0]
            
            # Forward recursion
            for t in range(1, T):
                alpha[t] = (alpha[t-1] @ self.A) * B[t]
                scale[t] = np.sum(alpha[t])
                alpha[t] /= scale[t]
            
            return alpha, scale
        
        def backward(self, X: np.ndarray, scale: np.ndarray) -> np.ndarray:
            """Backward algorithm."""
            T = len(X)
            beta = np.zeros((T, self.K))
            
            B = self._emission_probs(X)
            
            # Initialize
            beta[T-1] = 1.0
            
            # Backward recursion
            for t in range(T-2, -1, -1):
                beta[t] = self.A @ (B[t+1] * beta[t+1])
                beta[t] /= scale[t+1]
            
            return beta
        
        def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
            """Viterbi decoding."""
            T = len(X)
            delta = np.zeros((T, self.K))
            psi = np.zeros((T, self.K), dtype=int)
            
            B = self._emission_probs(X)
            
            # Initialize (log domain)
            delta[0] = np.log(self.pi + 1e-10) + np.log(B[0] + 1e-10)
            
            # Forward
            log_A = np.log(self.A + 1e-10)
            for t in range(1, T):
                for j in range(self.K):
                    scores = delta[t-1] + log_A[:, j]
                    psi[t, j] = np.argmax(scores)
                    delta[t, j] = scores[psi[t, j]] + np.log(B[t, j] + 1e-10)
            
            # Backtrack
            path = np.zeros(T, dtype=int)
            path[T-1] = np.argmax(delta[T-1])
            log_prob = delta[T-1, path[T-1]]
            
            for t in range(T-2, -1, -1):
                path[t] = psi[t+1, path[t+1]]
            
            return path, log_prob
        
        def fit(self, X: np.ndarray, n_iter: int = 100) -> List[float]:
            """Baum-Welch algorithm."""
            T = len(X)
            
            # Initialize
            self.pi = np.ones(self.K) / self.K
            self.A = np.random.dirichlet(np.ones(self.K), size=self.K)
            
            # Initialize means using quantiles
            sorted_X = np.sort(X)
            self.mu = np.array([sorted_X[int(i * T / self.K)] for i in range(self.K)])
            self.sigma = np.std(X) * np.ones(self.K)
            
            ll_history = []
            
            for iteration in range(n_iter):
                B = self._emission_probs(X)
                
                # E-step
                alpha, scale = self.forward(X)
                beta = self.backward(X, scale)
                
                # Gamma: P(z_t = k | X)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                
                # Xi: P(z_t = i, z_{t+1} = j | X)
                xi = np.zeros((T-1, self.K, self.K))
                for t in range(T-1):
                    xi[t] = np.outer(alpha[t], B[t+1] * beta[t+1]) * self.A
                    xi[t] /= xi[t].sum()
                
                # M-step
                self.pi = gamma[0]
                self.A = xi.sum(axis=0)
                self.A /= self.A.sum(axis=1, keepdims=True)
                
                for k in range(self.K):
                    self.mu[k] = np.sum(gamma[:, k] * X) / np.sum(gamma[:, k])
                    self.sigma[k] = np.sqrt(
                        np.sum(gamma[:, k] * (X - self.mu[k])**2) / np.sum(gamma[:, k])
                    )
                    self.sigma[k] = max(self.sigma[k], 1e-6)
                
                # Log-likelihood
                ll = np.sum(np.log(scale))
                ll_history.append(ll)
                
                if iteration > 0 and abs(ll - ll_history[-2]) < 1e-6:
                    break
            
            return ll_history
    
    # Test
    np.random.seed(42)
    
    T = 200
    true_A = np.array([[0.9, 0.1], [0.2, 0.8]])
    true_pi = np.array([0.6, 0.4])
    true_mu = np.array([0.0, 3.0])
    true_sigma = np.array([0.5, 1.0])
    
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T)
    
    states[0] = np.random.choice(2, p=true_pi)
    observations[0] = true_mu[states[0]] + true_sigma[states[0]] * np.random.randn()
    
    for t in range(1, T):
        states[t] = np.random.choice(2, p=true_A[states[t-1]])
        observations[t] = true_mu[states[t]] + true_sigma[states[t]] * np.random.randn()
    
    # Fit HMM
    hmm = GaussianHMM(n_states=2)
    ll_history = hmm.fit(observations, n_iter=50)
    
    print("True parameters:")
    print(f"  mu: {true_mu}")
    print(f"  sigma: {true_sigma}")
    
    print("\nLearned parameters:")
    print(f"  mu: {hmm.mu}")
    print(f"  sigma: {hmm.sigma}")
    
    # Viterbi decoding
    path, _ = hmm.viterbi(observations)
    
    # Handle label switching
    acc1 = np.mean(path == states)
    acc2 = np.mean(path == (1 - states))
    accuracy = max(acc1, acc2)
    
    print(f"\nViterbi accuracy: {accuracy:.3f}")


# =============================================================================
# Exercise 5: Gaussian Process Classification
# =============================================================================

def exercise5_gp_classification():
    """
    Implement Gaussian Process classification using Laplace approximation.
    
    Model:
        f ~ GP(0, k)
        y | f ~ Bernoulli(sigmoid(f))
    
    Laplace approximation:
        q(f | D) = N(f | f_hat, (K^-1 + W)^-1)
    
    where f_hat maximizes p(y|f)p(f) and W = diag(sigmoid(f)(1-sigmoid(f)))
    
    Tasks:
    1. Implement log posterior and gradient
    2. Implement Newton's method for MAP f_hat
    3. Compute predictive distribution
    """
    
    def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """RBF kernel."""
        # YOUR CODE HERE
        pass
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        # YOUR CODE HERE
        pass
    
    def log_posterior(f: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
        """Compute log posterior (up to constant)."""
        # YOUR CODE HERE
        pass
    
    def grad_log_posterior(f: np.ndarray, y: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
        """Gradient of log posterior."""
        # YOUR CODE HERE
        pass
    
    def hessian_log_posterior(f: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
        """Hessian of log posterior."""
        # YOUR CODE HERE
        pass
    
    def find_mode(y: np.ndarray, K: np.ndarray, n_iter: int = 20) -> np.ndarray:
        """Find MAP estimate using Newton's method."""
        # YOUR CODE HERE
        pass
    
    def predict(
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, f_hat: np.ndarray,
        length_scale: float = 1.0
    ) -> np.ndarray:
        """Predict probabilities at test points."""
        # YOUR CODE HERE
        pass
    
    # Test
    np.random.seed(42)
    
    # Generate classification data
    n = 50
    X_train = np.random.randn(n, 2)
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(float)
    
    print("Test GP classification implementation...")


def solution5_gp_classification():
    """Solution for Exercise 5."""
    
    def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """RBF kernel."""
        sq_dist = (
            np.sum(X1**2, axis=1, keepdims=True) +
            np.sum(X2**2, axis=1) -
            2 * X1 @ X2.T
        )
        return np.exp(-0.5 * sq_dist / length_scale**2)
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def log_posterior(f: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
        """Compute log posterior."""
        # Log likelihood
        pi = sigmoid(f)
        log_lik = np.sum(y * np.log(pi + 1e-10) + (1 - y) * np.log(1 - pi + 1e-10))
        
        # Log prior
        K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
        log_prior = -0.5 * f @ K_inv @ f
        
        return log_lik + log_prior
    
    def grad_log_posterior(f: np.ndarray, y: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
        """Gradient of log posterior."""
        pi = sigmoid(f)
        return y - pi - K_inv @ f
    
    def hessian_log_posterior(f: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
        """Hessian of log posterior."""
        pi = sigmoid(f)
        W = np.diag(pi * (1 - pi))
        return -W - K_inv
    
    def find_mode(y: np.ndarray, K: np.ndarray, n_iter: int = 20) -> np.ndarray:
        """Find MAP estimate using Newton's method."""
        n = len(y)
        f = np.zeros(n)
        
        K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))
        
        for _ in range(n_iter):
            grad = grad_log_posterior(f, y, K_inv)
            hess = hessian_log_posterior(f, K_inv)
            
            # Newton step
            delta = np.linalg.solve(hess, grad)
            f = f - delta
            
            if np.linalg.norm(delta) < 1e-6:
                break
        
        return f
    
    def predict(
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, f_hat: np.ndarray,
        length_scale: float = 1.0
    ) -> np.ndarray:
        """Predict probabilities at test points."""
        n = len(X_train)
        
        K = rbf_kernel(X_train, X_train, length_scale) + 1e-6 * np.eye(n)
        K_inv = np.linalg.inv(K)
        
        pi_hat = sigmoid(f_hat)
        W = np.diag(pi_hat * (1 - pi_hat))
        
        # Posterior covariance
        post_cov_inv = K_inv + W
        post_cov = np.linalg.inv(post_cov_inv)
        
        # Predictive mean for latent function
        K_s = rbf_kernel(X_test, X_train, length_scale)
        f_mean = K_s @ (y_train - pi_hat)
        
        # Predictive variance
        v = np.linalg.solve(np.linalg.cholesky(K + np.linalg.inv(W)), K_s.T)
        f_var = 1.0 - np.sum(v**2, axis=0)
        f_var = np.maximum(f_var, 1e-6)
        
        # Probit approximation for predictive probabilities
        kappa = 1 / np.sqrt(1 + np.pi * f_var / 8)
        return sigmoid(kappa * f_mean)
    
    # Test
    np.random.seed(42)
    
    n = 50
    X_train = np.random.randn(n, 2)
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(float)
    
    # Find mode
    K = rbf_kernel(X_train, X_train, length_scale=1.0) + 1e-6 * np.eye(n)
    f_hat = find_mode(y_train, K)
    
    # Predict
    X_test = np.random.randn(20, 2)
    y_test = ((X_test[:, 0] + X_test[:, 1]) > 0).astype(float)
    
    probs = predict(X_train, y_train, X_test, f_hat, length_scale=1.0)
    preds = (probs > 0.5).astype(float)
    
    accuracy = np.mean(preds == y_test)
    
    print(f"GP Classification accuracy: {accuracy:.3f}")
    print(f"Prediction confidence (mean): {np.mean(np.abs(probs - 0.5) + 0.5):.3f}")


# =============================================================================
# Exercise 6: Metropolis-Hastings with Adaptive Proposal
# =============================================================================

def exercise6_adaptive_mcmc():
    """
    Implement adaptive Metropolis-Hastings algorithm.
    
    The proposal distribution is adapted during sampling:
        q(x'|x) = N(x, sigma^2 * C)
    
    where C is updated based on sample covariance.
    
    Tasks:
    1. Implement basic MH step
    2. Implement covariance adaptation
    3. Compute effective sample size
    4. Diagnose convergence
    """
    
    def target_log_prob(x: np.ndarray) -> float:
        """
        Target: Banana-shaped distribution
        
        p(x) propto exp(-0.5 * (x[0]^2 / 100 + (x[1] + 0.03*x[0]^2 - 3)^2))
        """
        # YOUR CODE HERE
        pass
    
    def mh_step(
        x: np.ndarray,
        log_prob: Callable,
        proposal_cov: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Single Metropolis-Hastings step.
        
        Returns:
            new_x: New state
            accepted: Whether proposal was accepted
        """
        # YOUR CODE HERE
        pass
    
    def adapt_covariance(
        samples: np.ndarray,
        initial_cov: np.ndarray,
        adapt_start: int = 100,
        adapt_end: int = 500
    ) -> np.ndarray:
        """
        Adapt proposal covariance based on samples.
        
        Returns:
            Adapted covariance matrix
        """
        # YOUR CODE HERE
        pass
    
    def effective_sample_size(samples: np.ndarray) -> float:
        """Compute effective sample size."""
        # YOUR CODE HERE
        pass
    
    def run_adaptive_mcmc(
        target_log_prob: Callable,
        initial_x: np.ndarray,
        n_samples: int = 5000,
        burnin: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Run adaptive MCMC.
        
        Returns:
            samples: MCMC samples
            acceptance_rate: Overall acceptance rate
        """
        # YOUR CODE HERE
        pass
    
    # Test
    print("Test adaptive MCMC implementation...")


def solution6_adaptive_mcmc():
    """Solution for Exercise 6."""
    
    def target_log_prob(x: np.ndarray) -> float:
        """Banana-shaped target distribution."""
        return -0.5 * (x[0]**2 / 100 + (x[1] + 0.03 * x[0]**2 - 3)**2)
    
    def mh_step(
        x: np.ndarray,
        log_prob: Callable,
        proposal_cov: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Single MH step."""
        # Propose
        x_prop = np.random.multivariate_normal(x, proposal_cov)
        
        # Compute acceptance probability
        log_alpha = log_prob(x_prop) - log_prob(x)
        
        # Accept/reject
        if np.log(np.random.rand()) < log_alpha:
            return x_prop, True
        return x, False
    
    def adapt_covariance(
        samples: np.ndarray,
        initial_cov: np.ndarray,
        adapt_start: int = 100,
        adapt_end: int = 500
    ) -> np.ndarray:
        """Adapt proposal covariance."""
        n = len(samples)
        
        if n < adapt_start:
            return initial_cov
        
        # Compute sample covariance
        samples_use = samples[adapt_start:min(n, adapt_end)]
        if len(samples_use) < 10:
            return initial_cov
        
        sample_cov = np.cov(samples_use.T)
        
        # Optimal scaling for Gaussian target (2.38^2 / d)
        d = samples.shape[1]
        scale = 2.38**2 / d
        
        # Mix with initial covariance for robustness
        weight = min(1.0, (n - adapt_start) / (adapt_end - adapt_start))
        
        return (1 - weight) * initial_cov + weight * scale * sample_cov
    
    def effective_sample_size(samples: np.ndarray) -> float:
        """Compute effective sample size using autocorrelation."""
        n = len(samples)
        
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
        
        ess_per_dim = []
        for d in range(samples.shape[1]):
            x = samples[:, d]
            x = x - np.mean(x)
            
            # Autocorrelation
            acf = np.correlate(x, x, mode='full')[n-1:]
            acf = acf / acf[0]
            
            # Sum until first negative
            tau = 1.0
            for k in range(1, n // 2):
                if acf[k] < 0:
                    break
                tau += 2 * acf[k]
            
            ess_per_dim.append(n / tau)
        
        return np.min(ess_per_dim)
    
    def run_adaptive_mcmc(
        target_log_prob: Callable,
        initial_x: np.ndarray,
        n_samples: int = 5000,
        burnin: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """Run adaptive MCMC."""
        d = len(initial_x)
        
        # Initial proposal covariance
        proposal_cov = 0.1 * np.eye(d)
        
        # Storage
        samples = np.zeros((n_samples + burnin, d))
        samples[0] = initial_x
        
        accepts = 0
        
        for i in range(1, n_samples + burnin):
            # Adapt proposal
            if i < burnin:
                proposal_cov = adapt_covariance(
                    samples[:i], 0.1 * np.eye(d),
                    adapt_start=100, adapt_end=burnin
                )
            
            # MH step
            samples[i], accepted = mh_step(samples[i-1], target_log_prob, proposal_cov)
            accepts += int(accepted)
        
        acceptance_rate = accepts / (n_samples + burnin - 1)
        
        return samples[burnin:], acceptance_rate
    
    # Test
    np.random.seed(42)
    
    initial_x = np.array([0.0, 0.0])
    samples, acc_rate = run_adaptive_mcmc(target_log_prob, initial_x, n_samples=5000, burnin=1000)
    
    print(f"Acceptance rate: {acc_rate:.3f}")
    print(f"Sample mean: {np.mean(samples, axis=0)}")
    print(f"Sample std: {np.std(samples, axis=0)}")
    print(f"Effective sample size: {effective_sample_size(samples):.1f}")


# =============================================================================
# Exercise 7: Normalizing Flow (Planar Flow)
# =============================================================================

def exercise7_planar_flow():
    """
    Implement planar normalizing flow.
    
    Transformation:
        f(z) = z + u * h(w^T z + b)
    
    where h is tanh and u, w, b are learnable parameters.
    
    Log-det Jacobian:
        log|det(df/dz)| = log|1 + u^T * h'(w^T z + b) * w|
    
    Tasks:
    1. Implement single planar layer
    2. Stack multiple layers
    3. Train to match target distribution
    """
    
    class PlanarLayer:
        def __init__(self, dim: int):
            self.dim = dim
            # YOUR CODE HERE: Initialize u, w, b
            pass
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Forward transformation.
            
            Returns:
                z': Transformed samples
                log_det: Log determinant of Jacobian
            """
            # YOUR CODE HERE
            pass
        
        def backward(self, grad_output: np.ndarray, z: np.ndarray):
            """
            Compute gradients for training.
            
            Returns:
                Gradients w.r.t. u, w, b
            """
            # YOUR CODE HERE
            pass
    
    class PlanarFlow:
        def __init__(self, dim: int, n_layers: int):
            self.layers = [PlanarLayer(dim) for _ in range(n_layers)]
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Transform through all layers."""
            # YOUR CODE HERE
            pass
        
        def log_prob(self, x: np.ndarray) -> np.ndarray:
            """Compute log probability of samples."""
            # YOUR CODE HERE
            pass
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Sample from the flow."""
            # YOUR CODE HERE
            pass
    
    # Test
    print("Test planar flow implementation...")


def solution7_planar_flow():
    """Solution for Exercise 7."""
    
    class PlanarLayer:
        def __init__(self, dim: int):
            self.dim = dim
            self.u = np.random.randn(dim) * 0.1
            self.w = np.random.randn(dim) * 0.1
            self.b = 0.0
        
        def _h(self, x: np.ndarray) -> np.ndarray:
            """Activation function (tanh)."""
            return np.tanh(x)
        
        def _h_prime(self, x: np.ndarray) -> np.ndarray:
            """Derivative of activation."""
            return 1 - np.tanh(x)**2
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Forward transformation."""
            # Ensure invertibility
            wu = self.w @ self.u
            if wu < -1:
                u_hat = self.u + (-1 - wu) * self.w / (self.w @ self.w)
            else:
                u_hat = self.u
            
            # Compute transformation
            linear = z @ self.w + self.b
            h_val = self._h(linear)
            z_new = z + np.outer(h_val, u_hat)
            
            # Log determinant
            h_prime = self._h_prime(linear)
            psi = h_prime[:, None] * self.w
            log_det = np.log(np.abs(1 + psi @ u_hat) + 1e-10)
            
            return z_new, log_det
    
    class PlanarFlow:
        def __init__(self, dim: int, n_layers: int):
            self.dim = dim
            self.layers = [PlanarLayer(dim) for _ in range(n_layers)]
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Transform through all layers."""
            log_det_total = np.zeros(len(z))
            
            for layer in self.layers:
                z, log_det = layer.forward(z)
                log_det_total += log_det
            
            return z, log_det_total
        
        def log_prob(self, x: np.ndarray) -> np.ndarray:
            """
            Compute log probability.
            
            Note: This requires inverse, which is not tractable for planar flows.
            In practice, we optimize the reverse KL divergence.
            """
            # Forward KL: need inverse
            # We'll compute using samples from base distribution
            raise NotImplementedError("Use sample-based training instead")
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Sample from the flow."""
            z = np.random.randn(n_samples, self.dim)
            x, _ = self.forward(z)
            return x
        
        def train_step(self, n_samples: int, target_log_prob: Callable, lr: float = 0.01):
            """
            Training step using reverse KL divergence.
            
            Minimize E_q[log q(x) - log p(x)]
            """
            # Sample from base
            z = np.random.randn(n_samples, self.dim)
            
            # Forward pass with intermediate values stored
            log_det_total = np.zeros(n_samples)
            zs = [z]
            
            for layer in self.layers:
                z, log_det = layer.forward(z)
                log_det_total += log_det
                zs.append(z)
            
            # Base log prob
            log_q_z = -0.5 * (self.dim * np.log(2 * np.pi) + np.sum(zs[0]**2, axis=1))
            
            # Transformed log prob (including Jacobian)
            log_q_x = log_q_z - log_det_total
            
            # Target log prob
            log_p_x = np.array([target_log_prob(x) for x in z])
            
            # Loss: reverse KL (approximated)
            loss = np.mean(log_q_x - log_p_x)
            
            return loss
    
    # Test
    np.random.seed(42)
    
    # Target: mixture of Gaussians
    def target_log_prob(x):
        p1 = np.exp(-0.5 * np.sum((x - np.array([2, 2]))**2))
        p2 = np.exp(-0.5 * np.sum((x - np.array([-2, -2]))**2))
        return np.log(0.5 * p1 + 0.5 * p2 + 1e-10)
    
    flow = PlanarFlow(dim=2, n_layers=8)
    
    # Generate samples
    samples = flow.sample(1000)
    
    print(f"Flow sample mean: {np.mean(samples, axis=0)}")
    print(f"Flow sample std: {np.std(samples, axis=0)}")


# =============================================================================
# Exercise 8: Variational Autoencoder Training
# =============================================================================

def exercise8_vae_training():
    """
    Implement VAE training with β-VAE objective.
    
    Loss = Reconstruction + β * KL
    
    Tasks:
    1. Implement encoder network
    2. Implement decoder network
    3. Implement ELBO loss with reparameterization
    4. Train with gradient descent
    """
    
    class BetaVAE:
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            beta: float = 1.0
        ):
            self.beta = beta
            # YOUR CODE HERE: Initialize networks
            pass
        
        def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Encode to latent distribution parameters.
            
            Returns:
                mu, log_var
            """
            # YOUR CODE HERE
            pass
        
        def decode(self, z: np.ndarray) -> np.ndarray:
            """Decode from latent space."""
            # YOUR CODE HERE
            pass
        
        def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
            """Reparameterization trick."""
            # YOUR CODE HERE
            pass
        
        def loss(
            self, x: np.ndarray
        ) -> Tuple[float, float, float]:
            """
            Compute β-VAE loss.
            
            Returns:
                total_loss, recon_loss, kl_loss
            """
            # YOUR CODE HERE
            pass
        
        def train_step(self, x: np.ndarray, lr: float = 0.001) -> float:
            """Training step with gradient descent."""
            # YOUR CODE HERE
            pass
    
    # Test
    print("Test VAE training implementation...")


def solution8_vae_training():
    """Solution for Exercise 8."""
    
    class BetaVAE:
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            latent_dim: int,
            beta: float = 1.0
        ):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.latent_dim = latent_dim
            self.beta = beta
            
            # He initialization
            def init(fan_in, fan_out):
                return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
            
            # Encoder
            self.W_enc = init(input_dim, hidden_dim)
            self.b_enc = np.zeros(hidden_dim)
            self.W_mu = init(hidden_dim, latent_dim)
            self.b_mu = np.zeros(latent_dim)
            self.W_logvar = init(hidden_dim, latent_dim)
            self.b_logvar = np.zeros(latent_dim)
            
            # Decoder
            self.W_dec1 = init(latent_dim, hidden_dim)
            self.b_dec1 = np.zeros(hidden_dim)
            self.W_dec2 = init(hidden_dim, input_dim)
            self.b_dec2 = np.zeros(input_dim)
        
        def _relu(self, x):
            return np.maximum(0, x)
        
        def _sigmoid(self, x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Encode to latent distribution parameters."""
            h = self._relu(x @ self.W_enc + self.b_enc)
            mu = h @ self.W_mu + self.b_mu
            log_var = h @ self.W_logvar + self.b_logvar
            return mu, log_var
        
        def decode(self, z: np.ndarray) -> np.ndarray:
            """Decode from latent space."""
            h = self._relu(z @ self.W_dec1 + self.b_dec1)
            return self._sigmoid(h @ self.W_dec2 + self.b_dec2)
        
        def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
            """Reparameterization trick."""
            std = np.exp(0.5 * log_var)
            eps = np.random.randn(*mu.shape)
            return mu + std * eps
        
        def loss(self, x: np.ndarray) -> Tuple[float, float, float]:
            """Compute β-VAE loss."""
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decode(z)
            
            # Reconstruction loss (BCE)
            eps = 1e-8
            recon_loss = -np.mean(np.sum(
                x * np.log(x_recon + eps) + (1 - x) * np.log(1 - x_recon + eps),
                axis=1
            ))
            
            # KL divergence
            kl_loss = -0.5 * np.mean(np.sum(
                1 + log_var - mu**2 - np.exp(log_var),
                axis=1
            ))
            
            total_loss = recon_loss + self.beta * kl_loss
            
            return total_loss, recon_loss, kl_loss
        
        def train_step(self, x: np.ndarray, lr: float = 0.001) -> float:
            """Training step with backpropagation."""
            batch_size = len(x)
            
            # Forward pass
            h_enc = self._relu(x @ self.W_enc + self.b_enc)
            mu = h_enc @ self.W_mu + self.b_mu
            log_var = h_enc @ self.W_logvar + self.b_logvar
            
            std = np.exp(0.5 * log_var)
            eps = np.random.randn(*mu.shape)
            z = mu + std * eps
            
            h_dec = self._relu(z @ self.W_dec1 + self.b_dec1)
            x_recon = self._sigmoid(h_dec @ self.W_dec2 + self.b_dec2)
            
            # Backward pass
            eps_small = 1e-8
            
            # Reconstruction gradient
            d_recon = -(x / (x_recon + eps_small) - (1 - x) / (1 - x_recon + eps_small))
            d_recon = d_recon * x_recon * (1 - x_recon) / batch_size
            
            # Decoder gradients
            d_W_dec2 = h_dec.T @ d_recon
            d_b_dec2 = np.sum(d_recon, axis=0)
            
            d_h_dec = d_recon @ self.W_dec2.T * (h_dec > 0)
            d_W_dec1 = z.T @ d_h_dec
            d_b_dec1 = np.sum(d_h_dec, axis=0)
            
            # Gradient w.r.t. z
            d_z = d_h_dec @ self.W_dec1.T
            
            # Reparameterization gradients
            d_mu = d_z + self.beta * mu / batch_size
            d_log_var = d_z * eps * 0.5 * std + self.beta * 0.5 * (np.exp(log_var) - 1) / batch_size
            
            # Encoder gradients
            d_W_mu = h_enc.T @ d_mu
            d_b_mu = np.sum(d_mu, axis=0)
            d_W_logvar = h_enc.T @ d_log_var
            d_b_logvar = np.sum(d_log_var, axis=0)
            
            d_h_enc = (d_mu @ self.W_mu.T + d_log_var @ self.W_logvar.T) * (h_enc > 0)
            d_W_enc = x.T @ d_h_enc
            d_b_enc = np.sum(d_h_enc, axis=0)
            
            # Update parameters
            self.W_enc -= lr * d_W_enc
            self.b_enc -= lr * d_b_enc
            self.W_mu -= lr * d_W_mu
            self.b_mu -= lr * d_b_mu
            self.W_logvar -= lr * d_W_logvar
            self.b_logvar -= lr * d_b_logvar
            self.W_dec1 -= lr * d_W_dec1
            self.b_dec1 -= lr * d_b_dec1
            self.W_dec2 -= lr * d_W_dec2
            self.b_dec2 -= lr * d_b_dec2
            
            total_loss, _, _ = self.loss(x)
            return total_loss
    
    # Test
    np.random.seed(42)
    
    # Generate simple data
    n_samples = 500
    data = np.random.binomial(1, 0.5, (n_samples, 20)).astype(float)
    
    vae = BetaVAE(input_dim=20, hidden_dim=32, latent_dim=5, beta=1.0)
    
    print("Training β-VAE...")
    for epoch in range(50):
        loss = vae.train_step(data, lr=0.01)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")


# =============================================================================
# Exercise 9: Importance Sampling
# =============================================================================

def exercise9_importance_sampling():
    """
    Implement importance sampling for expectation estimation.
    
    E_p[f(x)] ≈ (1/N) * sum_i f(x_i) * w_i
    
    where w_i = p(x_i) / q(x_i) and x_i ~ q
    
    Tasks:
    1. Implement basic importance sampling
    2. Implement self-normalized importance sampling
    3. Compute effective sample size
    4. Implement adaptive importance sampling
    """
    
    def importance_sampling(
        target_log_prob: Callable,
        proposal_log_prob: Callable,
        proposal_sample: Callable,
        f: Callable,
        n_samples: int
    ) -> Tuple[float, float]:
        """
        Basic importance sampling.
        
        Returns:
            estimate: Estimated expectation
            variance: Variance of estimate
        """
        # YOUR CODE HERE
        pass
    
    def self_normalized_is(
        target_log_prob: Callable,
        proposal_log_prob: Callable,
        proposal_sample: Callable,
        f: Callable,
        n_samples: int
    ) -> Tuple[float, float, float]:
        """
        Self-normalized importance sampling.
        
        Returns:
            estimate, variance, effective_sample_size
        """
        # YOUR CODE HERE
        pass
    
    def adaptive_is(
        target_log_prob: Callable,
        f: Callable,
        n_samples: int,
        n_rounds: int = 5
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Adaptive importance sampling with Gaussian mixture proposal.
        
        Returns:
            estimate, final_means, final_covs
        """
        # YOUR CODE HERE
        pass
    
    # Test
    print("Test importance sampling implementation...")


def solution9_importance_sampling():
    """Solution for Exercise 9."""
    
    def importance_sampling(
        target_log_prob: Callable,
        proposal_log_prob: Callable,
        proposal_sample: Callable,
        f: Callable,
        n_samples: int
    ) -> Tuple[float, float]:
        """Basic importance sampling."""
        # Sample from proposal
        samples = proposal_sample(n_samples)
        
        # Compute log weights
        log_weights = np.array([
            target_log_prob(x) - proposal_log_prob(x)
            for x in samples
        ])
        
        # Compute f values
        f_values = np.array([f(x) for x in samples])
        
        # For stability, work in log space
        log_w_max = np.max(log_weights)
        weights = np.exp(log_weights - log_w_max)
        
        # Estimate
        estimate = np.mean(f_values * weights) * np.exp(log_w_max)
        
        # Variance estimate
        variance = np.var(f_values * weights) * np.exp(2 * log_w_max) / n_samples
        
        return estimate, variance
    
    def self_normalized_is(
        target_log_prob: Callable,
        proposal_log_prob: Callable,
        proposal_sample: Callable,
        f: Callable,
        n_samples: int
    ) -> Tuple[float, float, float]:
        """Self-normalized importance sampling."""
        samples = proposal_sample(n_samples)
        
        # Log weights
        log_weights = np.array([
            target_log_prob(x) - proposal_log_prob(x)
            for x in samples
        ])
        
        # Normalize (log-sum-exp trick)
        log_w_max = np.max(log_weights)
        weights = np.exp(log_weights - log_w_max)
        weights = weights / np.sum(weights)
        
        # f values
        f_values = np.array([f(x) for x in samples])
        
        # Self-normalized estimate
        estimate = np.sum(weights * f_values)
        
        # Variance using bootstrap-like approximation
        variance = np.sum(weights * (f_values - estimate)**2)
        
        # Effective sample size
        ess = 1.0 / np.sum(weights**2)
        
        return estimate, variance, ess
    
    def adaptive_is(
        target_log_prob: Callable,
        f: Callable,
        n_samples: int,
        n_rounds: int = 5,
        dim: int = 2
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Adaptive importance sampling with Gaussian mixture."""
        n_components = 3
        
        # Initialize proposal
        means = np.random.randn(n_components, dim) * 2
        covs = np.array([np.eye(dim) for _ in range(n_components)])
        weights = np.ones(n_components) / n_components
        
        for round_idx in range(n_rounds):
            # Sample from mixture
            samples_per_comp = n_samples // n_components
            all_samples = []
            all_comp = []
            
            for k in range(n_components):
                samples = np.random.multivariate_normal(
                    means[k], covs[k], samples_per_comp
                )
                all_samples.append(samples)
                all_comp.extend([k] * samples_per_comp)
            
            samples = np.vstack(all_samples)
            comp = np.array(all_comp)
            
            # Compute proposal log prob (mixture)
            def proposal_log_prob(x, k):
                diff = x - means[k]
                return -0.5 * (diff @ np.linalg.inv(covs[k]) @ diff + \
                              np.linalg.slogdet(covs[k])[1] + dim * np.log(2 * np.pi))
            
            log_q = np.zeros(len(samples))
            for i, x in enumerate(samples):
                log_q[i] = np.log(np.sum([
                    weights[k] * np.exp(proposal_log_prob(x, k))
                    for k in range(n_components)
                ]) + 1e-10)
            
            # Compute target log prob and weights
            log_p = np.array([target_log_prob(x) for x in samples])
            log_weights = log_p - log_q
            
            # Normalize weights
            log_w_max = np.max(log_weights)
            norm_weights = np.exp(log_weights - log_w_max)
            norm_weights = norm_weights / np.sum(norm_weights)
            
            # Update proposal using weighted samples
            for k in range(n_components):
                mask = comp == k
                if np.sum(mask) > 0:
                    w_k = norm_weights[mask]
                    w_k = w_k / np.sum(w_k)
                    
                    means[k] = np.sum(w_k[:, None] * samples[mask], axis=0)
                    diff = samples[mask] - means[k]
                    covs[k] = diff.T @ (w_k[:, None] * diff) + 0.1 * np.eye(dim)
        
        # Final estimate
        f_values = np.array([f(x) for x in samples])
        estimate = np.sum(norm_weights * f_values)
        
        return estimate, means, covs
    
    # Test
    np.random.seed(42)
    
    # Target: mixture of Gaussians
    def target_log_prob(x):
        p1 = np.exp(-0.5 * np.sum((x - np.array([2, 0]))**2))
        p2 = np.exp(-0.5 * np.sum((x - np.array([-2, 0]))**2))
        return np.log(0.5 * p1 + 0.5 * p2 + 1e-10)
    
    # Proposal: standard Gaussian
    def proposal_log_prob(x):
        return -0.5 * (len(x) * np.log(2 * np.pi) + np.sum(x**2))
    
    def proposal_sample(n):
        return np.random.randn(n, 2)
    
    # Function to estimate: E[x[0]^2]
    def f(x):
        return x[0]**2
    
    # True value (by symmetry): E[x[0]^2] = 4 + 1 = 5
    
    # Test importance sampling
    est, var = importance_sampling(
        target_log_prob, proposal_log_prob, proposal_sample, f, 10000
    )
    print(f"Basic IS estimate: {est:.3f} (true: 5.0)")
    
    # Self-normalized
    est, var, ess = self_normalized_is(
        target_log_prob, proposal_log_prob, proposal_sample, f, 10000
    )
    print(f"Self-normalized IS: {est:.3f}, ESS: {ess:.1f}")


# =============================================================================
# Exercise 10: Complete Probabilistic Modeling Pipeline
# =============================================================================

def exercise10_complete_pipeline():
    """
    Implement a complete probabilistic modeling pipeline.
    
    Tasks:
    1. Data preprocessing
    2. Model selection (GMM vs Factor Analysis)
    3. Training with early stopping
    4. Posterior inference
    5. Model comparison using BIC
    6. Visualization of results
    """
    
    def preprocess_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize data.
        
        Returns:
            X_scaled, mean, std
        """
        # YOUR CODE HERE
        pass
    
    def train_gmm(
        X: np.ndarray, K: int, n_iter: int = 100
    ) -> Tuple[Dict, float]:
        """
        Train GMM and compute BIC.
        
        Returns:
            params, bic
        """
        # YOUR CODE HERE
        pass
    
    def train_factor_analysis(
        X: np.ndarray, q: int, n_iter: int = 100
    ) -> Tuple[Dict, float]:
        """
        Train Factor Analysis and compute BIC.
        
        Returns:
            params, bic
        """
        # YOUR CODE HERE
        pass
    
    def select_model(X: np.ndarray, max_K: int = 10, max_q: int = 5) -> Dict:
        """
        Select best model using BIC.
        
        Returns:
            best_model_info
        """
        # YOUR CODE HERE
        pass
    
    def posterior_inference(X: np.ndarray, model_params: Dict) -> np.ndarray:
        """
        Compute posterior assignments/projections.
        
        Returns:
            posterior (cluster assignments or latent factors)
        """
        # YOUR CODE HERE
        pass
    
    # Test
    print("Test complete pipeline implementation...")


def solution10_complete_pipeline():
    """Solution for Exercise 10."""
    
    def preprocess_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize data."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-6] = 1.0  # Prevent division by zero
        X_scaled = (X - mean) / std
        return X_scaled, mean, std
    
    def train_gmm(
        X: np.ndarray, K: int, n_iter: int = 100
    ) -> Tuple[Dict, float]:
        """Train GMM and compute BIC."""
        n, d = X.shape
        
        # Initialize
        idx = np.random.choice(n, K, replace=False)
        mu = X[idx].copy()
        sigma = np.ones((K, d)) * np.std(X, axis=0)
        pi = np.ones(K) / K
        
        for _ in range(n_iter):
            # E-step
            log_resp = np.zeros((n, K))
            for k in range(K):
                diff = X - mu[k]
                log_resp[:, k] = np.log(pi[k]) - 0.5 * np.sum(
                    np.log(sigma[k]**2) + diff**2 / sigma[k]**2, axis=1
                )
            
            log_resp -= np.max(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)
            
            # M-step
            N_k = resp.sum(axis=0)
            pi = N_k / n
            
            for k in range(K):
                mu[k] = resp[:, k] @ X / N_k[k]
                diff = X - mu[k]
                sigma[k] = np.sqrt(resp[:, k] @ (diff**2) / N_k[k])
                sigma[k] = np.maximum(sigma[k], 1e-6)
        
        # Compute log-likelihood
        ll = 0
        for i in range(n):
            p = 0
            for k in range(K):
                diff = X[i] - mu[k]
                p += pi[k] * np.exp(-0.5 * np.sum(
                    np.log(sigma[k]**2) + diff**2 / sigma[k]**2
                )) / np.sqrt((2 * np.pi)**d)
            ll += np.log(p + 1e-10)
        
        # BIC = -2 * ll + k * log(n), k = number of parameters
        n_params = K - 1 + K * d + K * d  # pi, mu, sigma
        bic = -2 * ll + n_params * np.log(n)
        
        params = {'type': 'gmm', 'K': K, 'pi': pi, 'mu': mu, 'sigma': sigma}
        return params, bic
    
    def train_factor_analysis(
        X: np.ndarray, q: int, n_iter: int = 100
    ) -> Tuple[Dict, float]:
        """Train Factor Analysis and compute BIC."""
        n, d = X.shape
        
        # Initialize
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        W = np.random.randn(d, q) * 0.1
        Psi = np.var(X_centered, axis=0)
        
        for _ in range(n_iter):
            # E-step
            Psi_inv = 1.0 / Psi
            M = np.eye(q) + W.T @ (Psi_inv[:, None] * W)
            M_inv = np.linalg.inv(M)
            
            E_z = X_centered @ (Psi_inv[:, None] * W) @ M_inv.T
            
            # M-step
            sum_E_zz = n * M_inv + E_z.T @ E_z
            W = X_centered.T @ E_z @ np.linalg.inv(sum_E_zz)
            
            Psi = np.mean(X_centered**2, axis=0) - \
                  np.mean(E_z @ W.T * X_centered, axis=0) * 2 + \
                  np.diag(W @ sum_E_zz @ W.T / n)
            Psi = np.maximum(Psi, 1e-6)
        
        # Log-likelihood
        C = W @ W.T + np.diag(Psi)
        C_inv = np.linalg.inv(C)
        log_det = np.linalg.slogdet(C)[1]
        
        ll = -0.5 * n * (d * np.log(2 * np.pi) + log_det)
        ll -= 0.5 * np.sum(X_centered @ C_inv * X_centered)
        
        # BIC
        n_params = d + d * q + d  # mu, W, Psi
        bic = -2 * ll + n_params * np.log(n)
        
        params = {'type': 'fa', 'q': q, 'mu': mu, 'W': W, 'Psi': Psi}
        return params, bic
    
    def select_model(X: np.ndarray, max_K: int = 5, max_q: int = 5) -> Dict:
        """Select best model using BIC."""
        best_bic = np.inf
        best_params = None
        
        # Try GMMs
        for K in range(1, max_K + 1):
            try:
                params, bic = train_gmm(X, K)
                if bic < best_bic:
                    best_bic = bic
                    best_params = params
            except:
                pass
        
        # Try Factor Analysis
        for q in range(1, min(max_q + 1, X.shape[1])):
            try:
                params, bic = train_factor_analysis(X, q)
                if bic < best_bic:
                    best_bic = bic
                    best_params = params
            except:
                pass
        
        return {'params': best_params, 'bic': best_bic}
    
    def posterior_inference(X: np.ndarray, model_params: Dict) -> np.ndarray:
        """Compute posterior assignments/projections."""
        if model_params['type'] == 'gmm':
            K = model_params['K']
            pi = model_params['pi']
            mu = model_params['mu']
            sigma = model_params['sigma']
            
            n = len(X)
            log_resp = np.zeros((n, K))
            for k in range(K):
                diff = X - mu[k]
                log_resp[:, k] = np.log(pi[k]) - 0.5 * np.sum(
                    np.log(sigma[k]**2) + diff**2 / sigma[k]**2, axis=1
                )
            
            return np.argmax(log_resp, axis=1)
        
        else:  # Factor Analysis
            mu = model_params['mu']
            W = model_params['W']
            Psi = model_params['Psi']
            q = model_params['q']
            
            X_centered = X - mu
            Psi_inv = 1.0 / Psi
            M = np.eye(q) + W.T @ (Psi_inv[:, None] * W)
            M_inv = np.linalg.inv(M)
            
            return X_centered @ (Psi_inv[:, None] * W) @ M_inv.T
    
    # Test
    np.random.seed(42)
    
    # Generate data
    n = 300
    # Mixture of low-rank + noise
    true_W = np.random.randn(10, 2)
    z = np.random.randn(n, 2)
    X = z @ true_W.T + 0.5 * np.random.randn(n, 10)
    
    # Preprocess
    X_scaled, mean, std = preprocess_data(X)
    
    # Model selection
    print("Selecting model...")
    result = select_model(X_scaled, max_K=5, max_q=5)
    
    print(f"\nBest model: {result['params']['type']}")
    print(f"BIC: {result['bic']:.2f}")
    
    if result['params']['type'] == 'gmm':
        print(f"Number of components: {result['params']['K']}")
    else:
        print(f"Number of factors: {result['params']['q']}")
    
    # Posterior inference
    posterior = posterior_inference(X_scaled, result['params'])
    print(f"Posterior shape: {posterior.shape}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("PROBABILISTIC MODELS: EXERCISES")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Exercise 1: Bayesian Inference with Conjugate Priors")
    print("=" * 70)
    solution1_bayesian_inference()
    
    print("\n" + "=" * 70)
    print("Exercise 2: EM for Gaussian Mixture Model")
    print("=" * 70)
    solution2_em_gmm()
    
    print("\n" + "=" * 70)
    print("Exercise 3: Variational Factor Analysis")
    print("=" * 70)
    solution3_variational_factor_analysis()
    
    print("\n" + "=" * 70)
    print("Exercise 4: Hidden Markov Model")
    print("=" * 70)
    solution4_hmm()
    
    print("\n" + "=" * 70)
    print("Exercise 5: Gaussian Process Classification")
    print("=" * 70)
    solution5_gp_classification()
    
    print("\n" + "=" * 70)
    print("Exercise 6: Adaptive MCMC")
    print("=" * 70)
    solution6_adaptive_mcmc()
    
    print("\n" + "=" * 70)
    print("Exercise 7: Planar Normalizing Flow")
    print("=" * 70)
    solution7_planar_flow()
    
    print("\n" + "=" * 70)
    print("Exercise 8: β-VAE Training")
    print("=" * 70)
    solution8_vae_training()
    
    print("\n" + "=" * 70)
    print("Exercise 9: Importance Sampling")
    print("=" * 70)
    solution9_importance_sampling()
    
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Pipeline")
    print("=" * 70)
    solution10_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
