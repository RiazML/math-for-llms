"""
Probabilistic Models: Implementation Examples
============================================

From-scratch implementations of fundamental probabilistic models.
"""

import numpy as np
from typing import Tuple, List, Callable, Optional, Dict
from dataclasses import dataclass


# =============================================================================
# Example 1: Bayesian Inference - Conjugate Priors
# =============================================================================

class BayesianInference:
    """
    Bayesian inference with conjugate priors.
    
    Conjugate pairs allow closed-form posterior updates.
    """
    
    @staticmethod
    def beta_binomial(
        successes: int,
        trials: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> Tuple[float, float, float, float]:
        """
        Beta-Binomial conjugate pair.
        
        Prior: Beta(alpha, beta)
        Likelihood: Binomial(n, p)
        Posterior: Beta(alpha + successes, beta + failures)
        
        Returns:
            posterior_alpha, posterior_beta, posterior_mean, posterior_var
        """
        failures = trials - successes
        
        post_alpha = prior_alpha + successes
        post_beta = prior_beta + failures
        
        # Posterior statistics
        post_mean = post_alpha / (post_alpha + post_beta)
        post_var = (post_alpha * post_beta) / (
            (post_alpha + post_beta)**2 * (post_alpha + post_beta + 1)
        )
        
        return post_alpha, post_beta, post_mean, post_var
    
    @staticmethod
    def normal_normal(
        data: np.ndarray,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        likelihood_var: float = 1.0
    ) -> Tuple[float, float]:
        """
        Normal-Normal conjugate pair (known variance).
        
        Prior: N(mu_0, sigma_0^2)
        Likelihood: N(mu, sigma^2)
        Posterior: N(mu_n, sigma_n^2)
        
        Returns:
            posterior_mean, posterior_var
        """
        n = len(data)
        data_mean = np.mean(data)
        
        # Precision (inverse variance)
        prior_prec = 1.0 / prior_var
        likelihood_prec = n / likelihood_var
        
        # Posterior precision and variance
        post_prec = prior_prec + likelihood_prec
        post_var = 1.0 / post_prec
        
        # Posterior mean (precision-weighted average)
        post_mean = post_var * (prior_prec * prior_mean + likelihood_prec * data_mean)
        
        return post_mean, post_var
    
    @staticmethod
    def gamma_poisson(
        data: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Gamma-Poisson conjugate pair.
        
        Prior: Gamma(alpha, beta)
        Likelihood: Poisson(lambda)
        Posterior: Gamma(alpha + sum(x), beta + n)
        
        Returns:
            posterior_alpha, posterior_beta, posterior_mean
        """
        n = len(data)
        data_sum = np.sum(data)
        
        post_alpha = prior_alpha + data_sum
        post_beta = prior_beta + n
        
        # Posterior mean = alpha/beta for Gamma
        post_mean = post_alpha / post_beta
        
        return post_alpha, post_beta, post_mean


def example_bayesian_inference():
    """Demonstrate Bayesian inference with conjugate priors."""
    print("=" * 70)
    print("Example 1: Bayesian Inference with Conjugate Priors")
    print("=" * 70)
    
    bi = BayesianInference()
    
    # Beta-Binomial: Coin flip estimation
    print("\n1. Beta-Binomial (Coin Fairness):")
    print("-" * 40)
    
    # Observe 7 heads in 10 flips
    successes, trials = 7, 10
    
    # Uniform prior (no prior belief)
    alpha, beta, mean, var = bi.beta_binomial(successes, trials, 1, 1)
    print(f"Data: {successes} heads in {trials} flips")
    print(f"Uniform prior Beta(1,1):")
    print(f"  Posterior: Beta({alpha:.1f}, {beta:.1f})")
    print(f"  Posterior mean: {mean:.3f}")
    print(f"  95% credible interval: [{mean - 1.96*np.sqrt(var):.3f}, {mean + 1.96*np.sqrt(var):.3f}]")
    
    # Informative prior (believe coin is fair)
    alpha, beta, mean, var = bi.beta_binomial(successes, trials, 10, 10)
    print(f"Informative prior Beta(10,10):")
    print(f"  Posterior: Beta({alpha:.1f}, {beta:.1f})")
    print(f"  Posterior mean: {mean:.3f} (pulled toward 0.5)")
    
    # Normal-Normal: Estimating mean
    print("\n2. Normal-Normal (Mean Estimation):")
    print("-" * 40)
    
    np.random.seed(42)
    true_mean = 5.0
    data = np.random.normal(true_mean, 1.0, 20)
    
    post_mean, post_var = bi.normal_normal(data, prior_mean=0, prior_var=10, likelihood_var=1)
    print(f"True mean: {true_mean}")
    print(f"Sample mean: {np.mean(data):.3f}")
    print(f"Prior: N(0, 10)")
    print(f"Posterior: N({post_mean:.3f}, {post_var:.4f})")
    print(f"Posterior 95% CI: [{post_mean - 1.96*np.sqrt(post_var):.3f}, {post_mean + 1.96*np.sqrt(post_var):.3f}]")
    
    # Gamma-Poisson: Estimating rate
    print("\n3. Gamma-Poisson (Rate Estimation):")
    print("-" * 40)
    
    true_lambda = 3.0
    data = np.random.poisson(true_lambda, 50)
    
    post_alpha, post_beta, post_mean = bi.gamma_poisson(data, 1, 1)
    print(f"True rate: {true_lambda}")
    print(f"Sample mean: {np.mean(data):.3f}")
    print(f"Posterior mean: {post_mean:.3f}")


# =============================================================================
# Example 2: Gaussian Mixture Model with EM
# =============================================================================

class GaussianMixtureModel:
    """
    Gaussian Mixture Model with Expectation-Maximization.
    
    Model: p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
    """
    
    def __init__(self, n_components: int, covariance_type: str = 'full'):
        """
        Args:
            n_components: Number of mixture components
            covariance_type: 'full', 'diag', or 'spherical'
        """
        self.K = n_components
        self.cov_type = covariance_type
        self.weights_ = None  # pi_k
        self.means_ = None    # mu_k
        self.covariances_ = None  # Sigma_k
        
    def _initialize(self, X: np.ndarray):
        """Initialize parameters using k-means++."""
        n_samples, n_features = X.shape
        
        # Initialize means using k-means++ selection
        self.means_ = np.zeros((self.K, n_features))
        
        # First center: random
        idx = np.random.randint(n_samples)
        self.means_[0] = X[idx]
        
        # Subsequent centers: probability proportional to squared distance
        for k in range(1, self.K):
            dists = np.min([
                np.sum((X - self.means_[j])**2, axis=1) 
                for j in range(k)
            ], axis=0)
            probs = dists / np.sum(dists)
            idx = np.random.choice(n_samples, p=probs)
            self.means_[k] = X[idx]
        
        # Initialize covariances
        if self.cov_type == 'full':
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.K)])
        elif self.cov_type == 'diag':
            self.covariances_ = np.ones((self.K, n_features))
        else:  # spherical
            self.covariances_ = np.ones(self.K)
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.K) / self.K
    
    def _gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute Gaussian PDF for all samples."""
        n_features = X.shape[1]
        
        if self.cov_type == 'full':
            diff = X - mean
            # Add small regularization for numerical stability
            cov_reg = cov + 1e-6 * np.eye(n_features)
            
            # Using Cholesky decomposition for stability
            L = np.linalg.cholesky(cov_reg)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve L @ z = diff.T
            z = np.linalg.solve(L, diff.T)
            mahal = np.sum(z**2, axis=0)
            
        elif self.cov_type == 'diag':
            diff = X - mean
            cov_reg = cov + 1e-6
            log_det = np.sum(np.log(cov_reg))
            mahal = np.sum(diff**2 / cov_reg, axis=1)
            
        else:  # spherical
            diff = X - mean
            cov_reg = cov + 1e-6
            log_det = n_features * np.log(cov_reg)
            mahal = np.sum(diff**2, axis=1) / cov_reg
        
        log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahal)
        return np.exp(log_prob)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E-step: Compute responsibilities.
        
        gamma_nk = p(z_n = k | x_n) = pi_k * N(x_n | mu_k, Sigma_k) / sum_j ...
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.K))
        
        for k in range(self.K):
            responsibilities[:, k] = self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )
        
        # Normalize
        total = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities /= total + 1e-10
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        M-step: Update parameters.
        
        N_k = sum_n gamma_nk
        pi_k = N_k / N
        mu_k = (1/N_k) * sum_n gamma_nk * x_n
        Sigma_k = (1/N_k) * sum_n gamma_nk * (x_n - mu_k)(x_n - mu_k)^T
        """
        n_samples, n_features = X.shape
        
        # Effective number of points per component
        N_k = np.sum(responsibilities, axis=0)
        
        # Update weights
        self.weights_ = N_k / n_samples
        
        # Update means
        self.means_ = responsibilities.T @ X / N_k[:, np.newaxis]
        
        # Update covariances
        for k in range(self.K):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            
            if self.cov_type == 'full':
                self.covariances_[k] = weighted_diff.T @ diff / N_k[k]
            elif self.cov_type == 'diag':
                self.covariances_[k] = np.sum(weighted_diff * diff, axis=0) / N_k[k]
            else:  # spherical
                self.covariances_[k] = np.sum(weighted_diff * diff) / (N_k[k] * n_features)
    
    def fit(self, X: np.ndarray, n_iter: int = 100, tol: float = 1e-6) -> 'GaussianMixtureModel':
        """Fit GMM using EM algorithm."""
        self._initialize(X)
        
        prev_ll = -np.inf
        
        for iteration in range(n_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            ll = self.score(X)
            
            # Check convergence
            if abs(ll - prev_ll) < tol:
                print(f"Converged at iteration {iteration}")
                break
            
            prev_ll = ll
        
        return self
    
    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood."""
        n_samples = X.shape[0]
        log_prob = np.zeros(n_samples)
        
        for k in range(self.K):
            log_prob += self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )
        
        return np.sum(np.log(log_prob + 1e-10))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments."""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)


def example_gmm():
    """Demonstrate Gaussian Mixture Model."""
    print("\n" + "=" * 70)
    print("Example 2: Gaussian Mixture Model with EM")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data from 3 Gaussians
    n_samples = 300
    
    # True parameters
    true_means = [[-2, -2], [0, 3], [3, 0]]
    true_covs = [
        [[1, 0.5], [0.5, 1]],
        [[1, -0.3], [-0.3, 0.5]],
        [[0.5, 0], [0, 1.5]]
    ]
    
    X = np.vstack([
        np.random.multivariate_normal(true_means[i], true_covs[i], n_samples // 3)
        for i in range(3)
    ])
    
    # Fit GMM
    gmm = GaussianMixtureModel(n_components=3, covariance_type='full')
    gmm.fit(X, n_iter=50)
    
    print("\nFitted Parameters:")
    print("-" * 40)
    
    for k in range(3):
        print(f"\nComponent {k + 1}:")
        print(f"  Weight: {gmm.weights_[k]:.3f}")
        print(f"  Mean: {gmm.means_[k]}")
        print(f"  Covariance:\n{gmm.covariances_[k]}")
    
    # Predictions
    labels = gmm.predict(X)
    print(f"\nCluster distribution: {np.bincount(labels)}")
    print(f"Log-likelihood: {gmm.score(X):.2f}")


# =============================================================================
# Example 3: Variational Inference - Mean Field
# =============================================================================

class MeanFieldVI:
    """
    Mean-field variational inference for Bayesian linear regression.
    
    Model:
        w ~ N(0, alpha^-1 * I)  (prior)
        y ~ N(X @ w, beta^-1)   (likelihood)
    
    Approximate posterior:
        q(w) = N(m, S)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            alpha: Prior precision on weights
            beta: Observation noise precision
        """
        self.alpha = alpha
        self.beta = beta
        self.m = None  # Posterior mean
        self.S = None  # Posterior covariance
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MeanFieldVI':
        """
        Compute variational posterior.
        
        Closed form for Gaussian:
            S = (alpha * I + beta * X^T X)^-1
            m = beta * S @ X^T @ y
        """
        n_features = X.shape[1]
        
        # Posterior covariance
        self.S = np.linalg.inv(
            self.alpha * np.eye(n_features) + self.beta * X.T @ X
        )
        
        # Posterior mean
        self.m = self.beta * self.S @ X.T @ y
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """
        Predict with uncertainty quantification.
        
        Predictive distribution:
            p(y* | x*) = N(x* @ m, x* @ S @ x*^T + beta^-1)
        """
        mean = X @ self.m
        
        if return_std:
            # Epistemic uncertainty (model uncertainty)
            var_epistemic = np.sum((X @ self.S) * X, axis=1)
            # Aleatoric uncertainty (observation noise)
            var_aleatoric = 1.0 / self.beta
            # Total variance
            var_total = var_epistemic + var_aleatoric
            
            return mean, np.sqrt(var_total)
        
        return mean
    
    def elbo(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Evidence Lower Bound.
        
        ELBO = E_q[log p(y|w)] - KL(q(w) || p(w))
        """
        n_samples, n_features = X.shape
        
        # Expected log likelihood
        pred = X @ self.m
        residual = y - pred
        
        # E_q[(y - Xw)^T (y - Xw)] = (y - Xm)^T(y - Xm) + tr(X^T X S)
        expected_sq_error = residual @ residual + np.trace(X.T @ X @ self.S)
        
        E_log_lik = 0.5 * n_samples * np.log(self.beta / (2 * np.pi))
        E_log_lik -= 0.5 * self.beta * expected_sq_error
        
        # KL divergence between q(w) and p(w)
        # KL(N(m, S) || N(0, alpha^-1 I)) = 0.5 * (alpha * (m^T m + tr(S)) - n - log|alpha * S|)
        KL = 0.5 * (
            self.alpha * (self.m @ self.m + np.trace(self.S))
            - n_features
            - np.linalg.slogdet(self.alpha * self.S)[1]
        )
        
        return E_log_lik - KL


def example_variational_inference():
    """Demonstrate variational inference for Bayesian linear regression."""
    print("\n" + "=" * 70)
    print("Example 3: Mean-Field Variational Inference")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    n_samples = 100
    true_w = np.array([2.0, -1.5, 0.5])
    
    X = np.random.randn(n_samples, 3)
    y = X @ true_w + 0.5 * np.random.randn(n_samples)
    
    # Fit variational posterior
    vi = MeanFieldVI(alpha=1.0, beta=4.0)  # beta = 1/sigma^2 = 1/0.25 = 4
    vi.fit(X, y)
    
    print("\nTrue weights:", true_w)
    print("Posterior mean:", vi.m)
    print("Posterior std:", np.sqrt(np.diag(vi.S)))
    
    # Predictions with uncertainty
    X_test = np.random.randn(5, 3)
    y_test = X_test @ true_w
    
    pred_mean, pred_std = vi.predict(X_test, return_std=True)
    
    print("\nPredictions with uncertainty:")
    print("-" * 40)
    for i in range(5):
        print(f"True: {y_test[i]:7.3f}, Pred: {pred_mean[i]:7.3f} ± {1.96*pred_std[i]:.3f}")
    
    print(f"\nELBO: {vi.elbo(X, y):.2f}")


# =============================================================================
# Example 4: Variational Autoencoder
# =============================================================================

class VAE:
    """
    Variational Autoencoder with Gaussian encoder and decoder.
    
    Encoder: q(z|x) = N(mu(x), diag(sigma^2(x)))
    Decoder: p(x|z) = N(f(z), sigma^2 * I)
    Prior: p(z) = N(0, I)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        learning_rate: float = 0.001
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        
        # Encoder: x -> hidden -> (mu, log_var)
        scale = 0.1
        self.W_enc1 = scale * np.random.randn(input_dim, hidden_dim)
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_mu = scale * np.random.randn(hidden_dim, latent_dim)
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = scale * np.random.randn(hidden_dim, latent_dim)
        self.b_logvar = np.zeros(latent_dim)
        
        # Decoder: z -> hidden -> x_recon
        self.W_dec1 = scale * np.random.randn(latent_dim, hidden_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = scale * np.random.randn(hidden_dim, input_dim)
        self.b_dec2 = np.zeros(input_dim)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode input to latent distribution parameters.
        
        Returns:
            mu, log_var, hidden activation
        """
        h = self._relu(x @ self.W_enc1 + self.b_enc1)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        
        return mu, log_var, h
    
    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        This allows gradients to flow through the sampling operation.
        """
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps
    
    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode latent to reconstruction.
        
        Returns:
            reconstruction, hidden activation
        """
        h = self._relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = self._sigmoid(h @ self.W_dec2 + self.b_dec2)
        
        return x_recon, h
    
    def loss(
        self, x: np.ndarray, x_recon: np.ndarray,
        mu: np.ndarray, log_var: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute VAE loss = Reconstruction loss + KL divergence.
        
        Reconstruction: -E_q[log p(x|z)] ≈ BCE(x, x_recon)
        KL: D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        """
        # Binary cross-entropy for reconstruction
        eps = 1e-8
        recon_loss = -np.mean(
            x * np.log(x_recon + eps) + (1 - x) * np.log(1 - x_recon + eps)
        )
        
        # KL divergence (closed form for Gaussian)
        kl_loss = -0.5 * np.mean(
            1 + log_var - mu**2 - np.exp(log_var)
        )
        
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def forward(self, x: np.ndarray) -> Dict:
        """Full forward pass."""
        mu, log_var, h_enc = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon, h_dec = self.decode(z)
        
        return {
            'mu': mu,
            'log_var': log_var,
            'z': z,
            'x_recon': x_recon,
            'h_enc': h_enc,
            'h_dec': h_dec
        }
    
    def train_step(self, x: np.ndarray) -> float:
        """Single training step with backprop."""
        batch_size = x.shape[0]
        
        # Forward pass
        out = self.forward(x)
        mu = out['mu']
        log_var = out['log_var']
        z = out['z']
        x_recon = out['x_recon']
        h_enc = out['h_enc']
        h_dec = out['h_dec']
        
        # Compute loss
        total_loss, _, _ = self.loss(x, x_recon, mu, log_var)
        
        # Backward pass
        eps = 1e-8
        
        # Gradient of reconstruction loss w.r.t. x_recon
        d_x_recon = -(x / (x_recon + eps) - (1 - x) / (1 - x_recon + eps)) / batch_size
        
        # Through sigmoid
        d_logits = d_x_recon * x_recon * (1 - x_recon)
        
        # Decoder gradients
        d_W_dec2 = h_dec.T @ d_logits
        d_b_dec2 = np.sum(d_logits, axis=0)
        
        d_h_dec = d_logits @ self.W_dec2.T
        d_h_dec *= self._relu_grad(z @ self.W_dec1 + self.b_dec1)
        
        d_W_dec1 = z.T @ d_h_dec
        d_b_dec1 = np.sum(d_h_dec, axis=0)
        
        # Gradient w.r.t. z
        d_z = d_h_dec @ self.W_dec1.T
        
        # Reparameterization: z = mu + exp(0.5 * log_var) * eps
        std = np.exp(0.5 * log_var)
        eps_sample = (z - mu) / (std + 1e-8)
        
        d_mu = d_z
        d_log_var = d_z * eps_sample * 0.5 * std
        
        # Add KL gradients
        d_mu += mu / batch_size
        d_log_var += 0.5 * (np.exp(log_var) - 1) / batch_size
        
        # Encoder gradients
        d_W_mu = h_enc.T @ d_mu
        d_b_mu = np.sum(d_mu, axis=0)
        
        d_W_logvar = h_enc.T @ d_log_var
        d_b_logvar = np.sum(d_log_var, axis=0)
        
        d_h_enc = d_mu @ self.W_mu.T + d_log_var @ self.W_logvar.T
        d_h_enc *= self._relu_grad(x @ self.W_enc1 + self.b_enc1)
        
        d_W_enc1 = x.T @ d_h_enc
        d_b_enc1 = np.sum(d_h_enc, axis=0)
        
        # Update parameters
        self.W_enc1 -= self.lr * d_W_enc1
        self.b_enc1 -= self.lr * d_b_enc1
        self.W_mu -= self.lr * d_W_mu
        self.b_mu -= self.lr * d_b_mu
        self.W_logvar -= self.lr * d_W_logvar
        self.b_logvar -= self.lr * d_b_logvar
        
        self.W_dec1 -= self.lr * d_W_dec1
        self.b_dec1 -= self.lr * d_b_dec1
        self.W_dec2 -= self.lr * d_W_dec2
        self.b_dec2 -= self.lr * d_b_dec2
        
        return total_loss
    
    def generate(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples from prior."""
        z = np.random.randn(n_samples, self.latent_dim)
        x_gen, _ = self.decode(z)
        return x_gen


def example_vae():
    """Demonstrate Variational Autoencoder."""
    print("\n" + "=" * 70)
    print("Example 4: Variational Autoencoder")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate simple binary data (8x8 patterns)
    n_samples = 500
    input_dim = 64
    
    # Create structured patterns
    X = np.zeros((n_samples, input_dim))
    for i in range(n_samples):
        pattern = np.zeros((8, 8))
        # Random vertical or horizontal line
        if np.random.rand() > 0.5:
            col = np.random.randint(8)
            pattern[:, col] = 1
        else:
            row = np.random.randint(8)
            pattern[row, :] = 1
        # Add noise
        pattern = (pattern + 0.1 * np.random.rand(8, 8)) > 0.5
        X[i] = pattern.flatten().astype(float)
    
    # Create and train VAE
    vae = VAE(input_dim=64, hidden_dim=32, latent_dim=2, learning_rate=0.01)
    
    print("\nTraining VAE...")
    print("-" * 40)
    
    batch_size = 50
    n_epochs = 100
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        total_loss = 0
        
        for i in range(0, n_samples, batch_size):
            batch = X[indices[i:i+batch_size]]
            loss = vae.train_step(batch)
            total_loss += loss
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / (n_samples // batch_size)
            print(f"Epoch {epoch + 1:3d}: Loss = {avg_loss:.4f}")
    
    # Evaluate
    out = vae.forward(X[:10])
    _, recon_loss, kl_loss = vae.loss(X[:10], out['x_recon'], out['mu'], out['log_var'])
    
    print(f"\nFinal reconstruction loss: {recon_loss:.4f}")
    print(f"Final KL divergence: {kl_loss:.4f}")
    print(f"Latent space mean magnitude: {np.mean(np.abs(out['mu'])):.4f}")
    
    # Generate samples
    samples = vae.generate(5)
    print(f"\nGenerated sample range: [{samples.min():.3f}, {samples.max():.3f}]")


# =============================================================================
# Example 5: Hidden Markov Model
# =============================================================================

class HiddenMarkovModel:
    """
    Hidden Markov Model with discrete emissions.
    
    - States: z_t in {1, ..., K}
    - Observations: x_t in {1, ..., M}
    - Transition: A[i,j] = P(z_t = j | z_{t-1} = i)
    - Emission: B[j,k] = P(x_t = k | z_t = j)
    - Initial: pi[i] = P(z_1 = i)
    """
    
    def __init__(self, n_states: int, n_obs: int):
        """
        Args:
            n_states: Number of hidden states K
            n_obs: Number of observation symbols M
        """
        self.K = n_states
        self.M = n_obs
        
        # Initialize randomly
        self.pi = np.ones(n_states) / n_states
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.B = np.random.dirichlet(np.ones(n_obs), size=n_states)
    
    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward algorithm.
        
        alpha_t(j) = P(x_1, ..., x_t, z_t = j)
        
        Returns:
            alpha: Forward probabilities (T, K)
            scale: Scaling factors for numerical stability
        """
        T = len(observations)
        alpha = np.zeros((T, self.K))
        scale = np.zeros(T)
        
        # Initialize
        alpha[0] = self.pi * self.B[:, observations[0]]
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0]
        
        # Forward recursion
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, observations[t]]
            scale[t] = np.sum(alpha[t])
            alpha[t] /= scale[t]
        
        return alpha, scale
    
    def backward(self, observations: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Backward algorithm.
        
        beta_t(i) = P(x_{t+1}, ..., x_T | z_t = i)
        
        Returns:
            beta: Backward probabilities (T, K)
        """
        T = len(observations)
        beta = np.zeros((T, self.K))
        
        # Initialize
        beta[T-1] = 1.0
        
        # Backward recursion
        for t in range(T-2, -1, -1):
            beta[t] = self.A @ (self.B[:, observations[t+1]] * beta[t+1])
            beta[t] /= scale[t+1]
        
        return beta
    
    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi algorithm for most likely state sequence.
        
        delta_t(j) = max_{z_1,...,z_{t-1}} P(z_1,...,z_{t-1}, z_t=j, x_1,...,x_t)
        
        Returns:
            path: Most likely state sequence
            log_prob: Log probability of the path
        """
        T = len(observations)
        delta = np.zeros((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        
        # Initialize (log domain for stability)
        delta[0] = np.log(self.pi + 1e-10) + np.log(self.B[:, observations[0]] + 1e-10)
        
        # Forward recursion
        log_A = np.log(self.A + 1e-10)
        log_B = np.log(self.B + 1e-10)
        
        for t in range(1, T):
            for j in range(self.K):
                scores = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_B[j, observations[t]]
        
        # Backtrack
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[T-1])
        log_prob = delta[T-1, path[T-1]]
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, log_prob
    
    def fit(
        self, observations: np.ndarray, n_iter: int = 100
    ) -> List[float]:
        """
        Baum-Welch algorithm (EM for HMM).
        
        Returns:
            log_likelihoods: Log-likelihood history
        """
        T = len(observations)
        log_likelihoods = []
        
        for iteration in range(n_iter):
            # E-step: compute responsibilities
            alpha, scale = self.forward(observations)
            beta = self.backward(observations, scale)
            
            # Gamma: P(z_t = i | x)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            
            # Xi: P(z_t = i, z_{t+1} = j | x)
            xi = np.zeros((T-1, self.K, self.K))
            for t in range(T-1):
                xi[t] = (alpha[t:t+1].T @ 
                        (beta[t+1:t+2] * self.B[:, observations[t+1]]))
                xi[t] *= self.A
                xi[t] /= xi[t].sum()
            
            # M-step: update parameters
            self.pi = gamma[0]
            
            self.A = xi.sum(axis=0)
            self.A /= self.A.sum(axis=1, keepdims=True)
            
            for k in range(self.M):
                mask = (observations == k).astype(float)
                self.B[:, k] = np.sum(gamma * mask[:, np.newaxis], axis=0)
            self.B /= self.B.sum(axis=1, keepdims=True)
            
            # Log-likelihood
            ll = np.sum(np.log(scale))
            log_likelihoods.append(ll)
            
            # Convergence check
            if iteration > 0 and abs(ll - log_likelihoods[-2]) < 1e-6:
                break
        
        return log_likelihoods


def example_hmm():
    """Demonstrate Hidden Markov Model."""
    print("\n" + "=" * 70)
    print("Example 5: Hidden Markov Model")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True HMM parameters (2 states, 3 observations)
    true_pi = np.array([0.7, 0.3])
    true_A = np.array([[0.8, 0.2],
                       [0.3, 0.7]])
    true_B = np.array([[0.7, 0.2, 0.1],
                       [0.1, 0.3, 0.6]])
    
    # Generate data
    T = 200
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T, dtype=int)
    
    states[0] = np.random.choice(2, p=true_pi)
    observations[0] = np.random.choice(3, p=true_B[states[0]])
    
    for t in range(1, T):
        states[t] = np.random.choice(2, p=true_A[states[t-1]])
        observations[t] = np.random.choice(3, p=true_B[states[t]])
    
    print("\nTrue parameters:")
    print(f"Initial: {true_pi}")
    print(f"Transition:\n{true_A}")
    print(f"Emission:\n{true_B}")
    
    # Fit HMM
    hmm = HiddenMarkovModel(n_states=2, n_obs=3)
    ll_history = hmm.fit(observations, n_iter=50)
    
    print("\nLearned parameters:")
    print(f"Initial: {hmm.pi}")
    print(f"Transition:\n{hmm.A}")
    print(f"Emission:\n{hmm.B}")
    
    # Viterbi decoding
    path, log_prob = hmm.viterbi(observations)
    accuracy = np.mean(path == states)
    
    print(f"\nViterbi accuracy: {accuracy:.3f}")
    print(f"Path log-probability: {log_prob:.2f}")
    print(f"Final log-likelihood: {ll_history[-1]:.2f}")


# =============================================================================
# Example 6: Gaussian Process Regression
# =============================================================================

class GaussianProcessRegressor:
    """
    Gaussian Process Regression with various kernels.
    
    f(x) ~ GP(m(x), k(x, x'))
    y = f(x) + epsilon, epsilon ~ N(0, sigma^2)
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        length_scale: float = 1.0,
        signal_var: float = 1.0,
        noise_var: float = 0.01
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == 'rbf':
            # RBF (Squared Exponential)
            sqdist = (
                np.sum(X1**2, axis=1, keepdims=True) +
                np.sum(X2**2, axis=1) -
                2 * X1 @ X2.T
            )
            return self.signal_var * np.exp(-0.5 * sqdist / self.length_scale**2)
        
        elif self.kernel == 'matern32':
            # Matérn 3/2
            dist = np.sqrt(np.maximum(
                np.sum(X1**2, axis=1, keepdims=True) +
                np.sum(X2**2, axis=1) -
                2 * X1 @ X2.T,
                0
            ))
            r = np.sqrt(3) * dist / self.length_scale
            return self.signal_var * (1 + r) * np.exp(-r)
        
        elif self.kernel == 'linear':
            return self.signal_var * X1 @ X2.T
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessRegressor':
        """Fit GP to training data."""
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        K = self._kernel(X, X) + self.noise_var * np.eye(len(X))
        
        # Cholesky decomposition for stability
        L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
        
        # Solve K @ alpha = y
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        # Store inverse for log likelihood
        self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
        self.L = L
        
        return self
    
    def predict(
        self, X: np.ndarray, return_std: bool = False, return_cov: bool = False
    ):
        """
        Predict at new points.
        
        mu* = K(X*, X) @ K(X, X)^-1 @ y
        cov* = K(X*, X*) - K(X*, X) @ K(X, X)^-1 @ K(X, X*)
        """
        K_s = self._kernel(X, self.X_train)
        
        # Posterior mean
        mu = K_s @ self.alpha
        
        if return_std or return_cov:
            # Posterior covariance
            K_ss = self._kernel(X, X)
            v = np.linalg.solve(self.L, K_s.T)
            cov = K_ss - v.T @ v
            
            if return_cov:
                return mu, cov
            else:
                std = np.sqrt(np.maximum(np.diag(cov), 0))
                return mu, std
        
        return mu
    
    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood.
        
        log p(y|X) = -0.5 * y^T K^-1 y - 0.5 * log|K| - n/2 * log(2*pi)
        """
        n = len(self.y_train)
        
        # Using Cholesky decomposition
        log_det = 2 * np.sum(np.log(np.diag(self.L)))
        data_fit = self.y_train @ self.alpha
        
        return -0.5 * (data_fit + log_det + n * np.log(2 * np.pi))


def example_gaussian_process():
    """Demonstrate Gaussian Process Regression."""
    print("\n" + "=" * 70)
    print("Example 6: Gaussian Process Regression")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True function
    def f(x):
        return np.sin(3 * x) + 0.5 * x
    
    # Generate training data
    X_train = np.random.uniform(-2, 2, 20).reshape(-1, 1)
    y_train = f(X_train.ravel()) + 0.2 * np.random.randn(20)
    
    # Test points
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = f(X_test.ravel())
    
    # Fit GP with different kernels
    kernels = ['rbf', 'matern32']
    
    for kernel in kernels:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            length_scale=0.5,
            signal_var=1.0,
            noise_var=0.04
        )
        gp.fit(X_train, y_train)
        
        mu, std = gp.predict(X_test, return_std=True)
        
        rmse = np.sqrt(np.mean((mu - y_true)**2))
        lml = gp.log_marginal_likelihood()
        
        print(f"\nKernel: {kernel}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Log marginal likelihood: {lml:.2f}")
        print(f"  Mean uncertainty: {np.mean(std):.4f}")


# =============================================================================
# Example 7: Gibbs Sampling for Bayesian Linear Regression
# =============================================================================

class GibbsSamplerBLR:
    """
    Gibbs sampling for Bayesian linear regression.
    
    Model:
        y = X @ w + epsilon, epsilon ~ N(0, sigma^2)
        w ~ N(0, tau^2 * I)
        sigma^2 ~ InvGamma(a0, b0)
    """
    
    def __init__(
        self,
        prior_tau: float = 1.0,
        prior_a: float = 1.0,
        prior_b: float = 1.0
    ):
        self.tau = prior_tau
        self.a0 = prior_a
        self.b0 = prior_b
        
        self.samples_w = None
        self.samples_sigma2 = None
    
    def sample(
        self, X: np.ndarray, y: np.ndarray,
        n_samples: int = 1000, burnin: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Gibbs sampler.
        
        Alternates between:
        1. Sample w | sigma^2, y
        2. Sample sigma^2 | w, y
        """
        n, d = X.shape
        
        # Initialize
        sigma2 = 1.0
        
        # Storage
        self.samples_w = np.zeros((n_samples, d))
        self.samples_sigma2 = np.zeros(n_samples)
        
        total_iters = n_samples + burnin
        
        for i in range(total_iters):
            # Sample w | sigma^2, y
            # Posterior: N(m_n, S_n)
            # S_n = (tau^-2 * I + sigma^-2 * X^T X)^-1
            # m_n = S_n @ sigma^-2 @ X^T @ y
            
            prior_prec = (1.0 / self.tau**2) * np.eye(d)
            likelihood_prec = (1.0 / sigma2) * X.T @ X
            
            S_n = np.linalg.inv(prior_prec + likelihood_prec)
            m_n = S_n @ (1.0 / sigma2) @ X.T @ y
            
            # Sample from multivariate normal
            w = np.random.multivariate_normal(m_n, S_n)
            
            # Sample sigma^2 | w, y
            # Posterior: InvGamma(a_n, b_n)
            # a_n = a0 + n/2
            # b_n = b0 + 0.5 * ||y - Xw||^2
            
            residual = y - X @ w
            a_n = self.a0 + n / 2
            b_n = self.b0 + 0.5 * residual @ residual
            
            # InvGamma(a, b) = 1 / Gamma(a, 1/b)
            sigma2 = 1.0 / np.random.gamma(a_n, 1.0 / b_n)
            
            # Store after burn-in
            if i >= burnin:
                idx = i - burnin
                self.samples_w[idx] = w
                self.samples_sigma2[idx] = sigma2
        
        return self.samples_w, self.samples_sigma2
    
    def posterior_summary(self) -> Dict:
        """Compute posterior summary statistics."""
        w_mean = np.mean(self.samples_w, axis=0)
        w_std = np.std(self.samples_w, axis=0)
        
        sigma2_mean = np.mean(self.samples_sigma2)
        sigma2_std = np.std(self.samples_sigma2)
        
        return {
            'w_mean': w_mean,
            'w_std': w_std,
            'w_95_ci': (np.percentile(self.samples_w, 2.5, axis=0),
                       np.percentile(self.samples_w, 97.5, axis=0)),
            'sigma2_mean': sigma2_mean,
            'sigma2_std': sigma2_std
        }


def example_gibbs_sampling():
    """Demonstrate Gibbs sampling for Bayesian linear regression."""
    print("\n" + "=" * 70)
    print("Example 7: Gibbs Sampling for Bayesian Linear Regression")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data
    n = 100
    true_w = np.array([2.0, -1.0, 0.5])
    true_sigma2 = 0.5
    
    X = np.random.randn(n, 3)
    y = X @ true_w + np.sqrt(true_sigma2) * np.random.randn(n)
    
    # Run Gibbs sampler
    sampler = GibbsSamplerBLR(prior_tau=10.0, prior_a=1.0, prior_b=1.0)
    samples_w, samples_sigma2 = sampler.sample(X, y, n_samples=2000, burnin=500)
    
    summary = sampler.posterior_summary()
    
    print("\nTrue parameters:")
    print(f"  w: {true_w}")
    print(f"  sigma^2: {true_sigma2}")
    
    print("\nPosterior estimates:")
    print(f"  w mean: {summary['w_mean']}")
    print(f"  w std: {summary['w_std']}")
    print(f"  sigma^2 mean: {summary['sigma2_mean']:.4f} ± {summary['sigma2_std']:.4f}")
    
    print("\n95% Credible Intervals for w:")
    for i in range(3):
        print(f"  w[{i}]: [{summary['w_95_ci'][0][i]:.3f}, {summary['w_95_ci'][1][i]:.3f}]")


# =============================================================================
# Example 8: Probabilistic PCA
# =============================================================================

class ProbabilisticPCA:
    """
    Probabilistic PCA using EM algorithm.
    
    Model:
        z ~ N(0, I)                     (latent, q-dim)
        x | z ~ N(W @ z + mu, sigma^2 * I)  (observed, d-dim)
    
    Marginal:
        x ~ N(mu, W @ W^T + sigma^2 * I)
    """
    
    def __init__(self, n_components: int):
        self.q = n_components
        self.W = None
        self.mu = None
        self.sigma2 = None
    
    def fit(self, X: np.ndarray, n_iter: int = 100) -> 'ProbabilisticPCA':
        """Fit PPCA using EM algorithm."""
        n, d = X.shape
        
        # Center data
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu
        
        # Initialize
        self.W = np.random.randn(d, self.q) * 0.1
        self.sigma2 = 1.0
        
        for iteration in range(n_iter):
            # E-step: compute E[z] and E[zz^T]
            M = self.W.T @ self.W + self.sigma2 * np.eye(self.q)
            M_inv = np.linalg.inv(M)
            
            # E[z | x] = M^-1 @ W^T @ (x - mu)
            E_z = X_centered @ self.W @ M_inv.T  # (n, q)
            
            # E[zz^T | x] = sigma^2 * M^-1 + E[z]E[z]^T
            E_zz_sum = n * self.sigma2 * M_inv + E_z.T @ E_z
            
            # M-step: update W and sigma^2
            # W_new = (sum_n x_n E[z_n]^T) @ (sum_n E[z_n z_n^T])^-1
            self.W = X_centered.T @ E_z @ np.linalg.inv(E_zz_sum)
            
            # sigma^2_new = (1/nd) * sum_n ||x_n||^2 - 2 E[z_n]^T W^T x_n + tr(E[z_n z_n^T] W^T W)
            recon = E_z @ self.W.T
            self.sigma2 = np.mean(np.sum((X_centered - recon)**2, axis=1))
            self.sigma2 += np.trace(E_zz_sum @ self.W.T @ self.W) / n
            self.sigma2 = max(self.sigma2, 1e-6)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data to latent space."""
        X_centered = X - self.mu
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.q)
        M_inv = np.linalg.inv(M)
        return X_centered @ self.W @ M_inv.T
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct from latent space."""
        return Z @ self.W.T + self.mu
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the model."""
        z = np.random.randn(n_samples, self.q)
        x_mean = z @ self.W.T + self.mu
        x = x_mean + np.sqrt(self.sigma2) * np.random.randn(n_samples, self.W.shape[0])
        return x


def example_probabilistic_pca():
    """Demonstrate Probabilistic PCA."""
    print("\n" + "=" * 70)
    print("Example 8: Probabilistic PCA")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data from a low-dimensional subspace
    n = 200
    true_d = 5  # Observed dimension
    true_q = 2  # True latent dimension
    
    # True generative process
    true_W = np.random.randn(true_d, true_q)
    true_mu = np.random.randn(true_d)
    true_sigma2 = 0.1
    
    z = np.random.randn(n, true_q)
    X = z @ true_W.T + true_mu + np.sqrt(true_sigma2) * np.random.randn(n, true_d)
    
    # Fit PPCA
    ppca = ProbabilisticPCA(n_components=2)
    ppca.fit(X, n_iter=50)
    
    # Evaluate
    Z = ppca.transform(X)
    X_recon = ppca.inverse_transform(Z)
    
    recon_error = np.mean(np.sum((X - X_recon)**2, axis=1))
    
    print(f"\nTrue sigma^2: {true_sigma2}")
    print(f"Estimated sigma^2: {ppca.sigma2:.4f}")
    print(f"Reconstruction error: {recon_error:.4f}")
    print(f"Latent variance explained: {1 - ppca.sigma2 / np.var(X):.2%}")
    
    # Generate samples
    samples = ppca.sample(5)
    print(f"\nGenerated sample shape: {samples.shape}")
    print(f"Sample mean distance from data mean: {np.linalg.norm(np.mean(samples, axis=0) - ppca.mu):.4f}")


# =============================================================================
# Example 9: Normalizing Flow (RealNVP)
# =============================================================================

class RealNVP:
    """
    Real-valued Non-Volume Preserving (RealNVP) normalizing flow.
    
    Affine coupling layer:
        y_1 = x_1
        y_2 = x_2 * exp(s(x_1)) + t(x_1)
    
    Log-det Jacobian = sum(s(x_1))
    """
    
    def __init__(self, dim: int, hidden_dim: int, n_layers: int = 4):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Each layer has s (scale) and t (translation) networks
        self.layers = []
        for i in range(n_layers):
            # Alternate which half is transformed
            mask = np.array([i % 2] * (dim // 2) + [(i + 1) % 2] * (dim - dim // 2))
            
            # Simple linear networks for s and t
            layer = {
                'mask': mask,
                'W_s1': 0.1 * np.random.randn(dim // 2, hidden_dim),
                'b_s1': np.zeros(hidden_dim),
                'W_s2': 0.1 * np.random.randn(hidden_dim, dim - dim // 2),
                'b_s2': np.zeros(dim - dim // 2),
                'W_t1': 0.1 * np.random.randn(dim // 2, hidden_dim),
                'b_t1': np.zeros(hidden_dim),
                'W_t2': 0.1 * np.random.randn(hidden_dim, dim - dim // 2),
                'b_t2': np.zeros(dim - dim // 2),
            }
            self.layers.append(layer)
    
    def _scale_and_translate(self, x_masked: np.ndarray, layer: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scale and translation."""
        h_s = np.tanh(x_masked @ layer['W_s1'] + layer['b_s1'])
        s = h_s @ layer['W_s2'] + layer['b_s2']
        s = np.tanh(s)  # Bound scale
        
        h_t = np.tanh(x_masked @ layer['W_t1'] + layer['b_t1'])
        t = h_t @ layer['W_t2'] + layer['b_t2']
        
        return s, t
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Transform from data to latent space.
        
        Returns:
            z: Latent representation
            log_det: Log determinant of Jacobian
        """
        z = x.copy()
        log_det_total = 0.0
        
        for layer in self.layers:
            mask = layer['mask']
            x_masked = z[:, mask == 0]
            x_transform = z[:, mask == 1]
            
            s, t = self._scale_and_translate(x_masked, layer)
            
            # Apply transformation
            z_transform = x_transform * np.exp(s) + t
            
            # Update z
            z[:, mask == 0] = x_masked
            z[:, mask == 1] = z_transform
            
            # Accumulate log det
            log_det_total += np.sum(s, axis=1)
        
        return z, log_det_total
    
    def inverse(self, z: np.ndarray) -> np.ndarray:
        """Transform from latent to data space."""
        x = z.copy()
        
        for layer in reversed(self.layers):
            mask = layer['mask']
            x_masked = x[:, mask == 0]
            x_transform = x[:, mask == 1]
            
            s, t = self._scale_and_translate(x_masked, layer)
            
            # Inverse transformation
            x_inv = (x_transform - t) * np.exp(-s)
            
            x[:, mask == 0] = x_masked
            x[:, mask == 1] = x_inv
        
        return x
    
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log probability using change of variables.
        
        log p(x) = log p(z) + log |det(dz/dx)|
        """
        z, log_det = self.forward(x)
        
        # Standard normal prior
        log_p_z = -0.5 * (self.dim * np.log(2 * np.pi) + np.sum(z**2, axis=1))
        
        return log_p_z + log_det
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from the model."""
        z = np.random.randn(n_samples, self.dim)
        return self.inverse(z)


def example_normalizing_flow():
    """Demonstrate RealNVP normalizing flow."""
    print("\n" + "=" * 70)
    print("Example 9: Normalizing Flow (RealNVP)")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create a simple 2D flow
    flow = RealNVP(dim=2, hidden_dim=16, n_layers=4)
    
    # Test invertibility
    x = np.random.randn(100, 2)
    z, log_det = flow.forward(x)
    x_recon = flow.inverse(z)
    
    recon_error = np.mean(np.abs(x - x_recon))
    
    print(f"\nInvertibility check:")
    print(f"  Reconstruction error: {recon_error:.6f}")
    
    # Compute log probabilities
    log_probs = flow.log_prob(x)
    
    print(f"\nLog probability statistics:")
    print(f"  Mean: {np.mean(log_probs):.4f}")
    print(f"  Std: {np.std(log_probs):.4f}")
    
    # Generate samples
    samples = flow.sample(100)
    
    print(f"\nGenerated samples:")
    print(f"  Mean: {np.mean(samples, axis=0)}")
    print(f"  Std: {np.std(samples, axis=0)}")


# =============================================================================
# Example 10: Bayesian Neural Network (Monte Carlo Dropout)
# =============================================================================

class MCDropoutBNN:
    """
    Bayesian Neural Network using Monte Carlo Dropout.
    
    Dropout at test time approximates variational inference.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        dropout_rate: float = 0.1,
        learning_rate: float = 0.01
    ):
        self.layers = []
        self.dropout_rate = dropout_rate
        self.lr = learning_rate
        
        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)  # He initialization
            
            self.layers.append({
                'W': scale * np.random.randn(fan_in, fan_out),
                'b': np.zeros(fan_out)
            })
    
    def _forward(self, X: np.ndarray, training: bool = True) -> List[np.ndarray]:
        """Forward pass with dropout."""
        activations = [X]
        h = X
        
        for i, layer in enumerate(self.layers):
            z = h @ layer['W'] + layer['b']
            
            # Apply dropout (except last layer)
            if training and i < len(self.layers) - 1:
                mask = (np.random.rand(*z.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                z = z * mask
            
            # Activation (ReLU except last layer)
            if i < len(self.layers) - 1:
                h = np.maximum(0, z)
            else:
                h = z
            
            activations.append(h)
        
        return activations
    
    def predict(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Predict with single forward pass."""
        return self._forward(X, training=False)[-1]
    
    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty using MC Dropout.
        
        Returns:
            mean: Mean prediction
            epistemic_uncertainty: Model uncertainty (from dropout)
            aleatoric_uncertainty: Data uncertainty (placeholder)
        """
        predictions = []
        
        for _ in range(n_samples):
            pred = self._forward(X, training=True)[-1]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis=0)
        epistemic = np.std(predictions, axis=0)
        
        return mean, epistemic, np.zeros_like(mean)  # No aleatoric for regression
    
    def fit(
        self, X: np.ndarray, y: np.ndarray,
        n_epochs: int = 100, batch_size: int = 32
    ) -> List[float]:
        """Train network with backpropagation."""
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(n_epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                activations = self._forward(X_batch, training=True)
                
                # Compute loss (MSE)
                pred = activations[-1]
                loss = np.mean((pred - y_batch)**2)
                epoch_loss += loss
                
                # Backward pass
                grad = 2 * (pred - y_batch) / len(X_batch)
                
                for j in range(len(self.layers) - 1, -1, -1):
                    h = activations[j]
                    
                    # Gradients for weights and biases
                    dW = h.T @ grad
                    db = np.sum(grad, axis=0)
                    
                    # Gradient for previous layer
                    if j > 0:
                        grad = grad @ self.layers[j]['W'].T
                        grad = grad * (activations[j] > 0)  # ReLU derivative
                    
                    # Update
                    self.layers[j]['W'] -= self.lr * dW
                    self.layers[j]['b'] -= self.lr * db
            
            losses.append(epoch_loss / (n_samples // batch_size))
        
        return losses


def example_bayesian_nn():
    """Demonstrate Bayesian Neural Network with MC Dropout."""
    print("\n" + "=" * 70)
    print("Example 10: Bayesian Neural Network (MC Dropout)")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate data with noise
    def f(x):
        return np.sin(2 * np.pi * x) + 0.1 * x
    
    n = 100
    X_train = np.random.uniform(-1, 1, n).reshape(-1, 1)
    y_train = f(X_train) + 0.1 * np.random.randn(n, 1)
    
    # Test points (including out-of-distribution)
    X_test = np.linspace(-1.5, 1.5, 50).reshape(-1, 1)
    y_true = f(X_test)
    
    # Create and train BNN
    bnn = MCDropoutBNN(
        layer_sizes=[1, 50, 50, 1],
        dropout_rate=0.1,
        learning_rate=0.01
    )
    
    print("\nTraining Bayesian Neural Network...")
    losses = bnn.fit(X_train, y_train, n_epochs=200, batch_size=32)
    
    print(f"Final training loss: {losses[-1]:.4f}")
    
    # Predict with uncertainty
    mean, epistemic, _ = bnn.predict_with_uncertainty(X_test, n_samples=100)
    
    # Evaluate
    in_dist_mask = (X_test.ravel() >= -1) & (X_test.ravel() <= 1)
    out_dist_mask = ~in_dist_mask
    
    rmse_in = np.sqrt(np.mean((mean[in_dist_mask] - y_true[in_dist_mask])**2))
    rmse_out = np.sqrt(np.mean((mean[out_dist_mask] - y_true[out_dist_mask])**2))
    
    unc_in = np.mean(epistemic[in_dist_mask])
    unc_out = np.mean(epistemic[out_dist_mask])
    
    print(f"\nResults:")
    print(f"  In-distribution RMSE: {rmse_in:.4f}")
    print(f"  Out-of-distribution RMSE: {rmse_out:.4f}")
    print(f"  In-distribution uncertainty: {unc_in:.4f}")
    print(f"  Out-of-distribution uncertainty: {unc_out:.4f}")
    print(f"\n  Uncertainty increase factor: {unc_out/unc_in:.2f}x")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("PROBABILISTIC MODELS: IMPLEMENTATION EXAMPLES")
    print("=" * 70)
    
    example_bayesian_inference()
    example_gmm()
    example_variational_inference()
    example_vae()
    example_hmm()
    example_gaussian_process()
    example_gibbs_sampling()
    example_probabilistic_pca()
    example_normalizing_flow()
    example_bayesian_nn()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
