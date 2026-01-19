"""
Sampling Methods - Examples
===========================

Comprehensive examples demonstrating various sampling methods for ML.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict
from scipy import stats


# =============================================================================
# Example 1: Basic Sampling Methods
# =============================================================================

def example_basic_sampling():
    """Demonstrate inverse transform and Box-Muller sampling."""
    print("=" * 70)
    print("Example 1: Basic Sampling Methods")
    print("=" * 70)
    
    # Inverse Transform Sampling for Exponential Distribution
    def inverse_transform_exponential(n_samples: int, rate: float = 1.0) -> np.ndarray:
        """Sample from exponential using inverse CDF."""
        u = np.random.uniform(0, 1, n_samples)
        return -np.log(1 - u) / rate
    
    np.random.seed(42)
    samples_exp = inverse_transform_exponential(10000, rate=2.0)
    
    print("Inverse Transform Sampling (Exponential):")
    print(f"  Rate = 2.0")
    print(f"  Sample mean: {np.mean(samples_exp):.4f} (theoretical: 0.5)")
    print(f"  Sample var:  {np.var(samples_exp):.4f} (theoretical: 0.25)")
    
    # Box-Muller Transform for Normal Distribution
    def box_muller(n_samples: int) -> np.ndarray:
        """Generate standard normal samples using Box-Muller."""
        n_pairs = (n_samples + 1) // 2
        u1 = np.random.uniform(0, 1, n_pairs)
        u2 = np.random.uniform(0, 1, n_pairs)
        
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        
        return np.concatenate([z1, z2])[:n_samples]
    
    samples_normal = box_muller(10000)
    
    print(f"\nBox-Muller Transform (Standard Normal):")
    print(f"  Sample mean: {np.mean(samples_normal):.4f} (theoretical: 0)")
    print(f"  Sample std:  {np.std(samples_normal):.4f} (theoretical: 1)")


# =============================================================================
# Example 2: Rejection Sampling
# =============================================================================

def example_rejection_sampling():
    """Demonstrate rejection sampling."""
    print("\n" + "=" * 70)
    print("Example 2: Rejection Sampling")
    print("=" * 70)
    
    def rejection_sampling(
        target_pdf: Callable,
        proposal_sampler: Callable,
        proposal_pdf: Callable,
        M: float,
        n_samples: int
    ) -> Tuple[np.ndarray, float]:
        """
        Rejection sampling.
        
        Returns: (samples, acceptance_rate)
        """
        samples = []
        n_proposed = 0
        
        while len(samples) < n_samples:
            # Propose
            x = proposal_sampler()
            n_proposed += 1
            
            # Accept/reject
            u = np.random.uniform()
            if u < target_pdf(x) / (M * proposal_pdf(x)):
                samples.append(x)
        
        return np.array(samples), n_samples / n_proposed
    
    # Target: Beta(2, 5) distribution
    # Proposal: Uniform(0, 1)
    alpha, beta_param = 2, 5
    
    def target_pdf(x):
        return stats.beta.pdf(x, alpha, beta_param)
    
    def proposal_sampler():
        return np.random.uniform(0, 1)
    
    def proposal_pdf(x):
        return 1.0
    
    # Find M: maximum of target_pdf
    x_grid = np.linspace(0, 1, 1000)
    M = np.max(target_pdf(x_grid)) * 1.01
    
    np.random.seed(42)
    samples, acceptance_rate = rejection_sampling(
        target_pdf, proposal_sampler, proposal_pdf, M, 5000
    )
    
    print(f"Target: Beta({alpha}, {beta_param})")
    print(f"Proposal: Uniform(0, 1)")
    print(f"Bounding constant M: {M:.4f}")
    print(f"Acceptance rate: {acceptance_rate:.2%} (theoretical: {1/M:.2%})")
    print(f"Sample mean: {np.mean(samples):.4f} (theoretical: {alpha/(alpha+beta_param):.4f})")


# =============================================================================
# Example 3: Metropolis-Hastings Algorithm
# =============================================================================

def example_metropolis_hastings():
    """Demonstrate Metropolis-Hastings MCMC."""
    print("\n" + "=" * 70)
    print("Example 3: Metropolis-Hastings MCMC")
    print("=" * 70)
    
    def metropolis_hastings(
        log_target: Callable,
        proposal_std: float,
        n_samples: int,
        x0: np.ndarray,
        burn_in: int = 1000
    ) -> Tuple[np.ndarray, float]:
        """
        Metropolis-Hastings with Gaussian proposal.
        """
        d = len(x0)
        samples = np.zeros((n_samples + burn_in, d))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_samples + burn_in):
            # Propose
            x_current = samples[t-1]
            x_proposed = x_current + proposal_std * np.random.randn(d)
            
            # Acceptance ratio (log scale)
            log_alpha = log_target(x_proposed) - log_target(x_current)
            
            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                samples[t] = x_proposed
                if t >= burn_in:
                    n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples[burn_in:], n_accepted / n_samples
    
    # Target: 2D Gaussian mixture
    def log_target(x):
        # Mixture of two Gaussians
        mu1, mu2 = np.array([-2, 0]), np.array([2, 0])
        sigma = 1.0
        
        log_p1 = -0.5 * np.sum((x - mu1)**2) / sigma**2
        log_p2 = -0.5 * np.sum((x - mu2)**2) / sigma**2
        
        return np.logaddexp(log_p1, log_p2) - np.log(2)
    
    np.random.seed(42)
    x0 = np.array([0.0, 0.0])
    
    samples, acc_rate = metropolis_hastings(
        log_target, proposal_std=1.0, n_samples=10000, x0=x0, burn_in=2000
    )
    
    print(f"Target: Gaussian mixture (modes at [-2,0] and [2,0])")
    print(f"Proposal: N(x, I)")
    print(f"Acceptance rate: {acc_rate:.2%}")
    print(f"Sample mean: [{np.mean(samples[:, 0]):.4f}, {np.mean(samples[:, 1]):.4f}]")
    print(f"Sample std:  [{np.std(samples[:, 0]):.4f}, {np.std(samples[:, 1]):.4f}]")
    
    # Analyze mode exploration
    left_mode = np.mean(samples[:, 0] < 0)
    print(f"Time in left mode: {left_mode:.2%}")


# =============================================================================
# Example 4: Gibbs Sampling
# =============================================================================

def example_gibbs_sampling():
    """Demonstrate Gibbs sampling."""
    print("\n" + "=" * 70)
    print("Example 4: Gibbs Sampling")
    print("=" * 70)
    
    def gibbs_sampling_bivariate_normal(
        mu: np.ndarray,
        sigma: np.ndarray,
        n_samples: int,
        x0: np.ndarray,
        burn_in: int = 500
    ) -> np.ndarray:
        """
        Gibbs sampling for bivariate normal.
        Uses conditional distributions directly.
        """
        samples = np.zeros((n_samples + burn_in, 2))
        samples[0] = x0
        
        # Extract parameters
        mu1, mu2 = mu
        sigma1, sigma2 = np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1])
        rho = sigma[0, 1] / (sigma1 * sigma2)
        
        for t in range(1, n_samples + burn_in):
            x_prev = samples[t-1]
            
            # Sample x1 | x2
            cond_mean_1 = mu1 + rho * (sigma1/sigma2) * (x_prev[1] - mu2)
            cond_std_1 = sigma1 * np.sqrt(1 - rho**2)
            x1_new = np.random.normal(cond_mean_1, cond_std_1)
            
            # Sample x2 | x1
            cond_mean_2 = mu2 + rho * (sigma2/sigma1) * (x1_new - mu1)
            cond_std_2 = sigma2 * np.sqrt(1 - rho**2)
            x2_new = np.random.normal(cond_mean_2, cond_std_2)
            
            samples[t] = np.array([x1_new, x2_new])
        
        return samples[burn_in:]
    
    # Target: correlated bivariate normal
    mu = np.array([1.0, 2.0])
    sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    
    np.random.seed(42)
    samples = gibbs_sampling_bivariate_normal(
        mu, sigma, n_samples=5000, x0=np.array([0.0, 0.0])
    )
    
    print(f"Target: Bivariate Normal")
    print(f"  μ = {mu}")
    print(f"  Σ = \n{sigma}")
    
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples.T)
    
    print(f"\nGibbs Sampling Results:")
    print(f"  Sample mean: {sample_mean}")
    print(f"  Sample cov:\n{np.round(sample_cov, 4)}")


# =============================================================================
# Example 5: Hamiltonian Monte Carlo
# =============================================================================

def example_hmc():
    """Demonstrate Hamiltonian Monte Carlo."""
    print("\n" + "=" * 70)
    print("Example 5: Hamiltonian Monte Carlo (HMC)")
    print("=" * 70)
    
    def hmc(
        log_prob: Callable,
        grad_log_prob: Callable,
        n_samples: int,
        x0: np.ndarray,
        step_size: float = 0.1,
        n_leapfrog: int = 20,
        burn_in: int = 500
    ) -> Tuple[np.ndarray, float]:
        """
        Hamiltonian Monte Carlo.
        """
        d = len(x0)
        samples = np.zeros((n_samples + burn_in, d))
        samples[0] = x0
        n_accepted = 0
        
        def leapfrog(x, p, step_size, n_steps):
            """Leapfrog integrator."""
            x = x.copy()
            p = p.copy()
            
            # Half step for momentum
            p = p + 0.5 * step_size * grad_log_prob(x)
            
            # Full steps
            for _ in range(n_steps - 1):
                x = x + step_size * p
                p = p + step_size * grad_log_prob(x)
            
            # Final position and half momentum
            x = x + step_size * p
            p = p + 0.5 * step_size * grad_log_prob(x)
            
            return x, -p  # Negate momentum for reversibility
        
        def hamiltonian(x, p):
            """Compute Hamiltonian H = U + K."""
            U = -log_prob(x)  # Potential energy
            K = 0.5 * np.dot(p, p)  # Kinetic energy
            return U + K
        
        for t in range(1, n_samples + burn_in):
            x_current = samples[t-1]
            
            # Sample momentum
            p_current = np.random.randn(d)
            
            # Leapfrog integration
            x_proposed, p_proposed = leapfrog(x_current, p_current, step_size, n_leapfrog)
            
            # Metropolis acceptance
            H_current = hamiltonian(x_current, p_current)
            H_proposed = hamiltonian(x_proposed, p_proposed)
            
            if np.log(np.random.uniform()) < H_current - H_proposed:
                samples[t] = x_proposed
                if t >= burn_in:
                    n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples[burn_in:], n_accepted / n_samples
    
    # Target: 2D funnel distribution (challenging)
    def log_prob(x):
        v = x[0]
        return -0.5 * v**2 - 0.5 * np.sum(x[1:]**2) / np.exp(v)
    
    def grad_log_prob(x):
        v = x[0]
        grad = np.zeros_like(x)
        grad[0] = -v + 0.5 * np.sum(x[1:]**2) / np.exp(v)
        grad[1:] = -x[1:] / np.exp(v)
        return grad
    
    np.random.seed(42)
    d = 5
    x0 = np.zeros(d)
    
    samples, acc_rate = hmc(
        log_prob, grad_log_prob, 
        n_samples=3000, x0=x0,
        step_size=0.1, n_leapfrog=20,
        burn_in=500
    )
    
    print(f"Target: Neal's Funnel (d={d})")
    print(f"HMC Parameters: step_size=0.1, n_leapfrog=20")
    print(f"Acceptance rate: {acc_rate:.2%}")
    print(f"x[0] (log-variance): mean={np.mean(samples[:, 0]):.4f}, std={np.std(samples[:, 0]):.4f}")
    print(f"x[1] | x[0]=0: theoretical std = exp(0/2) = 1")
    print(f"Sample std of x[1]: {np.std(samples[:, 1]):.4f}")


# =============================================================================
# Example 6: Importance Sampling
# =============================================================================

def example_importance_sampling():
    """Demonstrate importance sampling."""
    print("\n" + "=" * 70)
    print("Example 6: Importance Sampling")
    print("=" * 70)
    
    def importance_sampling(
        f: Callable,
        target_pdf: Callable,
        proposal_sampler: Callable,
        proposal_pdf: Callable,
        n_samples: int
    ) -> Tuple[float, float, float]:
        """
        Importance sampling estimate of E_p[f(x)].
        
        Returns: (estimate, std_error, effective_sample_size)
        """
        samples = np.array([proposal_sampler() for _ in range(n_samples)])
        
        # Compute importance weights
        weights = np.array([target_pdf(x) / proposal_pdf(x) for x in samples])
        
        # Normalize weights
        normalized_weights = weights / np.sum(weights)
        
        # Compute estimate
        f_values = np.array([f(x) for x in samples])
        estimate = np.sum(normalized_weights * f_values)
        
        # Effective sample size
        ess = 1.0 / np.sum(normalized_weights**2)
        
        # Estimate standard error (using importance sampling variance)
        weighted_var = np.sum(normalized_weights * (f_values - estimate)**2)
        std_error = np.sqrt(weighted_var / n_samples)
        
        return estimate, std_error, ess
    
    # Example: Estimate E[x^2] where x ~ N(0, 1) using N(0, 2) proposal
    def f(x):
        return x**2
    
    def target_pdf(x):
        return stats.norm.pdf(x, 0, 1)
    
    def proposal_sampler():
        return np.random.normal(0, 2)
    
    def proposal_pdf(x):
        return stats.norm.pdf(x, 0, 2)
    
    np.random.seed(42)
    estimate, std_err, ess = importance_sampling(
        f, target_pdf, proposal_sampler, proposal_pdf, n_samples=10000
    )
    
    print(f"Estimating E[X²] where X ~ N(0,1)")
    print(f"Proposal: N(0, 2)")
    print(f"True value: 1.0")
    print(f"IS estimate: {estimate:.4f} ± {std_err:.4f}")
    print(f"Effective Sample Size: {ess:.0f} / 10000 ({100*ess/10000:.1f}%)")
    
    # Bad proposal example
    def bad_proposal_sampler():
        return np.random.normal(5, 0.5)  # Far from target mass
    
    def bad_proposal_pdf(x):
        return stats.norm.pdf(x, 5, 0.5)
    
    estimate_bad, std_err_bad, ess_bad = importance_sampling(
        f, target_pdf, bad_proposal_sampler, bad_proposal_pdf, n_samples=10000
    )
    
    print(f"\nBad proposal: N(5, 0.5)")
    print(f"IS estimate: {estimate_bad:.4f} ± {std_err_bad:.4f}")
    print(f"Effective Sample Size: {ess_bad:.0f} / 10000 ({100*ess_bad/10000:.1f}%)")


# =============================================================================
# Example 7: Sequential Monte Carlo (Particle Filter)
# =============================================================================

def example_smc():
    """Demonstrate Sequential Monte Carlo / Particle Filter."""
    print("\n" + "=" * 70)
    print("Example 7: Sequential Monte Carlo (Particle Filter)")
    print("=" * 70)
    
    def particle_filter(
        observations: np.ndarray,
        initial_sampler: Callable,
        transition: Callable,
        likelihood: Callable,
        n_particles: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic bootstrap particle filter.
        
        Returns: (filtered_means, log_marginal_likelihood)
        """
        T = len(observations)
        particles = np.zeros((T, n_particles))
        weights = np.zeros((T, n_particles))
        filtered_means = np.zeros(T)
        log_ml = 0
        
        # Initialize
        particles[0] = np.array([initial_sampler() for _ in range(n_particles)])
        weights[0] = np.array([likelihood(observations[0], particles[0, i]) 
                               for i in range(n_particles)])
        
        # Normalize
        log_ml += np.log(np.mean(weights[0]))
        weights[0] /= np.sum(weights[0])
        filtered_means[0] = np.sum(weights[0] * particles[0])
        
        for t in range(1, T):
            # Resample
            indices = np.random.choice(n_particles, size=n_particles, p=weights[t-1])
            
            # Propagate
            particles[t] = np.array([transition(particles[t-1, indices[i]]) 
                                     for i in range(n_particles)])
            
            # Weight
            weights[t] = np.array([likelihood(observations[t], particles[t, i])
                                   for i in range(n_particles)])
            
            # Normalize
            log_ml += np.log(np.mean(weights[t]))
            weights[t] /= np.sum(weights[t])
            filtered_means[t] = np.sum(weights[t] * particles[t])
        
        return filtered_means, log_ml
    
    # State-space model: x_t = 0.9 * x_{t-1} + noise, y_t = x_t + noise
    np.random.seed(42)
    
    T = 50
    true_states = np.zeros(T)
    observations = np.zeros(T)
    
    process_noise = 1.0
    obs_noise = 0.5
    
    true_states[0] = np.random.normal(0, 1)
    observations[0] = true_states[0] + np.random.normal(0, obs_noise)
    
    for t in range(1, T):
        true_states[t] = 0.9 * true_states[t-1] + np.random.normal(0, process_noise)
        observations[t] = true_states[t] + np.random.normal(0, obs_noise)
    
    # Run particle filter
    def initial_sampler():
        return np.random.normal(0, 1)
    
    def transition(x):
        return 0.9 * x + np.random.normal(0, process_noise)
    
    def likelihood(y, x):
        return stats.norm.pdf(y, x, obs_noise)
    
    filtered, log_ml = particle_filter(
        observations, initial_sampler, transition, likelihood, n_particles=1000
    )
    
    print(f"State-space model: x_t = 0.9*x_{t-1} + ε, y_t = x_t + η")
    print(f"T = {T}, n_particles = 1000")
    print(f"\nFiltering accuracy:")
    rmse = np.sqrt(np.mean((filtered - true_states)**2))
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation: {np.corrcoef(filtered, true_states)[0,1]:.4f}")
    print(f"  Log marginal likelihood: {log_ml:.2f}")


# =============================================================================
# Example 8: Reparameterization Trick
# =============================================================================

def example_reparameterization():
    """Demonstrate reparameterization trick for VAE-style training."""
    print("\n" + "=" * 70)
    print("Example 8: Reparameterization Trick")
    print("=" * 70)
    
    def sample_with_reparam(mu: np.ndarray, log_var: np.ndarray, 
                            n_samples: int = 1) -> np.ndarray:
        """
        Sample from N(mu, diag(exp(log_var))) using reparameterization.
        z = mu + std * epsilon, epsilon ~ N(0, I)
        """
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(n_samples, len(mu))
        return mu + std * epsilon
    
    def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
        """KL(q(z|x) || p(z)) for Gaussian prior p(z) = N(0, I)."""
        return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    
    # Simulate gradient computation
    def numerical_gradient(f: Callable, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
    np.random.seed(42)
    d = 3
    mu = np.array([1.0, -0.5, 0.5])
    log_var = np.array([0.0, -1.0, 0.5])
    
    print(f"Encoder output: μ = {mu}, log_var = {log_var}")
    
    # Sample
    z_samples = sample_with_reparam(mu, log_var, n_samples=5000)
    
    print(f"\nSamples from q(z|x):")
    print(f"  Sample mean: {np.mean(z_samples, axis=0).round(4)}")
    print(f"  Sample std:  {np.std(z_samples, axis=0).round(4)}")
    print(f"  Expected std: {np.exp(0.5 * log_var).round(4)}")
    
    # KL divergence
    kl = kl_divergence(mu, log_var)
    print(f"\nKL(q||p): {kl:.4f}")
    
    # Gradient through samples (illustration)
    def loss_fn(params):
        mu_p, log_var_p = params[:d], params[d:]
        np.random.seed(0)  # Fix randomness for gradient
        z = sample_with_reparam(mu_p, log_var_p, 1)[0]
        # Example loss: (z - target)²
        target = np.zeros(d)
        return np.sum((z - target)**2) + 0.1 * kl_divergence(mu_p, log_var_p)
    
    params = np.concatenate([mu, log_var])
    grad = numerical_gradient(loss_fn, params)
    
    print(f"\nGradients through reparameterization:")
    print(f"  ∂L/∂μ: {grad[:d].round(4)}")
    print(f"  ∂L/∂log_var: {grad[d:].round(4)}")


# =============================================================================
# Example 9: Gumbel-Softmax (Concrete Distribution)
# =============================================================================

def example_gumbel_softmax():
    """Demonstrate Gumbel-Softmax for differentiable categorical sampling."""
    print("\n" + "=" * 70)
    print("Example 9: Gumbel-Softmax (Concrete) Distribution")
    print("=" * 70)
    
    def gumbel_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Sample from Gumbel-Softmax distribution.
        Returns soft one-hot vector.
        """
        # Sample Gumbel noise
        u = np.random.uniform(0, 1, logits.shape)
        g = -np.log(-np.log(u + 1e-10) + 1e-10)
        
        # Add noise and apply softmax with temperature
        y = (logits + g) / temperature
        return np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
    
    def straight_through_gumbel(logits: np.ndarray, 
                                 temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Straight-through Gumbel-Softmax.
        Returns: (hard_sample, soft_sample)
        """
        soft = gumbel_softmax(logits, temperature)
        hard = np.zeros_like(soft)
        hard[np.argmax(soft)] = 1.0
        return hard, soft
    
    np.random.seed(42)
    
    # Category probabilities
    logits = np.array([1.0, 2.0, 0.5])  # Unnormalized log-probs
    true_probs = np.exp(logits) / np.sum(np.exp(logits))
    
    print(f"Logits: {logits}")
    print(f"True probabilities: {true_probs.round(4)}")
    
    # Sample at different temperatures
    for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
        samples = np.array([gumbel_softmax(logits, temp) for _ in range(5000)])
        
        # Average "softness"
        entropy = -np.mean(np.sum(samples * np.log(samples + 1e-10), axis=1))
        
        # Hard sample distribution
        hard_samples = np.argmax(samples, axis=1)
        hard_probs = np.bincount(hard_samples, minlength=3) / len(hard_samples)
        
        print(f"\nTemperature = {temp}:")
        print(f"  Mean entropy: {entropy:.4f} (max={np.log(3):.4f})")
        print(f"  Hard sample probs: {hard_probs.round(4)}")
        print(f"  Mean sample: {np.mean(samples, axis=0).round(4)}")


# =============================================================================
# Example 10: MCMC Diagnostics
# =============================================================================

def example_mcmc_diagnostics():
    """Demonstrate MCMC convergence diagnostics."""
    print("\n" + "=" * 70)
    print("Example 10: MCMC Diagnostics")
    print("=" * 70)
    
    def gelman_rubin(chains: np.ndarray) -> float:
        """
        Compute Gelman-Rubin R-hat statistic.
        chains: (n_chains, n_samples)
        """
        n_chains, n_samples = chains.shape
        
        # Between-chain variance
        chain_means = np.mean(chains, axis=1)
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Within-chain variance
        W = np.mean(np.var(chains, axis=1, ddof=1))
        
        # Pooled variance
        var_plus = (n_samples - 1) / n_samples * W + B / n_samples
        
        # R-hat
        R_hat = np.sqrt(var_plus / W)
        
        return R_hat
    
    def effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> float:
        """Compute effective sample size using autocorrelation."""
        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)
        
        if var == 0:
            return n
        
        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            autocorr[lag] = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean)) / var
        
        # Sum until autocorr becomes negative
        for i in range(1, max_lag):
            if autocorr[i] < 0:
                break
        
        ess = n / (1 + 2 * np.sum(autocorr[1:i]))
        return max(ess, 1)
    
    # Run multiple chains on a challenging target
    def log_target(x):
        # Banana-shaped distribution
        return -0.5 * (x[0]**2 / 100 + (x[1] - 0.1 * x[0]**2 + 10)**2)
    
    def metropolis_chain(n_samples, x0, proposal_std):
        """Run single Metropolis chain."""
        samples = np.zeros((n_samples, 2))
        samples[0] = x0
        
        for t in range(1, n_samples):
            x_current = samples[t-1]
            x_proposed = x_current + proposal_std * np.random.randn(2)
            
            log_alpha = log_target(x_proposed) - log_target(x_current)
            if np.log(np.random.uniform()) < log_alpha:
                samples[t] = x_proposed
            else:
                samples[t] = x_current
        
        return samples
    
    np.random.seed(42)
    n_chains = 4
    n_samples = 5000
    
    # Run chains from different starting points
    chains_x0 = np.zeros((n_chains, n_samples))
    chains_x1 = np.zeros((n_chains, n_samples))
    
    for i in range(n_chains):
        x0 = np.random.randn(2) * 10  # Dispersed starts
        chain = metropolis_chain(n_samples, x0, proposal_std=2.5)
        chains_x0[i] = chain[:, 0]
        chains_x1[i] = chain[:, 1]
    
    # Diagnostics
    print("Banana Distribution - MCMC Diagnostics:")
    
    # After burn-in
    burn_in = 1000
    chains_x0_burned = chains_x0[:, burn_in:]
    chains_x1_burned = chains_x1[:, burn_in:]
    
    r_hat_x0 = gelman_rubin(chains_x0_burned)
    r_hat_x1 = gelman_rubin(chains_x1_burned)
    
    print(f"\nGelman-Rubin R-hat (target < 1.01):")
    print(f"  x[0]: {r_hat_x0:.4f}")
    print(f"  x[1]: {r_hat_x1:.4f}")
    
    # ESS
    ess_x0 = effective_sample_size(chains_x0_burned[0])
    ess_x1 = effective_sample_size(chains_x1_burned[0])
    
    print(f"\nEffective Sample Size (chain 1):")
    print(f"  x[0]: {ess_x0:.0f} / {n_samples - burn_in} ({100*ess_x0/(n_samples-burn_in):.1f}%)")
    print(f"  x[1]: {ess_x1:.0f} / {n_samples - burn_in} ({100*ess_x1/(n_samples-burn_in):.1f}%)")
    
    # Summary statistics
    all_samples = chains_x0_burned.flatten()
    print(f"\nSummary (x[0]):")
    print(f"  Mean: {np.mean(all_samples):.4f}")
    print(f"  Std:  {np.std(all_samples):.4f}")
    print(f"  2.5% quantile: {np.percentile(all_samples, 2.5):.4f}")
    print(f"  97.5% quantile: {np.percentile(all_samples, 97.5):.4f}")


def main():
    """Run all examples."""
    print("SAMPLING METHODS - EXAMPLES")
    print("=" * 70)
    
    example_basic_sampling()
    example_rejection_sampling()
    example_metropolis_hastings()
    example_gibbs_sampling()
    example_hmc()
    example_importance_sampling()
    example_smc()
    example_reparameterization()
    example_gumbel_softmax()
    example_mcmc_diagnostics()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
