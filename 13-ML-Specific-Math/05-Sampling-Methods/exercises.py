"""
Sampling Methods - Exercises
============================

Practice exercises for implementing and understanding sampling methods.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict
from scipy import stats


# =============================================================================
# Exercise 1: Inverse Transform and Rejection Sampling
# =============================================================================

def exercise_basic_sampling():
    """
    Exercise: Implement basic sampling methods.
    
    Tasks:
    1. Inverse transform for exponential distribution
    2. Box-Muller for normal distribution
    3. Rejection sampling for custom distribution
    """
    print("=" * 70)
    print("Exercise 1: Basic Sampling Methods")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def inverse_transform_exponential(n: int, rate: float) -> np.ndarray:
        """Sample from Exp(rate) using inverse transform."""
        pass
    
    def box_muller(n: int) -> np.ndarray:
        """Sample from N(0,1) using Box-Muller."""
        pass
    
    def rejection_sample(target_pdf: Callable, proposal_sampler: Callable,
                         proposal_pdf: Callable, M: float, n: int) -> np.ndarray:
        """Generic rejection sampling."""
        pass


def solution_basic_sampling():
    """Reference solution for basic sampling."""
    print("\n--- Solution ---\n")
    
    def inverse_transform_exponential(n: int, rate: float) -> np.ndarray:
        u = np.random.uniform(0, 1, n)
        return -np.log(1 - u) / rate
    
    def box_muller(n: int) -> np.ndarray:
        n_pairs = (n + 1) // 2
        u1 = np.random.uniform(0, 1, n_pairs)
        u2 = np.random.uniform(0, 1, n_pairs)
        
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        
        return np.concatenate([z1, z2])[:n]
    
    def rejection_sample(target_pdf: Callable, proposal_sampler: Callable,
                         proposal_pdf: Callable, M: float, n: int) -> np.ndarray:
        samples = []
        n_proposed = 0
        
        while len(samples) < n:
            x = proposal_sampler()
            n_proposed += 1
            u = np.random.uniform()
            if u < target_pdf(x) / (M * proposal_pdf(x)):
                samples.append(x)
        
        return np.array(samples), n / n_proposed
    
    # Test exponential
    np.random.seed(42)
    rate = 2.0
    samples_exp = inverse_transform_exponential(10000, rate)
    
    print(f"Exponential(λ={rate}):")
    print(f"  Mean: {np.mean(samples_exp):.4f} (expected: {1/rate})")
    print(f"  Var:  {np.var(samples_exp):.4f} (expected: {1/rate**2})")
    
    # Test Box-Muller
    samples_normal = box_muller(10000)
    print(f"\nStandard Normal (Box-Muller):")
    print(f"  Mean: {np.mean(samples_normal):.4f} (expected: 0)")
    print(f"  Std:  {np.std(samples_normal):.4f} (expected: 1)")
    
    # Test rejection sampling for truncated normal
    def target(x):
        if 0 <= x <= 3:
            return stats.norm.pdf(x, 0, 1)
        return 0
    
    def proposal_sampler():
        return np.random.uniform(0, 3)
    
    def proposal_pdf(x):
        return 1/3 if 0 <= x <= 3 else 0
    
    M = stats.norm.pdf(0, 0, 1) * 3 * 1.01
    
    samples_rej, acc_rate = rejection_sample(target, proposal_sampler, proposal_pdf, M, 5000)
    print(f"\nTruncated Normal [0, 3] (Rejection):")
    print(f"  Acceptance rate: {acc_rate:.2%}")
    print(f"  Mean: {np.mean(samples_rej):.4f}")


# =============================================================================
# Exercise 2: Metropolis-Hastings Implementation
# =============================================================================

def exercise_metropolis_hastings():
    """
    Exercise: Implement Metropolis-Hastings MCMC.
    
    Tasks:
    1. Implement basic MH with Gaussian proposal
    2. Sample from a multimodal distribution
    3. Analyze acceptance rate vs proposal variance
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Metropolis-Hastings")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def metropolis_hastings(log_target: Callable, proposal_std: float,
                            n_samples: int, x0: np.ndarray,
                            burn_in: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Metropolis-Hastings with Gaussian proposal.
        Returns: (samples, acceptance_rate)
        """
        pass


def solution_metropolis_hastings():
    """Reference solution for Metropolis-Hastings."""
    print("\n--- Solution ---\n")
    
    def metropolis_hastings(log_target: Callable, proposal_std: float,
                            n_samples: int, x0: np.ndarray,
                            burn_in: int = 1000) -> Tuple[np.ndarray, float]:
        d = len(x0)
        samples = np.zeros((n_samples + burn_in, d))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_samples + burn_in):
            x_current = samples[t-1]
            x_proposed = x_current + proposal_std * np.random.randn(d)
            
            log_alpha = log_target(x_proposed) - log_target(x_current)
            
            if np.log(np.random.uniform()) < log_alpha:
                samples[t] = x_proposed
                if t >= burn_in:
                    n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples[burn_in:], n_accepted / n_samples
    
    # Target: bimodal distribution
    def log_target(x):
        mu1, mu2 = np.array([-3.0]), np.array([3.0])
        sigma = 1.0
        log_p1 = -0.5 * np.sum((x - mu1)**2) / sigma**2
        log_p2 = -0.5 * np.sum((x - mu2)**2) / sigma**2
        return np.logaddexp(log_p1, log_p2) - np.log(2)
    
    np.random.seed(42)
    
    # Try different proposal scales
    print("Acceptance rate vs proposal std:")
    for std in [0.1, 0.5, 1.0, 2.0, 5.0]:
        samples, acc_rate = metropolis_hastings(
            log_target, std, n_samples=5000, x0=np.array([0.0])
        )
        mode_switches = np.sum(np.abs(np.diff(np.sign(samples.flatten()))) > 0)
        print(f"  σ={std:.1f}: acceptance={acc_rate:.2%}, mode switches={mode_switches}")
    
    # Best settings
    samples, acc_rate = metropolis_hastings(
        log_target, proposal_std=2.0, n_samples=10000, x0=np.array([0.0])
    )
    
    print(f"\nSample statistics (σ=2.0):")
    print(f"  Mean: {np.mean(samples):.4f} (expected: ~0)")
    print(f"  Time in left mode: {np.mean(samples < 0):.2%}")


# =============================================================================
# Exercise 3: Gibbs Sampling
# =============================================================================

def exercise_gibbs_sampling():
    """
    Exercise: Implement Gibbs sampling.
    
    Tasks:
    1. Implement for bivariate normal
    2. Implement for categorical-Gaussian mixture
    3. Compare mixing with MH
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Gibbs Sampling")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def gibbs_bivariate_normal(mu: np.ndarray, sigma: np.ndarray,
                                n_samples: int, burn_in: int = 500) -> np.ndarray:
        """Gibbs sampling for bivariate normal."""
        pass


def solution_gibbs_sampling():
    """Reference solution for Gibbs sampling."""
    print("\n--- Solution ---\n")
    
    def gibbs_bivariate_normal(mu: np.ndarray, sigma: np.ndarray,
                                n_samples: int, burn_in: int = 500) -> np.ndarray:
        mu1, mu2 = mu
        sigma1, sigma2 = np.sqrt(sigma[0,0]), np.sqrt(sigma[1,1])
        rho = sigma[0,1] / (sigma1 * sigma2)
        
        samples = np.zeros((n_samples + burn_in, 2))
        samples[0] = np.zeros(2)
        
        for t in range(1, n_samples + burn_in):
            # Sample x1 | x2
            cond_mean_1 = mu1 + rho * (sigma1/sigma2) * (samples[t-1,1] - mu2)
            cond_std_1 = sigma1 * np.sqrt(1 - rho**2)
            samples[t,0] = np.random.normal(cond_mean_1, cond_std_1)
            
            # Sample x2 | x1
            cond_mean_2 = mu2 + rho * (sigma2/sigma1) * (samples[t,0] - mu1)
            cond_std_2 = sigma2 * np.sqrt(1 - rho**2)
            samples[t,1] = np.random.normal(cond_mean_2, cond_std_2)
        
        return samples[burn_in:]
    
    # Test
    np.random.seed(42)
    mu = np.array([1.0, 2.0])
    sigma = np.array([[1.0, 0.9], [0.9, 1.0]])  # High correlation
    
    samples = gibbs_bivariate_normal(mu, sigma, n_samples=10000)
    
    print(f"Target: Bivariate Normal")
    print(f"  True μ: {mu}")
    print(f"  True Σ:\n{sigma}")
    
    print(f"\nGibbs Results:")
    print(f"  Sample μ: {np.mean(samples, axis=0).round(4)}")
    print(f"  Sample Σ:\n{np.round(np.cov(samples.T), 4)}")
    
    # Gibbs for mixture
    print("\n\nGibbs for Gaussian Mixture:")
    
    def gibbs_mixture(data: np.ndarray, K: int, n_iter: int):
        """Gibbs sampling for Gaussian mixture."""
        n = len(data)
        
        # Initialize
        z = np.random.randint(0, K, n)
        pi = np.ones(K) / K
        mu = np.random.randn(K)
        sigma = np.ones(K)
        
        traces = {'mu': [], 'pi': []}
        
        for _ in range(n_iter):
            # Sample z | data, params
            log_probs = np.zeros((n, K))
            for k in range(K):
                log_probs[:, k] = np.log(pi[k]) - 0.5 * ((data - mu[k])/sigma[k])**2
            
            probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            
            z = np.array([np.random.choice(K, p=probs[i]) for i in range(n)])
            
            # Sample params | z, data
            counts = np.bincount(z, minlength=K)
            
            # Update pi
            pi = np.random.dirichlet(counts + 1)
            
            # Update mu (with prior N(0,10))
            for k in range(K):
                if counts[k] > 0:
                    data_k = data[z == k]
                    post_var = 1 / (counts[k]/sigma[k]**2 + 0.01)
                    post_mean = post_var * np.sum(data_k)/sigma[k]**2
                    mu[k] = np.random.normal(post_mean, np.sqrt(post_var))
            
            traces['mu'].append(mu.copy())
            traces['pi'].append(pi.copy())
        
        return traces
    
    # Generate mixture data
    true_pi = [0.3, 0.7]
    true_mu = [-2.0, 2.0]
    n_data = 200
    
    z_true = np.random.choice(2, n_data, p=true_pi)
    data = np.array([np.random.normal(true_mu[z], 1.0) for z in z_true])
    
    traces = gibbs_mixture(data, K=2, n_iter=200)
    
    print(f"  True π: {true_pi}")
    print(f"  True μ: {true_mu}")
    print(f"  Posterior μ (last 50 iters): {np.mean(traces['mu'][-50:], axis=0).round(3)}")
    print(f"  Posterior π (last 50 iters): {np.mean(traces['pi'][-50:], axis=0).round(3)}")


# =============================================================================
# Exercise 4: Hamiltonian Monte Carlo
# =============================================================================

def exercise_hmc():
    """
    Exercise: Implement Hamiltonian Monte Carlo.
    
    Tasks:
    1. Implement leapfrog integrator
    2. Implement HMC sampling
    3. Compare with random walk MH
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Hamiltonian Monte Carlo")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def leapfrog(x: np.ndarray, p: np.ndarray, grad_U: Callable,
                 step_size: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """Leapfrog integrator."""
        pass
    
    def hmc(log_prob: Callable, grad_log_prob: Callable,
            n_samples: int, x0: np.ndarray,
            step_size: float = 0.1, n_leapfrog: int = 20) -> Tuple[np.ndarray, float]:
        """HMC sampler."""
        pass


def solution_hmc():
    """Reference solution for HMC."""
    print("\n--- Solution ---\n")
    
    def leapfrog(x: np.ndarray, p: np.ndarray, grad_U: Callable,
                 step_size: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        x = x.copy()
        p = p.copy()
        
        p = p - 0.5 * step_size * grad_U(x)
        
        for _ in range(n_steps - 1):
            x = x + step_size * p
            p = p - step_size * grad_U(x)
        
        x = x + step_size * p
        p = p - 0.5 * step_size * grad_U(x)
        
        return x, -p
    
    def hmc(log_prob: Callable, grad_log_prob: Callable,
            n_samples: int, x0: np.ndarray,
            step_size: float = 0.1, n_leapfrog: int = 20,
            burn_in: int = 500) -> Tuple[np.ndarray, float]:
        
        grad_U = lambda x: -grad_log_prob(x)
        
        d = len(x0)
        samples = np.zeros((n_samples + burn_in, d))
        samples[0] = x0
        n_accepted = 0
        
        for t in range(1, n_samples + burn_in):
            x_current = samples[t-1]
            p_current = np.random.randn(d)
            
            x_proposed, p_proposed = leapfrog(x_current, p_current, grad_U, step_size, n_leapfrog)
            
            H_current = -log_prob(x_current) + 0.5 * np.dot(p_current, p_current)
            H_proposed = -log_prob(x_proposed) + 0.5 * np.dot(p_proposed, p_proposed)
            
            if np.log(np.random.uniform()) < H_current - H_proposed:
                samples[t] = x_proposed
                if t >= burn_in:
                    n_accepted += 1
            else:
                samples[t] = x_current
        
        return samples[burn_in:], n_accepted / n_samples
    
    # Target: correlated 2D Gaussian
    rho = 0.95
    cov = np.array([[1, rho], [rho, 1]])
    cov_inv = np.linalg.inv(cov)
    
    def log_prob(x):
        return -0.5 * x @ cov_inv @ x
    
    def grad_log_prob(x):
        return -cov_inv @ x
    
    np.random.seed(42)
    
    # HMC
    samples_hmc, acc_hmc = hmc(log_prob, grad_log_prob, 
                               n_samples=2000, x0=np.zeros(2),
                               step_size=0.15, n_leapfrog=20)
    
    # Compare with MH
    def mh(log_target, n_samples, x0, proposal_std, burn_in=500):
        d = len(x0)
        samples = np.zeros((n_samples + burn_in, d))
        samples[0] = x0
        n_acc = 0
        
        for t in range(1, n_samples + burn_in):
            x_curr = samples[t-1]
            x_prop = x_curr + proposal_std * np.random.randn(d)
            
            if np.log(np.random.uniform()) < log_target(x_prop) - log_target(x_curr):
                samples[t] = x_prop
                if t >= burn_in:
                    n_acc += 1
            else:
                samples[t] = x_curr
        
        return samples[burn_in:], n_acc / n_samples
    
    samples_mh, acc_mh = mh(log_prob, n_samples=2000, x0=np.zeros(2), proposal_std=0.3)
    
    # Compute ESS
    def ess(samples):
        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)
        if var == 0:
            return n
        autocorr = np.correlate(samples - mean, samples - mean, 'full')[n-1:] / (var * n)
        for i in range(1, len(autocorr)):
            if autocorr[i] < 0:
                break
        return n / (1 + 2 * np.sum(autocorr[1:i]))
    
    print(f"Highly correlated 2D Gaussian (ρ={rho}):")
    print(f"\nHMC:")
    print(f"  Acceptance: {acc_hmc:.2%}")
    print(f"  ESS (dim 0): {ess(samples_hmc[:,0]):.0f} / 2000")
    print(f"  Sample cov:\n{np.round(np.cov(samples_hmc.T), 4)}")
    
    print(f"\nRandom Walk MH:")
    print(f"  Acceptance: {acc_mh:.2%}")
    print(f"  ESS (dim 0): {ess(samples_mh[:,0]):.0f} / 2000")
    print(f"  Sample cov:\n{np.round(np.cov(samples_mh.T), 4)}")


# =============================================================================
# Exercise 5: Importance Sampling
# =============================================================================

def exercise_importance_sampling():
    """
    Exercise: Implement importance sampling.
    
    Tasks:
    1. Basic importance sampling
    2. Self-normalized importance sampling
    3. Compute effective sample size
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Importance Sampling")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def importance_sampling(f: Callable, target_pdf: Callable,
                            proposal_sampler: Callable, proposal_pdf: Callable,
                            n: int) -> Tuple[float, float, float]:
        """
        Returns: (estimate, variance, effective_sample_size)
        """
        pass


def solution_importance_sampling():
    """Reference solution for importance sampling."""
    print("\n--- Solution ---\n")
    
    def importance_sampling(f: Callable, target_pdf: Callable,
                            proposal_sampler: Callable, proposal_pdf: Callable,
                            n: int) -> Tuple[float, float, float]:
        samples = np.array([proposal_sampler() for _ in range(n)])
        
        weights = np.array([target_pdf(x) / proposal_pdf(x) for x in samples])
        normalized_weights = weights / np.sum(weights)
        
        f_vals = np.array([f(x) for x in samples])
        
        estimate = np.sum(normalized_weights * f_vals)
        variance = np.sum(normalized_weights * (f_vals - estimate)**2)
        ess = 1.0 / np.sum(normalized_weights**2)
        
        return estimate, variance, ess
    
    # Example: E[X²] for X ~ N(0,1)
    np.random.seed(42)
    
    def f(x): return x**2
    def target(x): return stats.norm.pdf(x, 0, 1)
    
    # Good proposal
    def good_sampler(): return np.random.normal(0, 1.5)
    def good_pdf(x): return stats.norm.pdf(x, 0, 1.5)
    
    est1, var1, ess1 = importance_sampling(f, target, good_sampler, good_pdf, 10000)
    
    print(f"E[X²] where X ~ N(0,1), true value = 1.0")
    print(f"\nGood proposal N(0, 1.5):")
    print(f"  Estimate: {est1:.4f}")
    print(f"  Variance: {var1:.6f}")
    print(f"  ESS: {ess1:.0f} / 10000 ({100*ess1/10000:.1f}%)")
    
    # Bad proposal
    def bad_sampler(): return np.random.normal(3, 0.5)
    def bad_pdf(x): return stats.norm.pdf(x, 3, 0.5)
    
    est2, var2, ess2 = importance_sampling(f, target, bad_sampler, bad_pdf, 10000)
    
    print(f"\nBad proposal N(3, 0.5):")
    print(f"  Estimate: {est2:.4f}")
    print(f"  Variance: {var2:.6f}")
    print(f"  ESS: {ess2:.0f} / 10000 ({100*ess2/10000:.1f}%)")
    
    # Optimal proposal for this problem
    print(f"\n(Optimal proposal would be proportional to |f(x)|p(x))")


# =============================================================================
# Exercise 6: Particle Filter
# =============================================================================

def exercise_particle_filter():
    """
    Exercise: Implement a particle filter.
    
    Tasks:
    1. Bootstrap particle filter
    2. Systematic resampling
    3. Estimate log-likelihood
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Particle Filter")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def particle_filter(observations: np.ndarray, 
                        transition: Callable,
                        likelihood: Callable,
                        prior_sampler: Callable,
                        n_particles: int) -> Tuple[np.ndarray, float]:
        """
        Bootstrap particle filter.
        Returns: (filtered_means, log_likelihood)
        """
        pass


def solution_particle_filter():
    """Reference solution for particle filter."""
    print("\n--- Solution ---\n")
    
    def systematic_resample(weights: np.ndarray) -> np.ndarray:
        """Systematic resampling."""
        n = len(weights)
        cumsum = np.cumsum(weights)
        u0 = np.random.uniform(0, 1/n)
        u = u0 + np.arange(n) / n
        indices = np.searchsorted(cumsum, u)
        return indices
    
    def particle_filter(observations: np.ndarray,
                        transition: Callable,
                        likelihood: Callable,
                        prior_sampler: Callable,
                        n_particles: int) -> Tuple[np.ndarray, float]:
        T = len(observations)
        particles = np.zeros((T, n_particles))
        filtered_means = np.zeros(T)
        log_ml = 0
        
        # Initialize
        particles[0] = np.array([prior_sampler() for _ in range(n_particles)])
        weights = np.array([likelihood(observations[0], particles[0,i]) 
                           for i in range(n_particles)])
        
        log_ml += np.log(np.mean(weights) + 1e-300)
        weights /= np.sum(weights)
        filtered_means[0] = np.sum(weights * particles[0])
        
        for t in range(1, T):
            # Resample
            indices = systematic_resample(weights)
            
            # Propagate
            particles[t] = np.array([transition(particles[t-1, indices[i]])
                                    for i in range(n_particles)])
            
            # Weight
            weights = np.array([likelihood(observations[t], particles[t,i])
                               for i in range(n_particles)])
            
            log_ml += np.log(np.mean(weights) + 1e-300)
            weights /= np.sum(weights)
            filtered_means[t] = np.sum(weights * particles[t])
        
        return filtered_means, log_ml
    
    # Test: linear Gaussian state-space model
    np.random.seed(42)
    
    T = 100
    process_var = 1.0
    obs_var = 0.5
    
    # Generate data
    true_states = np.zeros(T)
    observations = np.zeros(T)
    
    true_states[0] = np.random.normal(0, 1)
    observations[0] = true_states[0] + np.random.normal(0, np.sqrt(obs_var))
    
    for t in range(1, T):
        true_states[t] = 0.9 * true_states[t-1] + np.random.normal(0, np.sqrt(process_var))
        observations[t] = true_states[t] + np.random.normal(0, np.sqrt(obs_var))
    
    # Run filter
    def prior_sampler():
        return np.random.normal(0, 1)
    
    def transition(x):
        return 0.9 * x + np.random.normal(0, np.sqrt(process_var))
    
    def likelihood(y, x):
        return stats.norm.pdf(y, x, np.sqrt(obs_var))
    
    filtered, log_ml = particle_filter(observations, transition, likelihood,
                                        prior_sampler, n_particles=1000)
    
    rmse = np.sqrt(np.mean((filtered - true_states)**2))
    corr = np.corrcoef(filtered, true_states)[0, 1]
    
    print(f"Linear Gaussian SSM (T={T}):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  Log marginal likelihood: {log_ml:.2f}")


# =============================================================================
# Exercise 7: Reparameterization Trick
# =============================================================================

def exercise_reparameterization():
    """
    Exercise: Implement reparameterization for variational inference.
    
    Tasks:
    1. Gaussian reparameterization
    2. Gradient computation
    3. KL divergence
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Reparameterization Trick")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def sample_gaussian(mu: np.ndarray, log_var: np.ndarray, 
                        n: int = 1) -> np.ndarray:
        """Sample using reparameterization."""
        pass
    
    def kl_gaussian(mu: np.ndarray, log_var: np.ndarray) -> float:
        """KL(N(mu, exp(log_var)) || N(0, I))"""
        pass


def solution_reparameterization():
    """Reference solution for reparameterization."""
    print("\n--- Solution ---\n")
    
    def sample_gaussian(mu: np.ndarray, log_var: np.ndarray,
                        n: int = 1) -> np.ndarray:
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(n, len(mu))
        return mu + std * eps
    
    def kl_gaussian(mu: np.ndarray, log_var: np.ndarray) -> float:
        return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    
    # Test
    np.random.seed(42)
    d = 5
    mu = np.array([1.0, -0.5, 0.5, 0.0, -1.0])
    log_var = np.array([0.0, -1.0, 0.5, -0.5, 0.0])
    
    samples = sample_gaussian(mu, log_var, n=10000)
    
    print(f"Reparameterization: z = μ + σ⊙ε, ε ~ N(0,I)")
    print(f"\nEncoder output:")
    print(f"  μ: {mu}")
    print(f"  log_var: {log_var}")
    print(f"  σ: {np.exp(0.5*log_var).round(4)}")
    
    print(f"\nSample statistics:")
    print(f"  Sample mean: {np.mean(samples, axis=0).round(4)}")
    print(f"  Sample std:  {np.std(samples, axis=0).round(4)}")
    
    kl = kl_gaussian(mu, log_var)
    print(f"\nKL divergence: {kl:.4f}")
    
    # Gradient check
    def elbo_term(params, target=np.zeros(d)):
        mu_p = params[:d]
        log_var_p = params[d:]
        np.random.seed(0)
        z = sample_gaussian(mu_p, log_var_p, 1)[0]
        reconstruction = -np.sum((z - target)**2)
        kl = kl_gaussian(mu_p, log_var_p)
        return reconstruction - 0.1 * kl
    
    params = np.concatenate([mu, log_var])
    eps = 1e-5
    grad = np.zeros(2*d)
    for i in range(2*d):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad[i] = (elbo_term(p_plus) - elbo_term(p_minus)) / (2*eps)
    
    print(f"\nNumerical gradients (ELBO):")
    print(f"  ∂/∂μ: {grad[:d].round(4)}")
    print(f"  ∂/∂log_var: {grad[d:].round(4)}")


# =============================================================================
# Exercise 8: Gumbel-Softmax
# =============================================================================

def exercise_gumbel_softmax():
    """
    Exercise: Implement Gumbel-Softmax for differentiable discrete sampling.
    
    Tasks:
    1. Gumbel-Softmax sampling
    2. Straight-through estimator
    3. Temperature annealing
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Gumbel-Softmax")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def gumbel_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Sample from Gumbel-Softmax distribution."""
        pass
    
    def straight_through(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Straight-through Gumbel-Softmax."""
        pass


def solution_gumbel_softmax():
    """Reference solution for Gumbel-Softmax."""
    print("\n--- Solution ---\n")
    
    def sample_gumbel(shape):
        u = np.random.uniform(0, 1, shape)
        return -np.log(-np.log(u + 1e-10) + 1e-10)
    
    def gumbel_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        g = sample_gumbel(logits.shape)
        y = (logits + g) / temperature
        return np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
    
    def straight_through(logits: np.ndarray, temperature: float) -> np.ndarray:
        soft = gumbel_softmax(logits, temperature)
        hard = np.zeros_like(soft)
        hard[np.argmax(soft)] = 1.0
        # For autograd: return hard - soft.detach() + soft
        return hard, soft
    
    # Test
    np.random.seed(42)
    logits = np.array([2.0, 1.0, 0.5])
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    print(f"Logits: {logits}")
    print(f"True probabilities: {probs.round(4)}")
    
    print(f"\nGumbel-Softmax samples at different temperatures:")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        samples = np.array([gumbel_softmax(logits, temp) for _ in range(10000)])
        
        entropy = -np.mean(np.sum(samples * np.log(samples + 1e-10), axis=1))
        hard = np.bincount(np.argmax(samples, axis=1), minlength=3) / 10000
        
        print(f"  τ={temp}: entropy={entropy:.3f}, hard probs={hard.round(3)}")
    
    # Temperature annealing
    print(f"\nTemperature annealing schedule:")
    for epoch in [0, 10, 50, 100, 200]:
        temp = max(0.5, np.exp(-epoch / 50))
        print(f"  Epoch {epoch:3d}: τ = {temp:.3f}")


# =============================================================================
# Exercise 9: MCMC Diagnostics
# =============================================================================

def exercise_mcmc_diagnostics():
    """
    Exercise: Implement MCMC convergence diagnostics.
    
    Tasks:
    1. Gelman-Rubin R-hat
    2. Effective sample size
    3. Trace plot analysis
    """
    print("\n" + "=" * 70)
    print("Exercise 9: MCMC Diagnostics")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def gelman_rubin(chains: np.ndarray) -> float:
        """
        Compute R-hat from multiple chains.
        chains: (n_chains, n_samples)
        """
        pass
    
    def effective_sample_size(samples: np.ndarray) -> float:
        """Compute ESS using autocorrelation."""
        pass


def solution_mcmc_diagnostics():
    """Reference solution for MCMC diagnostics."""
    print("\n--- Solution ---\n")
    
    def gelman_rubin(chains: np.ndarray) -> float:
        n_chains, n = chains.shape
        
        # Between-chain variance
        chain_means = np.mean(chains, axis=1)
        B = n * np.var(chain_means, ddof=1)
        
        # Within-chain variance  
        W = np.mean(np.var(chains, axis=1, ddof=1))
        
        # Pooled variance estimate
        var_plus = (n - 1) / n * W + B / n
        
        return np.sqrt(var_plus / W)
    
    def effective_sample_size(samples: np.ndarray) -> float:
        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)
        
        if var == 0:
            return n
        
        # Autocorrelation
        max_lag = min(n // 2, 100)
        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            autocorr[lag] = np.mean((samples[:n-lag] - mean) * (samples[lag:] - mean)) / var
        
        # Sum until negative
        for i in range(1, max_lag):
            if autocorr[i] < 0:
                break
        
        return n / (1 + 2 * np.sum(autocorr[1:i]))
    
    # Run multiple chains
    def log_target(x):
        return -0.5 * x**2  # N(0, 1)
    
    def mh_chain(n, x0, prop_std):
        samples = np.zeros(n)
        samples[0] = x0
        for t in range(1, n):
            x_prop = samples[t-1] + prop_std * np.random.randn()
            if np.log(np.random.uniform()) < log_target(x_prop) - log_target(samples[t-1]):
                samples[t] = x_prop
            else:
                samples[t] = samples[t-1]
        return samples
    
    np.random.seed(42)
    n_chains = 4
    n_samples = 2000
    burn_in = 500
    
    chains = np.zeros((n_chains, n_samples - burn_in))
    for i in range(n_chains):
        x0 = np.random.randn() * 10  # Dispersed start
        full_chain = mh_chain(n_samples, x0, prop_std=1.0)
        chains[i] = full_chain[burn_in:]
    
    r_hat = gelman_rubin(chains)
    ess_vals = [effective_sample_size(chains[i]) for i in range(n_chains)]
    
    print(f"MCMC Diagnostics (target: N(0,1)):")
    print(f"\nGelman-Rubin R-hat: {r_hat:.4f} (target: < 1.01)")
    print(f"\nEffective Sample Size per chain:")
    for i, ess in enumerate(ess_vals):
        print(f"  Chain {i}: {ess:.0f} / {n_samples - burn_in} ({100*ess/(n_samples-burn_in):.1f}%)")
    
    print(f"\nPooled statistics:")
    all_samples = chains.flatten()
    print(f"  Mean: {np.mean(all_samples):.4f} (true: 0)")
    print(f"  Std:  {np.std(all_samples):.4f} (true: 1)")


# =============================================================================
# Exercise 10: Complete Sampler
# =============================================================================

def exercise_complete_sampler():
    """
    Exercise: Build a complete sampler with automatic tuning.
    
    Tasks:
    1. Adaptive proposal (for MH)
    2. Automatic burn-in detection
    3. Convergence checking
    """
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Adaptive Sampler")
    print("=" * 70)
    
    # YOUR CODE HERE
    pass


def solution_complete_sampler():
    """Reference solution for complete sampler."""
    print("\n--- Solution ---\n")
    
    class AdaptiveMH:
        """Adaptive Metropolis-Hastings sampler."""
        
        def __init__(self, log_target: Callable, d: int,
                     target_acceptance: float = 0.234):
            self.log_target = log_target
            self.d = d
            self.target_acceptance = target_acceptance
            
            # Adaptive proposal
            self.proposal_cov = np.eye(d)
            self.proposal_scale = 2.4**2 / d
            
            # Adaptation settings
            self.adapt_start = 100
            self.adapt_interval = 50
        
        def sample(self, n_samples: int, x0: np.ndarray,
                   n_warmup: int = 1000) -> Dict:
            """Run sampler with adaptation during warmup."""
            
            samples = np.zeros((n_samples + n_warmup, self.d))
            samples[0] = x0
            
            log_prob_current = self.log_target(x0)
            n_accepted = 0
            acceptance_history = []
            
            # Running mean and covariance
            running_mean = x0.copy()
            running_cov = np.zeros((self.d, self.d))
            
            for t in range(1, n_samples + n_warmup):
                # Propose
                chol = np.linalg.cholesky(self.proposal_scale * self.proposal_cov + 1e-6 * np.eye(self.d))
                x_proposed = samples[t-1] + chol @ np.random.randn(self.d)
                
                # Accept/reject
                log_prob_proposed = self.log_target(x_proposed)
                
                if np.log(np.random.uniform()) < log_prob_proposed - log_prob_current:
                    samples[t] = x_proposed
                    log_prob_current = log_prob_proposed
                    n_accepted += 1
                else:
                    samples[t] = samples[t-1]
                
                # Update running statistics
                delta = samples[t] - running_mean
                running_mean = running_mean + delta / (t + 1)
                running_cov = (t - 1) / t * running_cov + delta.reshape(-1, 1) @ delta.reshape(1, -1) / t
                
                # Adapt proposal during warmup
                if t < n_warmup and t >= self.adapt_start and t % self.adapt_interval == 0:
                    self.proposal_cov = running_cov + 1e-6 * np.eye(self.d)
                    
                    # Adapt scale based on acceptance
                    recent_accept = n_accepted / t
                    if recent_accept < self.target_acceptance:
                        self.proposal_scale *= 0.9
                    else:
                        self.proposal_scale *= 1.1
                
                if t == n_warmup - 1:
                    acceptance_history.append(n_accepted / n_warmup)
                    n_accepted = 0
            
            final_acceptance = n_accepted / n_samples
            
            return {
                'samples': samples[n_warmup:],
                'acceptance_rate': final_acceptance,
                'proposal_cov': self.proposal_cov,
                'proposal_scale': self.proposal_scale
            }
    
    # Test on correlated 2D Gaussian
    rho = 0.9
    true_cov = np.array([[1, rho], [rho, 1]])
    true_cov_inv = np.linalg.inv(true_cov)
    
    def log_target(x):
        return -0.5 * x @ true_cov_inv @ x
    
    np.random.seed(42)
    
    sampler = AdaptiveMH(log_target, d=2)
    result = sampler.sample(n_samples=5000, x0=np.array([5.0, -5.0]), n_warmup=2000)
    
    samples = result['samples']
    
    print(f"Adaptive Metropolis-Hastings")
    print(f"Target: 2D Gaussian with ρ={rho}")
    print(f"\nAdapted parameters:")
    print(f"  Proposal scale: {result['proposal_scale']:.4f}")
    print(f"  Proposal cov:\n{np.round(result['proposal_cov'], 4)}")
    
    print(f"\nSampling results:")
    print(f"  Acceptance rate: {result['acceptance_rate']:.2%}")
    print(f"  Sample cov:\n{np.round(np.cov(samples.T), 4)}")
    print(f"  True cov:\n{true_cov}")


def main():
    """Run all exercises with solutions."""
    print("SAMPLING METHODS - EXERCISES")
    print("=" * 70)
    
    exercise_basic_sampling()
    solution_basic_sampling()
    
    exercise_metropolis_hastings()
    solution_metropolis_hastings()
    
    exercise_gibbs_sampling()
    solution_gibbs_sampling()
    
    exercise_hmc()
    solution_hmc()
    
    exercise_importance_sampling()
    solution_importance_sampling()
    
    exercise_particle_filter()
    solution_particle_filter()
    
    exercise_reparameterization()
    solution_reparameterization()
    
    exercise_gumbel_softmax()
    solution_gumbel_softmax()
    
    exercise_mcmc_diagnostics()
    solution_mcmc_diagnostics()
    
    exercise_complete_sampler()
    solution_complete_sampler()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
