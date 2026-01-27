"""
Generative Models: Mathematical Foundations - Examples
======================================================

Comprehensive implementations of generative model concepts.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass


# =============================================================================
# Example 1: Autoregressive Model (MADE-style)
# =============================================================================

def example1_autoregressive():
    """
    Autoregressive model using masked connections.
    
    p(x) = ∏ p(x_i | x_{<i})
    
    Uses masked weights to ensure autoregressive property.
    """
    print("Example 1: Autoregressive Model (MADE)")
    print("=" * 50)
    
    class MADE:
        """Masked Autoencoder for Distribution Estimation."""
        
        def __init__(self, input_dim: int, hidden_dims: List[int], seed: int = 42):
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.rng = np.random.RandomState(seed)
            
            # Assign ordering to inputs
            self.m = {}
            self.m['input'] = np.arange(input_dim)
            
            # Initialize masks and weights
            self.masks = []
            self.weights = []
            self.biases = []
            
            self._build_network()
        
        def _build_network(self):
            """Build network with masks."""
            prev_dim = self.input_dim
            
            for i, hidden_dim in enumerate(self.hidden_dims):
                # Assign degrees to hidden units
                min_degree = min(self.m['input']) if i == 0 else 0
                self.m[f'hidden_{i}'] = self.rng.randint(
                    min_degree, self.input_dim - 1, size=hidden_dim
                )
                
                # Create mask
                if i == 0:
                    mask = (self.m[f'hidden_{i}'][:, None] >= self.m['input'][None, :]).T
                else:
                    mask = (self.m[f'hidden_{i}'][:, None] >= self.m[f'hidden_{i-1}'][None, :]).T
                
                self.masks.append(mask.astype(float))
                
                # Initialize weights
                W = self.rng.randn(prev_dim, hidden_dim) * 0.1
                b = np.zeros(hidden_dim)
                self.weights.append(W)
                self.biases.append(b)
                
                prev_dim = hidden_dim
            
            # Output layer
            output_mask = (self.m['input'][:, None] > self.m[f'hidden_{len(self.hidden_dims)-1}'][None, :]).T
            self.masks.append(output_mask.astype(float))
            
            W = self.rng.randn(prev_dim, self.input_dim) * 0.1
            b = np.zeros(self.input_dim)
            self.weights.append(W)
            self.biases.append(b)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass through masked network."""
            h = x
            
            for i in range(len(self.weights) - 1):
                W_masked = self.weights[i] * self.masks[i]
                h = np.tanh(h @ W_masked + self.biases[i])
            
            # Output layer (logits for each dimension)
            W_masked = self.weights[-1] * self.masks[-1]
            logits = h @ W_masked + self.biases[-1]
            
            return logits
        
        def log_prob(self, x: np.ndarray) -> np.ndarray:
            """Compute log probability of x (Bernoulli)."""
            logits = self.forward(x)
            
            # Binary cross-entropy (treating as Bernoulli)
            probs = 1 / (1 + np.exp(-logits))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            
            log_probs = x * np.log(probs) + (1 - x) * np.log(1 - probs)
            return np.sum(log_probs, axis=1)
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Autoregressive sampling."""
            samples = np.zeros((n_samples, self.input_dim))
            
            for i in range(self.input_dim):
                logits = self.forward(samples)
                probs = 1 / (1 + np.exp(-logits[:, i]))
                samples[:, i] = (np.random.random(n_samples) < probs).astype(float)
            
            return samples
    
    # Test MADE
    np.random.seed(42)
    input_dim = 5
    made = MADE(input_dim=input_dim, hidden_dims=[10, 10])
    
    # Check mask validity (ensures autoregressive property)
    print("Checking autoregressive property via masks...")
    
    # Test with random binary data
    x = np.random.randint(0, 2, size=(10, input_dim)).astype(float)
    log_probs = made.log_prob(x)
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"Mean log prob: {np.mean(log_probs):.4f}")
    
    # Sample
    samples = made.sample(5)
    print(f"Samples:\n{samples}")


# =============================================================================
# Example 2: Variational Autoencoder (VAE)
# =============================================================================

def example2_vae():
    """
    Variational Autoencoder with ELBO optimization.
    
    ELBO = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
    """
    print("\nExample 2: Variational Autoencoder")
    print("=" * 50)
    
    class VAE:
        """Simple VAE with Gaussian encoder/decoder."""
        
        def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            hidden_dim: int = 64,
            seed: int = 42
        ):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            
            np.random.seed(seed)
            
            # Encoder weights
            self.W_enc1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b_enc1 = np.zeros(hidden_dim)
            
            self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
            self.b_mu = np.zeros(latent_dim)
            
            self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
            self.b_logvar = np.zeros(latent_dim)
            
            # Decoder weights
            self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.1
            self.b_dec1 = np.zeros(hidden_dim)
            
            self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.1
            self.b_dec2 = np.zeros(input_dim)
        
        def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Encode to latent distribution parameters."""
            h = np.tanh(x @ self.W_enc1 + self.b_enc1)
            mu = h @ self.W_mu + self.b_mu
            logvar = h @ self.W_logvar + self.b_logvar
            return mu, logvar
        
        def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
            """Reparameterization trick: z = mu + sigma * epsilon."""
            std = np.exp(0.5 * logvar)
            eps = np.random.randn(*mu.shape)
            return mu + std * eps
        
        def decode(self, z: np.ndarray) -> np.ndarray:
            """Decode latent to reconstruction."""
            h = np.tanh(z @ self.W_dec1 + self.b_dec1)
            x_recon = z @ self.W_dec2 + self.b_dec2  # Linear output
            return x_recon
        
        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Full forward pass."""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
        
        def elbo(self, x: np.ndarray) -> Tuple[float, float, float]:
            """
            Compute ELBO = Reconstruction - KL divergence.
            
            Returns:
                elbo, reconstruction_loss, kl_divergence
            """
            x_recon, mu, logvar = self.forward(x)
            
            # Reconstruction loss (MSE as Gaussian negative log-likelihood)
            recon_loss = np.mean(np.sum((x - x_recon) ** 2, axis=1))
            
            # KL divergence: D_KL(q(z|x) || p(z))
            # For Gaussian: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
            
            elbo_val = -recon_loss - kl
            return elbo_val, recon_loss, kl
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Sample from prior and decode."""
            z = np.random.randn(n_samples, self.latent_dim)
            return self.decode(z)
    
    # Test VAE
    input_dim = 10
    latent_dim = 2
    
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    
    # Generate synthetic data
    np.random.seed(42)
    x = np.random.randn(100, input_dim)
    
    # Compute ELBO
    elbo_val, recon, kl = vae.elbo(x)
    print(f"ELBO: {elbo_val:.4f}")
    print(f"Reconstruction loss: {recon:.4f}")
    print(f"KL divergence: {kl:.4f}")
    
    # Sample from model
    samples = vae.sample(5)
    print(f"Sample shape: {samples.shape}")
    
    # Encode and decode
    mu, logvar = vae.encode(x[:5])
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent means:\n{mu}")


# =============================================================================
# Example 3: Normalizing Flow (RealNVP Coupling Layer)
# =============================================================================

def example3_normalizing_flow():
    """
    RealNVP normalizing flow with coupling layers.
    
    Forward: y_a = x_a, y_b = x_b * exp(s(x_a)) + t(x_a)
    Inverse: x_a = y_a, x_b = (y_b - t(y_a)) * exp(-s(y_a))
    """
    print("\nExample 3: RealNVP Normalizing Flow")
    print("=" * 50)
    
    class CouplingLayer:
        """Affine coupling layer."""
        
        def __init__(self, dim: int, hidden_dim: int = 32, mask_type: str = 'odd'):
            self.dim = dim
            self.hidden_dim = hidden_dim
            
            # Mask: which dimensions are transformed
            if mask_type == 'odd':
                self.mask = np.array([i % 2 for i in range(dim)])
            else:
                self.mask = np.array([1 - i % 2 for i in range(dim)])
            
            # Neural network for scale and translation
            input_dim = np.sum(1 - self.mask)
            output_dim = np.sum(self.mask)
            
            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W_s = np.random.randn(hidden_dim, output_dim) * 0.01
            self.b_s = np.zeros(output_dim)
            self.W_t = np.random.randn(hidden_dim, output_dim) * 0.1
            self.b_t = np.zeros(output_dim)
        
        def _get_scale_translate(self, x_masked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Compute scale and translation from conditioner."""
            h = np.tanh(x_masked @ self.W1 + self.b1)
            s = h @ self.W_s + self.b_s  # Scale (log)
            t = h @ self.W_t + self.b_t  # Translation
            return s, t
        
        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Forward transformation with log determinant."""
            # Split based on mask
            x_masked = x[:, self.mask == 0]
            x_transform = x[:, self.mask == 1]
            
            s, t = self._get_scale_translate(x_masked)
            
            # Transform
            y_transform = x_transform * np.exp(s) + t
            
            # Combine
            y = np.zeros_like(x)
            y[:, self.mask == 0] = x_masked
            y[:, self.mask == 1] = y_transform
            
            # Log determinant
            log_det = np.sum(s, axis=1)
            
            return y, log_det
        
        def inverse(self, y: np.ndarray) -> np.ndarray:
            """Inverse transformation."""
            y_masked = y[:, self.mask == 0]
            y_transform = y[:, self.mask == 1]
            
            s, t = self._get_scale_translate(y_masked)
            
            # Inverse transform
            x_transform = (y_transform - t) * np.exp(-s)
            
            x = np.zeros_like(y)
            x[:, self.mask == 0] = y_masked
            x[:, self.mask == 1] = x_transform
            
            return x
    
    class RealNVP:
        """RealNVP flow with multiple coupling layers."""
        
        def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 32):
            self.dim = dim
            self.layers = []
            
            for i in range(n_layers):
                mask_type = 'odd' if i % 2 == 0 else 'even'
                self.layers.append(CouplingLayer(dim, hidden_dim, mask_type))
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Transform from z to x with total log determinant."""
            x = z
            total_log_det = np.zeros(z.shape[0])
            
            for layer in self.layers:
                x, log_det = layer.forward(x)
                total_log_det += log_det
            
            return x, total_log_det
        
        def inverse(self, x: np.ndarray) -> np.ndarray:
            """Transform from x to z."""
            z = x
            for layer in reversed(self.layers):
                z = layer.inverse(z)
            return z
        
        def log_prob(self, x: np.ndarray) -> np.ndarray:
            """Compute log probability using change of variables."""
            z = self.inverse(x)
            
            # Prior: standard normal
            log_pz = -0.5 * np.sum(z**2, axis=1) - 0.5 * self.dim * np.log(2 * np.pi)
            
            # Forward pass for log det
            _, log_det = self.forward(z)
            
            return log_pz + log_det
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Sample from the flow."""
            z = np.random.randn(n_samples, self.dim)
            x, _ = self.forward(z)
            return x
    
    # Test RealNVP
    np.random.seed(42)
    dim = 4
    flow = RealNVP(dim=dim, n_layers=4)
    
    # Sample
    samples = flow.sample(100)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {np.mean(samples, axis=0).round(3)}")
    print(f"Sample std: {np.std(samples, axis=0).round(3)}")
    
    # Check invertibility
    z_orig = np.random.randn(10, dim)
    x, _ = flow.forward(z_orig)
    z_recon = flow.inverse(x)
    
    recon_error = np.max(np.abs(z_orig - z_recon))
    print(f"Reconstruction error (should be ~0): {recon_error:.10f}")
    
    # Log probability
    log_probs = flow.log_prob(samples)
    print(f"Mean log probability: {np.mean(log_probs):.4f}")


# =============================================================================
# Example 4: GAN (Generative Adversarial Network)
# =============================================================================

def example4_gan():
    """
    Simple GAN implementation.
    
    min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
    """
    print("\nExample 4: Generative Adversarial Network")
    print("=" * 50)
    
    class Generator:
        """Simple generator network."""
        
        def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 32):
            self.latent_dim = latent_dim
            
            self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
            self.b2 = np.zeros(hidden_dim)
            self.W3 = np.random.randn(hidden_dim, output_dim) * 0.1
            self.b3 = np.zeros(output_dim)
        
        def forward(self, z: np.ndarray) -> np.ndarray:
            """Generate samples from latent."""
            h = np.maximum(0, z @ self.W1 + self.b1)  # ReLU
            h = np.maximum(0, h @ self.W2 + self.b2)
            x = h @ self.W3 + self.b3
            return x
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Sample from generator."""
            z = np.random.randn(n_samples, self.latent_dim)
            return self.forward(z)
    
    class Discriminator:
        """Simple discriminator network."""
        
        def __init__(self, input_dim: int, hidden_dim: int = 32):
            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
            self.b2 = np.zeros(hidden_dim)
            self.W3 = np.random.randn(hidden_dim, 1) * 0.1
            self.b3 = np.zeros(1)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Discriminate real vs fake."""
            h = np.maximum(0, x @ self.W1 + self.b1)
            h = np.maximum(0, h @ self.W2 + self.b2)
            logits = h @ self.W3 + self.b3
            return logits.flatten()
        
        def predict(self, x: np.ndarray) -> np.ndarray:
            """Return probability of real."""
            logits = self.forward(x)
            return 1 / (1 + np.exp(-logits))
    
    def gan_loss(
        D: Discriminator,
        G: Generator,
        x_real: np.ndarray,
        batch_size: int
    ) -> Tuple[float, float]:
        """
        Compute GAN losses.
        
        D_loss = -E[log D(x)] - E[log(1-D(G(z)))]
        G_loss = -E[log D(G(z))]  # Non-saturating
        """
        # Real samples
        D_real = D.predict(x_real)
        
        # Fake samples
        z = np.random.randn(batch_size, G.latent_dim)
        x_fake = G.forward(z)
        D_fake = D.predict(x_fake)
        
        # Discriminator loss
        D_loss = -np.mean(np.log(D_real + 1e-10)) - np.mean(np.log(1 - D_fake + 1e-10))
        
        # Generator loss (non-saturating)
        G_loss = -np.mean(np.log(D_fake + 1e-10))
        
        return D_loss, G_loss
    
    # Test GAN
    np.random.seed(42)
    
    data_dim = 2
    latent_dim = 2
    
    G = Generator(latent_dim=latent_dim, output_dim=data_dim)
    D = Discriminator(input_dim=data_dim)
    
    # Generate "real" data (2D Gaussian mixture)
    n_samples = 100
    centers = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]])
    x_real = np.vstack([
        c + 0.5 * np.random.randn(n_samples // 4, 2)
        for c in centers
    ])
    
    # Compute losses
    D_loss, G_loss = gan_loss(D, G, x_real, batch_size=32)
    print(f"Discriminator loss: {D_loss:.4f}")
    print(f"Generator loss: {G_loss:.4f}")
    
    # Sample from generator
    fake_samples = G.sample(100)
    print(f"Fake samples mean: {np.mean(fake_samples, axis=0).round(3)}")
    print(f"Real samples mean: {np.mean(x_real, axis=0).round(3)}")


# =============================================================================
# Example 5: Wasserstein GAN with Gradient Penalty
# =============================================================================

def example5_wgan_gp():
    """
    WGAN with gradient penalty.
    
    Wasserstein distance: W(p_r, p_g) = sup_{||f||_L <= 1} E[f(x_r)] - E[f(x_g)]
    """
    print("\nExample 5: Wasserstein GAN with Gradient Penalty")
    print("=" * 50)
    
    def compute_gradient_penalty(
        D,
        x_real: np.ndarray,
        x_fake: np.ndarray,
        eps: float = 1e-5
    ) -> float:
        """
        Compute gradient penalty for WGAN-GP.
        
        GP = E[(||∇_x D(x_interp)||_2 - 1)^2]
        """
        # Interpolate
        alpha = np.random.random((x_real.shape[0], 1))
        x_interp = alpha * x_real + (1 - alpha) * x_fake
        
        # Compute gradient norm via finite differences
        grad_norms = []
        for i in range(x_interp.shape[0]):
            grads = []
            for j in range(x_interp.shape[1]):
                x_plus = x_interp[i:i+1].copy()
                x_minus = x_interp[i:i+1].copy()
                x_plus[0, j] += eps
                x_minus[0, j] -= eps
                
                grad_j = (D.forward(x_plus) - D.forward(x_minus)) / (2 * eps)
                grads.append(grad_j[0])
            
            grad_norm = np.sqrt(np.sum(np.array(grads) ** 2))
            grad_norms.append(grad_norm)
        
        gradient_penalty = np.mean((np.array(grad_norms) - 1) ** 2)
        return gradient_penalty
    
    class Critic:
        """WGAN critic (no sigmoid)."""
        
        def __init__(self, input_dim: int, hidden_dim: int = 32):
            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, 1) * 0.1
            self.b2 = np.zeros(1)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            h = np.maximum(0.01 * (x @ self.W1 + self.b1), x @ self.W1 + self.b1)  # LeakyReLU
            return (h @ self.W2 + self.b2).flatten()
    
    # Test
    np.random.seed(42)
    
    critic = Critic(input_dim=2)
    
    x_real = np.random.randn(32, 2)
    x_fake = np.random.randn(32, 2) * 2 + 1
    
    # Wasserstein loss
    W_loss = np.mean(critic.forward(x_real)) - np.mean(critic.forward(x_fake))
    print(f"Wasserstein estimate: {W_loss:.4f}")
    
    # Gradient penalty
    gp = compute_gradient_penalty(critic, x_real, x_fake)
    print(f"Gradient penalty: {gp:.4f}")
    
    # Total critic loss
    lambda_gp = 10
    critic_loss = -W_loss + lambda_gp * gp
    print(f"Total critic loss: {critic_loss:.4f}")


# =============================================================================
# Example 6: Diffusion Model (DDPM)
# =============================================================================

def example6_ddpm():
    """
    Denoising Diffusion Probabilistic Model.
    
    Forward: q(x_t|x_0) = N(sqrt(α̅_t)x_0, (1-α̅_t)I)
    Reverse: p(x_{t-1}|x_t) - learned denoising
    """
    print("\nExample 6: Denoising Diffusion (DDPM)")
    print("=" * 50)
    
    class SimpleDDPM:
        """Simple 1D diffusion model."""
        
        def __init__(
            self,
            n_steps: int = 100,
            beta_start: float = 0.0001,
            beta_end: float = 0.02
        ):
            self.n_steps = n_steps
            
            # Noise schedule
            self.betas = np.linspace(beta_start, beta_end, n_steps)
            self.alphas = 1 - self.betas
            self.alpha_bars = np.cumprod(self.alphas)
            
            # Simple noise predictor (for demonstration)
            self.W1 = np.random.randn(2, 64) * 0.1  # Input: [x_t, t_embedding]
            self.b1 = np.zeros(64)
            self.W2 = np.random.randn(64, 1) * 0.1
            self.b2 = np.zeros(1)
        
        def get_time_embedding(self, t: int) -> float:
            """Simple time embedding (normalized)."""
            return t / self.n_steps
        
        def noise_predictor(self, x_t: np.ndarray, t: int) -> np.ndarray:
            """Predict noise ε from x_t and t."""
            t_emb = np.full(x_t.shape, self.get_time_embedding(t))
            inputs = np.stack([x_t.flatten(), t_emb.flatten()], axis=1)
            
            h = np.tanh(inputs @ self.W1 + self.b1)
            eps_pred = (h @ self.W2 + self.b2).flatten()
            
            return eps_pred.reshape(x_t.shape)
        
        def forward_process(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Add noise to x_0 according to forward process.
            
            x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
            """
            alpha_bar = self.alpha_bars[t]
            eps = np.random.randn(*x_0.shape)
            x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * eps
            return x_t, eps
        
        def training_loss(self, x_0: np.ndarray) -> float:
            """
            Simplified training loss: ||ε - ε_θ(x_t, t)||²
            """
            t = np.random.randint(0, self.n_steps)
            x_t, eps_true = self.forward_process(x_0, t)
            eps_pred = self.noise_predictor(x_t, t)
            
            return np.mean((eps_true - eps_pred) ** 2)
        
        def reverse_step(self, x_t: np.ndarray, t: int) -> np.ndarray:
            """
            Single reverse (denoising) step.
            """
            if t == 0:
                return x_t
            
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t-1] if t > 0 else 1.0
            
            # Predict noise
            eps_pred = self.noise_predictor(x_t, t)
            
            # Compute mean
            coef1 = 1 / np.sqrt(alpha)
            coef2 = (1 - alpha) / np.sqrt(1 - alpha_bar)
            mu = coef1 * (x_t - coef2 * eps_pred)
            
            # Add noise (except for t=0)
            if t > 0:
                sigma = np.sqrt(self.betas[t])
                z = np.random.randn(*x_t.shape)
                x_t_minus_1 = mu + sigma * z
            else:
                x_t_minus_1 = mu
            
            return x_t_minus_1
        
        def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
            """Generate samples by running reverse process."""
            x = np.random.randn(*shape)
            
            for t in reversed(range(self.n_steps)):
                x = self.reverse_step(x, t)
            
            return x
    
    # Test DDPM
    np.random.seed(42)
    
    ddpm = SimpleDDPM(n_steps=100)
    
    # Forward process demonstration
    x_0 = np.array([1.0, 2.0, 3.0])
    
    print("Forward process (noise levels at different t):")
    for t in [0, 25, 50, 75, 99]:
        x_t, _ = ddpm.forward_process(x_0, t)
        alpha_bar = ddpm.alpha_bars[t]
        print(f"  t={t:3d}: α̅={alpha_bar:.4f}, x_t mean={np.mean(x_t):.3f}")
    
    # Training loss
    loss = ddpm.training_loss(x_0)
    print(f"\nTraining loss: {loss:.4f}")
    
    # Sample (just for demonstration - untrained model)
    samples = ddpm.sample((5,))
    print(f"Samples (untrained): {samples.round(3)}")


# =============================================================================
# Example 7: Score Matching
# =============================================================================

def example7_score_matching():
    """
    Score matching for learning ∇_x log p(x).
    
    Denoising score matching:
    L = E[(s_θ(x̃) - ∇_x̃ log q(x̃|x))²]
    """
    print("\nExample 7: Score Matching")
    print("=" * 50)
    
    class ScoreNetwork:
        """Simple score network."""
        
        def __init__(self, dim: int, hidden_dim: int = 64):
            self.dim = dim
            self.W1 = np.random.randn(dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, dim) * 0.1
            self.b2 = np.zeros(dim)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Predict score ∇_x log p(x)."""
            h = np.tanh(x @ self.W1 + self.b1)
            score = h @ self.W2 + self.b2
            return score
    
    def denoising_score_matching_loss(
        score_net: ScoreNetwork,
        x_clean: np.ndarray,
        sigma: float = 0.1
    ) -> float:
        """
        Denoising score matching loss.
        
        For Gaussian noise q(x̃|x) = N(x, σ²I):
        ∇_x̃ log q(x̃|x) = -(x̃ - x) / σ²
        """
        # Add noise
        noise = np.random.randn(*x_clean.shape)
        x_noisy = x_clean + sigma * noise
        
        # True score
        true_score = -noise / sigma  # = -(x_noisy - x_clean) / sigma²
        
        # Predicted score
        pred_score = score_net.forward(x_noisy)
        
        # MSE loss
        loss = np.mean(np.sum((pred_score - true_score) ** 2, axis=1))
        
        return loss
    
    def sliced_score_matching_loss(
        score_net: ScoreNetwork,
        x: np.ndarray,
        n_projections: int = 10,
        eps: float = 1e-5
    ) -> float:
        """
        Sliced score matching (no need for ∇_x log p(x)).
        
        L = E_v[v^T s(x) v + 0.5 (v^T s(x))²]
        """
        total_loss = 0
        
        for _ in range(n_projections):
            # Random projection direction
            v = np.random.randn(*x.shape)
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            
            # Score
            s = score_net.forward(x)
            
            # v^T s(x)
            vs = np.sum(v * s, axis=1)
            
            # Compute ∂/∂x (v^T s(x)) via finite differences
            div_term = np.zeros(x.shape[0])
            for i in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[:, i] += eps
                s_plus = score_net.forward(x_plus)
                vs_plus = np.sum(v * s_plus, axis=1)
                div_term += v[:, i] * (vs_plus - vs) / eps
            
            loss = div_term + 0.5 * vs ** 2
            total_loss += np.mean(loss)
        
        return total_loss / n_projections
    
    # Test
    np.random.seed(42)
    dim = 2
    score_net = ScoreNetwork(dim=dim)
    
    # Generate data from mixture of Gaussians
    n_samples = 100
    x = np.vstack([
        np.random.randn(n_samples // 2, dim) + np.array([2, 2]),
        np.random.randn(n_samples // 2, dim) + np.array([-2, -2])
    ])
    
    # Denoising score matching loss
    dsm_loss = denoising_score_matching_loss(score_net, x, sigma=0.5)
    print(f"Denoising score matching loss: {dsm_loss:.4f}")
    
    # Sliced score matching loss
    ssm_loss = sliced_score_matching_loss(score_net, x[:20])
    print(f"Sliced score matching loss: {ssm_loss:.4f}")


# =============================================================================
# Example 8: Energy-Based Model
# =============================================================================

def example8_energy_based():
    """
    Energy-based model with contrastive divergence.
    
    p(x) = exp(-E(x)) / Z
    """
    print("\nExample 8: Energy-Based Model")
    print("=" * 50)
    
    class EnergyModel:
        """Simple energy-based model."""
        
        def __init__(self, dim: int, hidden_dim: int = 32):
            self.dim = dim
            self.W1 = np.random.randn(dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, 1) * 0.1
            self.b2 = np.zeros(1)
        
        def energy(self, x: np.ndarray) -> np.ndarray:
            """Compute energy E(x)."""
            h = np.tanh(x @ self.W1 + self.b1)
            E = (h @ self.W2 + self.b2).flatten()
            return E
        
        def score(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
            """Compute score ∇_x log p(x) = -∇_x E(x)."""
            scores = np.zeros_like(x)
            
            for i in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[:, i] += eps
                x_minus = x.copy()
                x_minus[:, i] -= eps
                
                E_plus = self.energy(x_plus)
                E_minus = self.energy(x_minus)
                
                scores[:, i] = -(E_plus - E_minus) / (2 * eps)
            
            return scores
        
        def langevin_sample(
            self,
            n_samples: int,
            n_steps: int = 100,
            step_size: float = 0.01
        ) -> np.ndarray:
            """Sample using Langevin dynamics."""
            x = np.random.randn(n_samples, self.dim)
            
            for _ in range(n_steps):
                score = self.score(x)
                noise = np.random.randn(*x.shape)
                x = x + step_size * score + np.sqrt(2 * step_size) * noise
            
            return x
    
    def contrastive_divergence_loss(
        model: EnergyModel,
        x_data: np.ndarray,
        k: int = 1,  # CD-k
        step_size: float = 0.01
    ) -> float:
        """
        Contrastive divergence loss.
        
        L ≈ E_data[E(x)] - E_model[E(x̃)]
        """
        # Positive phase: energy on data
        E_data = np.mean(model.energy(x_data))
        
        # Negative phase: run k steps of MCMC starting from data
        x_sample = x_data.copy()
        for _ in range(k):
            score = model.score(x_sample)
            noise = np.random.randn(*x_sample.shape)
            x_sample = x_sample + step_size * score + np.sqrt(2 * step_size) * noise
        
        E_model = np.mean(model.energy(x_sample))
        
        # CD loss (want to minimize E_data and maximize E_model)
        loss = E_data - E_model
        
        return loss
    
    # Test
    np.random.seed(42)
    dim = 2
    ebm = EnergyModel(dim=dim)
    
    # Generate data
    x_data = np.random.randn(100, dim) * 0.5 + np.array([1, 1])
    
    # Energy values
    E = ebm.energy(x_data)
    print(f"Mean energy on data: {np.mean(E):.4f}")
    
    # Contrastive divergence loss
    cd_loss = contrastive_divergence_loss(ebm, x_data, k=10)
    print(f"CD-10 loss: {cd_loss:.4f}")
    
    # Sample via Langevin dynamics
    samples = ebm.langevin_sample(n_samples=50, n_steps=100)
    print(f"Samples mean: {np.mean(samples, axis=0).round(3)}")


# =============================================================================
# Example 9: Flow Matching
# =============================================================================

def example9_flow_matching():
    """
    Flow matching for generative modeling.
    
    Learn vector field v_t(x) such that ODE dx/dt = v_t(x)
    transforms p_0 to p_1.
    """
    print("\nExample 9: Flow Matching")
    print("=" * 50)
    
    class VectorFieldNetwork:
        """Simple vector field predictor."""
        
        def __init__(self, dim: int, hidden_dim: int = 64):
            self.dim = dim
            # Input: [x, t]
            self.W1 = np.random.randn(dim + 1, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, dim) * 0.1
            self.b2 = np.zeros(dim)
        
        def forward(self, x: np.ndarray, t: float) -> np.ndarray:
            """Predict velocity v_t(x)."""
            t_broadcast = np.full((x.shape[0], 1), t)
            inputs = np.concatenate([x, t_broadcast], axis=1)
            
            h = np.tanh(inputs @ self.W1 + self.b1)
            v = h @ self.W2 + self.b2
            return v
    
    def conditional_flow_matching_loss(
        model: VectorFieldNetwork,
        x_0: np.ndarray,  # Noise
        x_1: np.ndarray   # Data
    ) -> float:
        """
        Conditional flow matching loss.
        
        For linear interpolation: x_t = (1-t)x_0 + t*x_1
        Target velocity: u_t(x|x_1) = x_1 - x_0
        """
        # Random time
        t = np.random.random()
        
        # Interpolate
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target velocity (for linear path)
        u_target = x_1 - x_0
        
        # Predicted velocity
        v_pred = model.forward(x_t, t)
        
        # MSE loss
        loss = np.mean(np.sum((v_pred - u_target) ** 2, axis=1))
        
        return loss
    
    def ode_sample(
        model: VectorFieldNetwork,
        x_0: np.ndarray,
        n_steps: int = 100
    ) -> np.ndarray:
        """Sample by integrating ODE from t=0 to t=1."""
        dt = 1.0 / n_steps
        x = x_0.copy()
        
        for step in range(n_steps):
            t = step * dt
            v = model.forward(x, t)
            x = x + dt * v  # Euler integration
        
        return x
    
    # Test
    np.random.seed(42)
    dim = 2
    model = VectorFieldNetwork(dim=dim)
    
    # Generate data
    n_samples = 100
    x_0 = np.random.randn(n_samples, dim)  # Noise (source)
    x_1 = np.random.randn(n_samples, dim) * 0.5 + np.array([2, 2])  # Data (target)
    
    # CFM loss
    loss = conditional_flow_matching_loss(model, x_0, x_1)
    print(f"CFM loss: {loss:.4f}")
    
    # Sample
    z = np.random.randn(20, dim)
    samples = ode_sample(model, z, n_steps=100)
    print(f"Samples mean (untrained): {np.mean(samples, axis=0).round(3)}")
    print(f"Target mean: {np.mean(x_1, axis=0).round(3)}")


# =============================================================================
# Example 10: Evaluation Metrics
# =============================================================================

def example10_evaluation_metrics():
    """
    Evaluation metrics for generative models.
    
    - FID (Fréchet Inception Distance)
    - IS (Inception Score) - simplified
    """
    print("\nExample 10: Evaluation Metrics")
    print("=" * 50)
    
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def matrix_sqrt(A: np.ndarray) -> np.ndarray:
        """Compute matrix square root via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure positive
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    
    def frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray
    ) -> float:
        """
        Fréchet distance between two Gaussians.
        
        FID = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1Σ2)^{1/2})
        """
        # Mean difference
        diff = mu1 - mu2
        mean_term = np.sum(diff ** 2)
        
        # Covariance term
        sqrt_product = matrix_sqrt(sigma1 @ sigma2)
        cov_term = np.trace(sigma1 + sigma2 - 2 * sqrt_product)
        
        return mean_term + cov_term
    
    def precision_recall(
        real_features: np.ndarray,
        fake_features: np.ndarray,
        k: int = 5
    ) -> Tuple[float, float]:
        """
        Compute precision and recall using k-nearest neighbors.
        
        Precision: fraction of fake samples close to real manifold
        Recall: fraction of real manifold covered by fake samples
        """
        from scipy.spatial.distance import cdist
        
        # Distance from fake to real
        D_fake_real = cdist(fake_features, real_features)
        
        # Distance from real to fake
        D_real_fake = cdist(real_features, fake_features)
        
        # For each fake, find k-th nearest real neighbor
        kth_distances_fake = np.sort(D_fake_real, axis=1)[:, k-1]
        
        # For each real, find k-th nearest fake neighbor
        kth_distances_real = np.sort(D_real_fake, axis=1)[:, k-1]
        
        # Threshold (use median distances in manifold)
        D_real_real = cdist(real_features, real_features)
        np.fill_diagonal(D_real_real, np.inf)
        threshold = np.median(np.min(D_real_real, axis=1))
        
        # Precision: fake samples that are close to real
        precision = np.mean(kth_distances_fake < threshold)
        
        # Recall: real samples that are close to fake
        recall = np.mean(kth_distances_real < threshold)
        
        return precision, recall
    
    # Test with synthetic features
    np.random.seed(42)
    
    feature_dim = 64
    n_real = 1000
    n_fake = 1000
    
    # Real features (from some distribution)
    real_features = np.random.randn(n_real, feature_dim)
    
    # Fake features (slightly different distribution)
    fake_features = np.random.randn(n_fake, feature_dim) * 1.1 + 0.2
    
    # Compute FID
    mu_real, sigma_real = compute_statistics(real_features)
    mu_fake, sigma_fake = compute_statistics(fake_features)
    
    fid = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"FID: {fid:.2f}")
    
    # Identical distributions should have FID ≈ 0
    fid_same = frechet_distance(mu_real, sigma_real, mu_real, sigma_real)
    print(f"FID (same distribution): {fid_same:.6f}")
    
    # Precision and Recall
    prec, rec = precision_recall(real_features, fake_features)
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("GENERATIVE MODELS: MATHEMATICAL FOUNDATIONS")
    print("=" * 70)
    
    example1_autoregressive()
    example2_vae()
    example3_normalizing_flow()
    example4_gan()
    example5_wgan_gp()
    example6_ddpm()
    example7_score_matching()
    example8_energy_based()
    example9_flow_matching()
    example10_evaluation_metrics()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
