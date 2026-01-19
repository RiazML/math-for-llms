"""
Generative Models: Exercises
============================

Practice implementing generative models from scratch.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass


# =============================================================================
# Exercise 1: Implement VAE from Scratch
# =============================================================================

def exercise1_vae():
    """
    Implement a complete Variational Autoencoder.
    
    Tasks:
    1. Implement encoder (recognition network)
    2. Implement reparameterization trick
    3. Implement decoder (generative network)
    4. Compute ELBO loss
    5. Implement training step with gradients
    """
    
    class VAE:
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            # YOUR CODE HERE: Initialize weights
            pass
        
        def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Return mean and log variance of q(z|x)."""
            # YOUR CODE HERE
            pass
        
        def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
            """Sample z using reparameterization trick."""
            # YOUR CODE HERE
            pass
        
        def decode(self, z: np.ndarray) -> np.ndarray:
            """Reconstruct x from z."""
            # YOUR CODE HERE
            pass
        
        def elbo(self, x: np.ndarray) -> Tuple[float, float, float]:
            """Compute ELBO = reconstruction - KL."""
            # YOUR CODE HERE
            pass
    
    print("Test VAE implementation...")


def solution1_vae():
    """Solution for Exercise 1."""
    
    class VAE:
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            
            np.random.seed(42)
            
            # Encoder
            self.W_enc1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b_enc1 = np.zeros(hidden_dim)
            self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
            self.b_mu = np.zeros(latent_dim)
            self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
            self.b_logvar = np.zeros(latent_dim)
            
            # Decoder
            self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.1
            self.b_dec1 = np.zeros(hidden_dim)
            self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.1
            self.b_dec2 = np.zeros(input_dim)
        
        def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            h = np.tanh(x @ self.W_enc1 + self.b_enc1)
            mu = h @ self.W_mu + self.b_mu
            logvar = h @ self.W_logvar + self.b_logvar
            return mu, logvar
        
        def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
            std = np.exp(0.5 * logvar)
            eps = np.random.randn(*mu.shape)
            return mu + std * eps
        
        def decode(self, z: np.ndarray) -> np.ndarray:
            h = np.tanh(z @ self.W_dec1 + self.b_dec1)
            x_recon = h @ self.W_dec2 + self.b_dec2
            return x_recon
        
        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
        
        def elbo(self, x: np.ndarray) -> Tuple[float, float, float]:
            x_recon, mu, logvar = self.forward(x)
            
            # Reconstruction (negative log likelihood)
            recon_loss = np.mean(np.sum((x - x_recon) ** 2, axis=1))
            
            # KL divergence
            kl = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
            
            elbo_val = -recon_loss - kl
            return elbo_val, recon_loss, kl
        
        def sample(self, n_samples: int) -> np.ndarray:
            z = np.random.randn(n_samples, self.latent_dim)
            return self.decode(z)
    
    # Test
    vae = VAE(input_dim=10, latent_dim=2, hidden_dim=32)
    x = np.random.randn(100, 10)
    
    elbo_val, recon, kl = vae.elbo(x)
    print(f"ELBO: {elbo_val:.4f}")
    print(f"Reconstruction: {recon:.4f}")
    print(f"KL divergence: {kl:.4f}")
    
    samples = vae.sample(5)
    print(f"Sample shape: {samples.shape}")


# =============================================================================
# Exercise 2: Implement Normalizing Flow
# =============================================================================

def exercise2_normalizing_flow():
    """
    Implement a normalizing flow with planar layers.
    
    f(z) = z + u * h(w^T z + b)
    
    Tasks:
    1. Implement planar flow layer
    2. Compute log determinant of Jacobian
    3. Stack multiple layers
    4. Compute log probability
    """
    
    class PlanarFlow:
        def __init__(self, dim: int, n_layers: int):
            self.dim = dim
            self.n_layers = n_layers
            # YOUR CODE HERE: Initialize parameters
            pass
        
        def forward_layer(
            self, z: np.ndarray, w: np.ndarray, u: np.ndarray, b: float
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Single planar layer with log det Jacobian."""
            # YOUR CODE HERE
            pass
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Full flow with total log det."""
            # YOUR CODE HERE
            pass
        
        def log_prob(self, x: np.ndarray) -> np.ndarray:
            """Compute log probability."""
            # YOUR CODE HERE
            pass
    
    print("Test normalizing flow implementation...")


def solution2_normalizing_flow():
    """Solution for Exercise 2."""
    
    class PlanarFlow:
        def __init__(self, dim: int, n_layers: int = 4):
            self.dim = dim
            self.n_layers = n_layers
            
            np.random.seed(42)
            
            # Parameters for each layer
            self.w = [np.random.randn(dim) * 0.1 for _ in range(n_layers)]
            self.u = [np.random.randn(dim) * 0.1 for _ in range(n_layers)]
            self.b = [np.random.randn() * 0.1 for _ in range(n_layers)]
        
        def h(self, x: np.ndarray) -> np.ndarray:
            """Activation function (tanh)."""
            return np.tanh(x)
        
        def h_prime(self, x: np.ndarray) -> np.ndarray:
            """Derivative of activation."""
            return 1 - np.tanh(x) ** 2
        
        def forward_layer(
            self, z: np.ndarray, w: np.ndarray, u: np.ndarray, b: float
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Planar layer: f(z) = z + u * h(w^T z + b)
            log det = log|1 + u^T * h'(w^T z + b) * w|
            """
            wzb = z @ w + b  # (batch,)
            h_val = self.h(wzb)  # (batch,)
            
            # Output
            z_out = z + np.outer(h_val, u)  # (batch, dim)
            
            # Log determinant
            h_prime_val = self.h_prime(wzb)  # (batch,)
            psi = h_prime_val[:, None] * w[None, :]  # (batch, dim)
            log_det = np.log(np.abs(1 + psi @ u) + 1e-10)  # (batch,)
            
            return z_out, log_det
        
        def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Full flow."""
            total_log_det = np.zeros(z.shape[0])
            
            for i in range(self.n_layers):
                z, log_det = self.forward_layer(z, self.w[i], self.u[i], self.b[i])
                total_log_det += log_det
            
            return z, total_log_det
        
        def log_prob(self, x: np.ndarray) -> np.ndarray:
            """
            For planar flows, we don't have easy inverse.
            So we compute p(x) = p(z) |det df/dz|
            by running forward and using the relationship.
            
            Here we assume x = f(z) for some z we have.
            """
            # For demonstration, we compute log p(z) for samples
            z = np.random.randn(x.shape[0], self.dim)  # Base samples
            x_transformed, log_det = self.forward(z)
            
            # Log prior
            log_pz = -0.5 * np.sum(z**2, axis=1) - 0.5 * self.dim * np.log(2 * np.pi)
            
            # Log probability (this is for the transformed samples)
            return log_pz + log_det
        
        def sample(self, n_samples: int) -> np.ndarray:
            """Sample from the flow."""
            z = np.random.randn(n_samples, self.dim)
            x, _ = self.forward(z)
            return x
    
    # Test
    flow = PlanarFlow(dim=2, n_layers=4)
    
    # Sample
    samples = flow.sample(100)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {np.mean(samples, axis=0).round(3)}")
    
    # Log probability
    log_probs = flow.log_prob(samples)
    print(f"Mean log prob: {np.mean(log_probs):.4f}")


# =============================================================================
# Exercise 3: Implement GAN Training Loop
# =============================================================================

def exercise3_gan_training():
    """
    Implement GAN training with alternating updates.
    
    Tasks:
    1. Implement discriminator update
    2. Implement generator update
    3. Track training metrics
    4. Implement gradient penalty (WGAN-GP)
    """
    
    def train_gan(
        generator,
        discriminator,
        x_real: np.ndarray,
        n_epochs: int,
        lr: float = 0.001
    ) -> Dict:
        """
        Train GAN with alternating updates.
        
        Returns:
            Training history
        """
        # YOUR CODE HERE
        pass
    
    print("Test GAN training implementation...")


def solution3_gan_training():
    """Solution for Exercise 3."""
    
    class Generator:
        def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 32):
            self.latent_dim = latent_dim
            self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
            self.b2 = np.zeros(output_dim)
        
        def forward(self, z: np.ndarray) -> np.ndarray:
            h = np.maximum(0, z @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
        
        def sample(self, n: int) -> np.ndarray:
            z = np.random.randn(n, self.latent_dim)
            return self.forward(z)
    
    class Discriminator:
        def __init__(self, input_dim: int, hidden_dim: int = 32):
            self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, 1) * 0.1
            self.b2 = np.zeros(1)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            h = np.maximum(0, x @ self.W1 + self.b1)
            return (h @ self.W2 + self.b2).flatten()
        
        def predict(self, x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-self.forward(x)))
    
    def numerical_gradient(f, params, eps=1e-5):
        """Compute numerical gradient."""
        grads = []
        for p in params:
            grad = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_val = p[idx]
                
                p[idx] = old_val + eps
                fxh1 = f()
                p[idx] = old_val - eps
                fxh2 = f()
                
                grad[idx] = (fxh1 - fxh2) / (2 * eps)
                p[idx] = old_val
                it.iternext()
            grads.append(grad)
        return grads
    
    def train_gan(
        G: Generator,
        D: Discriminator,
        x_real: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.01,
        n_critic: int = 5
    ) -> Dict:
        """Train GAN."""
        
        history = {'d_loss': [], 'g_loss': []}
        n_samples = x_real.shape[0]
        
        for epoch in range(n_epochs):
            # Shuffle data
            perm = np.random.permutation(n_samples)
            x_real = x_real[perm]
            
            epoch_d_loss = 0
            epoch_g_loss = 0
            n_batches = n_samples // batch_size
            
            for i in range(n_batches):
                x_batch = x_real[i*batch_size:(i+1)*batch_size]
                
                # Train discriminator
                for _ in range(n_critic):
                    z = np.random.randn(batch_size, G.latent_dim)
                    x_fake = G.forward(z)
                    
                    D_real = D.predict(x_batch)
                    D_fake = D.predict(x_fake)
                    
                    d_loss = -np.mean(np.log(D_real + 1e-10)) - np.mean(np.log(1 - D_fake + 1e-10))
                    
                    # Simple gradient update (numerical for demonstration)
                    def d_loss_fn():
                        return -np.mean(np.log(D.predict(x_batch) + 1e-10)) - \
                               np.mean(np.log(1 - D.predict(x_fake) + 1e-10))
                    
                    grads = numerical_gradient(d_loss_fn, [D.W1, D.b1, D.W2, D.b2])
                    D.W1 -= lr * grads[0]
                    D.b1 -= lr * grads[1]
                    D.W2 -= lr * grads[2]
                    D.b2 -= lr * grads[3]
                
                # Train generator
                z = np.random.randn(batch_size, G.latent_dim)
                x_fake = G.forward(z)
                D_fake = D.predict(x_fake)
                g_loss = -np.mean(np.log(D_fake + 1e-10))
                
                # Update generator (simplified)
                def g_loss_fn():
                    x_f = G.forward(z)
                    return -np.mean(np.log(D.predict(x_f) + 1e-10))
                
                grads = numerical_gradient(g_loss_fn, [G.W1, G.b1, G.W2, G.b2])
                G.W1 -= lr * grads[0]
                G.b1 -= lr * grads[1]
                G.W2 -= lr * grads[2]
                G.b2 -= lr * grads[3]
                
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
            
            history['d_loss'].append(epoch_d_loss / n_batches)
            history['g_loss'].append(epoch_g_loss / n_batches)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: D_loss={history['d_loss'][-1]:.4f}, G_loss={history['g_loss'][-1]:.4f}")
        
        return history
    
    # Test
    np.random.seed(42)
    
    # Generate real data (2D Gaussian)
    x_real = np.random.randn(200, 2) * 0.5 + np.array([2, 2])
    
    G = Generator(latent_dim=2, output_dim=2, hidden_dim=16)
    D = Discriminator(input_dim=2, hidden_dim=16)
    
    # Train for a few epochs (simplified)
    history = train_gan(G, D, x_real, n_epochs=40, lr=0.01, n_critic=1)
    
    # Check generated samples
    fake_samples = G.sample(50)
    print(f"Real mean: {np.mean(x_real, axis=0).round(3)}")
    print(f"Fake mean: {np.mean(fake_samples, axis=0).round(3)}")


# =============================================================================
# Exercise 4: Implement Diffusion Forward Process
# =============================================================================

def exercise4_diffusion_forward():
    """
    Implement the forward diffusion process.
    
    q(x_t | x_0) = N(sqrt(α̅_t) x_0, (1-α̅_t) I)
    
    Tasks:
    1. Implement noise schedule
    2. Implement forward sampling
    3. Compute noisy samples at any timestep
    """
    
    class DiffusionProcess:
        def __init__(self, n_steps: int, beta_start: float, beta_end: float):
            self.n_steps = n_steps
            # YOUR CODE HERE: Compute noise schedule
            pass
        
        def q_sample(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
            """Sample x_t given x_0."""
            # YOUR CODE HERE
            pass
        
        def posterior_mean_variance(
            self, x_0: np.ndarray, x_t: np.ndarray, t: int
        ) -> Tuple[np.ndarray, float]:
            """Compute q(x_{t-1} | x_t, x_0)."""
            # YOUR CODE HERE
            pass
    
    print("Test diffusion forward process...")


def solution4_diffusion_forward():
    """Solution for Exercise 4."""
    
    class DiffusionProcess:
        def __init__(
            self,
            n_steps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02
        ):
            self.n_steps = n_steps
            
            # Linear noise schedule
            self.betas = np.linspace(beta_start, beta_end, n_steps)
            self.alphas = 1 - self.betas
            self.alpha_bars = np.cumprod(self.alphas)
            
            # For posterior
            self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])
        
        def q_sample(self, x_0: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Sample from q(x_t | x_0).
            
            x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
            """
            alpha_bar_t = self.alpha_bars[t]
            
            noise = np.random.randn(*x_0.shape)
            x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * noise
            
            return x_t, noise
        
        def posterior_mean_variance(
            self, x_0: np.ndarray, x_t: np.ndarray, t: int
        ) -> Tuple[np.ndarray, float]:
            """
            Compute q(x_{t-1} | x_t, x_0).
            
            This is a Gaussian with:
            μ = (sqrt(α̅_{t-1}) β_t / (1 - α̅_t)) x_0 + 
                (sqrt(α_t)(1 - α̅_{t-1}) / (1 - α̅_t)) x_t
            σ² = β_t (1 - α̅_{t-1}) / (1 - α̅_t)
            """
            if t == 0:
                return x_0, 0.0
            
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_t_prev = self.alpha_bars_prev[t]
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            
            # Posterior mean coefficients
            coef1 = np.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
            coef2 = np.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            
            mean = coef1 * x_0 + coef2 * x_t
            
            # Posterior variance
            variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            
            return mean, variance
        
        def visualize_forward(self, x_0: np.ndarray, steps: List[int]):
            """Show noise levels at different timesteps."""
            print("Forward process noise levels:")
            for t in steps:
                x_t, _ = self.q_sample(x_0, t)
                signal = np.sqrt(self.alpha_bars[t])
                noise_level = np.sqrt(1 - self.alpha_bars[t])
                print(f"  t={t:4d}: α̅={self.alpha_bars[t]:.4f}, "
                      f"signal={signal:.3f}, noise={noise_level:.3f}, "
                      f"x_t std={np.std(x_t):.3f}")
    
    # Test
    diffusion = DiffusionProcess(n_steps=1000)
    
    # Original data
    x_0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    # Forward process at different timesteps
    diffusion.visualize_forward(x_0[0], [0, 100, 250, 500, 750, 999])
    
    # Posterior
    t = 500
    x_t, _ = diffusion.q_sample(x_0, t)
    mean, var = diffusion.posterior_mean_variance(x_0, x_t, t)
    print(f"\nPosterior at t={t}:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Variance: {var:.6f}")


# =============================================================================
# Exercise 5: Implement Denoising Score Matching
# =============================================================================

def exercise5_score_matching():
    """
    Implement denoising score matching.
    
    L = E[ ||s_θ(x̃) - ∇_x̃ log q(x̃|x)||² ]
    
    For Gaussian noise: ∇_x̃ log q(x̃|x) = -(x̃ - x) / σ²
    
    Tasks:
    1. Add noise to data
    2. Compute target score
    3. Train score network
    """
    
    def denoising_score_matching(
        model,
        x_data: np.ndarray,
        sigma: float,
        n_steps: int = 1000,
        lr: float = 0.01
    ) -> List[float]:
        """
        Train score network with DSM.
        
        Returns:
            Loss history
        """
        # YOUR CODE HERE
        pass
    
    print("Test denoising score matching...")


def solution5_score_matching():
    """Solution for Exercise 5."""
    
    class ScoreNetwork:
        def __init__(self, dim: int, hidden_dim: int = 64):
            self.dim = dim
            np.random.seed(42)
            self.W1 = np.random.randn(dim, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, dim) * 0.1
            self.b2 = np.zeros(dim)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            h = np.tanh(x @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
    
    def denoising_score_matching(
        model: ScoreNetwork,
        x_data: np.ndarray,
        sigma: float = 0.5,
        n_steps: int = 500,
        lr: float = 0.001,
        batch_size: int = 32
    ) -> List[float]:
        """Train with DSM."""
        
        losses = []
        n_samples = x_data.shape[0]
        
        for step in range(n_steps):
            # Sample batch
            idx = np.random.choice(n_samples, batch_size)
            x = x_data[idx]
            
            # Add noise
            noise = np.random.randn(*x.shape)
            x_noisy = x + sigma * noise
            
            # Target score: -(x_noisy - x) / sigma² = -noise / sigma
            target_score = -noise / sigma
            
            # Predicted score
            pred_score = model.forward(x_noisy)
            
            # Loss
            loss = np.mean(np.sum((pred_score - target_score) ** 2, axis=1))
            losses.append(loss)
            
            # Gradient (numerical for simplicity)
            eps = 1e-5
            
            # Update W2
            grad_W2 = np.zeros_like(model.W2)
            for i in range(model.W2.shape[0]):
                for j in range(model.W2.shape[1]):
                    model.W2[i, j] += eps
                    loss_plus = np.mean(np.sum((model.forward(x_noisy) - target_score) ** 2, axis=1))
                    model.W2[i, j] -= 2 * eps
                    loss_minus = np.mean(np.sum((model.forward(x_noisy) - target_score) ** 2, axis=1))
                    model.W2[i, j] += eps
                    grad_W2[i, j] = (loss_plus - loss_minus) / (2 * eps)
            
            model.W2 -= lr * grad_W2
            
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}: Loss = {loss:.4f}")
        
        return losses
    
    # Test
    np.random.seed(42)
    dim = 2
    
    # Generate data
    x_data = np.vstack([
        np.random.randn(100, dim) * 0.3 + np.array([2, 2]),
        np.random.randn(100, dim) * 0.3 + np.array([-2, -2])
    ])
    
    model = ScoreNetwork(dim=dim, hidden_dim=32)
    
    # Train
    losses = denoising_score_matching(model, x_data, sigma=0.5, n_steps=300)
    
    print(f"Final loss: {losses[-1]:.4f}")


# =============================================================================
# Exercise 6: Implement DDPM Reverse Process
# =============================================================================

def exercise6_ddpm_reverse():
    """
    Implement the reverse denoising process.
    
    p_θ(x_{t-1}|x_t) = N(μ_θ(x_t, t), σ_t²I)
    
    Tasks:
    1. Predict noise ε_θ(x_t, t)
    2. Compute mean μ_θ from ε_θ
    3. Sample x_{t-1}
    """
    
    class DDPMReverse:
        def __init__(self, n_steps: int, noise_predictor):
            self.n_steps = n_steps
            self.noise_predictor = noise_predictor
            # YOUR CODE HERE: Set up schedule
            pass
        
        def p_sample(self, x_t: np.ndarray, t: int) -> np.ndarray:
            """Sample x_{t-1} from p(x_{t-1}|x_t)."""
            # YOUR CODE HERE
            pass
        
        def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
            """Generate samples from x_T ~ N(0,I)."""
            # YOUR CODE HERE
            pass
    
    print("Test DDPM reverse process...")


def solution6_ddpm_reverse():
    """Solution for Exercise 6."""
    
    class NoisePredictor:
        """Simple noise predictor."""
        def __init__(self, dim: int, n_steps: int):
            self.dim = dim
            self.n_steps = n_steps
            np.random.seed(42)
            self.W1 = np.random.randn(dim + 1, 64) * 0.1
            self.b1 = np.zeros(64)
            self.W2 = np.random.randn(64, dim) * 0.1
            self.b2 = np.zeros(dim)
        
        def forward(self, x_t: np.ndarray, t: int) -> np.ndarray:
            t_emb = np.full((x_t.shape[0], 1), t / self.n_steps)
            inputs = np.concatenate([x_t, t_emb], axis=1)
            h = np.tanh(inputs @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
    
    class DDPMReverse:
        def __init__(
            self,
            n_steps: int,
            noise_predictor,
            beta_start: float = 0.0001,
            beta_end: float = 0.02
        ):
            self.n_steps = n_steps
            self.noise_predictor = noise_predictor
            
            self.betas = np.linspace(beta_start, beta_end, n_steps)
            self.alphas = 1 - self.betas
            self.alpha_bars = np.cumprod(self.alphas)
            self.alpha_bars_prev = np.append(1.0, self.alpha_bars[:-1])
        
        def p_sample(self, x_t: np.ndarray, t: int) -> np.ndarray:
            """
            Sample x_{t-1} ~ p(x_{t-1}|x_t).
            
            μ = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-α̅_t)) * ε_θ(x_t, t))
            """
            if t == 0:
                return x_t
            
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            
            # Predict noise
            eps_pred = self.noise_predictor.forward(x_t, t)
            
            # Compute mean
            coef1 = 1 / np.sqrt(alpha_t)
            coef2 = beta_t / np.sqrt(1 - alpha_bar_t)
            mean = coef1 * (x_t - coef2 * eps_pred)
            
            # Variance (simplified: use β_t)
            sigma = np.sqrt(beta_t)
            
            # Sample
            noise = np.random.randn(*x_t.shape)
            x_t_minus_1 = mean + sigma * noise
            
            return x_t_minus_1
        
        def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
            """Generate samples."""
            # Start from noise
            x = np.random.randn(*shape)
            
            # Reverse process
            for t in reversed(range(self.n_steps)):
                x = self.p_sample(x, t)
            
            return x
    
    # Test
    dim = 2
    n_steps = 100
    
    noise_pred = NoisePredictor(dim=dim, n_steps=n_steps)
    ddpm = DDPMReverse(n_steps=n_steps, noise_predictor=noise_pred)
    
    # Sample
    samples = ddpm.sample((10, dim))
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {np.mean(samples, axis=0).round(3)}")
    print(f"Sample std: {np.std(samples, axis=0).round(3)}")


# =============================================================================
# Exercise 7: Implement β-VAE
# =============================================================================

def exercise7_beta_vae():
    """
    Implement β-VAE for disentanglement.
    
    L = E[log p(x|z)] - β * D_KL(q(z|x) || p(z))
    
    Tasks:
    1. Modify VAE loss with β parameter
    2. Implement training with different β values
    3. Compare latent representations
    """
    
    class BetaVAE:
        def __init__(self, input_dim: int, latent_dim: int, beta: float = 4.0):
            self.beta = beta
            # YOUR CODE HERE
            pass
        
        def loss(self, x: np.ndarray) -> Tuple[float, float, float]:
            """Compute β-VAE loss."""
            # YOUR CODE HERE
            pass
    
    print("Test β-VAE implementation...")


def solution7_beta_vae():
    """Solution for Exercise 7."""
    
    class BetaVAE:
        def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            hidden_dim: int = 64,
            beta: float = 4.0
        ):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.beta = beta
            
            np.random.seed(42)
            
            # Encoder
            self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.1
            self.b_enc = np.zeros(hidden_dim)
            self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
            self.b_mu = np.zeros(latent_dim)
            self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
            self.b_logvar = np.zeros(latent_dim)
            
            # Decoder
            self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.1
            self.b_dec1 = np.zeros(hidden_dim)
            self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.1
            self.b_dec2 = np.zeros(input_dim)
        
        def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            h = np.tanh(x @ self.W_enc + self.b_enc)
            mu = h @ self.W_mu + self.b_mu
            logvar = h @ self.W_logvar + self.b_logvar
            return mu, logvar
        
        def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
            std = np.exp(0.5 * logvar)
            eps = np.random.randn(*mu.shape)
            return mu + std * eps
        
        def decode(self, z: np.ndarray) -> np.ndarray:
            h = np.tanh(z @ self.W_dec1 + self.b_dec1)
            return h @ self.W_dec2 + self.b_dec2
        
        def loss(self, x: np.ndarray) -> Tuple[float, float, float]:
            """β-VAE loss."""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            
            # Reconstruction
            recon_loss = np.mean(np.sum((x - x_recon) ** 2, axis=1))
            
            # KL divergence (weighted by β)
            kl = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
            
            # Total loss
            total_loss = recon_loss + self.beta * kl
            
            return total_loss, recon_loss, kl
    
    # Test with different β values
    x = np.random.randn(100, 10)
    
    print("Comparing different β values:")
    for beta in [0.1, 1.0, 4.0, 10.0]:
        vae = BetaVAE(input_dim=10, latent_dim=5, beta=beta)
        total, recon, kl = vae.loss(x)
        print(f"  β={beta:4.1f}: Total={total:.2f}, Recon={recon:.2f}, KL={kl:.4f}")


# =============================================================================
# Exercise 8: Implement VQ-VAE Quantization
# =============================================================================

def exercise8_vqvae():
    """
    Implement VQ-VAE vector quantization.
    
    z_q = argmin_k ||z_e - e_k||²
    
    Tasks:
    1. Implement codebook
    2. Implement nearest neighbor lookup
    3. Implement straight-through gradient
    4. Compute commitment loss
    """
    
    class VectorQuantizer:
        def __init__(self, n_embeddings: int, embedding_dim: int):
            self.n_embeddings = n_embeddings
            self.embedding_dim = embedding_dim
            # YOUR CODE HERE
            pass
        
        def quantize(self, z_e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Quantize encoder output."""
            # YOUR CODE HERE
            pass
        
        def loss(self, z_e: np.ndarray, z_q: np.ndarray) -> float:
            """Compute VQ loss (commitment + codebook)."""
            # YOUR CODE HERE
            pass
    
    print("Test VQ-VAE quantization...")


def solution8_vqvae():
    """Solution for Exercise 8."""
    
    class VectorQuantizer:
        def __init__(
            self,
            n_embeddings: int,
            embedding_dim: int,
            commitment_cost: float = 0.25
        ):
            self.n_embeddings = n_embeddings
            self.embedding_dim = embedding_dim
            self.commitment_cost = commitment_cost
            
            # Initialize codebook
            np.random.seed(42)
            self.embeddings = np.random.randn(n_embeddings, embedding_dim) * 0.1
        
        def quantize(self, z_e: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Quantize encoder output to nearest codebook entry.
            
            Returns:
                z_q: Quantized vectors
                indices: Codebook indices used
            """
            # Flatten if needed (assume batch, dim)
            flat_z = z_e.reshape(-1, self.embedding_dim)
            
            # Compute distances
            # ||z - e||² = ||z||² + ||e||² - 2 z·e
            z_sq = np.sum(flat_z ** 2, axis=1, keepdims=True)
            e_sq = np.sum(self.embeddings ** 2, axis=1, keepdims=True).T
            distances = z_sq + e_sq - 2 * flat_z @ self.embeddings.T
            
            # Find nearest
            indices = np.argmin(distances, axis=1)
            
            # Get quantized vectors
            z_q = self.embeddings[indices]
            z_q = z_q.reshape(z_e.shape)
            
            return z_q, indices
        
        def loss(self, z_e: np.ndarray, z_q: np.ndarray) -> Tuple[float, float]:
            """
            VQ-VAE loss:
            - Codebook loss: ||sg(z_e) - e||² (updates codebook)
            - Commitment loss: ||z_e - sg(e)||² (updates encoder)
            
            sg = stop gradient
            """
            # Codebook loss (pretend z_e is constant)
            codebook_loss = np.mean((z_e - z_q) ** 2)
            
            # Commitment loss (pretend z_q is constant)
            commitment_loss = np.mean((z_e - z_q) ** 2)
            
            total = codebook_loss + self.commitment_cost * commitment_loss
            
            return total, codebook_loss
        
        def straight_through(self, z_e: np.ndarray, z_q: np.ndarray) -> np.ndarray:
            """
            Straight-through gradient estimator.
            Forward: use z_q
            Backward: copy gradients to z_e
            
            z_q_st = z_e + sg(z_q - z_e)
            """
            # In practice, this is: z_q with gradients flowing to z_e
            return z_e + (z_q - z_e)  # z_q - z_e is treated as constant in backprop
    
    # Test
    vq = VectorQuantizer(n_embeddings=64, embedding_dim=8)
    
    # Encoder outputs
    z_e = np.random.randn(16, 8)
    
    # Quantize
    z_q, indices = vq.quantize(z_e)
    
    print(f"Input shape: {z_e.shape}")
    print(f"Quantized shape: {z_q.shape}")
    print(f"Unique indices used: {len(np.unique(indices))}")
    
    # Loss
    total_loss, codebook_loss = vq.loss(z_e, z_q)
    print(f"VQ loss: {total_loss:.4f}")
    print(f"Codebook loss: {codebook_loss:.4f}")
    
    # Reconstruction error
    recon_error = np.mean((z_e - z_q) ** 2)
    print(f"Quantization error: {recon_error:.4f}")


# =============================================================================
# Exercise 9: Implement Flow Matching Loss
# =============================================================================

def exercise9_flow_matching():
    """
    Implement conditional flow matching.
    
    For linear interpolation: x_t = (1-t)x_0 + t*x_1
    Target velocity: u_t = x_1 - x_0
    
    Tasks:
    1. Sample time t
    2. Compute interpolated points
    3. Compute target velocity
    4. Train velocity network
    """
    
    def flow_matching_loss(
        model,
        x_0: np.ndarray,
        x_1: np.ndarray
    ) -> float:
        """
        CFM loss: E[ ||v_θ(x_t, t) - u_t||² ]
        """
        # YOUR CODE HERE
        pass
    
    print("Test flow matching loss...")


def solution9_flow_matching():
    """Solution for Exercise 9."""
    
    class VelocityNetwork:
        def __init__(self, dim: int, hidden_dim: int = 64):
            self.dim = dim
            np.random.seed(42)
            self.W1 = np.random.randn(dim + 1, hidden_dim) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, dim) * 0.1
            self.b2 = np.zeros(dim)
        
        def forward(self, x: np.ndarray, t: float) -> np.ndarray:
            t_vec = np.full((x.shape[0], 1), t)
            inputs = np.concatenate([x, t_vec], axis=1)
            h = np.tanh(inputs @ self.W1 + self.b1)
            return h @ self.W2 + self.b2
    
    def flow_matching_loss(
        model: VelocityNetwork,
        x_0: np.ndarray,
        x_1: np.ndarray
    ) -> float:
        """CFM loss."""
        batch_size = x_0.shape[0]
        
        # Sample random time
        t = np.random.random()
        
        # Linear interpolation
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target velocity
        u_target = x_1 - x_0
        
        # Predicted velocity
        v_pred = model.forward(x_t, t)
        
        # MSE loss
        loss = np.mean(np.sum((v_pred - u_target) ** 2, axis=1))
        
        return loss
    
    def ode_sample(
        model: VelocityNetwork,
        x_0: np.ndarray,
        n_steps: int = 100
    ) -> np.ndarray:
        """Sample by ODE integration."""
        dt = 1.0 / n_steps
        x = x_0.copy()
        
        for step in range(n_steps):
            t = step * dt
            v = model.forward(x, t)
            x = x + dt * v
        
        return x
    
    # Test
    dim = 2
    model = VelocityNetwork(dim=dim)
    
    # Source: standard normal
    x_0 = np.random.randn(100, dim)
    
    # Target: shifted Gaussian
    x_1 = np.random.randn(100, dim) * 0.5 + np.array([3, 3])
    
    # CFM loss
    loss = flow_matching_loss(model, x_0, x_1)
    print(f"CFM loss: {loss:.4f}")
    
    # Sample (untrained - just for demonstration)
    z = np.random.randn(20, dim)
    samples = ode_sample(model, z)
    print(f"Sample mean (untrained): {np.mean(samples, axis=0).round(3)}")


# =============================================================================
# Exercise 10: Complete Generative Model Pipeline
# =============================================================================

def exercise10_complete_pipeline():
    """
    Implement a complete generative model training pipeline.
    
    Tasks:
    1. Data loading and preprocessing
    2. Model initialization
    3. Training loop with logging
    4. Sampling and evaluation
    5. FID computation
    """
    
    class GenerativeModelPipeline:
        def __init__(self, model, data_dim: int):
            self.model = model
            self.data_dim = data_dim
            # YOUR CODE HERE
            pass
        
        def train(
            self,
            x_data: np.ndarray,
            n_epochs: int,
            batch_size: int
        ) -> Dict:
            """Full training loop."""
            # YOUR CODE HERE
            pass
        
        def evaluate(self, x_real: np.ndarray, n_samples: int) -> Dict:
            """Compute evaluation metrics."""
            # YOUR CODE HERE
            pass
    
    print("Test complete pipeline...")


def solution10_complete_pipeline():
    """Solution for Exercise 10."""
    
    class SimpleVAE:
        """Simple VAE for pipeline."""
        def __init__(self, input_dim: int, latent_dim: int):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            np.random.seed(42)
            self.W_enc = np.random.randn(input_dim, 32) * 0.1
            self.b_enc = np.zeros(32)
            self.W_mu = np.random.randn(32, latent_dim) * 0.1
            self.b_mu = np.zeros(latent_dim)
            self.W_logvar = np.random.randn(32, latent_dim) * 0.1
            self.b_logvar = np.zeros(latent_dim)
            self.W_dec1 = np.random.randn(latent_dim, 32) * 0.1
            self.b_dec1 = np.zeros(32)
            self.W_dec2 = np.random.randn(32, input_dim) * 0.1
            self.b_dec2 = np.zeros(input_dim)
        
        def forward(self, x):
            h = np.tanh(x @ self.W_enc + self.b_enc)
            mu = h @ self.W_mu + self.b_mu
            logvar = h @ self.W_logvar + self.b_logvar
            z = mu + np.exp(0.5 * logvar) * np.random.randn(*mu.shape)
            h_dec = np.tanh(z @ self.W_dec1 + self.b_dec1)
            x_recon = h_dec @ self.W_dec2 + self.b_dec2
            return x_recon, mu, logvar
        
        def loss(self, x):
            x_recon, mu, logvar = self.forward(x)
            recon = np.mean(np.sum((x - x_recon) ** 2, axis=1))
            kl = -0.5 * np.mean(np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=1))
            return recon + kl, recon, kl
        
        def sample(self, n):
            z = np.random.randn(n, self.latent_dim)
            h = np.tanh(z @ self.W_dec1 + self.b_dec1)
            return h @ self.W_dec2 + self.b_dec2
    
    class GenerativeModelPipeline:
        def __init__(self, model, data_dim: int):
            self.model = model
            self.data_dim = data_dim
            self.history = {'loss': [], 'recon': [], 'kl': []}
        
        def train(
            self,
            x_data: np.ndarray,
            n_epochs: int = 50,
            batch_size: int = 32,
            lr: float = 0.001
        ) -> Dict:
            """Training loop."""
            n_samples = x_data.shape[0]
            
            for epoch in range(n_epochs):
                perm = np.random.permutation(n_samples)
                x_data = x_data[perm]
                
                epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
                n_batches = n_samples // batch_size
                
                for i in range(n_batches):
                    x_batch = x_data[i*batch_size:(i+1)*batch_size]
                    loss, recon, kl = self.model.loss(x_batch)
                    epoch_loss += loss
                    epoch_recon += recon
                    epoch_kl += kl
                
                self.history['loss'].append(epoch_loss / n_batches)
                self.history['recon'].append(epoch_recon / n_batches)
                self.history['kl'].append(epoch_kl / n_batches)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: Loss={self.history['loss'][-1]:.4f}")
            
            return self.history
        
        def evaluate(self, x_real: np.ndarray, n_samples: int = 100) -> Dict:
            """Compute metrics."""
            # Generate samples
            x_fake = self.model.sample(n_samples)
            
            # Simple FID approximation (using direct statistics)
            mu_real = np.mean(x_real, axis=0)
            mu_fake = np.mean(x_fake, axis=0)
            
            sigma_real = np.cov(x_real, rowvar=False)
            sigma_fake = np.cov(x_fake, rowvar=False)
            
            # FID (simplified)
            mean_diff = np.sum((mu_real - mu_fake) ** 2)
            
            # Trace term approximation
            trace_term = np.trace(sigma_real) + np.trace(sigma_fake)
            
            fid_approx = mean_diff + trace_term
            
            return {
                'fid_approx': fid_approx,
                'mean_diff': mean_diff,
                'real_mean': mu_real,
                'fake_mean': mu_fake
            }
    
    # Test pipeline
    np.random.seed(42)
    
    # Generate data
    data_dim = 8
    x_data = np.vstack([
        np.random.randn(500, data_dim) * 0.5 + np.array([2] * data_dim),
        np.random.randn(500, data_dim) * 0.5 + np.array([-2] * data_dim)
    ])
    
    # Create model and pipeline
    model = SimpleVAE(input_dim=data_dim, latent_dim=2)
    pipeline = GenerativeModelPipeline(model, data_dim)
    
    # Train
    history = pipeline.train(x_data, n_epochs=30, batch_size=64)
    
    # Evaluate
    metrics = pipeline.evaluate(x_data, n_samples=100)
    
    print(f"\nEvaluation:")
    print(f"  FID approximation: {metrics['fid_approx']:.4f}")
    print(f"  Mean difference: {metrics['mean_diff']:.4f}")
    print(f"  Real mean: {metrics['real_mean'][:3].round(3)}...")
    print(f"  Fake mean: {metrics['fake_mean'][:3].round(3)}...")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("GENERATIVE MODELS: EXERCISES")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("Exercise 1: VAE Implementation")
    print("=" * 70)
    solution1_vae()
    
    print("\n" + "=" * 70)
    print("Exercise 2: Normalizing Flow")
    print("=" * 70)
    solution2_normalizing_flow()
    
    print("\n" + "=" * 70)
    print("Exercise 3: GAN Training")
    print("=" * 70)
    solution3_gan_training()
    
    print("\n" + "=" * 70)
    print("Exercise 4: Diffusion Forward Process")
    print("=" * 70)
    solution4_diffusion_forward()
    
    print("\n" + "=" * 70)
    print("Exercise 5: Denoising Score Matching")
    print("=" * 70)
    solution5_score_matching()
    
    print("\n" + "=" * 70)
    print("Exercise 6: DDPM Reverse Process")
    print("=" * 70)
    solution6_ddpm_reverse()
    
    print("\n" + "=" * 70)
    print("Exercise 7: β-VAE")
    print("=" * 70)
    solution7_beta_vae()
    
    print("\n" + "=" * 70)
    print("Exercise 8: VQ-VAE Quantization")
    print("=" * 70)
    solution8_vqvae()
    
    print("\n" + "=" * 70)
    print("Exercise 9: Flow Matching")
    print("=" * 70)
    solution9_flow_matching()
    
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Pipeline")
    print("=" * 70)
    solution10_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
