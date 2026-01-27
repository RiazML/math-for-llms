"""
Regularization Methods - Examples
=================================
Implementing and demonstrating regularization techniques.
"""

import numpy as np
from scipy import optimize


def example_l2_regularization():
    """L2 regularization (Ridge regression)."""
    print("=" * 60)
    print("EXAMPLE 1: L2 Regularization (Ridge)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data with multicollinearity
    n, p = 100, 5
    X = np.random.randn(n, p)
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # Correlated features
    
    true_w = np.array([1.0, 2.0, 0.5, -1.0, 0.3])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    def ridge_solution(X, y, lambda_):
        """Closed-form Ridge solution."""
        n_features = X.shape[1]
        return np.linalg.solve(X.T @ X + lambda_ * np.eye(n_features), X.T @ y)
    
    print("Ridge: w = (X'X + λI)^(-1) X'y")
    print(f"\nTrue weights: {true_w}")
    
    print(f"\n{'λ':>10} {'||w||_2':>12} {'Weights':>40}")
    print("-" * 65)
    
    for lambda_ in [0, 0.1, 1.0, 10.0, 100.0]:
        w = ridge_solution(X, y, lambda_)
        w_norm = np.linalg.norm(w)
        w_str = "[" + ", ".join(f"{wi:.2f}" for wi in w) + "]"
        print(f"{lambda_:>10.1f} {w_norm:>12.4f} {w_str:>40}")
    
    print("\nAs λ increases, weights shrink toward zero")


def example_l1_regularization():
    """L1 regularization (Lasso) with sparsity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: L1 Regularization (Lasso)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate sparse ground truth
    n, p = 100, 10
    X = np.random.randn(n, p)
    true_w = np.array([3.0, -2.0, 0, 0, 1.5, 0, 0, 0, -1.0, 0])
    y = X @ true_w + 0.3 * np.random.randn(n)
    
    def soft_threshold(w, threshold):
        """Soft thresholding operator."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
    
    def lasso_coordinate_descent(X, y, lambda_, max_iter=1000, tol=1e-6):
        """Coordinate descent for Lasso."""
        n, p = X.shape
        w = np.zeros(p)
        
        for iteration in range(max_iter):
            w_old = w.copy()
            
            for j in range(p):
                # Compute residual without feature j
                residual = y - X @ w + X[:, j] * w[j]
                
                # Coordinate update
                rho = X[:, j] @ residual
                w[j] = soft_threshold(rho, lambda_ * n) / (X[:, j] @ X[:, j])
            
            if np.linalg.norm(w - w_old) < tol:
                break
        
        return w
    
    print("Lasso induces sparsity (some weights become exactly 0)")
    print(f"\nTrue weights: {true_w}")
    print(f"Non-zero: {np.sum(true_w != 0)}")
    
    print(f"\n{'λ':>8} {'Non-zero':>10} {'||w||_1':>10} {'MSE':>10}")
    print("-" * 45)
    
    for lambda_ in [0.01, 0.1, 0.5, 1.0, 2.0]:
        w = lasso_coordinate_descent(X, y, lambda_)
        n_nonzero = np.sum(np.abs(w) > 1e-6)
        w_norm = np.linalg.norm(w, 1)
        mse = np.mean((y - X @ w) ** 2)
        print(f"{lambda_:>8.2f} {n_nonzero:>10} {w_norm:>10.4f} {mse:>10.4f}")
    
    # Show which features selected at λ=0.5
    w = lasso_coordinate_descent(X, y, 0.5)
    print(f"\nSelected features at λ=0.5: {np.where(np.abs(w) > 1e-6)[0]}")
    print(f"True non-zero features: {np.where(true_w != 0)[0]}")


def example_elastic_net():
    """Elastic Net combining L1 and L2."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Elastic Net")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Correlated features
    n, p = 100, 6
    X = np.random.randn(n, p)
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)
    X[:, 3] = X[:, 2] + 0.1 * np.random.randn(n)
    
    true_w = np.array([2.0, 2.0, 1.0, 1.0, 0, 0])
    y = X @ true_w + 0.3 * np.random.randn(n)
    
    def elastic_net_loss(w, X, y, lambda1, lambda2):
        """Elastic net objective."""
        residual = y - X @ w
        return 0.5 * np.sum(residual**2) + lambda1 * np.sum(np.abs(w)) + lambda2 * np.sum(w**2)
    
    def elastic_net_coord_descent(X, y, lambda1, lambda2, max_iter=1000):
        """Coordinate descent for Elastic Net."""
        n, p = X.shape
        w = np.zeros(p)
        
        for _ in range(max_iter):
            w_old = w.copy()
            
            for j in range(p):
                residual = y - X @ w + X[:, j] * w[j]
                rho = X[:, j] @ residual
                
                # Elastic net update
                denom = X[:, j] @ X[:, j] + 2 * lambda2
                if rho > lambda1:
                    w[j] = (rho - lambda1) / denom
                elif rho < -lambda1:
                    w[j] = (rho + lambda1) / denom
                else:
                    w[j] = 0
            
            if np.linalg.norm(w - w_old) < 1e-6:
                break
        
        return w
    
    print("Elastic Net: λ₁||w||₁ + λ₂||w||₂²")
    print("Groups correlated features together")
    
    print(f"\nTrue weights: {true_w}")
    print("(Features 0,1 correlated; Features 2,3 correlated)")
    
    # Compare Lasso vs Elastic Net
    lambda_total = 0.5
    
    # Pure Lasso
    w_lasso = elastic_net_coord_descent(X, y, lambda_total, 0)
    
    # Elastic Net (50-50 mix)
    w_enet = elastic_net_coord_descent(X, y, lambda_total/2, lambda_total/2)
    
    print(f"\n{'Method':>15} {'Weights':>50}")
    print("-" * 70)
    print(f"{'True':>15} {str(true_w.round(2)):>50}")
    print(f"{'Lasso':>15} {str(w_lasso.round(2)):>50}")
    print(f"{'Elastic Net':>15} {str(w_enet.round(2)):>50}")
    
    print("\nElastic Net keeps correlated features together")


def example_dropout_simulation():
    """Simulate dropout effect."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Dropout Simulation")
    print("=" * 60)
    
    np.random.seed(42)
    
    def dropout_forward(x, p_drop, training=True):
        """Apply dropout."""
        if not training or p_drop == 0:
            return x
        
        mask = np.random.binomial(1, 1 - p_drop, size=x.shape)
        return x * mask / (1 - p_drop)  # Inverted dropout
    
    # Simple 2-layer network simulation
    n_samples = 5
    n_hidden = 8
    
    x = np.random.randn(n_samples, 4)
    W1 = np.random.randn(4, n_hidden) * 0.5
    W2 = np.random.randn(n_hidden, 2) * 0.5
    
    def relu(x):
        return np.maximum(0, x)
    
    def forward(x, W1, W2, p_drop=0.0, training=True):
        h = relu(x @ W1)
        h = dropout_forward(h, p_drop, training)
        return h @ W2
    
    # Without dropout (inference)
    out_no_drop = forward(x, W1, W2, p_drop=0.0, training=False)
    
    # Multiple forward passes with dropout
    print("Effect of dropout on forward passes:")
    print(f"\nWithout dropout (single pass):")
    print(f"  Output[0]: {out_no_drop[0].round(3)}")
    
    print(f"\nWith 50% dropout (multiple passes):")
    outputs = []
    for i in range(5):
        out = forward(x, W1, W2, p_drop=0.5, training=True)
        outputs.append(out[0])
        print(f"  Pass {i+1}: {out[0].round(3)}")
    
    print(f"\nMean of dropout passes: {np.mean(outputs, axis=0).round(3)}")
    print(f"Std of dropout passes:  {np.std(outputs, axis=0).round(3)}")
    print("\nDropout adds stochasticity during training")


def example_early_stopping():
    """Early stopping demonstration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Early Stopping")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n_train, n_val = 50, 20
    
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    x_train = np.random.rand(n_train)
    y_train = true_function(x_train) + 0.3 * np.random.randn(n_train)
    
    x_val = np.random.rand(n_val)
    y_val = true_function(x_val) + 0.3 * np.random.randn(n_val)
    
    # Polynomial features
    def polynomial_features(x, degree):
        return np.column_stack([x**i for i in range(degree + 1)])
    
    degree = 15  # High degree to enable overfitting
    X_train = polynomial_features(x_train, degree)
    X_val = polynomial_features(x_val, degree)
    
    # Gradient descent with early stopping
    def train_with_early_stopping(X_train, y_train, X_val, y_val, 
                                   lr=0.001, max_epochs=1000, patience=50):
        n_features = X_train.shape[1]
        w = np.zeros(n_features)
        
        best_val_loss = float('inf')
        best_w = w.copy()
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            # Training loss and gradient
            pred_train = X_train @ w
            train_loss = np.mean((pred_train - y_train) ** 2)
            grad = 2 * X_train.T @ (pred_train - y_train) / len(y_train)
            
            # Update
            w = w - lr * grad
            
            # Validation loss
            pred_val = X_val @ w
            val_loss = np.mean((pred_val - y_val) ** 2)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_w = w.copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return best_w, best_epoch, train_losses, val_losses
    
    w, best_epoch, train_losses, val_losses = train_with_early_stopping(
        X_train, y_train, X_val, y_val, lr=0.0001, max_epochs=2000, patience=100
    )
    
    print(f"Best epoch: {best_epoch}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"Best val loss: {min(val_losses):.4f}")
    
    # Show loss progression
    print(f"\n{'Epoch':>8} {'Train Loss':>12} {'Val Loss':>12}")
    print("-" * 35)
    for i in [0, 100, 200, 500, best_epoch, len(train_losses)-1]:
        if i < len(train_losses):
            marker = " *" if i == best_epoch else ""
            print(f"{i:>8} {train_losses[i]:>12.4f} {val_losses[i]:>12.4f}{marker}")


def example_weight_decay_vs_l2():
    """Weight decay vs L2 regularization difference with Adam."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Weight Decay vs L2 with Adam")
    print("=" * 60)
    
    np.random.seed(42)
    
    # They differ when using adaptive optimizers!
    print("With SGD: Weight decay = L2 regularization")
    print("With Adam: Weight decay ≠ L2 regularization")
    
    def adam_l2(grad, w, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, 
                eps=1e-8, lambda_l2=0.01):
        """Adam with L2 regularization (in gradient)."""
        grad = grad + lambda_l2 * w  # L2 added to gradient
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        return w, m, v
    
    def adamw(grad, w, m, v, t, lr=0.01, beta1=0.9, beta2=0.999,
              eps=1e-8, weight_decay=0.01):
        """AdamW with decoupled weight decay."""
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        w = w - lr * weight_decay * w  # Decoupled weight decay
        return w, m, v
    
    # Simple optimization
    w_l2 = np.array([5.0, 3.0])
    w_wd = np.array([5.0, 3.0])
    m_l2, v_l2 = np.zeros(2), np.zeros(2)
    m_wd, v_wd = np.zeros(2), np.zeros(2)
    
    print(f"\nInitial weights: {w_l2}")
    print("\nOptimizing toward [0, 0] with regularization...")
    
    print(f"\n{'Step':>6} {'L2 Reg weights':>25} {'AdamW weights':>25}")
    print("-" * 60)
    
    for t in range(1, 101):
        # Gradient (toward origin, simple quadratic loss)
        grad = w_l2.copy()
        
        w_l2, m_l2, v_l2 = adam_l2(grad.copy(), w_l2, m_l2, v_l2, t)
        w_wd, m_wd, v_wd = adamw(grad.copy(), w_wd, m_wd, v_wd, t)
        
        if t in [1, 10, 25, 50, 100]:
            l2_str = f"[{w_l2[0]:.4f}, {w_l2[1]:.4f}]"
            wd_str = f"[{w_wd[0]:.4f}, {w_wd[1]:.4f}]"
            print(f"{t:>6} {l2_str:>25} {wd_str:>25}")
    
    print("\nAdamW (decoupled) provides more consistent regularization")


def example_batch_norm_regularization():
    """Batch normalization as implicit regularization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Batch Norm as Regularization")
    print("=" * 60)
    
    np.random.seed(42)
    
    def batch_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
        """Apply batch normalization."""
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, mean, var
    
    # Simulate mini-batches with different statistics
    n_features = 4
    
    print("Batch norm statistics vary across mini-batches:")
    print("This injects noise → regularization effect")
    
    print(f"\n{'Batch':>8} {'Mean':>30} {'Std':>30}")
    print("-" * 70)
    
    for batch_idx in range(5):
        # Different mini-batch
        batch = np.random.randn(32, n_features) * 2 + batch_idx
        
        _, mean, var = batch_norm(batch)
        
        mean_str = "[" + ", ".join(f"{m:.2f}" for m in mean) + "]"
        std_str = "[" + ", ".join(f"{np.sqrt(v):.2f}" for v in var) + "]"
        print(f"{batch_idx:>8} {mean_str:>30} {std_str:>30}")
    
    print("\nDifferent batches → different normalization → noise → regularization")


def example_data_augmentation():
    """Data augmentation as regularization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Data Augmentation")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate image augmentations on feature vectors
    def add_noise(x, std=0.1):
        return x + np.random.randn(*x.shape) * std
    
    def dropout_features(x, p=0.1):
        mask = np.random.binomial(1, 1-p, size=x.shape)
        return x * mask
    
    def mixup(x1, y1, x2, y2, alpha=0.2):
        """Mixup augmentation."""
        lam = np.random.beta(alpha, alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2
        return x_mix, y_mix, lam
    
    # Original sample
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 0.0])  # One-hot label
    
    print("Original sample:")
    print(f"  x = {x}")
    print(f"  y = {y}")
    
    print("\nAugmented versions:")
    
    # Noise
    x_noisy = add_noise(x)
    print(f"  With noise: {x_noisy.round(3)}")
    
    # Feature dropout
    x_dropped = dropout_features(x, p=0.25)
    print(f"  With dropout: {x_dropped}")
    
    # Mixup
    x2 = np.array([5.0, 4.0, 3.0, 2.0])
    y2 = np.array([0.0, 1.0])
    x_mix, y_mix, lam = mixup(x, y, x2, y2)
    print(f"  Mixup (λ={lam:.2f}): x={x_mix.round(3)}, y={y_mix.round(2)}")
    
    print("\nAugmentation increases effective dataset size → regularization")


def example_max_norm_constraint():
    """Max-norm weight constraint."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Max-Norm Constraint")
    print("=" * 60)
    
    def apply_max_norm(w, max_norm):
        """Apply max-norm constraint."""
        norm = np.linalg.norm(w)
        if norm > max_norm:
            w = w * max_norm / norm
        return w
    
    # Simulate weight updates with constraint
    w = np.random.randn(5) * 3
    max_norm = 3.0
    
    print(f"Max norm constraint: ||w||_2 ≤ {max_norm}")
    print(f"\nInitial weights: {w.round(3)}")
    print(f"Initial norm: {np.linalg.norm(w):.3f}")
    
    print("\nSimulating gradient updates with constraint:")
    print(f"{'Step':>6} {'||w|| before':>15} {'||w|| after':>15}")
    print("-" * 40)
    
    for step in range(5):
        # Gradient update (random for simulation)
        grad = np.random.randn(5) * 0.5
        w = w - 0.5 * grad
        
        norm_before = np.linalg.norm(w)
        w = apply_max_norm(w, max_norm)
        norm_after = np.linalg.norm(w)
        
        print(f"{step+1:>6} {norm_before:>15.3f} {norm_after:>15.3f}")
    
    print("\nMax-norm keeps weights bounded, preventing explosion")


def example_spectral_normalization():
    """Spectral normalization for weight matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Spectral Normalization")
    print("=" * 60)
    
    def power_iteration(W, num_iters=10):
        """Estimate largest singular value."""
        u = np.random.randn(W.shape[0])
        u = u / np.linalg.norm(u)
        
        for _ in range(num_iters):
            v = W.T @ u
            v = v / np.linalg.norm(v)
            u = W @ v
            u = u / np.linalg.norm(u)
        
        sigma = u @ W @ v
        return sigma
    
    def spectral_norm(W):
        """Apply spectral normalization."""
        sigma = power_iteration(W)
        return W / sigma
    
    # Random weight matrix
    W = np.random.randn(4, 4) * 2
    
    # True spectral norm (largest singular value)
    true_sigma = np.linalg.svd(W, compute_uv=False)[0]
    est_sigma = power_iteration(W)
    
    print("Spectral normalization: W_SN = W / σ(W)")
    print("where σ(W) is the largest singular value")
    
    print(f"\nOriginal W singular values: {np.linalg.svd(W, compute_uv=False).round(3)}")
    print(f"True spectral norm: {true_sigma:.4f}")
    print(f"Estimated (power iter): {est_sigma:.4f}")
    
    W_normalized = spectral_norm(W)
    new_sigma = np.linalg.svd(W_normalized, compute_uv=False)[0]
    
    print(f"\nAfter normalization:")
    print(f"New singular values: {np.linalg.svd(W_normalized, compute_uv=False).round(3)}")
    print(f"New spectral norm: {new_sigma:.4f} ≈ 1.0")
    
    print("\nUsed in GANs to stabilize discriminator training")


def example_regularization_comparison():
    """Compare different regularization methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Regularization Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n = 100
    X = np.random.randn(n, 8)
    true_w = np.array([2.0, -1.5, 0, 0, 1.0, 0, 0, -0.5])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    # Split
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]
    
    def ols(X, y):
        return np.linalg.lstsq(X, y, rcond=None)[0]
    
    def ridge(X, y, lambda_):
        return np.linalg.solve(X.T @ X + lambda_ * np.eye(X.shape[1]), X.T @ y)
    
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    print("Comparing: OLS, Ridge, Lasso (simulated)")
    print(f"\nTrue weights: {true_w}")
    print(f"Sparsity: {np.sum(true_w == 0)}/8 zeros")
    
    # OLS
    w_ols = ols(X_train, y_train)
    
    # Ridge
    w_ridge = ridge(X_train, y_train, 1.0)
    
    # Lasso (approximate with iterative soft thresholding)
    w_lasso = np.zeros(8)
    for _ in range(1000):
        grad = -X_train.T @ (y_train - X_train @ w_lasso) / len(y_train)
        w_lasso = w_lasso - 0.01 * grad
        w_lasso = np.sign(w_lasso) * np.maximum(np.abs(w_lasso) - 0.01, 0)
    
    print(f"\n{'Method':>10} {'Train MSE':>12} {'Test MSE':>12} {'# Non-zero':>12}")
    print("-" * 50)
    
    for name, w in [('OLS', w_ols), ('Ridge', w_ridge), ('Lasso', w_lasso)]:
        train_mse = mse(y_train, X_train @ w)
        test_mse = mse(y_test, X_test @ w)
        n_nonzero = np.sum(np.abs(w) > 0.01)
        print(f"{name:>10} {train_mse:>12.4f} {test_mse:>12.4f} {n_nonzero:>12}")
    
    print("\nLasso achieves sparsity, Ridge/OLS keep all features")


def example_cross_validation_lambda():
    """Cross-validation for regularization strength."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Cross-Validation for λ")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n = 100
    X = np.random.randn(n, 5)
    true_w = np.array([1.0, 2.0, 0.5, -1.0, 0.3])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    def ridge(X, y, lambda_):
        return np.linalg.solve(X.T @ X + lambda_ * np.eye(X.shape[1]), X.T @ y)
    
    def k_fold_cv(X, y, lambda_, k=5):
        """K-fold cross-validation."""
        n = len(y)
        fold_size = n // k
        scores = []
        
        for i in range(k):
            val_start = i * fold_size
            val_end = val_start + fold_size
            
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            X_train = np.vstack([X[:val_start], X[val_end:]])
            y_train = np.concatenate([y[:val_start], y[val_end:]])
            
            w = ridge(X_train, y_train, lambda_)
            mse = np.mean((y_val - X_val @ w) ** 2)
            scores.append(mse)
        
        return np.mean(scores), np.std(scores)
    
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print("5-fold Cross-Validation for Ridge λ")
    print(f"\n{'λ':>10} {'CV MSE':>12} {'Std':>10}")
    print("-" * 35)
    
    cv_scores = []
    for lambda_ in lambdas:
        mean_mse, std_mse = k_fold_cv(X, y, lambda_)
        cv_scores.append(mean_mse)
        print(f"{lambda_:>10.3f} {mean_mse:>12.4f} {std_mse:>10.4f}")
    
    best_idx = np.argmin(cv_scores)
    best_lambda = lambdas[best_idx]
    
    print(f"\nBest λ = {best_lambda}")
    print("Selected via minimum CV error")


if __name__ == "__main__":
    example_l2_regularization()
    example_l1_regularization()
    example_elastic_net()
    example_dropout_simulation()
    example_early_stopping()
    example_weight_decay_vs_l2()
    example_batch_norm_regularization()
    example_data_augmentation()
    example_max_norm_constraint()
    example_spectral_normalization()
    example_regularization_comparison()
    example_cross_validation_lambda()
