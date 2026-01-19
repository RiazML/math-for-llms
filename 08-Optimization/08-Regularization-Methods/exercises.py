"""
Regularization Methods - Exercises
==================================
Practice problems for mastering regularization techniques.
"""

import numpy as np


def exercise_1_ridge_from_scratch():
    """
    EXERCISE 1: Implement Ridge Regression
    ======================================
    
    Implement Ridge regression with:
    1. Closed-form solution: w = (X'X + λI)^(-1) X'y
    2. Gradient descent solution
    
    Tasks:
    a) Implement closed_form_ridge(X, y, lambda_)
    b) Implement gd_ridge(X, y, lambda_, lr, n_iters)
    c) Compare the two solutions
    
    Expected output:
    - Both methods should give similar weights
    - Larger λ should give smaller weights
    """
    print("=" * 60)
    print("EXERCISE 1: Implement Ridge Regression")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    true_w = np.array([2.0, -1.0, 0.5, 1.5, -0.8])
    y = X @ true_w + 0.3 * np.random.randn(n)
    
    # YOUR CODE HERE
    def closed_form_ridge(X, y, lambda_):
        """Ridge regression closed-form solution."""
        # TODO: Implement w = (X'X + λI)^(-1) X'y
        pass
    
    def gd_ridge(X, y, lambda_, lr=0.01, n_iters=1000):
        """Ridge regression via gradient descent."""
        # TODO: Implement gradient descent
        # Gradient: ∇L = X'(Xw - y) + λw
        pass
    
    # Test your implementation
    # w_closed = closed_form_ridge(X, y, 1.0)
    # w_gd = gd_ridge(X, y, 1.0, lr=0.01, n_iters=1000)
    # print(f"Closed-form: {w_closed.round(3)}")
    # print(f"GD solution: {w_gd.round(3)}")


def exercise_1_solution():
    """Solution for Exercise 1."""
    print("=" * 60)
    print("SOLUTION 1: Ridge Regression")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    true_w = np.array([2.0, -1.0, 0.5, 1.5, -0.8])
    y = X @ true_w + 0.3 * np.random.randn(n)
    
    def closed_form_ridge(X, y, lambda_):
        """Ridge regression closed-form solution."""
        n_features = X.shape[1]
        # w = (X'X + λI)^(-1) X'y
        XtX = X.T @ X
        XtY = X.T @ y
        w = np.linalg.solve(XtX + lambda_ * np.eye(n_features), XtY)
        return w
    
    def gd_ridge(X, y, lambda_, lr=0.01, n_iters=1000):
        """Ridge regression via gradient descent."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        
        for _ in range(n_iters):
            # Gradient: (1/n) * X'(Xw - y) + λw
            pred = X @ w
            grad = (X.T @ (pred - y)) / n_samples + lambda_ * w
            w = w - lr * grad
        
        return w
    
    print("Comparing closed-form vs gradient descent:")
    
    for lambda_ in [0.1, 1.0, 10.0]:
        w_closed = closed_form_ridge(X, y, lambda_)
        w_gd = gd_ridge(X, y, lambda_, lr=0.01, n_iters=2000)
        
        diff = np.linalg.norm(w_closed - w_gd)
        print(f"\nλ = {lambda_}")
        print(f"  Closed-form: {w_closed.round(3)}")
        print(f"  GD:          {w_gd.round(3)}")
        print(f"  Difference:  {diff:.6f}")


def exercise_2_lasso_proximal():
    """
    EXERCISE 2: Implement Lasso with Proximal Gradient
    =================================================
    
    Implement Lasso using proximal gradient descent:
    
    Algorithm (ISTA):
    1. w_temp = w - lr * ∇f(w)  where f(w) = ||y - Xw||²/2
    2. w = soft_threshold(w_temp, lr * λ)
    
    Soft thresholding:
    S_λ(w) = sign(w) * max(|w| - λ, 0)
    
    Tasks:
    a) Implement soft_threshold(w, threshold)
    b) Implement lasso_ista(X, y, lambda_, lr, n_iters)
    c) Verify sparsity increases with λ
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: Implement Lasso (ISTA)")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 10
    X = np.random.randn(n, p)
    true_w = np.array([2.0, 0, 0, -1.5, 0, 0, 1.0, 0, 0, 0])  # Sparse
    y = X @ true_w + 0.2 * np.random.randn(n)
    
    # YOUR CODE HERE
    def soft_threshold(w, threshold):
        """Soft thresholding operator."""
        # TODO: Implement S_λ(w) = sign(w) * max(|w| - λ, 0)
        pass
    
    def lasso_ista(X, y, lambda_, lr=0.01, n_iters=1000):
        """Lasso via ISTA (proximal gradient)."""
        # TODO: Implement ISTA algorithm
        pass
    
    # Test
    # for lambda_ in [0.1, 0.5, 1.0]:
    #     w = lasso_ista(X, y, lambda_)
    #     print(f"λ={lambda_}: {np.sum(np.abs(w) > 0.01)} non-zeros")


def exercise_2_solution():
    """Solution for Exercise 2."""
    print("\n" + "=" * 60)
    print("SOLUTION 2: Lasso (ISTA)")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 10
    X = np.random.randn(n, p)
    true_w = np.array([2.0, 0, 0, -1.5, 0, 0, 1.0, 0, 0, 0])
    y = X @ true_w + 0.2 * np.random.randn(n)
    
    def soft_threshold(w, threshold):
        """Soft thresholding operator."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
    
    def lasso_ista(X, y, lambda_, lr=0.001, n_iters=5000):
        """Lasso via ISTA (proximal gradient)."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        
        for _ in range(n_iters):
            # Gradient step on smooth part
            pred = X @ w
            grad = X.T @ (pred - y) / n_samples
            w_temp = w - lr * grad
            
            # Proximal step (soft thresholding)
            w = soft_threshold(w_temp, lr * lambda_)
        
        return w
    
    print(f"True weights: {true_w}")
    print(f"True non-zeros: indices {np.where(true_w != 0)[0]}")
    
    print(f"\n{'λ':>8} {'Non-zeros':>12} {'Weights':>50}")
    print("-" * 75)
    
    for lambda_ in [0.05, 0.1, 0.2, 0.5]:
        w = lasso_ista(X, y, lambda_)
        n_nonzero = np.sum(np.abs(w) > 0.01)
        w_str = "[" + ", ".join(f"{wi:.2f}" for wi in w) + "]"
        print(f"{lambda_:>8.2f} {n_nonzero:>12} {w_str:>50}")
    
    print("\nIncreasing λ increases sparsity")


def exercise_3_elastic_net():
    """
    EXERCISE 3: Implement Elastic Net
    =================================
    
    Elastic Net combines L1 and L2:
    L = ||y - Xw||² + α||w||₁ + β||w||₂²
    
    Use coordinate descent:
    For each feature j:
    1. Compute partial residual
    2. Apply soft thresholding with L2 modification
    
    Tasks:
    a) Implement elastic_net(X, y, alpha, beta, n_iters)
    b) Compare with pure Lasso (β=0) and pure Ridge (α=0)
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: Implement Elastic Net")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 8
    X = np.random.randn(n, p)
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)  # Correlated
    true_w = np.array([1.5, 1.5, 0, 0, -1.0, 0, 0, 0.5])
    y = X @ true_w + 0.3 * np.random.randn(n)
    
    # YOUR CODE HERE
    def elastic_net(X, y, alpha, beta, n_iters=1000):
        """Elastic net via coordinate descent."""
        # TODO: Implement
        # For each coordinate j:
        # r_j = y - X @ w + X[:,j] * w[j]  (residual without j)
        # rho = X[:,j]' @ r_j
        # w[j] = soft_threshold(rho, alpha) / (X[:,j]'X[:,j] + 2*beta)
        pass


def exercise_3_solution():
    """Solution for Exercise 3."""
    print("\n" + "=" * 60)
    print("SOLUTION 3: Elastic Net")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 8
    X = np.random.randn(n, p)
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n)
    true_w = np.array([1.5, 1.5, 0, 0, -1.0, 0, 0, 0.5])
    y = X @ true_w + 0.3 * np.random.randn(n)
    
    def soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def elastic_net(X, y, alpha, beta, n_iters=1000, tol=1e-6):
        """Elastic net via coordinate descent."""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        
        # Precompute X'X diagonal
        X_col_norms = np.sum(X ** 2, axis=0)
        
        for _ in range(n_iters):
            w_old = w.copy()
            
            for j in range(n_features):
                # Partial residual
                r_j = y - X @ w + X[:, j] * w[j]
                rho = X[:, j] @ r_j
                
                # Coordinate update
                w[j] = soft_threshold(rho, alpha * n_samples) / (X_col_norms[j] + 2 * beta * n_samples)
            
            if np.linalg.norm(w - w_old) < tol:
                break
        
        return w
    
    print(f"True weights: {true_w}")
    print("(Features 0,1 are correlated)")
    
    print(f"\n{'Method':>20} {'Weights':>55}")
    print("-" * 80)
    
    # Pure Ridge
    w_ridge = elastic_net(X, y, alpha=0, beta=0.5)
    print(f"{'Ridge (α=0, β=0.5)':>20} {str(w_ridge.round(2)):>55}")
    
    # Pure Lasso
    w_lasso = elastic_net(X, y, alpha=0.1, beta=0)
    print(f"{'Lasso (α=0.1, β=0)':>20} {str(w_lasso.round(2)):>55}")
    
    # Elastic Net
    w_enet = elastic_net(X, y, alpha=0.05, beta=0.25)
    print(f"{'ElasticNet':>20} {str(w_enet.round(2)):>55}")
    
    print("\nElastic Net balances sparsity (L1) and grouping (L2)")


def exercise_4_dropout():
    """
    EXERCISE 4: Implement Dropout Layer
    ===================================
    
    Implement dropout with:
    - Training mode: randomly zero out with probability p
    - Inference mode: no dropout, but scale appropriately
    
    Two common implementations:
    a) Standard: multiply by (1-p) at test time
    b) Inverted: divide by (1-p) at train time
    
    Tasks:
    a) Implement Dropout class with forward(x, training)
    b) Verify expected value is preserved
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: Implement Dropout")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    class Dropout:
        def __init__(self, p=0.5, inverted=True):
            """
            Args:
                p: probability of dropping a unit
                inverted: if True, scale during training (recommended)
            """
            self.p = p
            self.inverted = inverted
            self.mask = None
        
        def forward(self, x, training=True):
            """Forward pass."""
            # TODO: Implement
            # If training:
            #   - Create binary mask with P(1) = 1-p
            #   - If inverted: return x * mask / (1-p)
            #   - If not inverted: return x * mask
            # If not training:
            #   - If inverted: return x
            #   - If not inverted: return x * (1-p)
            pass
    
    # Test
    # dropout = Dropout(p=0.5)
    # x = np.ones((1, 10))
    # print("Training passes:", [dropout.forward(x, True).sum() for _ in range(5)])
    # print("Inference:", dropout.forward(x, False).sum())


def exercise_4_solution():
    """Solution for Exercise 4."""
    print("\n" + "=" * 60)
    print("SOLUTION 4: Dropout")
    print("=" * 60)
    
    class Dropout:
        def __init__(self, p=0.5, inverted=True):
            self.p = p
            self.inverted = inverted
            self.mask = None
        
        def forward(self, x, training=True):
            if not training:
                if self.inverted:
                    return x
                else:
                    return x * (1 - self.p)
            
            # Training mode
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            
            if self.inverted:
                return x * self.mask / (1 - self.p)
            else:
                return x * self.mask
    
    # Test
    np.random.seed(42)
    
    dropout = Dropout(p=0.5, inverted=True)
    x = np.ones((1, 100))
    
    print("Inverted Dropout (scale at train time):")
    print(f"Input sum: {x.sum()}")
    
    train_outputs = [dropout.forward(x.copy(), True).sum() for _ in range(10)]
    print(f"Training outputs (sum): {[f'{v:.1f}' for v in train_outputs]}")
    print(f"Training mean: {np.mean(train_outputs):.2f} (expect ~100)")
    
    test_output = dropout.forward(x.copy(), False).sum()
    print(f"Inference output: {test_output:.1f}")
    
    # Non-inverted version
    print("\nStandard Dropout (scale at test time):")
    dropout2 = Dropout(p=0.5, inverted=False)
    
    train_outputs2 = [dropout2.forward(x.copy(), True).sum() for _ in range(10)]
    print(f"Training mean: {np.mean(train_outputs2):.2f} (expect ~50)")
    
    test_output2 = dropout2.forward(x.copy(), False).sum()
    print(f"Inference output: {test_output2:.1f} (scaled by 0.5)")


def exercise_5_early_stopping():
    """
    EXERCISE 5: Implement Early Stopping
    ====================================
    
    Implement training with early stopping:
    - Track validation loss
    - Stop when validation loss doesn't improve for `patience` epochs
    - Return best weights
    
    Tasks:
    a) Implement early_stopping_train(...)
    b) Return best weights, not final weights
    c) Plot or report train/val curves
    """
    print("\n" + "=" * 60)
    print("EXERCISE 5: Early Stopping")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate polynomial regression data
    n = 80
    x = np.random.rand(n)
    y = np.sin(2 * np.pi * x) + 0.3 * np.random.randn(n)
    
    # Train/val split
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]
    
    # High-degree polynomial features
    def poly_features(x, degree):
        return np.column_stack([x**i for i in range(degree + 1)])
    
    degree = 15
    X_train = poly_features(x_train, degree)
    X_val = poly_features(x_val, degree)
    
    # YOUR CODE HERE
    def early_stopping_train(X_train, y_train, X_val, y_val,
                             lr=0.001, max_epochs=2000, patience=100):
        """
        Train with early stopping.
        
        Returns:
            best_w: weights at best validation loss
            best_epoch: epoch of best validation loss
            history: dict with 'train_loss' and 'val_loss' lists
        """
        # TODO: Implement
        pass


def exercise_5_solution():
    """Solution for Exercise 5."""
    print("\n" + "=" * 60)
    print("SOLUTION 5: Early Stopping")
    print("=" * 60)
    
    np.random.seed(42)
    
    n = 80
    x = np.random.rand(n)
    y = np.sin(2 * np.pi * x) + 0.3 * np.random.randn(n)
    
    x_train, x_val = x[:60], x[60:]
    y_train, y_val = y[:60], y[60:]
    
    def poly_features(x, degree):
        return np.column_stack([x**i for i in range(degree + 1)])
    
    degree = 15
    X_train = poly_features(x_train, degree)
    X_val = poly_features(x_val, degree)
    
    def early_stopping_train(X_train, y_train, X_val, y_val,
                             lr=0.0001, max_epochs=3000, patience=200):
        n_features = X_train.shape[1]
        w = np.zeros(n_features)
        
        best_val_loss = float('inf')
        best_w = w.copy()
        best_epoch = 0
        patience_counter = 0
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(max_epochs):
            # Forward and loss
            pred_train = X_train @ w
            train_loss = np.mean((pred_train - y_train) ** 2)
            
            pred_val = X_val @ w
            val_loss = np.mean((pred_val - y_val) ** 2)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Update
            grad = 2 * X_train.T @ (pred_train - y_train) / len(y_train)
            w = w - lr * grad
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_w = w.copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Stopped at epoch {epoch}")
                break
        
        return best_w, best_epoch, history
    
    best_w, best_epoch, history = early_stopping_train(
        X_train, y_train, X_val, y_val
    )
    
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    print(f"\n{'Epoch':>8} {'Train Loss':>12} {'Val Loss':>12}")
    print("-" * 35)
    epochs_to_show = [0, 100, 500, best_epoch, len(history['train_loss'])-1]
    for e in epochs_to_show:
        if e < len(history['train_loss']):
            marker = " *" if e == best_epoch else ""
            print(f"{e:>8} {history['train_loss'][e]:>12.4f} {history['val_loss'][e]:>12.4f}{marker}")


def exercise_6_batch_norm():
    """
    EXERCISE 6: Implement Batch Normalization
    =========================================
    
    Implement batch norm with:
    - Training: use batch statistics
    - Inference: use running statistics
    
    Forward pass:
    x_hat = (x - μ) / √(σ² + ε)
    y = γ * x_hat + β
    
    Tasks:
    a) Implement BatchNorm class
    b) Track running mean and variance
    c) Verify output has mean≈0, std≈1 during training
    """
    print("\n" + "=" * 60)
    print("EXERCISE 6: Batch Normalization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    class BatchNorm:
        def __init__(self, n_features, momentum=0.1, eps=1e-5):
            """
            Args:
                n_features: number of features
                momentum: for running statistics update
                eps: small constant for numerical stability
            """
            self.gamma = np.ones(n_features)
            self.beta = np.zeros(n_features)
            self.eps = eps
            self.momentum = momentum
            
            # Running statistics
            self.running_mean = np.zeros(n_features)
            self.running_var = np.ones(n_features)
            
            # Cache for backward pass
            self.cache = None
        
        def forward(self, x, training=True):
            """Forward pass."""
            # TODO: Implement
            # If training:
            #   - Compute batch mean and variance
            #   - Update running statistics
            #   - Normalize using batch statistics
            # If not training:
            #   - Normalize using running statistics
            pass


def exercise_6_solution():
    """Solution for Exercise 6."""
    print("\n" + "=" * 60)
    print("SOLUTION 6: Batch Normalization")
    print("=" * 60)
    
    class BatchNorm:
        def __init__(self, n_features, momentum=0.1, eps=1e-5):
            self.gamma = np.ones(n_features)
            self.beta = np.zeros(n_features)
            self.eps = eps
            self.momentum = momentum
            
            self.running_mean = np.zeros(n_features)
            self.running_var = np.ones(n_features)
        
        def forward(self, x, training=True):
            if training:
                # Batch statistics
                batch_mean = np.mean(x, axis=0)
                batch_var = np.var(x, axis=0)
                
                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                
                # Normalize
                x_hat = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            else:
                # Use running statistics
                x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
            return self.gamma * x_hat + self.beta
    
    np.random.seed(42)
    
    # Test
    bn = BatchNorm(4)
    
    print("Training mode statistics:")
    for batch_idx in range(5):
        x = np.random.randn(32, 4) * 3 + 2  # Non-standard input
        out = bn.forward(x, training=True)
        print(f"Batch {batch_idx}: mean={out.mean(axis=0).round(3)}, std={out.std(axis=0).round(3)}")
    
    print(f"\nRunning mean: {bn.running_mean.round(3)}")
    print(f"Running var:  {bn.running_var.round(3)}")
    
    print("\nInference mode:")
    x_test = np.random.randn(10, 4) * 3 + 2
    out_test = bn.forward(x_test, training=False)
    print(f"Output mean: {out_test.mean(axis=0).round(3)}")
    print(f"Output std:  {out_test.std(axis=0).round(3)}")


def exercise_7_mixup():
    """
    EXERCISE 7: Implement Mixup Augmentation
    =======================================
    
    Mixup creates virtual training examples:
    x̃ = λx_i + (1-λ)x_j
    ỹ = λy_i + (1-λ)y_j
    
    where λ ~ Beta(α, α)
    
    Tasks:
    a) Implement mixup_batch(x, y, alpha)
    b) Return mixed inputs, mixed targets, and lambda
    c) Test with one-hot encoded labels
    """
    print("\n" + "=" * 60)
    print("EXERCISE 7: Mixup Augmentation")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    def mixup_batch(x, y, alpha=0.2):
        """
        Apply mixup to a batch.
        
        Args:
            x: input batch (batch_size, features)
            y: labels (batch_size,) or (batch_size, n_classes)
            alpha: Beta distribution parameter
        
        Returns:
            x_mixed, y_mixed, lambda_values
        """
        # TODO: Implement
        # 1. Sample λ ~ Beta(α, α) for each sample
        # 2. Shuffle indices
        # 3. Mix: x̃ = λx + (1-λ)x[shuffled]
        pass


def exercise_7_solution():
    """Solution for Exercise 7."""
    print("\n" + "=" * 60)
    print("SOLUTION 7: Mixup Augmentation")
    print("=" * 60)
    
    def mixup_batch(x, y, alpha=0.2):
        batch_size = x.shape[0]
        
        # Sample lambda from Beta distribution
        if alpha > 0:
            lam = np.random.beta(alpha, alpha, size=batch_size)
        else:
            lam = np.ones(batch_size)
        
        # Reshape for broadcasting
        lam_x = lam.reshape(-1, 1)
        
        # Random permutation for mixing partners
        indices = np.random.permutation(batch_size)
        
        # Mix
        x_mixed = lam_x * x + (1 - lam_x) * x[indices]
        
        # Handle different y shapes
        if y.ndim == 1:
            y_mixed = lam * y + (1 - lam) * y[indices]
        else:
            lam_y = lam.reshape(-1, 1)
            y_mixed = lam_y * y + (1 - lam_y) * y[indices]
        
        return x_mixed, y_mixed, lam
    
    np.random.seed(42)
    
    # Test with one-hot labels
    batch_size = 4
    n_features = 5
    n_classes = 3
    
    x = np.random.randn(batch_size, n_features)
    y_onehot = np.eye(n_classes)[np.random.randint(0, n_classes, batch_size)]
    
    print("Original data:")
    print(f"x[0] = {x[0].round(2)}")
    print(f"y[0] = {y_onehot[0]}")
    
    x_mixed, y_mixed, lam = mixup_batch(x, y_onehot, alpha=0.4)
    
    print(f"\nMixed data (λ={lam[0]:.3f}):")
    print(f"x_mixed[0] = {x_mixed[0].round(2)}")
    print(f"y_mixed[0] = {y_mixed[0].round(2)}")
    
    print("\nMixup creates soft labels between classes")


def exercise_8_regularization_path():
    """
    EXERCISE 8: Compute Regularization Path
    ======================================
    
    Compute how weights change as regularization strength varies.
    
    For Ridge:
    - Solve for many λ values
    - Plot weight trajectories
    
    Tasks:
    a) Implement regularization_path(X, y, lambdas)
    b) Identify which weights shrink fastest
    c) Find optimal λ via cross-validation
    """
    print("\n" + "=" * 60)
    print("EXERCISE 8: Regularization Path")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 6
    X = np.random.randn(n, p)
    true_w = np.array([3.0, -2.0, 1.0, 0.5, 0.1, 0.01])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    # YOUR CODE HERE
    def ridge_path(X, y, lambdas):
        """
        Compute Ridge solutions for multiple λ values.
        
        Returns:
            weights: array of shape (len(lambdas), n_features)
        """
        # TODO: Implement
        pass


def exercise_8_solution():
    """Solution for Exercise 8."""
    print("\n" + "=" * 60)
    print("SOLUTION 8: Regularization Path")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 6
    X = np.random.randn(n, p)
    true_w = np.array([3.0, -2.0, 1.0, 0.5, 0.1, 0.01])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    def ridge_path(X, y, lambdas):
        n_features = X.shape[1]
        weights = np.zeros((len(lambdas), n_features))
        
        XtX = X.T @ X
        XtY = X.T @ y
        
        for i, lam in enumerate(lambdas):
            weights[i] = np.linalg.solve(XtX + lam * np.eye(n_features), XtY)
        
        return weights
    
    lambdas = np.logspace(-4, 4, 50)
    weights = ridge_path(X, y, lambdas)
    
    print(f"True weights: {true_w}")
    print("\nRegularization path (weights vs λ):")
    print(f"\n{'λ':>10}" + "".join(f"{'w'+str(i):>10}" for i in range(6)))
    print("-" * 70)
    
    for i in [0, 10, 20, 30, 40, 49]:
        lam = lambdas[i]
        w = weights[i]
        print(f"{lam:>10.4f}" + "".join(f"{w[j]:>10.3f}" for j in range(6)))
    
    print("\nSmaller true weights shrink faster")
    print(f"At λ=10000: w[5] is {weights[-1, 5]/true_w[5]*100:.1f}% of true")
    print(f"At λ=10000: w[0] is {weights[-1, 0]/true_w[0]*100:.1f}% of true")


def exercise_9_compare_regularizers():
    """
    EXERCISE 9: Compare Regularization Effects
    ==========================================
    
    Compare on noisy data:
    - No regularization
    - L2 regularization
    - L1 regularization
    - Elastic Net
    
    Measure:
    - Training error
    - Test error  
    - Sparsity
    - Weight norm
    
    Tasks:
    a) Train all models
    b) Create comparison table
    c) Identify best for different metrics
    """
    print("\n" + "=" * 60)
    print("EXERCISE 9: Compare Regularizers")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n_train, n_test = 80, 20
    p = 12
    
    X = np.random.randn(n_train + n_test, p)
    true_w = np.array([2.0, -1.5, 0, 0, 1.0, 0, 0, 0, -0.8, 0, 0, 0.3])
    y = X @ true_w + 0.5 * np.random.randn(n_train + n_test)
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # YOUR CODE HERE
    # Implement training with different regularizers
    # Compare results


def exercise_9_solution():
    """Solution for Exercise 9."""
    print("\n" + "=" * 60)
    print("SOLUTION 9: Compare Regularizers")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_train, n_test = 80, 20
    p = 12
    
    X = np.random.randn(n_train + n_test, p)
    true_w = np.array([2.0, -1.5, 0, 0, 1.0, 0, 0, 0, -0.8, 0, 0, 0.3])
    y = X @ true_w + 0.5 * np.random.randn(n_train + n_test)
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def ols(X, y):
        return np.linalg.lstsq(X, y, rcond=None)[0]
    
    def ridge(X, y, lam):
        return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    
    def soft_threshold(w, t):
        return np.sign(w) * np.maximum(np.abs(w) - t, 0)
    
    def lasso(X, y, lam, n_iter=5000, lr=0.001):
        w = np.zeros(X.shape[1])
        for _ in range(n_iter):
            grad = X.T @ (X @ w - y) / len(y)
            w = w - lr * grad
            w = soft_threshold(w, lr * lam)
        return w
    
    def elastic_net(X, y, alpha, beta, n_iter=5000, lr=0.001):
        w = np.zeros(X.shape[1])
        for _ in range(n_iter):
            grad = X.T @ (X @ w - y) / len(y) + 2 * beta * w
            w = w - lr * grad
            w = soft_threshold(w, lr * alpha)
        return w
    
    results = []
    
    # OLS
    w_ols = ols(X_train, y_train)
    results.append(('OLS', w_ols, mse(y_train, X_train @ w_ols),
                   mse(y_test, X_test @ w_ols),
                   np.sum(np.abs(w_ols) > 0.01),
                   np.linalg.norm(w_ols)))
    
    # Ridge
    w_ridge = ridge(X_train, y_train, 1.0)
    results.append(('Ridge', w_ridge, mse(y_train, X_train @ w_ridge),
                   mse(y_test, X_test @ w_ridge),
                   np.sum(np.abs(w_ridge) > 0.01),
                   np.linalg.norm(w_ridge)))
    
    # Lasso
    w_lasso = lasso(X_train, y_train, 0.1)
    results.append(('Lasso', w_lasso, mse(y_train, X_train @ w_lasso),
                   mse(y_test, X_test @ w_lasso),
                   np.sum(np.abs(w_lasso) > 0.01),
                   np.linalg.norm(w_lasso)))
    
    # Elastic Net
    w_enet = elastic_net(X_train, y_train, 0.05, 0.5)
    results.append(('ElasticNet', w_enet, mse(y_train, X_train @ w_enet),
                   mse(y_test, X_test @ w_enet),
                   np.sum(np.abs(w_enet) > 0.01),
                   np.linalg.norm(w_enet)))
    
    print(f"True sparsity: {np.sum(true_w != 0)}/12 non-zero")
    
    print(f"\n{'Method':>12} {'Train MSE':>12} {'Test MSE':>12} {'Non-zero':>10} {'||w||':>10}")
    print("-" * 60)
    
    for name, w, train_mse, test_mse, n_nonzero, w_norm in results:
        print(f"{name:>12} {train_mse:>12.4f} {test_mse:>12.4f} {n_nonzero:>10} {w_norm:>10.3f}")
    
    print("\nLasso achieves sparsity, Ridge has lowest test error")


def exercise_10_cross_validation():
    """
    EXERCISE 10: Implement K-Fold Cross-Validation
    ==============================================
    
    Implement cross-validation to select optimal λ:
    1. Split data into k folds
    2. For each λ, compute average validation error
    3. Select λ with lowest CV error
    
    Tasks:
    a) Implement k_fold_cv(X, y, lambdas, k)
    b) Return best λ and CV scores
    c) Compare with 1-standard-error rule
    """
    print("\n" + "=" * 60)
    print("EXERCISE 10: K-Fold Cross-Validation")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    true_w = np.array([1.0, 2.0, 0.5, -1.0, 0.3])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    # YOUR CODE HERE
    def k_fold_cv(X, y, lambdas, k=5):
        """
        K-fold cross-validation for Ridge regression.
        
        Returns:
            best_lambda: λ with lowest CV error
            cv_scores: mean CV error for each λ
            cv_stds: std of CV error for each λ
        """
        # TODO: Implement
        pass


def exercise_10_solution():
    """Solution for Exercise 10."""
    print("\n" + "=" * 60)
    print("SOLUTION 10: K-Fold Cross-Validation")
    print("=" * 60)
    
    np.random.seed(42)
    n, p = 100, 5
    X = np.random.randn(n, p)
    true_w = np.array([1.0, 2.0, 0.5, -1.0, 0.3])
    y = X @ true_w + 0.5 * np.random.randn(n)
    
    def ridge(X, y, lam):
        return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    
    def k_fold_cv(X, y, lambdas, k=5):
        n = len(y)
        fold_size = n // k
        
        cv_scores = []
        cv_stds = []
        
        for lam in lambdas:
            fold_errors = []
            
            for i in range(k):
                # Create fold
                val_start = i * fold_size
                val_end = val_start + fold_size
                
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate([np.arange(val_start), np.arange(val_end, n)])
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train and evaluate
                w = ridge(X_train, y_train, lam)
                mse = np.mean((y_val - X_val @ w) ** 2)
                fold_errors.append(mse)
            
            cv_scores.append(np.mean(fold_errors))
            cv_stds.append(np.std(fold_errors))
        
        best_idx = np.argmin(cv_scores)
        best_lambda = lambdas[best_idx]
        
        # 1-standard-error rule
        threshold = cv_scores[best_idx] + cv_stds[best_idx]
        lambda_1se_idx = np.where(np.array(cv_scores) <= threshold)[0][-1]
        lambda_1se = lambdas[lambda_1se_idx]
        
        return best_lambda, lambda_1se, cv_scores, cv_stds
    
    lambdas = np.logspace(-3, 3, 20)
    best_lam, lam_1se, scores, stds = k_fold_cv(X, y, lambdas)
    
    print(f"{'λ':>12} {'CV MSE':>12} {'Std':>12}")
    print("-" * 40)
    
    for i in range(0, len(lambdas), 3):
        marker = ""
        if lambdas[i] == best_lam:
            marker = " <- best"
        elif lambdas[i] == lam_1se:
            marker = " <- 1SE"
        print(f"{lambdas[i]:>12.4f} {scores[i]:>12.4f} {stds[i]:>12.4f}{marker}")
    
    print(f"\nBest λ (min CV): {best_lam:.4f}")
    print(f"λ (1-SE rule):  {lam_1se:.4f}")
    print("\n1-SE rule selects simpler model within 1 std of best")


def run_all_exercises():
    """Run all exercise solutions."""
    exercise_1_solution()
    exercise_2_solution()
    exercise_3_solution()
    exercise_4_solution()
    exercise_5_solution()
    exercise_6_solution()
    exercise_7_solution()
    exercise_8_solution()
    exercise_9_solution()
    exercise_10_solution()


if __name__ == "__main__":
    run_all_exercises()
