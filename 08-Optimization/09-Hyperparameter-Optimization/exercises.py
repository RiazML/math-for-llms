"""
Hyperparameter Optimization - Exercises
=======================================
Practice problems for mastering hyperparameter tuning.
"""

import numpy as np
from scipy import stats


def exercise_1_grid_search():
    """
    EXERCISE 1: Implement Grid Search
    =================================
    
    Implement a complete grid search with:
    1. Parameter grid specification
    2. Cross-validation evaluation
    3. Best parameter selection
    
    Tasks:
    a) Implement grid_search(param_grid, objective, cv_folds)
    b) Return best parameters and all results
    c) Handle mixed parameter types
    """
    print("=" * 60)
    print("EXERCISE 1: Implement Grid Search")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulated model evaluation
    def evaluate_model(params, X, y):
        """Simulate model evaluation (return error)."""
        lr, hidden = params['lr'], params['hidden']
        
        # Optimal around lr=0.01, hidden=64
        error = (np.log10(lr) + 2) ** 2 + ((hidden - 64) / 32) ** 2
        error += np.random.randn() * 0.1
        return max(0, error)
    
    # Generate dummy data
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'hidden': [32, 64, 128]
    }
    
    # YOUR CODE HERE
    def grid_search(param_grid, X, y, n_folds=5):
        """
        Perform grid search with cross-validation.
        
        Returns:
            best_params: dict of best parameters
            results: list of (params, mean_score, std_score)
        """
        # TODO: Implement
        pass


def exercise_1_solution():
    """Solution for Exercise 1."""
    print("=" * 60)
    print("SOLUTION 1: Grid Search")
    print("=" * 60)
    
    np.random.seed(42)
    
    def evaluate_model(params, X, y):
        lr, hidden = params['lr'], params['hidden']
        error = (np.log10(lr) + 2) ** 2 + ((hidden - 64) / 32) ** 2
        error += np.random.randn() * 0.1
        return max(0, error)
    
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'hidden': [32, 64, 128]
    }
    
    def grid_search(param_grid, X, y, n_folds=5):
        from itertools import product
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        results = []
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            # Cross-validation
            fold_size = len(X) // n_folds
            scores = []
            
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size
                
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                
                score = evaluate_model(params, X_val, y_val)
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results.append((params, mean_score, std_score))
        
        # Find best
        best = min(results, key=lambda x: x[1])
        
        return best[0], results
    
    best_params, results = grid_search(param_grid, X, y)
    
    print("Grid Search Results:")
    print(f"{'lr':>8} {'hidden':>8} {'Mean':>10} {'Std':>10}")
    print("-" * 40)
    
    for params, mean, std in sorted(results, key=lambda x: x[1]):
        print(f"{params['lr']:>8.3f} {params['hidden']:>8} {mean:>10.4f} {std:>10.4f}")
    
    print(f"\nBest: {best_params}")


def exercise_2_random_search():
    """
    EXERCISE 2: Implement Random Search
    ===================================
    
    Implement random search with:
    1. Support for different distributions
    2. Log-uniform sampling for learning rates
    3. Comparison with grid search efficiency
    
    Tasks:
    a) Implement random_search(search_space, n_trials, objective)
    b) Support uniform, log-uniform, and categorical distributions
    c) Compare samples per budget with grid search
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: Implement Random Search")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Define search space
    search_space = {
        'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
        'dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
        'hidden': {'type': 'categorical', 'choices': [32, 64, 128, 256]}
    }
    
    # YOUR CODE HERE
    def sample_from_space(space):
        """Sample one configuration from search space."""
        # TODO: Handle different distribution types
        pass
    
    def random_search(search_space, n_trials, objective):
        """
        Perform random search.
        
        Returns:
            best_params: dict of best parameters
            history: list of (params, score)
        """
        # TODO: Implement
        pass


def exercise_2_solution():
    """Solution for Exercise 2."""
    print("\n" + "=" * 60)
    print("SOLUTION 2: Random Search")
    print("=" * 60)
    
    np.random.seed(42)
    
    search_space = {
        'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
        'dropout': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
        'hidden': {'type': 'categorical', 'choices': [32, 64, 128, 256]}
    }
    
    def sample_from_space(space):
        config = {}
        for param, spec in space.items():
            if spec['type'] == 'uniform':
                config[param] = np.random.uniform(spec['low'], spec['high'])
            elif spec['type'] == 'log_uniform':
                log_low = np.log10(spec['low'])
                log_high = np.log10(spec['high'])
                config[param] = 10 ** np.random.uniform(log_low, log_high)
            elif spec['type'] == 'categorical':
                config[param] = np.random.choice(spec['choices'])
        return config
    
    def objective(params):
        lr, dropout, hidden = params['lr'], params['dropout'], params['hidden']
        error = (np.log10(lr) + 2) ** 2 + (dropout - 0.2) ** 2 + ((hidden - 64) / 64) ** 2
        error += np.random.randn() * 0.1
        return max(0, error)
    
    def random_search(search_space, n_trials, objective):
        history = []
        
        for _ in range(n_trials):
            params = sample_from_space(search_space)
            score = objective(params)
            history.append((params, score))
        
        best = min(history, key=lambda x: x[1])
        return best[0], history
    
    best_params, history = random_search(search_space, 20, objective)
    
    print("Random Search Results (sorted by score):")
    print(f"{'lr':>10} {'dropout':>10} {'hidden':>8} {'Score':>10}")
    print("-" * 45)
    
    for params, score in sorted(history, key=lambda x: x[1])[:10]:
        print(f"{params['lr']:>10.5f} {params['dropout']:>10.3f} {params['hidden']:>8} {score:>10.4f}")
    
    print(f"\nBest: lr={best_params['lr']:.5f}, dropout={best_params['dropout']:.3f}, hidden={best_params['hidden']}")


def exercise_3_gaussian_process():
    """
    EXERCISE 3: Implement Simple GP
    ===============================
    
    Implement a Gaussian Process for 1D regression:
    1. RBF kernel
    2. Fit to observations
    3. Predict mean and variance
    
    Tasks:
    a) Implement rbf_kernel(X1, X2, length_scale, variance)
    b) Implement GP.fit(X, y) and GP.predict(X_new)
    c) Visualize uncertainty
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: Gaussian Process")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    class GaussianProcess:
        def __init__(self, length_scale=1.0, variance=1.0, noise=1e-6):
            self.length_scale = length_scale
            self.variance = variance
            self.noise = noise
        
        def rbf_kernel(self, X1, X2):
            """RBF kernel: k(x, x') = σ² exp(-||x-x'||² / 2ℓ²)"""
            # TODO: Implement
            pass
        
        def fit(self, X, y):
            """Fit GP to data."""
            # TODO: Store data and compute K inverse
            pass
        
        def predict(self, X_new):
            """Predict mean and std at new points."""
            # TODO: Implement
            pass


def exercise_3_solution():
    """Solution for Exercise 3."""
    print("\n" + "=" * 60)
    print("SOLUTION 3: Gaussian Process")
    print("=" * 60)
    
    np.random.seed(42)
    
    class GaussianProcess:
        def __init__(self, length_scale=1.0, variance=1.0, noise=1e-6):
            self.length_scale = length_scale
            self.variance = variance
            self.noise = noise
            self.X_train = None
            self.y_train = None
            self.K_inv = None
        
        def rbf_kernel(self, X1, X2):
            X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
            X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
            
            dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return self.variance * np.exp(-dists / (2 * self.length_scale**2))
        
        def fit(self, X, y):
            self.X_train = np.atleast_1d(X)
            self.y_train = np.atleast_1d(y)
            
            K = self.rbf_kernel(self.X_train, self.X_train)
            self.K_inv = np.linalg.inv(K + self.noise * np.eye(len(X)))
        
        def predict(self, X_new):
            X_new = np.atleast_1d(X_new)
            
            K_star = self.rbf_kernel(X_new, self.X_train)
            K_star_star = self.rbf_kernel(X_new, X_new)
            
            mu = K_star @ self.K_inv @ self.y_train
            cov = K_star_star - K_star @ self.K_inv @ K_star.T
            std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            
            return mu, std
    
    # Test
    X_train = np.array([0.1, 0.3, 0.7, 0.9])
    y_train = np.sin(3 * X_train)
    
    gp = GaussianProcess(length_scale=0.3, variance=1.0)
    gp.fit(X_train, y_train)
    
    X_test = np.linspace(0, 1, 11)
    mu, std = gp.predict(X_test)
    
    print("GP Predictions:")
    print(f"Training points: {X_train}")
    
    print(f"\n{'x':>8} {'True':>10} {'μ':>10} {'σ':>10}")
    print("-" * 42)
    
    for x, m, s in zip(X_test, mu, std):
        true_val = np.sin(3 * x)
        print(f"{x:>8.2f} {true_val:>10.4f} {m:>10.4f} {s:>10.4f}")


def exercise_4_acquisition_functions():
    """
    EXERCISE 4: Implement Acquisition Functions
    ===========================================
    
    Implement common acquisition functions:
    1. Expected Improvement (EI)
    2. Upper Confidence Bound (UCB)
    3. Probability of Improvement (PI)
    
    Tasks:
    a) Implement each acquisition function
    b) Compare their behavior
    c) Identify exploration vs exploitation trade-offs
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: Acquisition Functions")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Given GP predictions
    X = np.linspace(0, 1, 21)
    mu = 0.5 - 0.3 * np.sin(5 * X)
    sigma = 0.1 + 0.15 * np.abs(X - 0.5)
    y_best = 0.3
    
    # YOUR CODE HERE
    def expected_improvement(mu, sigma, y_best, xi=0.01):
        """Expected Improvement."""
        # TODO: EI(x) = (y* - μ - ξ)Φ(Z) + σφ(Z)
        pass
    
    def upper_confidence_bound(mu, sigma, kappa=2.0):
        """Upper Confidence Bound (for minimization)."""
        # TODO: UCB(x) = -μ + κσ  (note: negative for minimization)
        pass
    
    def probability_of_improvement(mu, sigma, y_best, xi=0.01):
        """Probability of Improvement."""
        # TODO: PI(x) = Φ((y* - μ - ξ) / σ)
        pass


def exercise_4_solution():
    """Solution for Exercise 4."""
    print("\n" + "=" * 60)
    print("SOLUTION 4: Acquisition Functions")
    print("=" * 60)
    
    X = np.linspace(0, 1, 21)
    mu = 0.5 - 0.3 * np.sin(5 * X)
    sigma = 0.1 + 0.15 * np.abs(X - 0.5)
    y_best = 0.3
    
    def expected_improvement(mu, sigma, y_best, xi=0.01):
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mu - xi
            Z = imp / sigma
            ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            ei[sigma <= 0] = 0
        return ei
    
    def upper_confidence_bound(mu, sigma, kappa=2.0):
        # For minimization: lower is better, so we want low mu + exploration bonus
        return -mu + kappa * sigma
    
    def probability_of_improvement(mu, sigma, y_best, xi=0.01):
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (y_best - mu - xi) / sigma
            pi = stats.norm.cdf(Z)
            pi[sigma <= 0] = 0
        return pi
    
    ei = expected_improvement(mu, sigma, y_best)
    ucb = upper_confidence_bound(mu, sigma)
    pi = probability_of_improvement(mu, sigma, y_best)
    
    print("Acquisition Function Comparison:")
    print(f"{'x':>6} {'μ':>8} {'σ':>8} {'EI':>10} {'UCB':>10} {'PI':>10}")
    print("-" * 55)
    
    for i in range(0, len(X), 2):
        print(f"{X[i]:>6.2f} {mu[i]:>8.3f} {sigma[i]:>8.3f} {ei[i]:>10.4f} {ucb[i]:>10.4f} {pi[i]:>10.4f}")
    
    print(f"\nNext point suggestions:")
    print(f"  EI:  x = {X[np.argmax(ei)]:.2f}")
    print(f"  UCB: x = {X[np.argmax(ucb)]:.2f}")
    print(f"  PI:  x = {X[np.argmax(pi)]:.2f}")


def exercise_5_bayesian_optimization():
    """
    EXERCISE 5: Complete Bayesian Optimization
    ==========================================
    
    Implement a complete BO loop:
    1. Initialize with random samples
    2. Fit GP surrogate
    3. Optimize acquisition function
    4. Iterate
    
    Tasks:
    a) Implement BayesianOptimizer class
    b) Run optimization loop
    c) Track convergence
    """
    print("\n" + "=" * 60)
    print("EXERCISE 5: Bayesian Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x):
        """Objective to minimize."""
        return (x - 0.35) ** 2 + 0.1 * np.sin(15 * x)
    
    # YOUR CODE HERE
    class BayesianOptimizer:
        def __init__(self, bounds, length_scale=0.2):
            self.bounds = bounds
            self.length_scale = length_scale
            self.X_obs = []
            self.y_obs = []
        
        def suggest(self):
            """Suggest next point to evaluate."""
            # TODO: Use GP + EI
            pass
        
        def observe(self, x, y):
            """Record observation."""
            # TODO: Implement
            pass
        
        def optimize(self, objective, n_init=3, n_iter=10):
            """Run full optimization."""
            # TODO: Implement
            pass


def exercise_5_solution():
    """Solution for Exercise 5."""
    print("\n" + "=" * 60)
    print("SOLUTION 5: Bayesian Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x):
        return (x - 0.35) ** 2 + 0.1 * np.sin(15 * x)
    
    class BayesianOptimizer:
        def __init__(self, bounds, length_scale=0.2):
            self.bounds = bounds
            self.length_scale = length_scale
            self.X_obs = []
            self.y_obs = []
        
        def rbf_kernel(self, X1, X2):
            X1 = np.array(X1).reshape(-1, 1)
            X2 = np.array(X2).reshape(-1, 1)
            dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-dists / (2 * self.length_scale**2))
        
        def predict(self, X):
            if len(self.X_obs) == 0:
                return np.zeros(len(X)), np.ones(len(X))
            
            X = np.array(X)
            K = self.rbf_kernel(self.X_obs, self.X_obs) + 1e-6 * np.eye(len(self.X_obs))
            K_inv = np.linalg.inv(K)
            K_star = self.rbf_kernel(X, self.X_obs)
            K_star_star = self.rbf_kernel(X, X)
            
            mu = K_star @ K_inv @ np.array(self.y_obs)
            cov = K_star_star - K_star @ K_inv @ K_star.T
            std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            
            return mu, std
        
        def expected_improvement(self, X):
            mu, std = self.predict(X)
            if len(self.y_obs) == 0:
                return std
            
            y_best = min(self.y_obs)
            imp = y_best - mu - 0.01
            Z = np.where(std > 0, imp / std, 0)
            ei = imp * stats.norm.cdf(Z) + std * stats.norm.pdf(Z)
            return ei
        
        def suggest(self, n_candidates=100):
            X_cand = np.linspace(self.bounds[0], self.bounds[1], n_candidates)
            ei = self.expected_improvement(X_cand)
            return X_cand[np.argmax(ei)]
        
        def observe(self, x, y):
            self.X_obs.append(x)
            self.y_obs.append(y)
        
        def optimize(self, objective, n_init=3, n_iter=10):
            history = []
            
            # Random init
            for _ in range(n_init):
                x = np.random.uniform(*self.bounds)
                y = objective(x)
                self.observe(x, y)
                history.append((x, y))
            
            # BO iterations
            for _ in range(n_iter):
                x = self.suggest()
                y = objective(x)
                self.observe(x, y)
                history.append((x, y))
            
            return history
    
    bo = BayesianOptimizer(bounds=(0, 1))
    history = bo.optimize(objective, n_init=3, n_iter=12)
    
    print("Optimization Progress:")
    print(f"{'Iter':>6} {'x':>10} {'f(x)':>12} {'Best':>12}")
    print("-" * 45)
    
    best_so_far = float('inf')
    for i, (x, y) in enumerate(history):
        best_so_far = min(best_so_far, y)
        init_marker = " (init)" if i < 3 else ""
        print(f"{i+1:>6} {x:>10.4f} {y:>12.4f} {best_so_far:>12.4f}{init_marker}")
    
    best_x = bo.X_obs[np.argmin(bo.y_obs)]
    print(f"\nBest found: x = {best_x:.4f}")
    print(f"True optimum: x ≈ 0.35")


def exercise_6_successive_halving():
    """
    EXERCISE 6: Implement Successive Halving
    ========================================
    
    Implement the Successive Halving algorithm:
    1. Start with n configurations
    2. Train each for r resources
    3. Keep top half, double resources
    4. Repeat until one remains
    
    Tasks:
    a) Implement successive_halving(configs, train_fn, r_init, eta)
    b) Return best config and total compute used
    c) Compare with full training
    """
    print("\n" + "=" * 60)
    print("EXERCISE 6: Successive Halving")
    print("=" * 60)
    
    np.random.seed(42)
    
    def train_config(config, n_epochs):
        """Simulate training."""
        base = config['quality']  # Lower is better
        improvement = 0.1 * np.log(n_epochs + 1)
        noise = np.random.randn() * 0.05
        return max(0, base - improvement + noise)
    
    def generate_configs(n):
        """Generate random configs."""
        return [{'id': i, 'quality': np.random.uniform(0.5, 2.0)} for i in range(n)]
    
    # YOUR CODE HERE
    def successive_halving(configs, train_fn, r_init=1, eta=2):
        """
        Run Successive Halving.
        
        Returns:
            best_config: winning configuration
            total_compute: total resources used
        """
        # TODO: Implement
        pass


def exercise_6_solution():
    """Solution for Exercise 6."""
    print("\n" + "=" * 60)
    print("SOLUTION 6: Successive Halving")
    print("=" * 60)
    
    np.random.seed(42)
    
    def train_config(config, n_epochs):
        base = config['quality']
        improvement = 0.1 * np.log(n_epochs + 1)
        noise = np.random.randn() * 0.05
        return max(0, base - improvement + noise)
    
    def generate_configs(n):
        return [{'id': i, 'quality': np.random.uniform(0.5, 2.0)} for i in range(n)]
    
    def successive_halving(configs, train_fn, r_init=1, eta=2):
        remaining = configs.copy()
        budget = r_init
        total_compute = 0
        
        print(f"{'Round':>8} {'Configs':>10} {'Budget':>10} {'Compute':>12}")
        print("-" * 45)
        
        round_num = 0
        while len(remaining) > 1:
            round_num += 1
            
            # Train all configs
            results = []
            for config in remaining:
                score = train_fn(config, budget)
                results.append((config, score))
            
            compute = len(remaining) * budget
            total_compute += compute
            
            print(f"{round_num:>8} {len(remaining):>10} {budget:>10} {compute:>12}")
            
            # Keep top half
            results.sort(key=lambda x: x[1])
            n_keep = max(1, len(results) // eta)
            remaining = [config for config, _ in results[:n_keep]]
            
            budget *= eta
        
        return remaining[0], total_compute
    
    configs = generate_configs(16)
    best_config, total = successive_halving(configs, train_config, r_init=1, eta=2)
    
    print(f"\nBest config: id={best_config['id']}, quality={best_config['quality']:.4f}")
    print(f"Total compute: {total}")
    print(f"Full training (16 configs × 8 epochs): {16 * 8}")
    print(f"Speedup: {(16 * 8) / total:.2f}x")


def exercise_7_hyperband():
    """
    EXERCISE 7: Implement Hyperband
    ==============================
    
    Implement Hyperband which runs multiple brackets
    of Successive Halving with different n/r tradeoffs.
    
    Tasks:
    a) Calculate bracket parameters (n, r) for each s
    b) Run SH for each bracket
    c) Return overall best configuration
    """
    print("\n" + "=" * 60)
    print("EXERCISE 7: Hyperband")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    def hyperband(R, eta, get_config, train_fn):
        """
        Run Hyperband.
        
        Args:
            R: maximum resources
            eta: reduction factor
            get_config: function to generate config
            train_fn: function(config, budget) -> score
        
        Returns:
            best_config, total_compute
        """
        # TODO: Implement
        # s_max = floor(log_eta(R))
        # For each s from s_max down to 0:
        #   n = ceil(B/R * eta^s / (s+1))
        #   r = R * eta^(-s)
        #   Run successive halving with n configs starting at r resources
        pass


def exercise_7_solution():
    """Solution for Exercise 7."""
    print("\n" + "=" * 60)
    print("SOLUTION 7: Hyperband")
    print("=" * 60)
    
    np.random.seed(42)
    
    def train_fn(config, budget):
        base = config['quality']
        improvement = 0.1 * np.log(budget + 1)
        noise = np.random.randn() * 0.05
        return max(0, base - improvement + noise)
    
    def get_config():
        return {'quality': np.random.uniform(0.3, 2.0)}
    
    def hyperband(R, eta, get_config, train_fn):
        s_max = int(np.log(R) / np.log(eta))
        B = (s_max + 1) * R
        
        all_results = []
        total_compute = 0
        
        for s in range(s_max, -1, -1):
            print(f"\n--- Bracket s={s} ---")
            
            n = int(np.ceil(B / R * eta**s / (s + 1)))
            r = R * eta**(-s)
            
            configs = [get_config() for _ in range(n)]
            
            for i in range(s + 1):
                n_i = int(n * eta**(-i))
                r_i = int(r * eta**i)
                
                results = [(c, train_fn(c, r_i)) for c in configs]
                compute = n_i * r_i
                total_compute += compute
                
                print(f"  Round {i+1}: {n_i} configs × {r_i} epochs")
                
                all_results.extend(results)
                
                results.sort(key=lambda x: x[1])
                n_keep = max(1, int(n_i / eta))
                configs = [c for c, _ in results[:n_keep]]
        
        best = min(all_results, key=lambda x: x[1])
        return best[0], total_compute
    
    best_config, total = hyperband(R=27, eta=3, get_config=get_config, train_fn=train_fn)
    
    print(f"\nBest config quality: {best_config['quality']:.4f}")
    print(f"Total compute: {total}")


def exercise_8_tpe():
    """
    EXERCISE 8: Implement Tree Parzen Estimator
    ===========================================
    
    Implement simplified TPE:
    1. Split observations into good (l) and bad (g)
    2. Model l(x) and g(x) with KDEs
    3. Sample to maximize l(x)/g(x)
    
    Tasks:
    a) Implement splitting by quantile
    b) Implement KDE for l and g
    c) Sample new configuration
    """
    print("\n" + "=" * 60)
    print("EXERCISE 8: TPE")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    class SimpleTPE:
        def __init__(self, gamma=0.25, bandwidth=0.1):
            self.gamma = gamma
            self.bandwidth = bandwidth
            self.observations = []
        
        def observe(self, x, y):
            # TODO: Store observation
            pass
        
        def suggest(self, n_candidates=100):
            # TODO: Implement TPE suggestion
            # 1. If not enough observations, return random
            # 2. Split into good/bad by quantile
            # 3. Fit KDEs
            # 4. Return candidate maximizing l(x)/g(x)
            pass


def exercise_8_solution():
    """Solution for Exercise 8."""
    print("\n" + "=" * 60)
    print("SOLUTION 8: TPE")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x):
        return (x - 0.4) ** 2 + 0.05 * np.sin(20 * x)
    
    class SimpleTPE:
        def __init__(self, gamma=0.25, bandwidth=0.1):
            self.gamma = gamma
            self.bandwidth = bandwidth
            self.observations = []
        
        def observe(self, x, y):
            self.observations.append((x, y))
        
        def kde(self, samples, x):
            """Simple Gaussian KDE."""
            samples = np.array(samples)
            x = np.array(x)
            
            # Gaussian kernel
            diff = x.reshape(-1, 1) - samples.reshape(1, -1)
            kernel = np.exp(-diff**2 / (2 * self.bandwidth**2))
            return np.mean(kernel, axis=1)
        
        def suggest(self, n_candidates=100):
            if len(self.observations) < 5:
                return np.random.uniform(0, 1)
            
            # Sort and split
            sorted_obs = sorted(self.observations, key=lambda x: x[1])
            n_good = max(1, int(len(sorted_obs) * self.gamma))
            
            good = np.array([x for x, y in sorted_obs[:n_good]])
            bad = np.array([x for x, y in sorted_obs[n_good:]])
            
            # Generate candidates
            candidates = np.random.uniform(0, 1, n_candidates)
            
            # Compute l(x) / g(x)
            l_x = self.kde(good, candidates) + 1e-10
            g_x = self.kde(bad, candidates) + 1e-10
            
            ratios = l_x / g_x
            return candidates[np.argmax(ratios)]
    
    tpe = SimpleTPE()
    
    print("TPE Optimization:")
    print(f"{'Iter':>6} {'x':>10} {'f(x)':>12} {'Best':>12}")
    print("-" * 45)
    
    # Run optimization
    for i in range(20):
        x = tpe.suggest()
        y = objective(x)
        tpe.observe(x, y)
        
        best = min(obs[1] for obs in tpe.observations)
        print(f"{i+1:>6} {x:>10.4f} {y:>12.4f} {best:>12.4f}")
    
    best_x, best_y = min(tpe.observations, key=lambda x: x[1])
    print(f"\nBest: x = {best_x:.4f}, f(x) = {best_y:.4f}")


def exercise_9_pbt():
    """
    EXERCISE 9: Implement Population Based Training
    ==============================================
    
    Implement simplified PBT:
    1. Maintain population of agents
    2. Periodically exploit (copy from better) and explore (mutate)
    
    Tasks:
    a) Implement Agent class with train, copy_from, perturb
    b) Implement PBT loop
    c) Track population evolution
    """
    print("\n" + "=" * 60)
    print("EXERCISE 9: Population Based Training")
    print("=" * 60)
    
    np.random.seed(42)
    
    # YOUR CODE HERE
    class Agent:
        def __init__(self, agent_id, lr):
            # TODO: Initialize agent
            pass
        
        def train_step(self):
            # TODO: Simulate training
            pass
        
        def copy_from(self, other):
            # TODO: Copy weights
            pass
        
        def perturb(self, factor_range=(0.8, 1.2)):
            # TODO: Randomly perturb lr
            pass
    
    def pbt(n_agents, n_steps, exploit_interval):
        # TODO: Implement PBT loop
        pass


def exercise_9_solution():
    """Solution for Exercise 9."""
    print("\n" + "=" * 60)
    print("SOLUTION 9: Population Based Training")
    print("=" * 60)
    
    np.random.seed(42)
    
    class Agent:
        def __init__(self, agent_id, lr):
            self.id = agent_id
            self.lr = lr
            self.performance = np.random.uniform(0.8, 1.0)
            self.steps = 0
        
        def train_step(self):
            lr_quality = max(0, 1 - abs(np.log10(self.lr) + 2))
            improvement = 0.02 * lr_quality + np.random.randn() * 0.005
            self.performance = max(0, self.performance - improvement)
            self.steps += 1
        
        def copy_from(self, other):
            self.performance = other.performance
        
        def perturb(self, factor_range=(0.8, 1.2)):
            factor = np.random.uniform(*factor_range)
            self.lr = np.clip(self.lr * factor, 1e-5, 1.0)
    
    def pbt(n_agents, n_steps, exploit_interval):
        agents = [Agent(i, 10 ** np.random.uniform(-4, 0)) for i in range(n_agents)]
        
        print(f"Initial LRs: {[f'{a.lr:.4f}' for a in agents]}")
        
        for step in range(n_steps):
            for agent in agents:
                agent.train_step()
            
            if (step + 1) % exploit_interval == 0:
                sorted_agents = sorted(agents, key=lambda a: a.performance)
                
                # Bottom copies from top and perturbs
                n_replace = max(1, n_agents // 4)
                for i in range(n_replace):
                    sorted_agents[-(i+1)].copy_from(sorted_agents[i])
                    sorted_agents[-(i+1)].perturb()
                
                print(f"Step {step+1}: Perfs = {[f'{a.performance:.3f}' for a in agents]}")
        
        return min(agents, key=lambda a: a.performance)
    
    best = pbt(n_agents=4, n_steps=40, exploit_interval=10)
    print(f"\nBest agent: id={best.id}, perf={best.performance:.4f}, lr={best.lr:.5f}")


def exercise_10_multi_fidelity():
    """
    EXERCISE 10: Multi-Fidelity Optimization
    =======================================
    
    Implement multi-fidelity BO that considers
    both objective quality and evaluation cost.
    
    Tasks:
    a) Define fidelity-dependent objective and cost
    b) Implement acquisition with cost awareness
    c) Compare with single-fidelity approach
    """
    print("\n" + "=" * 60)
    print("EXERCISE 10: Multi-Fidelity Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x, fidelity):
        """Objective with fidelity parameter."""
        true_val = (x - 0.4) ** 2
        noise = np.random.randn() * (0.15 / np.sqrt(fidelity))
        bias = 0.05 * (1 - fidelity / 10)
        return true_val + noise + bias
    
    def cost(fidelity):
        return fidelity  # Linear cost
    
    # YOUR CODE HERE
    def multi_fidelity_search(objective, cost, budget, fidelities=[1, 3, 10]):
        """
        Run multi-fidelity random search.
        
        Strategy:
        1. Use low fidelity to screen many candidates
        2. Evaluate top candidates at high fidelity
        """
        # TODO: Implement
        pass


def exercise_10_solution():
    """Solution for Exercise 10."""
    print("\n" + "=" * 60)
    print("SOLUTION 10: Multi-Fidelity Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x, fidelity):
        true_val = (x - 0.4) ** 2
        noise = np.random.randn() * (0.15 / np.sqrt(fidelity))
        bias = 0.05 * (1 - fidelity / 10)
        return true_val + noise + bias
    
    def cost(fidelity):
        return fidelity
    
    def multi_fidelity_search(budget):
        results = []
        total_cost = 0
        
        # Phase 1: Screen with low fidelity
        n_screen = 20
        screen_fidelity = 1
        
        for _ in range(n_screen):
            x = np.random.uniform(0, 1)
            y = objective(x, screen_fidelity)
            results.append({'x': x, 'y_low': y, 'fidelity': screen_fidelity})
            total_cost += cost(screen_fidelity)
        
        print(f"Screening: {n_screen} configs at fidelity={screen_fidelity}")
        print(f"Cost so far: {total_cost}")
        
        # Phase 2: Evaluate top candidates at high fidelity
        sorted_results = sorted(results, key=lambda r: r['y_low'])
        remaining_budget = budget - total_cost
        high_fidelity = 10
        n_high = remaining_budget // cost(high_fidelity)
        
        high_results = []
        for r in sorted_results[:n_high]:
            y_high = objective(r['x'], high_fidelity)
            high_results.append((r['x'], y_high))
            total_cost += cost(high_fidelity)
        
        print(f"High-fidelity: {n_high} configs at fidelity={high_fidelity}")
        print(f"Total cost: {total_cost}")
        
        best_x, best_y = min(high_results, key=lambda r: r[1])
        return best_x, best_y
    
    # Single-fidelity comparison
    def single_fidelity_search(budget, fidelity=10):
        n_evals = budget // cost(fidelity)
        results = []
        
        for _ in range(n_evals):
            x = np.random.uniform(0, 1)
            y = objective(x, fidelity)
            results.append((x, y))
        
        return min(results, key=lambda r: r[1])
    
    budget = 50
    
    print("Multi-Fidelity Search:")
    mf_x, mf_y = multi_fidelity_search(budget)
    print(f"Best: x={mf_x:.4f}, f(x)={mf_y:.4f}")
    
    print("\nSingle-Fidelity Search:")
    sf_x, sf_y = single_fidelity_search(budget)
    print(f"Evaluations: {budget // 10}")
    print(f"Best: x={sf_x:.4f}, f(x)={sf_y:.4f}")
    
    print(f"\nTrue optimum: x=0.4")


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
