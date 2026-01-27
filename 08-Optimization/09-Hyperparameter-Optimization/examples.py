"""
Hyperparameter Optimization - Examples
======================================
Implementing and demonstrating hyperparameter search strategies.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def example_grid_search():
    """Grid search implementation."""
    print("=" * 60)
    print("EXAMPLE 1: Grid Search")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulated objective function (pretend it's model validation error)
    def objective(learning_rate, hidden_units, dropout):
        """Simulated validation error."""
        # Optimal around lr=0.01, hidden=64, dropout=0.3
        error = (
            (np.log10(learning_rate) + 2) ** 2 +  # optimal at 0.01
            ((hidden_units - 64) / 32) ** 2 +      # optimal at 64
            (dropout - 0.3) ** 2 * 5 +             # optimal at 0.3
            np.random.randn() * 0.1                # noise
        )
        return max(0.1, error)
    
    # Define grid
    learning_rates = [0.001, 0.01, 0.1]
    hidden_units_list = [32, 64, 128]
    dropout_rates = [0.1, 0.3, 0.5]
    
    print("Grid Search:")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Hidden units:   {hidden_units_list}")
    print(f"  Dropout rates:  {dropout_rates}")
    print(f"  Total combinations: {len(learning_rates) * len(hidden_units_list) * len(dropout_rates)}")
    
    best_error = float('inf')
    best_params = None
    results = []
    
    for lr in learning_rates:
        for hidden in hidden_units_list:
            for dropout in dropout_rates:
                error = objective(lr, hidden, dropout)
                results.append((lr, hidden, dropout, error))
                
                if error < best_error:
                    best_error = error
                    best_params = (lr, hidden, dropout)
    
    print(f"\nTop 5 configurations:")
    sorted_results = sorted(results, key=lambda x: x[3])
    for lr, hidden, dropout, error in sorted_results[:5]:
        print(f"  lr={lr}, hidden={hidden}, dropout={dropout}: error={error:.4f}")
    
    print(f"\nBest: lr={best_params[0]}, hidden={best_params[1]}, dropout={best_params[2]}")
    print(f"Best error: {best_error:.4f}")


def example_random_search():
    """Random search with comparison to grid search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Random Search")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x, y):
        """Objective where only x matters."""
        # Only x affects the result
        return (x - 0.3) ** 2 + np.random.randn() * 0.01
    
    n_trials = 9
    
    # Grid search: 3x3 grid
    grid_x = [0.0, 0.5, 1.0]
    grid_y = [0.0, 0.5, 1.0]
    
    grid_results = []
    for x in grid_x:
        for y in grid_y:
            error = objective(x, y)
            grid_results.append((x, y, error))
    
    # Random search: 9 random samples
    random_results = []
    for _ in range(n_trials):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        error = objective(x, y)
        random_results.append((x, y, error))
    
    print("When only one hyperparameter matters:")
    print(f"\nGrid search ({n_trials} trials):")
    print(f"  Unique x values: {sorted(set(r[0] for r in grid_results))}")
    print(f"  Best error: {min(r[2] for r in grid_results):.4f}")
    
    print(f"\nRandom search ({n_trials} trials):")
    print(f"  Unique x values: {[f'{r[0]:.3f}' for r in random_results]}")
    print(f"  Best error: {min(r[2] for r in random_results):.4f}")
    
    print("\nRandom search explores more values of the important HP!")


def example_latin_hypercube_sampling():
    """Latin Hypercube Sampling for initialization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Latin Hypercube Sampling")
    print("=" * 60)
    
    np.random.seed(42)
    
    def latin_hypercube_sample(n_samples, n_dims):
        """Generate LHS samples in [0, 1]^d."""
        samples = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            # Divide [0, 1] into n_samples equal intervals
            intervals = np.linspace(0, 1, n_samples + 1)
            
            # Sample uniformly within each interval
            for i in range(n_samples):
                samples[i, dim] = np.random.uniform(intervals[i], intervals[i + 1])
            
            # Randomly permute
            np.random.shuffle(samples[:, dim])
        
        return samples
    
    n_samples = 5
    n_dims = 2
    
    # Regular random sampling
    random_samples = np.random.rand(n_samples, n_dims)
    
    # Latin Hypercube Sampling
    lhs_samples = latin_hypercube_sample(n_samples, n_dims)
    
    print(f"Sampling {n_samples} points in 2D:")
    
    print("\nRandom sampling:")
    for i, (x, y) in enumerate(random_samples):
        print(f"  Point {i+1}: ({x:.3f}, {y:.3f})")
    
    print("\nLatin Hypercube Sampling:")
    for i, (x, y) in enumerate(lhs_samples):
        print(f"  Point {i+1}: ({x:.3f}, {y:.3f})")
    
    # Show coverage
    print("\nLHS ensures each row and column has exactly one point")
    print("(Better space coverage for initialization)")


def example_gaussian_process_surrogate():
    """Gaussian Process as surrogate model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Gaussian Process Surrogate")
    print("=" * 60)
    
    np.random.seed(42)
    
    class SimpleGP:
        """Simple Gaussian Process for 1D regression."""
        
        def __init__(self, length_scale=1.0, noise=1e-6):
            self.length_scale = length_scale
            self.noise = noise
            self.X_train = None
            self.y_train = None
        
        def rbf_kernel(self, X1, X2):
            """RBF kernel."""
            dists = np.subtract.outer(X1.ravel(), X2.ravel()) ** 2
            return np.exp(-dists / (2 * self.length_scale ** 2))
        
        def fit(self, X, y):
            """Fit GP to data."""
            self.X_train = np.array(X).reshape(-1)
            self.y_train = np.array(y).reshape(-1)
            
            # Compute kernel matrix
            K = self.rbf_kernel(self.X_train, self.X_train)
            self.K_inv = np.linalg.inv(K + self.noise * np.eye(len(X)))
        
        def predict(self, X):
            """Predict mean and std at X."""
            X = np.array(X).reshape(-1)
            
            # Kernel between test and train
            K_star = self.rbf_kernel(X, self.X_train)
            
            # Kernel of test points
            K_star_star = self.rbf_kernel(X, X)
            
            # Posterior mean
            mu = K_star @ self.K_inv @ self.y_train
            
            # Posterior covariance
            cov = K_star_star - K_star @ self.K_inv @ K_star.T
            std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            
            return mu, std
    
    # True function (unknown to optimizer)
    def true_function(x):
        return np.sin(3 * x) + x ** 2 - 0.5 * x
    
    # Initial observations
    X_obs = np.array([0.1, 0.4, 0.8])
    y_obs = np.array([true_function(x) for x in X_obs])
    
    # Fit GP
    gp = SimpleGP(length_scale=0.3)
    gp.fit(X_obs, y_obs)
    
    # Predict on grid
    X_test = np.linspace(0, 1, 11)
    mu, std = gp.predict(X_test)
    
    print("GP Surrogate Model:")
    print(f"Observations: {list(zip(X_obs.round(2), y_obs.round(2)))}")
    
    print(f"\n{'x':>8} {'True f(x)':>12} {'GP μ':>12} {'GP σ':>12}")
    print("-" * 48)
    
    for x, m, s in zip(X_test, mu, std):
        true_val = true_function(x)
        print(f"{x:>8.2f} {true_val:>12.4f} {m:>12.4f} {s:>12.4f}")
    
    print("\nGP provides uncertainty estimates (σ)")
    print("High σ = unexplored region → good for exploration")


def example_expected_improvement():
    """Expected Improvement acquisition function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Expected Improvement")
    print("=" * 60)
    
    np.random.seed(42)
    
    def expected_improvement(mu, sigma, y_best, xi=0.01):
        """Compute Expected Improvement."""
        with np.errstate(divide='warn'):
            imp = y_best - mu - xi
            Z = imp / sigma
            ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei
    
    # Simulated GP predictions
    X = np.linspace(0, 1, 21)
    
    # Pretend we have observations at x=0.2 and x=0.7
    # GP gives us these predictions
    mu = 0.5 - 0.3 * np.sin(5 * X) + 0.2 * X  # Mean prediction
    sigma = 0.1 + 0.2 * np.abs(X - 0.2) * np.abs(X - 0.7)  # Uncertainty
    
    # Best observed value
    y_best = 0.3
    
    # Compute EI
    ei = expected_improvement(mu, sigma, y_best)
    
    print("Expected Improvement (EI):")
    print(f"Current best: y* = {y_best}")
    print(f"\n{'x':>6} {'μ(x)':>10} {'σ(x)':>10} {'EI(x)':>10}")
    print("-" * 40)
    
    for i in range(0, len(X), 2):
        print(f"{X[i]:>6.2f} {mu[i]:>10.4f} {sigma[i]:>10.4f} {ei[i]:>10.4f}")
    
    best_idx = np.argmax(ei)
    print(f"\nNext point to evaluate: x = {X[best_idx]:.2f}")
    print(f"Max EI = {ei[best_idx]:.4f}")
    
    print("\nEI balances:")
    print("  - Low μ (exploitation)")
    print("  - High σ (exploration)")


def example_bayesian_optimization():
    """Complete Bayesian Optimization loop."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Bayesian Optimization Loop")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True function (unknown)
    def objective(x):
        """Objective to minimize."""
        return (x - 0.4) ** 2 + 0.1 * np.sin(10 * x)
    
    class SimpleBayesOpt:
        """Simple 1D Bayesian Optimization."""
        
        def __init__(self, bounds=(0, 1), length_scale=0.2):
            self.bounds = bounds
            self.length_scale = length_scale
            self.X_obs = []
            self.y_obs = []
        
        def rbf_kernel(self, X1, X2):
            X1 = np.array(X1).reshape(-1, 1)
            X2 = np.array(X2).reshape(-1, 1)
            dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-dists / (2 * self.length_scale ** 2))
        
        def predict(self, X):
            if len(self.X_obs) == 0:
                return np.zeros(len(X)), np.ones(len(X))
            
            K = self.rbf_kernel(self.X_obs, self.X_obs) + 1e-6 * np.eye(len(self.X_obs))
            K_inv = np.linalg.inv(K)
            K_star = self.rbf_kernel(X, self.X_obs)
            K_star_star = self.rbf_kernel(X, X)
            
            mu = K_star @ K_inv @ np.array(self.y_obs)
            cov = K_star_star - K_star @ K_inv @ K_star.T
            std = np.sqrt(np.maximum(np.diag(cov), 1e-10))
            
            return mu, std
        
        def acquisition(self, X, xi=0.01):
            mu, std = self.predict(X)
            if len(self.y_obs) == 0:
                return std  # Just explore
            
            y_best = min(self.y_obs)
            imp = y_best - mu - xi
            Z = np.where(std > 0, imp / std, 0)
            ei = imp * stats.norm.cdf(Z) + std * stats.norm.pdf(Z)
            return ei
        
        def suggest_next(self, n_candidates=100):
            X_cand = np.linspace(self.bounds[0], self.bounds[1], n_candidates)
            ei = self.acquisition(X_cand)
            return X_cand[np.argmax(ei)]
        
        def observe(self, x, y):
            self.X_obs.append(x)
            self.y_obs.append(y)
    
    # Run BO
    bo = SimpleBayesOpt()
    n_init = 2
    n_iter = 8
    
    print("Bayesian Optimization Progress:")
    print(f"{'Iter':>6} {'x':>10} {'f(x)':>12} {'Best so far':>15}")
    print("-" * 50)
    
    # Random initialization
    for i in range(n_init):
        x = np.random.uniform(0, 1)
        y = objective(x)
        bo.observe(x, y)
        print(f"{i+1:>6} {x:>10.4f} {y:>12.4f} {min(bo.y_obs):>15.4f}")
    
    # BO iterations
    for i in range(n_iter):
        x = bo.suggest_next()
        y = objective(x)
        bo.observe(x, y)
        print(f"{n_init+i+1:>6} {x:>10.4f} {y:>12.4f} {min(bo.y_obs):>15.4f}")
    
    best_idx = np.argmin(bo.y_obs)
    print(f"\nBest found: x = {bo.X_obs[best_idx]:.4f}, f(x) = {bo.y_obs[best_idx]:.4f}")
    print(f"True optimum: x ≈ 0.4")


def example_successive_halving():
    """Successive Halving algorithm."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Successive Halving")
    print("=" * 60)
    
    np.random.seed(42)
    
    def train_model(config, n_epochs):
        """Simulate training a model."""
        # config is learning rate; optimal around 0.01
        # Better configs converge faster
        base_error = (np.log10(config) + 2) ** 2 + np.random.randn() * 0.1
        improvement = np.log(n_epochs + 1) * 0.1
        return max(0.1, base_error - improvement)
    
    # Initial configurations
    n_configs = 16
    configs = 10 ** np.random.uniform(-3, 0, n_configs)  # Learning rates
    
    print("Successive Halving:")
    print(f"Initial configs: {n_configs}")
    print(f"Learning rates: {[f'{c:.4f}' for c in sorted(configs)]}")
    
    budget = 1
    remaining = list(enumerate(configs))
    
    print(f"\n{'Round':>8} {'Configs':>10} {'Epochs':>10} {'Total compute':>15}")
    print("-" * 50)
    
    round_num = 0
    total_compute = 0
    
    while len(remaining) > 1:
        round_num += 1
        n_remaining = len(remaining)
        
        # Train each config
        results = []
        for idx, config in remaining:
            error = train_model(config, budget)
            results.append((idx, config, error))
        
        compute = n_remaining * budget
        total_compute += compute
        
        print(f"{round_num:>8} {n_remaining:>10} {budget:>10} {compute:>15}")
        
        # Keep top half
        results.sort(key=lambda x: x[2])
        n_keep = max(1, len(results) // 2)
        remaining = [(idx, config) for idx, config, _ in results[:n_keep]]
        
        budget *= 2
    
    winner_idx, winner_config = remaining[0]
    print(f"\nWinner: config {winner_idx}, lr = {winner_config:.4f}")
    print(f"Total compute: {total_compute} config-epochs")
    print(f"Full training would be: {n_configs * budget // 2} config-epochs")


def example_hyperband():
    """Hyperband algorithm."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Hyperband")
    print("=" * 60)
    
    np.random.seed(42)
    
    def train_model(config, n_epochs):
        """Simulate training."""
        base_error = config['lr_error'] + np.random.randn() * 0.05
        improvement = np.log(n_epochs + 1) * 0.1
        return max(0.05, base_error - improvement)
    
    def get_random_config():
        lr = 10 ** np.random.uniform(-4, 0)
        lr_error = (np.log10(lr) + 2) ** 2  # Optimal at 0.01
        return {'lr': lr, 'lr_error': lr_error}
    
    # Hyperband parameters
    R = 27  # Maximum resources (epochs)
    eta = 3  # Reduction factor
    
    # Calculate brackets
    s_max = int(np.log(R) / np.log(eta))
    B = (s_max + 1) * R
    
    print(f"Hyperband settings:")
    print(f"  Max resources R = {R}")
    print(f"  Reduction factor η = {eta}")
    print(f"  Number of brackets = {s_max + 1}")
    
    all_results = []
    total_compute = 0
    
    for s in range(s_max, -1, -1):
        print(f"\n--- Bracket s={s} ---")
        
        # Initial number of configs and resources
        n = int(np.ceil(B / R * eta**s / (s + 1)))
        r = R * eta**(-s)
        
        # Generate configs
        configs = [get_random_config() for _ in range(n)]
        
        for i in range(s + 1):
            n_i = int(n * eta**(-i))
            r_i = int(r * eta**i)
            
            # Train
            results = []
            for config in configs:
                error = train_model(config, r_i)
                results.append((config, error))
                all_results.append((config['lr'], error))
            
            compute = n_i * r_i
            total_compute += compute
            
            print(f"  Round {i+1}: {n_i} configs × {r_i} epochs = {compute} compute")
            
            # Keep top 1/eta
            results.sort(key=lambda x: x[1])
            n_keep = max(1, int(n_i / eta))
            configs = [config for config, _ in results[:n_keep]]
    
    # Find best
    best_lr, best_error = min(all_results, key=lambda x: x[1])
    
    print(f"\nTotal compute: {total_compute}")
    print(f"Best learning rate: {best_lr:.4f}")
    print(f"Best error: {best_error:.4f}")


def example_tpe():
    """Tree Parzen Estimator (simplified)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Tree Parzen Estimator (TPE)")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x):
        """Objective to minimize."""
        return (x - 0.3) ** 2 + 0.1 * np.sin(20 * x)
    
    class SimpleTPE:
        """Simplified 1D TPE."""
        
        def __init__(self, gamma=0.25):
            self.gamma = gamma  # Percentile for splitting
            self.observations = []
        
        def observe(self, x, y):
            self.observations.append((x, y))
        
        def suggest_next(self, n_candidates=100):
            if len(self.observations) < 5:
                return np.random.uniform(0, 1)
            
            # Sort by objective
            sorted_obs = sorted(self.observations, key=lambda x: x[1])
            
            # Split into good (l) and bad (g)
            n_good = max(1, int(len(sorted_obs) * self.gamma))
            good = [x for x, y in sorted_obs[:n_good]]
            bad = [x for x, y in sorted_obs[n_good:]]
            
            # Fit KDEs (simplified: use normal distributions)
            good_mean, good_std = np.mean(good), max(0.1, np.std(good))
            bad_mean, bad_std = np.mean(bad), max(0.1, np.std(bad))
            
            # Sample candidates and compute l(x)/g(x)
            candidates = np.random.uniform(0, 1, n_candidates)
            
            l_x = stats.norm.pdf(candidates, good_mean, good_std)
            g_x = stats.norm.pdf(candidates, bad_mean, bad_std) + 1e-10
            
            # Return candidate with highest l(x)/g(x)
            ratios = l_x / g_x
            return candidates[np.argmax(ratios)]
    
    tpe = SimpleTPE()
    n_init = 5
    n_iter = 15
    
    print("TPE Optimization:")
    print(f"{'Iter':>6} {'x':>10} {'f(x)':>12} {'Best':>12}")
    print("-" * 45)
    
    # Random init
    for i in range(n_init):
        x = np.random.uniform(0, 1)
        y = objective(x)
        tpe.observe(x, y)
        best = min(obs[1] for obs in tpe.observations)
        print(f"{i+1:>6} {x:>10.4f} {y:>12.4f} {best:>12.4f}")
    
    # TPE iterations
    for i in range(n_iter):
        x = tpe.suggest_next()
        y = objective(x)
        tpe.observe(x, y)
        best = min(obs[1] for obs in tpe.observations)
        print(f"{n_init+i+1:>6} {x:>10.4f} {y:>12.4f} {best:>12.4f}")
    
    best_x, best_y = min(tpe.observations, key=lambda x: x[1])
    print(f"\nBest: x = {best_x:.4f}, f(x) = {best_y:.4f}")
    print(f"True optimum: x ≈ 0.3")


def example_pbt():
    """Population Based Training (simplified)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Population Based Training")
    print("=" * 60)
    
    np.random.seed(42)
    
    class Agent:
        def __init__(self, agent_id, lr):
            self.id = agent_id
            self.lr = lr
            self.performance = np.random.uniform(0.5, 1.0)  # Initial performance
            self.steps = 0
        
        def train_step(self):
            """Simulate one training step."""
            # Performance improves based on how good lr is
            # Optimal lr around 0.01
            lr_quality = 1 - abs(np.log10(self.lr) + 2)
            improvement = 0.01 * lr_quality + np.random.randn() * 0.005
            self.performance = max(0, self.performance - improvement)
            self.steps += 1
        
        def copy_from(self, other):
            """Copy weights from another agent."""
            self.performance = other.performance
        
        def perturb(self):
            """Perturb hyperparameters."""
            factor = np.random.choice([0.8, 1.2])
            self.lr = np.clip(self.lr * factor, 1e-5, 1.0)
    
    # Initialize population
    n_agents = 4
    agents = [Agent(i, 10 ** np.random.uniform(-4, 0)) for i in range(n_agents)]
    
    print("Population Based Training:")
    print(f"Initial learning rates: {[f'{a.lr:.4f}' for a in agents]}")
    
    n_steps = 20
    exploit_interval = 5
    
    print(f"\n{'Step':>6}" + "".join(f"{'Agent '+str(i):>12}" for i in range(n_agents)))
    print("-" * (6 + 12 * n_agents))
    
    for step in range(n_steps):
        # Train all agents
        for agent in agents:
            agent.train_step()
        
        if (step + 1) % exploit_interval == 0:
            # Exploit: bottom copies from top
            sorted_agents = sorted(agents, key=lambda a: a.performance)
            
            # Bottom 25% copies from top 25%
            n_replace = max(1, n_agents // 4)
            for i in range(n_replace):
                sorted_agents[-(i+1)].copy_from(sorted_agents[i])
                sorted_agents[-(i+1)].perturb()
        
        # Print progress
        if (step + 1) % 4 == 0:
            perfs = "".join(f"{a.performance:>12.4f}" for a in agents)
            print(f"{step+1:>6}{perfs}")
    
    best_agent = min(agents, key=lambda a: a.performance)
    print(f"\nBest agent: {best_agent.id}")
    print(f"Best performance: {best_agent.performance:.4f}")
    print(f"Best lr: {best_agent.lr:.4f}")


def example_multi_fidelity():
    """Multi-fidelity optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Multi-Fidelity Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(x, fidelity):
        """
        Objective with fidelity parameter.
        Higher fidelity = more accurate but more expensive.
        """
        true_value = (x - 0.4) ** 2
        
        # Noise decreases with fidelity
        noise = np.random.randn() * (0.2 / np.sqrt(fidelity))
        
        # Bias at low fidelity
        bias = 0.1 * (1 - fidelity / 10)
        
        return true_value + noise + bias
    
    def cost(fidelity):
        """Cost scales with fidelity."""
        return fidelity
    
    # Compare strategies
    budget = 50  # Total cost budget
    
    # Strategy 1: High fidelity only
    n_high = budget // 10
    high_results = []
    for _ in range(n_high):
        x = np.random.uniform(0, 1)
        y = objective(x, fidelity=10)
        high_results.append((x, y))
    
    # Strategy 2: Multi-fidelity
    multi_results = []
    
    # Use low fidelity to screen
    n_low = 20
    for _ in range(n_low):
        x = np.random.uniform(0, 1)
        y = objective(x, fidelity=1)
        multi_results.append((x, y, 1))
    
    # Evaluate top candidates at high fidelity
    remaining_budget = budget - n_low * 1
    top_candidates = sorted(multi_results, key=lambda r: r[1])[:3]
    
    for x, _, _ in top_candidates:
        y = objective(x, fidelity=10)
        multi_results.append((x, y, 10))
    
    print("Multi-Fidelity Comparison:")
    print(f"Total budget: {budget}")
    
    print(f"\nHigh-fidelity only ({n_high} evaluations at f=10):")
    best_high = min(high_results, key=lambda r: r[1])
    print(f"  Best x = {best_high[0]:.4f}, f(x) = {best_high[1]:.4f}")
    
    print(f"\nMulti-fidelity ({n_low} at f=1, 3 at f=10):")
    high_fidelity_results = [(x, y) for x, y, f in multi_results if f == 10]
    best_multi = min(high_fidelity_results, key=lambda r: r[1])
    print(f"  Best x = {best_multi[0]:.4f}, f(x) = {best_multi[1]:.4f}")
    
    print(f"\nTrue optimum: x = 0.4")


def example_hyperparameter_importance():
    """Analyze hyperparameter importance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Hyperparameter Importance")
    print("=" * 60)
    
    np.random.seed(42)
    
    def objective(lr, batch_size, dropout, momentum):
        """
        Objective where hyperparameters have different importance.
        """
        # Learning rate: VERY important (wide range)
        lr_effect = (np.log10(lr) + 2) ** 2 * 5
        
        # Dropout: Moderately important
        dropout_effect = (dropout - 0.3) ** 2 * 2
        
        # Batch size: Less important
        batch_effect = np.log2(batch_size / 64) ** 2 * 0.5
        
        # Momentum: Least important
        momentum_effect = (momentum - 0.9) ** 2 * 0.2
        
        noise = np.random.randn() * 0.1
        
        return lr_effect + dropout_effect + batch_effect + momentum_effect + noise
    
    # Run random search
    n_trials = 100
    results = []
    
    for _ in range(n_trials):
        lr = 10 ** np.random.uniform(-4, 0)
        batch_size = 2 ** np.random.randint(4, 8)  # 16 to 128
        dropout = np.random.uniform(0, 0.5)
        momentum = np.random.uniform(0.8, 0.99)
        
        error = objective(lr, batch_size, dropout, momentum)
        results.append({
            'lr': lr,
            'batch_size': batch_size,
            'dropout': dropout,
            'momentum': momentum,
            'error': error
        })
    
    # Compute importance via variance decomposition (simplified)
    errors = np.array([r['error'] for r in results])
    
    # For each HP, compute variance of mean error at each value
    importance = {}
    
    for hp in ['lr', 'batch_size', 'dropout', 'momentum']:
        values = np.array([r[hp] for r in results])
        
        # Bin values
        if hp == 'lr':
            bins = np.logspace(-4, 0, 5)
        elif hp == 'batch_size':
            bins = [16, 32, 64, 128, 256]
        else:
            bins = np.linspace(min(values), max(values), 5)
        
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (values >= bins[i]) & (values < bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(errors[mask]))
        
        importance[hp] = np.var(bin_means) if len(bin_means) > 1 else 0
    
    # Normalize
    total = sum(importance.values())
    importance = {k: v/total * 100 for k, v in importance.items()}
    
    print("Hyperparameter Importance (% of total variance):")
    print("-" * 40)
    
    for hp, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp / 5)
        print(f"{hp:>15}: {imp:>5.1f}% {bar}")
    
    print("\nLearning rate dominates! Focus tuning efforts there.")


if __name__ == "__main__":
    example_grid_search()
    example_random_search()
    example_latin_hypercube_sampling()
    example_gaussian_process_surrogate()
    example_expected_improvement()
    example_bayesian_optimization()
    example_successive_halving()
    example_hyperband()
    example_tpe()
    example_pbt()
    example_multi_fidelity()
    example_hyperparameter_importance()
