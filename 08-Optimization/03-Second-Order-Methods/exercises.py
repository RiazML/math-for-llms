"""
Second-Order Optimization Methods - Exercises
=============================================
Practice problems for second-order optimization.
"""

import numpy as np
from scipy import optimize


class SecondOrderExercises:
    """Exercises for second-order optimization methods."""
    
    def exercise_1_newton_step(self):
        """
        Exercise 1: Compute Newton Step
        
        Find the Newton step for a function.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Newton Step Computation")
        print("=" * 60)
        
        print("f(x,y) = x² + xy + 2y²")
        print("Find Newton step at (1, 1)")
        
        # Gradient
        print("\n∇f = [2x + y, x + 4y]")
        x = np.array([1.0, 1.0])
        g = np.array([2*x[0] + x[1], x[0] + 4*x[1]])
        print(f"∇f(1,1) = {g}")
        
        # Hessian
        print("\nH = [[2, 1], [1, 4]]")
        H = np.array([[2, 1], [1, 4]])
        
        # Newton step
        delta = np.linalg.solve(H, g)
        print(f"\nNewton step = H⁻¹g = {delta}")
        
        x_new = x - delta
        print(f"\nx_new = x - H⁻¹g = {x_new}")
        
        # Verify this is minimum
        g_new = np.array([2*x_new[0] + x_new[1], x_new[0] + 4*x_new[1]])
        print(f"\n∇f at x_new: {np.round(g_new, 10)}")
        print("(Should be zero - Newton finds minimum in 1 step for quadratic)")
    
    def exercise_2_convergence_analysis(self):
        """
        Exercise 2: Analyze Newton Convergence
        
        Track quadratic convergence.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Newton Convergence Analysis")
        print("=" * 60)
        
        # f(x) = e^x - x - 1, minimum at x = 0
        def f(x):
            return np.exp(x) - x - 1
        
        def fp(x):
            return np.exp(x) - 1
        
        def fpp(x):
            return np.exp(x)
        
        print("f(x) = eˣ - x - 1")
        print("Minimum at x = 0")
        
        x = 1.0
        errors = []
        
        print(f"\n{'Step':>4} {'x':>15} {'|x - 0|':>15} {'|e_t|/|e_{t-1}|²':>20}")
        print("-" * 60)
        
        for i in range(8):
            error = abs(x)
            errors.append(error)
            
            if i > 0 and errors[i-1] > 1e-15:
                ratio = error / (errors[i-1]**2)
            else:
                ratio = np.nan
            
            print(f"{i:>4} {x:>15.10f} {error:>15.10e} {ratio:>20.6f}")
            
            if abs(fp(x)) < 1e-15:
                break
            
            x = x - fp(x) / fpp(x)
        
        print("\nQuadratic convergence: error_t ≈ C × error_{t-1}²")
        print("The ratio |e_t|/|e_{t-1}|² approaches a constant")
    
    def exercise_3_implement_bfgs_step(self):
        """
        Exercise 3: BFGS Update
        
        Implement one BFGS update step.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: BFGS Update Step")
        print("=" * 60)
        
        # Current inverse Hessian approximation
        H = np.eye(2)
        
        # Suppose we took a step from x to x+s with gradient change y
        s = np.array([1.0, 0.5])  # Step taken
        y = np.array([2.0, 1.5])  # Gradient difference
        
        print("BFGS update formula:")
        print("H_{k+1} = (I - ρsy')H_k(I - ρys') + ρss'")
        print("where ρ = 1/(y's)")
        
        rho = 1.0 / (y @ s)
        print(f"\nρ = 1/(y's) = 1/{y @ s} = {rho:.4f}")
        
        I = np.eye(2)
        
        # BFGS update
        H_new = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        print(f"\nInitial H:\n{H}")
        print(f"\nUpdated H:\n{np.round(H_new, 4)}")
        
        # Verify secant condition: H_new @ y should equal s
        print(f"\nSecant condition check:")
        print(f"H_new @ y = {np.round(H_new @ y, 4)}")
        print(f"s = {s}")
        print(f"Satisfied: {np.allclose(H_new @ y, s)}")
    
    def exercise_4_gauss_newton(self):
        """
        Exercise 4: Gauss-Newton Method
        
        Apply to nonlinear least squares.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Gauss-Newton")
        print("=" * 60)
        
        # Fit y = a*x / (b + x) (Michaelis-Menten kinetics)
        np.random.seed(42)
        
        x_data = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        true_a, true_b = 5.0, 2.0
        y_data = true_a * x_data / (true_b + x_data) + np.random.randn(5) * 0.1
        
        print("Model: y = a*x / (b + x)")
        print(f"True parameters: a = {true_a}, b = {true_b}")
        print(f"Data: x = {x_data}")
        print(f"      y = {np.round(y_data, 3)}")
        
        def residuals(params):
            a, b = params
            return y_data - a * x_data / (b + x_data)
        
        def jacobian(params):
            a, b = params
            J = np.zeros((len(x_data), 2))
            J[:, 0] = -x_data / (b + x_data)  # ∂r/∂a
            J[:, 1] = a * x_data / (b + x_data)**2  # ∂r/∂b
            return J
        
        params = np.array([1.0, 1.0])
        
        print(f"\nGauss-Newton iterations:")
        print(f"{'Step':>4} {'a':>10} {'b':>10} {'||r||²':>12}")
        print("-" * 40)
        
        for i in range(10):
            r = residuals(params)
            J = jacobian(params)
            loss = 0.5 * np.sum(r**2)
            
            print(f"{i:>4} {params[0]:>10.4f} {params[1]:>10.4f} {loss:>12.6f}")
            
            if loss < 1e-6:
                break
            
            # Gauss-Newton: (J'J)⁻¹J'r
            delta = np.linalg.solve(J.T @ J, J.T @ r)
            params = params - delta
        
        print(f"\nEstimated: a = {params[0]:.4f}, b = {params[1]:.4f}")
    
    def exercise_5_levenberg_marquardt(self):
        """
        Exercise 5: Levenberg-Marquardt
        
        Implement damped Gauss-Newton.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Levenberg-Marquardt")
        print("=" * 60)
        
        # Same problem with worse initial guess
        np.random.seed(42)
        
        x_data = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        true_a, true_b = 5.0, 2.0
        y_data = true_a * x_data / (true_b + x_data) + np.random.randn(5) * 0.1
        
        def residuals(params):
            a, b = params
            return y_data - a * x_data / (b + x_data)
        
        def jacobian(params):
            a, b = params
            J = np.zeros((len(x_data), 2))
            J[:, 0] = -x_data / (b + x_data)
            J[:, 1] = a * x_data / (b + x_data)**2
            return J
        
        params = np.array([10.0, 0.1])  # Bad initial guess
        lam = 0.1
        
        print("Bad initial guess: a = 10, b = 0.1")
        print("\nLevenberg-Marquardt with adaptive λ:")
        print(f"{'Step':>4} {'a':>10} {'b':>10} {'||r||²':>12} {'λ':>10}")
        print("-" * 55)
        
        for i in range(20):
            r = residuals(params)
            J = jacobian(params)
            loss = 0.5 * np.sum(r**2)
            
            print(f"{i:>4} {params[0]:>10.4f} {params[1]:>10.4f} {loss:>12.6f} {lam:>10.4f}")
            
            if loss < 1e-6:
                break
            
            # LM step
            JTJ = J.T @ J
            delta = np.linalg.solve(JTJ + lam * np.eye(2), J.T @ r)
            
            new_params = params - delta
            new_loss = 0.5 * np.sum(residuals(new_params)**2)
            
            if new_loss < loss:
                params = new_params
                lam *= 0.5
            else:
                lam *= 2
        
        print(f"\nFinal: a = {params[0]:.4f}, b = {params[1]:.4f}")
    
    def exercise_6_condition_number(self):
        """
        Exercise 6: Condition Number Effects
        
        Analyze how condition number affects methods.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Condition Number Effects")
        print("=" * 60)
        
        def compare_methods(kappa, n_iter=100):
            A = np.diag([kappa, 1])
            b = np.ones(2)
            x_opt = np.linalg.solve(A, b)
            
            # GD
            x_gd = np.zeros(2)
            eta = 2 / (kappa + 1)  # Optimal learning rate
            
            for _ in range(n_iter):
                x_gd = x_gd - eta * (A @ x_gd - b)
            
            # Newton
            x_newton = np.zeros(2)
            x_newton = x_newton - np.linalg.solve(A, A @ x_newton - b)
            
            return np.linalg.norm(x_gd - x_opt), np.linalg.norm(x_newton - x_opt)
        
        print("Minimizing f(x) = 0.5*x'Ax - b'x")
        print("A = diag(κ, 1), varying condition number κ")
        print("\n100 GD iterations vs 1 Newton iteration:")
        
        print(f"\n{'κ':>8} {'GD error':>15} {'Newton error':>15} {'GD/Newton':>15}")
        print("-" * 60)
        
        for kappa in [1, 10, 100, 1000]:
            gd_err, newton_err = compare_methods(kappa)
            ratio = gd_err / (newton_err + 1e-15)
            print(f"{kappa:>8} {gd_err:>15.6e} {newton_err:>15.6e} {ratio:>15.2e}")
        
        print("\nNewton is invariant to conditioning!")
        print("GD convergence rate: (κ-1)/(κ+1)")
    
    def exercise_7_lbfgs_memory(self):
        """
        Exercise 7: L-BFGS Memory Effect
        
        Compare L-BFGS with different memory sizes.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: L-BFGS Memory Effect")
        print("=" * 60)
        
        np.random.seed(42)
        
        d = 20
        A = np.random.randn(d, d)
        A = A.T @ A + 0.1 * np.eye(d)
        b = np.random.randn(d)
        
        def f(x):
            return 0.5 * x @ A @ x - b @ x
        
        def grad(x):
            return A @ x - b
        
        x_opt = np.linalg.solve(A, b)
        f_opt = f(x_opt)
        
        def run_lbfgs(m, n_iter=50):
            """Simple L-BFGS."""
            x = np.zeros(d)
            s_list, y_list = [], []
            
            errors = []
            for _ in range(n_iter):
                errors.append(np.linalg.norm(x - x_opt))
                
                g = grad(x)
                
                # Two-loop recursion
                if len(s_list) == 0:
                    p = -g
                else:
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    
                    for i in range(len(s_list)-1, -1, -1):
                        rho = 1 / (y_list[i] @ s_list[i])
                        alpha[i] = rho * (s_list[i] @ q)
                        q = q - alpha[i] * y_list[i]
                    
                    gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
                    r = gamma * q
                    
                    for i in range(len(s_list)):
                        rho = 1 / (y_list[i] @ s_list[i])
                        beta = rho * (y_list[i] @ r)
                        r = r + s_list[i] * (alpha[i] - beta)
                    
                    p = -r
                
                # Line search
                alpha_ls = 1.0
                while f(x + alpha_ls * p) > f(x) - 1e-4 * alpha_ls * (g @ p):
                    alpha_ls *= 0.5
                
                s = alpha_ls * p
                x_new = x + s
                y = grad(x_new) - g
                
                if len(s_list) >= m:
                    s_list.pop(0)
                    y_list.pop(0)
                s_list.append(s)
                y_list.append(y)
                
                x = x_new
            
            return errors
        
        print(f"Problem dimension: d = {d}")
        print(f"\n{'Iter':>6}", end="")
        for m in [1, 3, 5, 10, 20]:
            print(f"{'m='+str(m):>12}", end="")
        print()
        print("-" * 70)
        
        results = {m: run_lbfgs(m) for m in [1, 3, 5, 10, 20]}
        
        for i in [0, 5, 10, 20, 30, 49]:
            print(f"{i:>6}", end="")
            for m in [1, 3, 5, 10, 20]:
                print(f"{results[m][i]:>12.2e}", end="")
            print()
        
        print("\nMore memory generally helps (up to a point)")
    
    def exercise_8_hessian_modification(self):
        """
        Exercise 8: Hessian Modification
        
        Handle non-PD Hessian.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Hessian Modification")
        print("=" * 60)
        
        # Function with saddle point
        def f(x):
            return x[0]**3 - 3*x[0]*x[1]**2 + x[0]**2 + x[1]**2
        
        def grad(x):
            return np.array([3*x[0]**2 - 3*x[1]**2 + 2*x[0], -6*x[0]*x[1] + 2*x[1]])
        
        def hessian(x):
            return np.array([[6*x[0] + 2, -6*x[1]], [-6*x[1], -6*x[0] + 2]])
        
        x = np.array([0.5, 0.5])
        
        print("Function with non-convex regions")
        print(f"Starting at x = {x}")
        
        print(f"\n{'Step':>4} {'x':>25} {'f(x)':>12} {'min(λ)':>12} {'Method':>10}")
        print("-" * 70)
        
        for i in range(15):
            g = grad(x)
            H = hessian(x)
            eigvals = np.linalg.eigvalsh(H)
            min_eig = np.min(eigvals)
            
            method = ""
            
            if min_eig > 0:
                # Pure Newton
                delta = np.linalg.solve(H, g)
                method = "Newton"
            else:
                # Modified Newton: add to make PD
                tau = max(0, -min_eig + 0.1)
                H_mod = H + tau * np.eye(2)
                delta = np.linalg.solve(H_mod, g)
                method = f"Mod(τ={tau:.2f})"
            
            print(f"{i:>4} {str(np.round(x, 4)):>25} {f(x):>12.6f} {min_eig:>12.4f} {method:>10}")
            
            if np.linalg.norm(g) < 1e-6:
                break
            
            # Line search
            alpha = 1.0
            while f(x - alpha * delta) > f(x) - 0.1 * alpha * (g @ delta):
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            x = x - alpha * delta
        
        print(f"\nFinal: x = {np.round(x, 6)}, f(x) = {f(x):.6f}")
    
    def exercise_9_trust_region(self):
        """
        Exercise 9: Trust Region
        
        Implement trust region update.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Trust Region")
        print("=" * 60)
        
        def f(x):
            return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        def grad(x):
            g0 = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
            g1 = 200*(x[1] - x[0]**2)
            return np.array([g0, g1])
        
        def hessian(x):
            h00 = 1200*x[0]**2 - 400*x[1] + 2
            h01 = -400*x[0]
            h11 = 200
            return np.array([[h00, h01], [h01, h11]])
        
        x = np.array([-1.0, 1.0])
        Delta = 1.0
        
        print("Rosenbrock function, trust region method")
        print(f"{'Step':>4} {'||x-x*||':>12} {'f(x)':>12} {'Δ':>8} {'ρ':>8}")
        print("-" * 50)
        
        for i in range(30):
            g = grad(x)
            H = hessian(x)
            
            dist = np.linalg.norm(x - np.array([1, 1]))
            
            # Solve trust region subproblem (cauchy point for simplicity)
            # More sophisticated: dogleg or exact TR solution
            p = -g
            if np.linalg.norm(p) > Delta:
                p = Delta * p / np.linalg.norm(p)
            
            # Actual vs predicted reduction
            actual = f(x) - f(x + p)
            predicted = -(g @ p + 0.5 * p @ H @ p)
            
            rho = actual / (predicted + 1e-10)
            
            print(f"{i:>4} {dist:>12.6f} {f(x):>12.4f} {Delta:>8.4f} {rho:>8.4f}")
            
            # Update trust region
            if rho > 0.75:
                Delta = min(2*Delta, 5)
            elif rho < 0.25:
                Delta = 0.5 * Delta
            
            # Accept if rho > 0
            if rho > 0:
                x = x + p
            
            if np.linalg.norm(g) < 1e-6:
                break
        
        print(f"\nFinal: x = {np.round(x, 6)}")
    
    def exercise_10_scipy_minimize(self):
        """
        Exercise 10: Using scipy.optimize.minimize
        
        Compare different methods in scipy.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: scipy.optimize Methods")
        print("=" * 60)
        
        np.random.seed(42)
        
        def rosenbrock(x):
            return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        x0 = np.zeros(5)
        
        methods = ['Nelder-Mead', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'trust-ncg']
        
        print("Rosenbrock (d=5), starting at origin")
        print("Optimal: all ones, f* = 0")
        
        print(f"\n{'Method':>15} {'Iters':>8} {'f(x*)':>15} {'||x*-1||':>12} {'Success':>10}")
        print("-" * 65)
        
        for method in methods:
            try:
                result = optimize.minimize(rosenbrock, x0, method=method, 
                                          options={'maxiter': 1000})
                dist = np.linalg.norm(result.x - 1)
                print(f"{method:>15} {result.nit:>8} {result.fun:>15.6e} "
                      f"{dist:>12.6e} {str(result.success):>10}")
            except Exception as e:
                print(f"{method:>15} {'Failed':>8}")
        
        print("\nNotes:")
        print("- BFGS/L-BFGS-B: quasi-Newton, good general purpose")
        print("- Newton-CG/trust-ncg: use Hessian (or Hessian-vector products)")
        print("- CG: conjugate gradient, first-order")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = SecondOrderExercises()
    
    print("SECOND-ORDER OPTIMIZATION EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    exercises.solution_2()
    exercises.solution_3()
    exercises.solution_4()
    exercises.solution_5()
    exercises.solution_6()
    exercises.solution_7()
    exercises.solution_8()
    exercises.solution_9()
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
