
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.abspath("oracle_2025A1"))
from oracle_2025A1 import oq1, oq2f, oq2g, oq3

# Question 1: Quad Minimization Func

def compute_gradient(Q_matrix, bias_vector, x_vector):
    """Gradient of f(x) = 0.5 x^T Q x + b^T x"""
    return Q_matrix @ x_vector + bias_vector

def evaluate_quadratic(Q_matrix, bias_vector, x_vector):
    """Evaluate quadratic function"""
    return 0.5 * x_vector.T @ Q_matrix @ x_vector + bias_vector.T @ x_vector

def gradient_descent_exact_line_search(Q_matrix, bias_vector, init_x, max_iterations=100000, tolerance=1e-8):

    x_current = init_x
    x_history = [x_current.copy()]
    analytical_x_star = np.linalg.solve(-Q_matrix, bias_vector)

    for _ in range(max_iterations):
        gradient = compute_gradient(Q_matrix, bias_vector, x_current)
        if np.linalg.norm(gradient) < tolerance:
            break
        step_size = (gradient.T @ gradient) / (gradient.T @ Q_matrix @ gradient)
        x_current = x_current - step_size * gradient
        x_history.append(x_current.copy())

    return np.array(x_history), analytical_x_star

def solve_analytical_minimizer(Q_matrix, bias_vector):
    """Analytical solution for x*"""
    return np.linalg.solve(-Q_matrix, bias_vector)

# Question 1 Main

def question1_main():
    sr_number = 25846
    bias_vector = np.array([1.0, 1.0])

    # Get 5 Q matrices from oracle
    oracle_Q_list = oq1(sr_number)

    plt.figure(figsize=(10, 6))

    for case_index, Q_matrix in enumerate(oracle_Q_list, start=1):
        print(f"\n=== Case {case_index} ===")

        # Analytical solution
        x_star_analytical = solve_analytical_minimizer(Q_matrix, bias_vector)
        f_star = evaluate_quadratic(Q_matrix, bias_vector, x_star_analytical)
        print("Analytical x*:", x_star_analytical)
        print("Analytical f(x*):", f_star)

        # Gradient descent with exact line search
        init_x = np.zeros(2)
        x_history, _ = gradient_descent_exact_line_search(Q_matrix, bias_vector, init_x)
        x_final = x_history[-1]
        error_norm = np.linalg.norm(x_final - x_star_analytical)
        f_final_gd = evaluate_quadratic(Q_matrix, bias_vector, x_final)
        print("x* from exact line search:", x_final)
        print("f(x*) from exact line search:", f_final_gd)
        print("||x_gd - x*||:", error_norm)


        # Plot ||x(k) - x*||
        convergence_errors = [np.linalg.norm(x - x_star_analytical) for x in x_history]
        plt.plot(convergence_errors, label=f"Case {case_index}")

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("||x(k) - x*|| (log scale)")
    plt.title("Convergence of Gradient Descent with Exact Line Search")
    plt.legend()
    plt.grid(True)
    plt.savefig("Q1_convergence_plot.png")  
    plt.show()

if __name__ == "__main__":
    question1_main()

# Question 2: Gradient Descent with Various Line Searches
def armijo_line_search(f_func, grad_func, x_current, p, alpha_init=1e-3, c1=1e-4, tau=2.0, max_iter=1000):
    """
    Forward expansion Armijo line search.
    We Start small and expands until condition fails, then rolls back.
    """
    alpha = alpha_init
    fx = f_func(x_current)
    grad_fx = grad_func(x_current)
    directional_deriv = grad_fx.T @ p
    oracle_calls = 1  # f(x_current) already computed

    if directional_deriv >= 0:
        return 0.0, oracle_calls

    last_good_alpha = 0.0
    for _ in range(max_iter):
        x_new = x_current + alpha * p
        fx_new = f_func(x_new)
        oracle_calls += 1

        if fx_new <= fx + c1 * alpha * directional_deriv:
            # Armijo holds → update last good and expand further
            last_good_alpha = alpha
            alpha *= tau
        else:
            # Condition broke → return last safe alpha
            return last_good_alpha, oracle_calls

    # If never violated, return the last good alpha
    return last_good_alpha, oracle_calls


def backtracking_line_search(f_func, grad_func, x_current, p, alpha_init=1.0, c=1e-4, rho=0.5, max_iter=1000):
    alpha = alpha_init
    fx = f_func(x_current)
    grad_fx = grad_func(x_current)
    oracle_calls = 2

    for _ in range(max_iter):
        x_new = x_current + alpha * p
        fx_new = f_func(x_new)
        oracle_calls += 1
        if fx_new <= fx + c * alpha * grad_fx.T @ p:
            break
        alpha *= rho
    return alpha, oracle_calls

def armijo_goldstein_line_search(f_func, grad_func, x_current, p, alpha_init=1.0, c1=1e-4, c2=0.9, rho=0.5, max_iter=1000):
    alpha = alpha_init
    fx = f_func(x_current)
    grad_fx = grad_func(x_current)
    oracle_calls = 2
    dphi0 = grad_fx.T @ p
    
    for i in range(max_iter):
        x_new = x_current + alpha * p
        fx_new = f_func(x_new)
        oracle_calls += 1
        if fx_new <= fx + c1 * alpha * dphi0:
            break
        alpha *= rho
        if i == max_iter - 1:
            return alpha, oracle_calls 

    # Once Armijo is satisfied, check Goldstein
    if fx_new >= fx + c2 * alpha * dphi0:
        return alpha, oracle_calls
    else:
        # If Goldstein is not satisfied, backtrack again from current alpha to find one that is
        alpha_new = alpha * 0.5
        for i in range(max_iter):
            x_new_2 = x_current + alpha_new * p
            fx_new_2 = f_func(x_new_2)
            oracle_calls += 1
            if fx_new_2 >= fx + c2 * alpha_new * dphi0:
                return alpha_new, oracle_calls
            alpha_new *= 0.5
    
    return alpha_new, oracle_calls


def wolfe_line_search(f_func, grad_func, x_current, p,
                      alpha_init=1.0, c1=1e-4, c2=0.9,
                      alpha_max=10.0, max_iter=50):

    fx = f_func(x_current)
    grad_fx = grad_func(x_current)
    dphi0 = grad_fx.T @ p
    oracle_calls = 2

    if dphi0 >= 0:
        return 0.0, oracle_calls

    alpha_prev = 0
    fx_prev = fx
    alpha = alpha_init

    # Bracketing phase
    for i in range(max_iter):
        x_new = x_current + alpha * p
        fx_new = f_func(x_new)
        oracle_calls += 1

        if (fx_new > fx + c1 * alpha * dphi0) or (i > 0 and fx_new >= fx_prev):
            return zoom(f_func, grad_func, x_current, p,
                        alpha_prev, alpha, fx, dphi0, c1, c2, oracle_calls)

        grad_new = grad_func(x_new)
        oracle_calls += 1
        dphi_new = grad_new.T @ p

        if abs(dphi_new) <= -c2 * dphi0:
            return alpha, oracle_calls

        if dphi_new >= 0:
            return zoom(f_func, grad_func, x_current, p,
                        alpha, alpha_prev, fx, dphi0, c1, c2, oracle_calls)

        alpha_prev = alpha
        fx_prev = fx_new
        alpha = min(2 * alpha, alpha_max)

    return alpha, oracle_calls


def zoom(f_func, grad_func, x_current, p,
         alpha_lo, alpha_hi, fx0, dphi0,
         c1, c2, oracle_calls, max_iter=50):

    for _ in range(max_iter):
        alpha = 0.5 * (alpha_lo + alpha_hi)
        x_new = x_current + alpha * p
        fx_new = f_func(x_new)
        oracle_calls += 1

        x_lo = x_current + alpha_lo * p
        fx_lo = f_func(x_lo)
        oracle_calls += 1

        if (fx_new > fx0 + c1 * alpha * dphi0) or (fx_new >= fx_lo):
            alpha_hi = alpha
        else:
            grad_new = grad_func(x_new)
            oracle_calls += 1
            dphi_new = grad_new.T @ p

            if abs(dphi_new) <= -c2 * dphi0:
                return alpha, oracle_calls

            if dphi_new * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

        if abs(alpha_hi - alpha_lo) < 1e-9:
            return alpha, oracle_calls

    return alpha, oracle_calls



# Gradient Descent Function

def gradient_descent_line_search(f_func, grad_func, x0, method='armijo', max_iterations=1000, tol=1e-9):
    x_current = x0.copy()
    step_sizes = []
    oracle_calls_total = 0

    for k in range(max_iterations):
        grad = grad_func(x_current)
        if np.linalg.norm(grad) < tol:
            break

        p = -grad  # search direction

        if method == 'armijo':
            alpha, calls = armijo_line_search(f_func, grad_func, x_current, p)
        elif method == 'backtracking':
            alpha, calls = backtracking_line_search(f_func, grad_func, x_current, p)
        elif method == 'armijo_goldstein':
            alpha, calls = armijo_goldstein_line_search(f_func, grad_func, x_current, p)
        elif method == 'wolfe':
            alpha, calls = wolfe_line_search(f_func, grad_func, x_current, p)
        else:
            raise ValueError("Unknown line search method")

        oracle_calls_total += calls
        step_sizes.append(alpha)
        x_current = x_current + alpha * p

    f_final = f_func(x_current)
    oracle_calls_total += 1  # final f(x) evaluation
    return x_current, f_final, step_sizes, oracle_calls_total


# Main Function for Question 2

def question2_main():
    sr_number = 25846
    dim = 5
    x0 = np.zeros((dim, 1))

    # Oracle wrapper
    f_oracle = lambda x: oq2f(sr_number, x)
    g_oracle = lambda x: oq2g(sr_number, x)

    methods = ['armijo', 'armijo_goldstein', 'wolfe', 'backtracking']
    results = {}

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    for method in methods:
        x_star, f_star, step_sizes, oracle_calls = gradient_descent_line_search(
            f_oracle, g_oracle, x0, method=method
        )
        results[method] = {
            'x_star': x_star,
            'f_star': f_star,
            'step_sizes': step_sizes,
            'oracle_calls': oracle_calls
        }
        print(f"\n=== Method: {method} ===")
        print("x*:", x_star)
        print("f(x*):", f_star)
        print("Oracle calls:", oracle_calls)

        plt.plot(step_sizes, label=method)

    plt.xlabel("Iteration")
    plt.ylabel("Step size α")
    plt.title("Step Sizes for Different Line Search Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig("Q2_step_sizes_plot.png")

 
    plt.figure(figsize=(10, 6))
    for method in methods:
        step_sizes = results[method]['step_sizes']
        
        
        plt.plot(step_sizes[:210], label=method)

    plt.xlabel("Iteration")
    plt.ylabel("Step size α")
    plt.title("Step Sizes for Different Line Search Methods (Zoomed In)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Q2_step_sizes_plot_zoomed.png")
    

    plt.show()

if __name__ == "__main__":
    question2_main()

# Question 3: Optimization Functions

# Objective and Gradient functions for the least squares problem
def objective_least_squares(A, b, x):
    return 0.5 * np.linalg.norm(A @ x - b)**2

def gradient_least_squares(A, b, x):
    return A.T @ (A @ x - b)

# Armijo (Backtracking) Line Search
def armijo_ls(f_func, grad_func, x_current, p, alpha_init=1.0, c1=1e-4, rho=0.5, max_iter=1000):
    alpha = alpha_init
    fx = f_func(x_current)
    grad_fx = grad_func(x_current)
    directional_deriv = grad_fx.T @ p
    
    for _ in range(max_iter):
        x_new = x_current + alpha * p
        fx_new = f_func(x_new)
        if fx_new <= fx + c1 * alpha * directional_deriv:
            break
        alpha *= rho
    return alpha

# Gradient Descent solver
def solve_with_gd(A, b, line_search_func, max_iterations=1000, tol=1e-9):
    m, n = A.shape
    x_current = np.zeros((n, 1))

    f_wrapper = lambda x: objective_least_squares(A, b, x)
    g_wrapper = lambda x: gradient_least_squares(A, b, x)
    
    for _ in range(max_iterations):
        grad = g_wrapper(x_current)
        if np.linalg.norm(grad) < tol:
            break
        
        p = -grad
        alpha = line_search_func(f_wrapper, g_wrapper, x_current, p)
        x_current += alpha * p

    return x_current

# Code to generate the time comparison table and CSV output
def run_benchmarks():
    
    SR_NUMBER = 25846
    
    def get_oracle_data(sr_number):
        np.random.seed(sr_number)
        m, n = 500, 300
        A = np.random.rand(m, n)
        b = np.random.rand(m, 1)
        return A, b
    
    A_main, b_main = oq3(SR_NUMBER)
    x_star = solve_with_gd(A_main, b_main, armijo_ls)

    np.savetxt('x_star_solution.csv', x_star, delimiter=',')
    print("Solution vector x* saved to x_star_solution.csv")

    sizes = 2**np.arange(1, 14)
    inversion_times = []
    optimization_times = []
    
    print("\nRunning benchmarks for time comparison...")
    for m in sizes:
        A = np.random.rand(m, m)
        b = np.random.rand(m, 1)

        # Time Matrix Inversion
        start_time = time.time()
        try:
            np.linalg.solve(A, b)
            inversion_times.append(time.time() - start_time)
        except np.linalg.LinAlgError:
            inversion_times.append(np.nan)

        # Time Optimization (Gradient Descent)
        start_time = time.time()
        solve_with_gd(A, b, armijo_ls)
        optimization_times.append(time.time() - start_time)
    

    print("\nTime Comparison Table:")
    print("-" * 50)
    print(f"{'Matrix Size (m)':<20s} | {'Inversion Time (s)':<20s} | {'Optimization Time (s)':<20s}")
    print("-" * 50)
    for i in range(len(sizes)):
        inv_time = f"{inversion_times[i]:.6f}" if not np.isnan(inversion_times[i]) else "N/A"
        opt_time = f"{optimization_times[i]:.6f}"
        print(f"{sizes[i]:<20d} | {inv_time:<20s} | {opt_time:<20s}")
    print("-" * 50)

if __name__ == "__main__":
   run_benchmarks()
