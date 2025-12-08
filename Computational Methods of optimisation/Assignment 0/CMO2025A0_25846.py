
import os
import sys

# Add oracle directory to sys.path for importing oracle module
sys.path.insert(0, os.path.abspath("oracle_2025A0"))
from oracle_2025A0 import oracle

# Function to query oracle for its value and slope
def derivative(x):
    f, f_dash = oracle(25846, x)  # SR number 25846
    return f, f_dash

# Gradient descent to find minimum of function from oracle
def my_func(start_x=0.0, lr=0.001, max_iters=1000000, tol=1e-8):
    x = start_x  # initial guess 
    for _ in range(max_iters):
        f_val, f_dash = derivative(x)      # get function value and slope at current x
        x_new = x - lr * f_dash            # update x using gradient descent step
        if abs(x_new - x) < tol:           # stop if change in x is very small (converged)
            break
        x = x_new
    f_val, _ = derivative(x)               # get final function value at minimum found
    return x, f_val

x_min, f_min = my_func()
print(f"x* = {x_min} and f(x*) = {f_min}")
