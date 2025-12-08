import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError, solve
from scipy.linalg import eigh

SRNO = 25846
sys.path.insert(0, os.path.abspath("oracle_CMO2025A2_py310"))
from oracle_CMO2025A2_py310.oracle_final_CMOA2 import f2, f5

# --- Rosenbrock Functions ---

def get_rosenbrock_grad(x):
    x0, x1 = x[0], x[1]
    grad = np.zeros(2)
    grad[0] = -400.0 * x0 * (x1 - x0**2) - 2.0 * (1.0 - x0)
    grad[1] = 200.0 * (x1 - x0**2)
    return grad

def get_rosenbrock_hessian(x):
    x0, x1 = x[0], x[1]
    H = np.zeros((2, 2))
    H[0, 0] = 2.0 - 400.0 * (x1 - 3.0 * x0**2)
    H[0, 1] = -400.0 * x0
    H[1, 0] = H[0, 1]
    H[1, 1] = 200.0
    return H

def get_rosenbrock_val(x):
    x0, x1 = x[0], x[1]
    return (1.0 - x0)**2 + 100.0 * (x1 - x0**2)**2

# --- Solvers ---

def solve_newton(grad_func, hess_func, x0, tol=1e-8, max_iter=100):
    x_curr = x0.copy()
    trajectory = [x_curr.copy()]
    iter_count = 0
    
    for i in range(max_iter):
        grad = grad_func(x_curr)
        if np.dot(grad, grad)**0.5 < tol:
            iter_count = i
            break
            
        hess = hess_func(x_curr)
        try:
            p_step = solve(hess, -grad)
        except LinAlgError:
            p_step = np.zeros_like(x_curr)
            
        x_curr = x_curr + p_step
        trajectory.append(x_curr.copy())
        iter_count = i + 1

    return x_curr, iter_count, trajectory

def solve_coordinate_descent(A, b, x0=None, max_iter=100):
    print("\n***** CD_SOLVE *****")
    print("Step | Alpha | Numerator | Lambda")
    print("-" * 50)

    n = len(b)
    x = np.zeros(n, dtype=float) if x0 is None else x0.copy()
    eig_vals, eig_vecs = eigh(A)
    
    alpha_log = []
    num_log = []
    eig_log = []
    
    steps = len(eig_vals)
    for step in range(steps):
        d = eig_vecs[:, step]
        r = b - A @ x
        
        num = r.T @ d
        den = d.T @ (A @ d)
        alpha = num / den
        
        x = x + alpha * d
        
        alpha_log.append(alpha)
        num_log.append(num)
        eig_log.append(eig_vals[step])
        
        if step < 7:
             print(f"{step} | {alpha:.4f} | {num:.4f} | {eig_vals[step]:.4f}")

    print("-" * 50)
    return x, alpha_log, num_log, eig_log

def gram_schmidt(vectors, A):
    m = len(vectors)
    D = [vectors[0].copy()]
    
    for k in range(1, m):
        p_k = vectors[k]
        d_k = p_k.copy()
        
        for i in range(k):
            d_i = D[i]
            Ad_i = A.dot(d_i)
            beta = (p_k.T @ Ad_i) / (d_i.T @ Ad_i)
            d_k = d_k - beta * d_i
                
        D.append(d_k)
    return D

def solve_pcg(A, b, tol=1e-6, max_iter=10000, use_rel_tol=False):
    x = np.zeros(len(b))
    diag_A = np.diag(A)
    M_inv = 1.0 / (diag_A + 1e-13) 
    
    r = b - A @ x
    z = M_inv * r
    p = z.copy()
    
    r_norm_sq = np.dot(r, r)
    r0_norm = np.sqrt(r_norm_sq)
    res = [r0_norm]
    
    for k in range(max_iter):
        r_curr = np.sqrt(r_norm_sq)
        if use_rel_tol and (r_curr / r0_norm < tol): break
        if not use_rel_tol and (r_curr < tol): break
                
        Ap = A @ p
        alpha = np.dot(r, z) / np.dot(p, Ap)
        
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        res.append(np.linalg.norm(r_new))
        
        z_new = M_inv * r_new
        beta = np.dot(r_new, z_new) / np.dot(r, z)
        
        p = z_new + beta * p
        r = r_new
        z = z_new
        r_norm_sq = np.dot(r, r)
        
    return x, k, np.array(res)

def solve_cg(A, b, tol=1e-6, max_iter=10000, log_dirs=False, use_rel_tol=False):
    x = np.zeros(len(b))
    r = b - A @ x
    p = r.copy()
    
    r_sq = np.dot(r, r)
    r0 = np.sqrt(r_sq)
    
    res = [r0]
    r_log, p_log = [], []
    
    if log_dirs:
        r_log.append(r.copy())
        p_log.append(p.copy())
        
    k = 0
    for k in range(max_iter):
        curr_norm = np.sqrt(r_sq)
        if use_rel_tol and (curr_norm / r0 < tol): break
        if not use_rel_tol and (curr_norm < tol): break
        
        Ap = A @ p
        alpha = r_sq / np.dot(p, Ap)
        
        x = x + alpha * p
        r = r - alpha * Ap
        
        r_sq_new = np.dot(r, r)
        res.append(np.sqrt(r_sq_new))
        
        beta = r_sq_new / r_sq
        p = r + beta * p
        r_sq = r_sq_new
        
        if log_dirs:
            r_log.append(r.copy())
            p_log.append(p.copy())
            
    if log_dirs:
        return x, k, np.array(res), r_log, p_log
    return x, k, np.array(res)

# --- Main ---

if __name__ == "__main__":
    print("| Q1 Part 1C |")
    A, b = f2(srno=SRNO, subq=True)
    
    x_cd, _, _, _ = solve_coordinate_descent(A, b, max_iter=7)

    print("\n***** Q1 Part-2: CG *****")
    x_cg, iters_cg, _, r_list, p_list = solve_cg(A, b, log_dirs=True)
    print(f"CG converged in {iters_cg} iterations.")

    D_list = gram_schmidt(p_list, A)
    np.savetxt("plist_ready.txt", np.vstack(p_list))
    np.savetxt("dlist_ready.txt", np.vstack(D_list))

    print("\n***** Q1 Part 4: Cosine Similarities *****")
    coss_sim = []
    fmt = {'float_kind': lambda x: f"{x:+.18e}"}
    
    for k in range(len(p_list)):
        p_k = p_list[k]
        d_k = D_list[k]
        
        Ap = A @ p_k
        Ad = A @ d_k
        
        num = p_k.T @ Ad
        den = np.sqrt(p_k.T @ Ap) * np.sqrt(d_k.T @ Ad)
        
        val = 0.0 if np.isclose(den, 0) else num / den
        coss_sim.append(val)
        
        print(f"k={k}: {np.array2string(np.array(coss_sim), formatter=fmt, separator=', ')}")

    print("\n***** Q1 Part-3: Matrix M *****")
    D_hat = []
    for d in D_list:
        n_A = np.sqrt(d.T @ (A @ d))
        D_hat.append(d / n_A if not np.isclose(n_A, 0) else d)
        
    m_size = len(p_list)
    M = np.zeros((m_size, m_size))
    
    for i in range(m_size):
        for j in range(m_size):
            M[i, j] = D_hat[i].T @ (A @ D_hat[j])
            
    print("Matrix M:")
    print(np.array2string(M, formatter=fmt, separator=', '))
    
    print("| Q2 |")
    A_op, b_op = f5(SRNO)

    print("\n***** Q2 Part-1: Std CG *****")
    _, i1, res1 = solve_cg(A_op, b_op, use_rel_tol=True)
    print(f"Std CG iters: {i1}")
    
    print("\n***** Q2 Part-2: Fast CG *****")
    _, i2, res2 = solve_pcg(A_op, b_op, use_rel_tol=True)
    print(f"Fast CG iters: {i2}")

    # Plots
    rng1 = np.arange(len(res1))
    plt.figure(figsize=(9, 5))
    plt.semilogy(rng1, res1, 's-', color='teal', markersize=4, lw=1.5, label=f'Normal CG ({i1})')
    plt.title("Q2 Part 1: Main CG Residuals")
    plt.ylabel("||r_k|| (log)")
    plt.xlabel("Iteration")
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig('maincg.png', dpi=150)
    plt.close()

    rng2 = np.arange(len(res2))
    plt.figure(figsize=(9, 5))
    plt.semilogy(rng2, res2, 'x--', color='navy', markersize=5, lw=1.2, label='Fast CG')
    plt.semilogy(rng1, res1, 'o-', color='darkorange', markersize=4, lw=0.5, label='Main CG')
    plt.ylabel("||r_k|| (log)")
    plt.xlabel("Iteration")
    plt.title("Q2 Part 2: CG vs Fast CG")
    plt.grid(True, which='both', linestyle=':', alpha=0.8)
    plt.legend()
    plt.savefig('comparison_cg_fast.png', dpi=150)
    plt.close()

    print("| Q3 |")
    starts = [
        np.array([2.0, 2.0]), np.array([5.0, 5.0]),
        np.array([50.0, 60.0]), np.array([-10.0, -4.0])
    ]
    x_star = np.array([1.0, 1.0])
    res_data = []

    for s in starts:
        xf, it, tr = solve_newton(get_rosenbrock_grad, get_rosenbrock_hessian, s)
        res_data.append((xf, it, tr, s))

    print("***** Q3: Error Plot *****")
    plt.figure(figsize=(11, 7))
    for (xf, it, tr, s) in res_data:
        errs = [np.linalg.norm(t - x_star) for t in tr]
        plt.semilogy(errs, marker="o", lw=1.3, label=f"start={s}")
        
    plt.title("Newton Convergence")
    plt.ylabel("||x - x*|| (log)")
    plt.xlabel("Iter")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig("error_plot.png", dpi=200)
    plt.close()

    print("***** Q3: Contour Plots *****")
    gx = np.linspace(-10, 10, 100)
    gy = np.linspace(-10, 10, 100)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    
    for i in range(100):
        for j in range(100):
            GZ[i, j] = get_rosenbrock_val([GX[i, j], GY[i, j]])

    cols = ['r', 'b', 'g', 'm']
    mrks = ['o', 's', '^', 'D']

    for i, (xf, it, tr, s) in enumerate(res_data):
        plt.figure(figsize=(8, 8))
        CS = plt.contour(GX, GY, GZ, levels=np.logspace(0, 4, 15), colors='blue', alpha=0.4)
        plt.clabel(CS, inline=True, fontsize=9, fmt='%1.0f')
        
        tr = np.array(tr)
        c = cols[(i + 2) % 4]
        m = mrks[(i + 3) % 4]
        
        plt.plot(tr[:, 0], tr[:, 1], f'{c}--', lw=1.5, label='Path')
        plt.plot(tr[:, 0], tr[:, 1], f'{c}{m}', markersize=7, label='Iterates')
        plt.plot(s[0], s[1], f'y{mrks[i%4]}', markersize=10, label='Start')
        plt.plot(x_star[0], x_star[1], 'k*', markersize=12, label='Min')
        
        plt.title(f'Start: {s} | Iters: {it}')
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend()
        plt.savefig(f'contor_plot_{i+1}.png', dpi=180)
        plt.close()

    print("Done.")