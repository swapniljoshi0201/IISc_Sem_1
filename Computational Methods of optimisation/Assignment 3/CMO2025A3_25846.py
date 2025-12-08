import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
from numpy.linalg import norm

SRNO = 25846
sys.path.insert(0, os.path.abspath("oracle_2025A3"))
from oracle_2025A3 import f1

def check_kkt(X, y, beta, lam, tol=1e-6):
    res = X @ beta - y
    grad = X.T @ res
    n_feat = X.shape[1]
    is_valid = True

    for i in range(n_feat):
        b_i = beta[i]
        g_i = grad[i]
        
        if np.abs(b_i) > tol:
            expected = -lam * np.sign(b_i)
            if not np.isclose(g_i, expected, atol=tol):
                print(f"KKT Fail (Non-zero): beta_{i}, Grad {g_i:.4f} != {expected:.4f}")
                is_valid = False
        else:
            if np.abs(g_i) > lam + tol:
                print(f"KKT Fail (Zero): beta_{i}, |Grad| {np.abs(g_i):.4f} > {lam}")
                is_valid = False
                
    return is_valid

def get_hyperplane():
    pt_a = np.array([1.0, 0.0])
    pt_b = np.array([3.0, 0.0])
    
    diff = pt_b - pt_a
    n = diff / norm(diff)
    mid = (pt_a + pt_b) * 0.5
    c = n @ mid
    
    return n, c, pt_a, pt_b

def check_farkas(A=None, b=None):
    if A is None or b is None:
        A = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        b = np.array([-1.0, 0.0, 0.0])

    m, n = A.shape
    x = cp.Variable(n)
    prob_p = cp.Problem(cp.Minimize(0), [A @ x <= b])
    prob_p.solve()
    
    if prob_p.status == 'optimal':
        return True, None, "Primal feasible"

    elif prob_p.status == 'infeasible':
        y = cp.Variable(m)
        consts = [y >= 0, A.T @ y == 0, b.T @ y <= -1e-7]
        prob_d = cp.Problem(cp.Minimize(0), consts)
        prob_d.solve()
        
        if prob_d.status == 'optimal':
            return False, y.value, "Primal infeasible, Farkas certificate found"
        
    return False, None, f"Status: {prob_p.status}"

def proj_box(y, low=None, high=None):
    if low is None: low = np.array([-3.0, 0.0])
    if high is None: high = np.array([3.0, 4.0])
    return np.clip(y, low, high)

def proj_circle(y, center=None, r=5.0):
    if center is None: center = np.array([0.0, 0.0])
    shift = y - center
    d = norm(shift)
    
    if d <= r:
        return y
    return center + shift * (r / d)

def run_q3():
    c1 = np.array([0.0, 0.0])
    r1 = 5.0
    pts_c1 = [np.array([6.0, 0.0]), np.array([0.0, -7.0]), 
              np.array([4.0, 3.0]), np.array([3.0, 3.0])]
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Circle(c1, r1, color='blue', alpha=0.2, label='Safe Zone C1'))
    
    for p in pts_c1:
        proj = proj_circle(p, c1, r1)
        lbl_o = 'Original' if p[0] == 6.0 else None
        lbl_p = 'Projection' if p[0] == 6.0 else None
        
        ax.plot(p[0], p[1], 'ro', label=lbl_o)
        ax.plot(proj[0], proj[1], 'go', label=lbl_p)
        ax.arrow(p[0], p[1], proj[0]-p[0], proj[1]-p[1], head_width=0.2, fc='k', ec='k', ls='--')

    ax.set_aspect('equal')
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
    ax.grid(True, ls=':')
    ax.legend(loc='lower right')
    ax.set_title('Q3.1(a) Circle Projection')
    plt.savefig('q3circle.png')
    plt.close(fig)

    low = np.array([-3.0, 0.0])
    high = np.array([3.0, 4.0])
    pts_c2 = [np.array([4.0, 2.0]), np.array([-4.0, 5.0]), 
              np.array([0.0, -1.0]), np.array([1.0, 1.0])]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_patch(plt.Rectangle(low, high[0]-low[0], high[1]-low[1], color='green', alpha=0.2, label='Safe Zone C2'))
    
    for p in pts_c2:
        proj = proj_box(p, low, high)
        lbl_o = 'Original' if p[0] == 4.0 else None
        lbl_p = 'Projection' if p[0] == 4.0 else None
        
        ax.plot(p[0], p[1], 'ro', label=lbl_o)
        ax.plot(proj[0], proj[1], 'go', label=lbl_p)
        ax.arrow(p[0], p[1], proj[0]-p[0], proj[1]-p[1], head_width=0.2, fc='k', ec='k', ls='--')

    ax.set_aspect('equal')
    ax.set_xlim(-5, 6); ax.set_ylim(-2, 6)
    ax.grid(True, ls=':')
    ax.legend(loc='lower right')
    ax.set_title('Q3.1(b) Box Projection')
    plt.savefig('q3box.png')
    plt.close(fig)

    n, c, a, b = get_hyperplane()
    print(f"Hyperplane n: {n}, c: {c}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_patch(plt.Circle((0, 0), 1.0, color='blue', alpha=0.3, label='Group A'))
    ax.fill_betweenx(np.linspace(-5, 5, 100), 3, 6, color='red', alpha=0.3, label='Group B')
    
    intercept = c / n[0]
    ax.axvline(x=intercept, color='k', ls='--', label=f'Hyperplane x={intercept}')
    ax.plot(a[0], a[1], 'bo'); ax.plot(b[0], b[1], 'ro')
    
    ax.set_xlim(-2, 6); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.grid(True, ls=':')
    ax.legend()
    ax.set_title('Q3.2 Separating Hyperplane')
    plt.savefig('q3_hyperplane.png')
    plt.close(fig)

    feasible, y_cert, info = check_farkas()
    print(info)

def run_q1(X, y):
    n, m = X.shape
    lams = [1e-2, 1e-1, 1]
    
    beta = cp.Variable(m)
    lam_p = cp.Parameter(nonneg=True)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(X @ beta - y) + lam_p * cp.norm(beta, 1)))
    
    nz_counts = []
    res_beta = {}
    
    for l in lams:
        lam_p.value = l
        prob.solve(solver=cp.SCS, eps=1e-8)
        
        if prob.status == 'optimal':
            b_val = beta.value
            res_beta[l] = b_val
            nz = np.sum(np.abs(b_val) > 1e-5)
            nz_counts.append(nz)
            print(f"Lambda: {l} | Non-zeros: {nz}")
            check_kkt(X, y, b_val, l)
        else:
            nz_counts.append(0)

    plt.figure(figsize=(10, 6))
    plt.plot(lams, nz_counts, 'o-')
    plt.xscale('log')
    plt.grid(True, linestyle='--')
    plt.savefig('q1_plot.png')
    plt.close()

    X_c = np.hstack((X, X[:, 0:1]))
    beta_c = cp.Variable(m + 1)
    lam_pc = cp.Parameter(nonneg=True)
    prob_c = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(X_c @ beta_c - y) + lam_pc * cp.norm(beta_c, 1)))
    
    for l in lams:
        lam_pc.value = l
        prob_c.solve(solver=cp.SCS, eps=1e-8)
        
        if prob_c.status == 'optimal':
            b = beta_c.value
            print(f"L={l} | b[0]:{b[0]:.4f}, b[last]:{b[-1]:.4f}, Sum:{b[0]+b[-1]:.4f}")

def run_q2(X, y):
    n, m = X.shape
    lams = [0.01, 0.1, 1.0]
    
    beta = cp.Variable(m)
    lam_p = cp.Parameter(nonneg=True)
    prob_p = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(X @ beta - y) + lam_p * cp.norm(beta, 1)))
    
    u = cp.Variable(n)
    lam_d = cp.Parameter(nonneg=True)
    prob_d = cp.Problem(cp.Maximize(-0.5 * cp.sum_squares(u) + y.T @ u), [cp.norm(X.T @ u, "inf") <= lam_d])
    
    for l in lams:
        lam_p.value = l
        prob_p.solve(solver=cp.SCS, eps=1e-8)
        
        lam_d.value = l
        prob_d.solve(solver=cp.SCS, eps=1e-8)
        
        if prob_p.status == 'optimal' and prob_d.status == 'optimal':
            gap = prob_p.value - prob_d.value
            err = norm(u.value - (y - X @ beta.value))
            print(f"L={l} | Primal:{prob_p.value:.4f} | Dual:{prob_d.value:.4f} | Gap:{gap:.4e} | Consist:{err:.4e}")

if __name__ == "__main__":
    f1(SRNO)
    data = pd.read_csv(f'data_{SRNO}.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    run_q1(X, y)
    run_q2(X, y)
    run_q3()