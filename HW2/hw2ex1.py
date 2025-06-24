import numpy as np
import math
import time

# --- Function and Exact Derivative ---
def u_func(x, k):
    """Calculates the function u(x) = exp(k*sin(x))."""
    return np.exp(k * np.sin(x))

def exact_derivative(x, k):
    """Calculates the exact derivative of u(x) = exp(k*sin(x))."""
    return k * np.cos(x) * np.exp(k * np.sin(x))

# --- Differentiation Matrix (Even Method - cot formula) ---
def build_diff_matrix_even(N_even):
    """
    Builds the 'even' differentiation matrix (cot formula).
    N_even is the number of points (typically even).
    Matrix D_ij acts on vector u to give derivative at point i.
    Uses (-1)**(i+j) in exponent as requested.
    """
    D = np.zeros((N_even, N_even))
    indices = np.arange(N_even)
    i_idx, j_idx = np.meshgrid(indices, indices, indexing='ij') # row i, col j

    diff = i_idx - j_idx
    sum_idx = i_idx + j_idx # For the exponent (-1)**(i+j)
    mask_diag = (diff == 0)
    mask_nondiag = ~mask_diag

    with np.errstate(divide='ignore', invalid='ignore'):
        angle = diff[mask_nondiag] * np.pi / N_even
        cot_term = 1.0 / np.tan(angle)
        # Formula: D_ij = 0.5 * (-1)^(i+j) * cot((i-j)*pi / N_even)
        D[mask_nondiag] = 0.5 * ((-1)**sum_idx[mask_nondiag]) * cot_term
    return D

# --- Differentiation Matrix (Odd Method - sin formula) ---
def build_diff_matrix_odd(N_odd):
    """
    Builds the 'odd' differentiation matrix (sin formula from Ex3).
    Input N_odd defines matrix size N+1 x N+1.
    Matrix D_ji acts on vector u to give derivative at point j.
    """
    num_points = N_odd + 1
    D = np.zeros((num_points, num_points))
    indices = np.arange(num_points)
    j_idx, i_idx = np.meshgrid(indices, indices, indexing='ij') # row j, col i

    diff = j_idx - i_idx
    sum_idx = j_idx + i_idx # For the exponent (-1)**(j+i)
    mask_diag = (diff == 0)
    mask_nondiag = ~mask_diag

    with np.errstate(divide='ignore', invalid='ignore'):
        angle = diff[mask_nondiag] * np.pi / num_points
        denom = 2.0 * np.sin(angle)
         # Handle potential division by zero if sin(angle) is exactly zero
        # This should only happen for i=j, which is masked out.
        # However, for safety:
        denom[np.abs(denom) < 1e-15] = 1e-15 # Avoid exact zero division

        # Formula: D_ji = (-1)**(j+i) / (2 * sin((j-i)*pi/(N_odd+1)) )
        D[mask_nondiag] = ((-1)**sum_idx[mask_nondiag]) / denom
    return D

# --- Error Calculation (common for both) ---
def calculate_max_relative_error(u_numerical, u_exact):
    """Calculates the maximum relative pointwise error."""
    abs_error = np.abs(u_numerical - u_exact)
    abs_exact = np.abs(u_exact)
    max_rel_err = 0.0
    idx_nonzero = abs_exact > 1e-12

    if np.any(idx_nonzero):
        rel_error = np.zeros_like(u_exact)
        rel_error[idx_nonzero] = abs_error[idx_nonzero] / abs_exact[idx_nonzero]
        max_rel_err = np.max(rel_error[idx_nonzero])

    idx_zero = ~idx_nonzero
    if np.any(idx_zero):
        max_abs_err_at_zero = np.max(abs_error[idx_zero])
        max_norm_exact = np.max(abs_exact)
        if max_norm_exact > 1e-12:
             max_rel_err = max(max_rel_err, max_abs_err_at_zero / max_norm_exact)
        elif max_abs_err_at_zero > 1e-12:
             max_rel_err = max(max_rel_err, np.inf)

    if np.max(abs_exact) < 1e-12:
       max_rel_err = np.max(abs_error)
    return max_rel_err

# --- Main Script ---
k_values = [2, 4, 6, 8, 10, 12]
target_error = 1e-5
N_limit = 200 # Upper limit for N search

min_N_results_even = {}
min_N_results_odd = {}

print("Calculating minimum points required for max relative error < 1e-5")
print(f"Target Error: {target_error:.1e}")
print("-" * 70)

# --- Calculate for Even Method ---
print("Running Even Method (cot formula)...")
start_time_even = time.time()
for k in k_values:
    print(f"  Processing k = {k}...")
    min_N_found = -1
    # Search N_even, starting at 4, incrementing by 2
    for N_even in range(4, N_limit + 1, 2):
        x = np.linspace(0, 2 * np.pi * (N_even - 1) / N_even, N_even)
        u = u_func(x, k)
        du_exact = exact_derivative(x, k)
        D_even = build_diff_matrix_even(N_even)
        du_numerical = D_even @ u
        max_rel_err = calculate_max_relative_error(du_numerical, du_exact)

        if max_rel_err < target_error:
            # Optional: Check N_even-2 if N_even > 4 to ensure minimum
            if N_even > 4:
                 x_prev = np.linspace(0, 2 * np.pi * (N_even - 3) / (N_even-2), N_even-2)
                 u_prev = u_func(x_prev, k)
                 du_exact_prev = exact_derivative(x_prev, k)
                 D_even_prev = build_diff_matrix_even(N_even-2)
                 du_numerical_prev = D_even_prev @ u_prev
                 prev_error = calculate_max_relative_error(du_numerical_prev, du_exact_prev)
                 if prev_error < target_error:
                      # If N-2 also worked, the actual minimum is N-2
                      min_N_found = N_even - 2
                      print(f"    Found min N_even = {min_N_found} for k = {k} (Error: {prev_error:.2e})")
                      break # Stop inner loop

            # If N-2 didn't work or N=4, then N is the minimum
            if min_N_found == -1:
                 min_N_found = N_even
                 print(f"    Found min N_even = {min_N_found} for k = {k} (Error: {max_rel_err:.2e})")
                 break # Stop inner loop
        if N_even == N_limit:
             print(f"    Minimum N_even not found for k = {k} below N={N_limit}.")

    min_N_results_even[k] = min_N_found if min_N_found != -1 else f'>{N_limit}'
end_time_even = time.time()
print(f"Even Method calculation finished in {end_time_even - start_time_even:.2f} seconds.\n")

# --- Calculate for Odd Method ---
print("Running Odd Method (sin formula)...")
start_time_odd = time.time()
for k in k_values:
    print(f"  Processing k = {k}...")
    min_N_found = -1
    # Search N_odd, starting at 4? incrementing by 2? (N+1 points used)
    for N_odd in range(4, N_limit + 1, 2):
        num_points = N_odd + 1
        # Grid x_j = 2*pi*j / (N_odd+1) for j=0...N_odd
        x = np.linspace(0, 2 * np.pi * N_odd / num_points, num_points)
        u = u_func(x, k)
        du_exact = exact_derivative(x, k)
        D_odd = build_diff_matrix_odd(N_odd)
        du_numerical = D_odd @ u
        max_rel_err = calculate_max_relative_error(du_numerical, du_exact)

        if max_rel_err < target_error:
             # Optional check for N_odd-2
            if N_odd > 4:
                 num_points_prev = N_odd - 2 + 1
                 x_prev = np.linspace(0, 2 * np.pi * (N_odd - 2) / num_points_prev, num_points_prev)
                 u_prev = u_func(x_prev, k)
                 du_exact_prev = exact_derivative(x_prev, k)
                 D_odd_prev = build_diff_matrix_odd(N_odd-2)
                 du_numerical_prev = D_odd_prev @ u_prev
                 prev_error = calculate_max_relative_error(du_numerical_prev, du_exact_prev)
                 if prev_error < target_error:
                      min_N_found = N_odd - 2
                      print(f"    Found min N_odd = {min_N_found} for k = {k} (Error: {prev_error:.2e}) ({min_N_found+1} points)")
                      break

            if min_N_found == -1:
                 min_N_found = N_odd
                 print(f"    Found min N_odd = {min_N_found} for k = {k} (Error: {max_rel_err:.2e}) ({min_N_found+1} points)")
                 break # Stop inner loop
        if N_odd == N_limit:
             print(f"    Minimum N_odd not found for k = {k} below N={N_limit}.")

    min_N_results_odd[k] = min_N_found if min_N_found != -1 else f'>{N_limit}'
end_time_odd = time.time()
print(f"Odd Method calculation finished in {end_time_odd - start_time_odd:.2f} seconds.\n")


# --- Comparison ---
print("\n--- Comparison ---")
print("Compares total grid points used (N_even vs N_odd+1)")
print("-" * 70)
print("k  | Min Points (Even Method, N_even) | Min Points (Odd Method, N_odd+1) | More Accurate?")
print("---|----------------------------------|-----------------------------------|---------------")
for k in k_values:
    n_even = min_N_results_even.get(k, 'N/A')
    n_odd_val = min_N_results_odd.get(k, 'N/A')
    n_odd_plus_1 = 'N/A'
    winner = 'N/A'

    if isinstance(n_odd_val, int):
        n_odd_plus_1 = n_odd_val + 1

    if isinstance(n_even, int) and isinstance(n_odd_plus_1, int):
        if n_even < n_odd_plus_1:
            winner = "Even"
        elif n_odd_plus_1 < n_even:
             winner = "Odd"
        else:
             winner = "Equal"

    print(f"{k:<3}| {n_even:<32} | {n_odd_plus_1:<33} | {winner}")
print("-" * 70)