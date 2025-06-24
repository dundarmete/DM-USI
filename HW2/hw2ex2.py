# hw2ex2_matrix_method_corrected.py
import numpy as np
import matplotlib.pyplot as plt
import warnings # Import warnings module

def fourier_diff_matrix(N, L=2*np.pi):
    """
    Constructs the Fourier differentiation matrix for N evenly spaced points
    in [0, L), using the cotangent formula for the 'even' case.
    """
    # This matrix formulation works best for N even.
    if N % 2 != 0:
         warnings.warn(f"N={N} is odd. The standard cotangent Fourier differentiation matrix formula is typically derived for and works best with even N. Results might be less accurate.", UserWarning)
         # Proceeding anyway, but be aware of potential issues.

    DN = np.zeros((N, N), dtype=float) # Ensure float type
    # h = L / N # Grid spacing, not directly needed here

    # Scaling factor (from chain rule: d/dx = (2pi/L) * d/dtheta)
    scale_factor = np.pi / L

    col, row = np.meshgrid(np.arange(N), np.arange(N))
    diff = row - col

    # Populate off-diagonal elements
    non_zero_diff_indices = (diff != 0) # Boolean mask for non-diagonal
    diff_vals = diff[non_zero_diff_indices] # Get the actual difference values

    # Calculate cotangent term, handle potential division by zero if sin(arg)=0
    # This happens when pi * diff_vals / N is a multiple of pi, i.e., diff_vals is a multiple of N
    # Since -N < diff_vals < N and diff_vals != 0, this cannot happen.
    cot_term = 1.0 / np.tan(np.pi * diff_vals / N)

    # Calculate (-1)^(j-k) term carefully
    # Use np.power with float base -1.0 to avoid integer power error
    sign_term = np.power(-1.0, diff_vals)

    DN[non_zero_diff_indices] = scale_factor * sign_term * cot_term

    # Diagonal elements are zero (already initialized)
    # np.fill_diagonal(DN, 0) # Redundant if initialized with zeros

    return DN

def fourier_derivative_matrix(f_vals, L=2*np.pi):
    """Computes the derivative using the Fourier differentiation matrix."""
    N = len(f_vals)
    # Ensure f_vals is a float array for matrix multiplication
    f_vals_float = np.asarray(f_vals, dtype=float)
    DN = fourier_diff_matrix(N, L)
    df_numeric = DN @ f_vals_float # Matrix-vector product
    return df_numeric

# --- Rest of the script remains the same ---

# Define functions and their exact derivatives
functions = {
    r'$\cos(10x)$': {'func': lambda x: np.cos(10*x), 'deriv': lambda x: -10*np.sin(10*x)},
    r'$\cos(x/2)$': {'func': lambda x: np.cos(x/2), 'deriv': lambda x: -0.5*np.sin(x/2)},
    r'$x$':        {'func': lambda x: x, 'deriv': lambda x: np.ones_like(x)}
}

# N values (prefer even N for this method)
# Reduced N range for performance, matrix method is O(N^2) to compute derivative, O(N^3) to build matrix initially
N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] # Further reduced for faster execution
# N_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


results = {name: {'N': [], 'L_inf': [], 'L_2': []} for name in functions}
L = 2 * np.pi

print("Calculating errors using Matrix method...")
for N in N_values:
    # Grid (excluding endpoint L, as it's identified with 0)
    x = np.linspace(0, L, N, endpoint=False)

    for name, data in functions.items():
        f = data['func']
        f_prime_exact = data['deriv']

        f_vals = f(x)
        # Use the new matrix-based derivative function
        df_numeric = fourier_derivative_matrix(f_vals, L=L)
        df_exact_vals = f_prime_exact(x)

        error_vec = df_numeric - df_exact_vals
        l_inf_error = np.max(np.abs(error_vec))
        l_2_error = np.sqrt(np.mean(error_vec**2)) # RMS error

        results[name]['N'].append(N)
        results[name]['L_inf'].append(l_inf_error)
        results[name]['L_2'].append(l_2_error)
    print(f"  Completed N={N}")


print("Done.\n")

# Print results table (optional)
print("| Function    | N    | L_infinity Error | L_2 Error (RMS) |")
print("|-------------|------|------------------|-----------------|")
# Find the maximum width needed for N to align the table
max_n_width = len(str(max(N_values))) if N_values else 4
# Find maximum function name length for alignment
max_name_width = max(len(name) for name in functions) if functions else 10

for name in functions:
    print(f"| {name:<{max_name_width}} | {'N':<{max_n_width}} | L_infinity Error | L_2 Error (RMS) |")
    print(f"|{'-'*(max_name_width+2)}|{'-'*(max_n_width+2)}|{'-'*18}|{'-'*17}|") # Adjust separator length
    for i in range(len(N_values)):
        N = results[name]['N'][i]
        l_inf = results[name]['L_inf'][i]
        l_2 = results[name]['L_2'][i]
        print(f"| {name:<{max_name_width}} | {N:<{max_n_width}} | {l_inf:<16.5e} | {l_2:<15.5e} |")
    print(f"|{'-'*(max_name_width+2)}|{'-'*(max_n_width+2)}|{'-'*18}|{'-'*17}|")


# Plotting errors
plt.figure(figsize=(12, 6))

# L_infinity plot
plt.subplot(1, 2, 1)
for name in functions:
    plt.loglog(results[name]['N'], results[name]['L_inf'], 'o-', label=f"{name} (Matrix)")
plt.xlabel('N (Number of points)')
plt.ylabel(r'$L_\infty$ Error') # Use LaTeX for infinity symbol
plt.title(r'$L_\infty$ Error vs N (Matrix Method)')
plt.grid(True, which="both", ls="--")
plt.legend()

# L_2 plot
plt.subplot(1, 2, 2)
for name in functions:
    plt.loglog(results[name]['N'], results[name]['L_2'], 's-', label=f"{name} (Matrix)")
plt.xlabel('N (Number of points)')
plt.ylabel(r'$L_2$ Error (RMS)')
plt.title(r'$L_2$ Error (RMS) vs N (Matrix Method)')
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()
plt.show()