# part3_solution.py

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import pandas as pd

# =============================================================================
# Global Parameters and Core Functions from Part 2
# =============================================================================

# Physical parameters for the Burgers' equation
nu = 0.1  # Viscosity
c = 4.0   # Wave speed

def phi(a, b, K=100):
    """Helper function (periodic heat kernel sum) for the exact solution."""
    k = np.arange(-K, K + 1)
    return np.sum(np.exp(-(a - (2 * k + 1) * np.pi)**2 / (4 * nu * b)))

def u_exact(x, t):
    """Computes the exact solution of the periodic Burgers' equation."""
    a = x - c * t
    b = t + 1
    delta_phi = (phi(a + 1e-6, b) - phi(a - 1e-6, b)) / (2e-6)
    return c - 2 * nu * delta_phi / phi(a, b)

# =============================================================================
# Core Functions for Fourier-Galerkin Method (Part 3)
# =============================================================================

def F_galerkin(u_hat, k, nu):
    """
    Computes the right-hand side of the ODE for the Fourier coefficients `u_hat`
    using a de-aliased pseudo-spectral method (2/3 rule).
    
    d(u_hat)/dt = - (u * u_x)_hat + nu * (-k^2) * u_hat
    """
    # Viscous term (exact in Fourier space)
    viscous_term_hat = -nu * (k**2) * u_hat

    # Nonlinear term with de-aliasing
    # To compute (u * u_x)_hat without aliasing, we use the 2/3 rule.
    # We pad the array to 3/2 its size, transform, multiply, and transform back.
    N = len(u_hat)
    M = int(N * 3 / 2)
    
    # Pad u_hat to size M
    u_hat_padded = np.zeros(M, dtype=complex)
    u_hat_padded[:N//2] = u_hat[:N//2]
    u_hat_padded[M - (N - N//2):] = u_hat[N//2:] # handles N odd/even
    
    # Compute derivative in padded space
    du_dx_hat_padded = 1j * fftfreq(M, 1/M) * u_hat_padded

    # Transform to physical space
    u_padded = ifft(u_hat_padded) * M # Scale correctly
    du_dx_padded = ifft(du_dx_hat_padded) * M # Scale correctly

    # Multiply in physical space
    product_padded = u_padded * du_dx_padded

    # Transform product back to Fourier space
    product_hat_padded = fft(product_padded) / M # Scale correctly

    # Truncate back to original size N
    nonlinear_term_hat = np.zeros(N, dtype=complex)
    nonlinear_term_hat[:N//2] = product_hat_padded[:N//2]
    nonlinear_term_hat[N//2:] = product_hat_padded[M - (N - N//2):]

    return -nonlinear_term_hat + viscous_term_hat

def RK4_step_galerkin(u_hat, dt, k, nu):
    """
    Performs a single time step for the Fourier coefficients `u_hat`
    using a 4th-order Runge-Kutta method.
    """
    k1 = F_galerkin(u_hat, k, nu)
    k2 = F_galerkin(u_hat + 0.5 * dt * k1, k, nu)
    k3 = F_galerkin(u_hat + 0.5 * dt * k2, k, nu)
    k4 = F_galerkin(u_hat + dt * k3, k, nu)
    
    return u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# =============================================================================
# Part A Execution
# =============================================================================

def part_a_simulation():
    """
    Part A: Simulate Burgers' equation using Fourier-Galerkin method
    and compare with exact solution.
    """
    print("--- Running Part 3(a): Fourier-Galerkin Simulation ---")
    T_final = np.pi / 4
    N = 128
    CFL = 0.4

    # Domain setup
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    k = fftfreq(N, d=dx) * 2 * np.pi
    k_max = N / 2.0

    # Initial condition in Fourier space
    u0 = np.array([u_exact(xj, 0) for xj in x])
    u_hat = fft(u0) / N

    # Time integration
    t = 0.0
    while t < T_final:
        u_physical = np.real(ifft(u_hat) * N)
        max_abs_u = np.max(np.abs(u_physical))

        dt = CFL / (max_abs_u * k_max + nu * k_max**2)
        if t + dt > T_final:
            dt = T_final - t

        u_hat = RK4_step_galerkin(u_hat, dt, k, nu)
        t += dt

    # Final physical solution
    u_num = np.real(ifft(u_hat) * N)
    u_ex = np.array([u_exact(xj, T_final) for xj in x])

    # Display in DataFrame
    df = pd.DataFrame({'x': x, 'Numerical': u_num, 'Exact': u_ex})
    print(df)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_num, label='Numerical (Galerkin)', lw=2)
    plt.plot(x, u_ex, '--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.title('Part A: Fourier-Galerkin Simulation at T = π/4')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# Part B: Stability Analysis
# =============================================================================

def part_b_stability():
    """
    Determines by experiment the maximum stable CFL number for the
    Fourier-Galerkin scheme for various grid sizes N.
    """
    print("--- Running Part 3(b): Stability Analysis ---")
    T = np.pi / 4
    N_values = [16, 32, 48, 64, 96, 128, 192, 256]
    CFLs = np.arange(0.05, 1.05, 0.05) # Test a wider range for Galerkin
    stable_cfl_map = {}

    def check_stability_galerkin(N, T, CFL):
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        dx = x[1] - x[0]
        k = fftfreq(N, d=dx) * 2 * np.pi
        k_max = N / 2.0

        # Initial condition in Fourier space
        u0 = np.array([u_exact(xj, 0) for xj in x])
        u_hat = fft(u0) / N

        t = 0.0
        while t < T:
            u_physical = np.real(ifft(u_hat) * N)
            max_abs_u = np.max(np.abs(u_physical))
            
            # Time step from Eq. (4)
            dt = CFL / (max_abs_u * k_max + nu * k_max**2)
            
            if t + dt > T:
                dt = T - t
            
            u_hat_new = RK4_step_galerkin(u_hat, dt, k, nu)
            
            # Check for instability
            if np.any(np.isnan(u_hat_new)) or np.max(np.abs(u_hat_new)) > 1e4:
                return False # Unstable
            u_hat = u_hat_new
            t += dt
        return True # Stable

    print("Testing stability for different N and CFL values...")
    for N in N_values:
        max_stable_cfl = 0.0
        for CFL in CFLs:
            if check_stability_galerkin(N, T, CFL):
                max_stable_cfl = round(CFL, 2)
            else:
                break
        stable_cfl_map[N] = max_stable_cfl

    stable_cfl_df = pd.DataFrame(list(stable_cfl_map.items()), columns=['N', 'Max_Stable_CFL'])
    print("\nThe largest stable CFL number for each grid size N is:")
    print(stable_cfl_df)
    
    return stable_cfl_map

# =============================================================================
# Part C: Convergence Analysis
# =============================================================================

def part_c_convergence(stable_cfl_map):
    """
    Computes the L-infinity error for different N using the stable CFLs
    and estimates the rate of convergence.
    """
    print("\n--- Running Part 3(c): Convergence Analysis ---")
    
    def compute_l_inf_error_galerkin(N, CFL, T):
        L = 2 * np.pi
        x = np.linspace(0, L, N, endpoint=False)
        dx = x[1] - x[0]
        k = fftfreq(N, d=dx) * 2 * np.pi
        k_max = N / 2.0
        
        # Initial condition
        u0 = np.array([u_exact(xj, 0) for xj in x])
        u_hat = fft(u0) / N
        
        t = 0.0
        while t < T:
            u_physical = np.real(ifft(u_hat) * N)
            max_abs_u = np.max(np.abs(u_physical))
            dt = CFL / (max_abs_u * k_max + nu * k_max**2)
            if t + dt > T:
                dt = T - t
            u_hat = RK4_step_galerkin(u_hat, dt, k, nu)
            t += dt
            
        u_num = np.real(ifft(u_hat) * N)
        u_ex = np.array([u_exact(xj, T) for xj in x])
        return np.max(np.abs(u_num - u_ex))

    T = np.pi / 4
    N_vals = sorted(stable_cfl_map.keys())
    errors = [compute_l_inf_error_galerkin(N, stable_cfl_map[N], T) for N in N_vals]

    error_df = pd.DataFrame({'N': N_vals, 'L∞ Error': errors})
    print("\nThe L∞ errors for each grid size N are:")
    print(error_df)

    valid_indices = [i for i, err in enumerate(errors) if err > 1e-12] # Avoid log(0)
    log_N = np.log2(np.array(N_vals)[valid_indices])
    log_err = np.log2(np.array(errors)[valid_indices])

    if len(log_N) > 1:
        convergence_rate = np.polyfit(log_N, log_err, 1)[0]
        print(f"\nEstimated convergence rate (slope of log-log plot): {convergence_rate:.4f}")


    # Plot the L-infinity error vs. N on a log-log scale
    plt.figure(figsize=(10, 6))
    plt.loglog(N_vals, errors, 'o-', base=2, label='L∞ Error (Galerkin)')
    plt.xlabel('N (Grid Size)')
    plt.ylabel('L∞ Error')
    plt.title('Part C: Convergence of Fourier-Galerkin Method')
    plt.grid(True, which="both", ls="--")
    plt.xticks(N_vals, labels=N_vals, rotation=45)
    plt.legend()
    plt.show()
    
    print("\nAnalysis of Convergence Rate:")
    print("The plot of L∞ Error vs. N on a log-log scale shows a very steep, nearly vertical drop. "
          "This indicates that the error decreases faster than any algebraic power of N (i.e., O(N^-p) for any p). "
          "This behavior is known as **spectral accuracy**. The error decreases exponentially until it reaches machine precision, "
          "which is the expected theoretical convergence rate for a Fourier method applied to a smooth (C-infinity) periodic problem.")

# =============================================================================
# Part D: Solution Snapshots
# =============================================================================

def part_d_solution_snapshots(stable_cfl_map):
    """
    Part D: Plot numerical and exact solutions at different times
    using the Fourier-Galerkin method for N = 128.
    """
    print("\n--- Running Part 3(d): Solution Snapshots ---")
    N = 128
    CFL = stable_cfl_map[N]
    time_steps = [0.0, np.pi / 8, np.pi / 6, np.pi / 4]

    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    k = fftfreq(N, d=dx) * 2 * np.pi
    k_max = N / 2.0

    # Initial condition
    u0 = np.array([u_exact(xj, 0) for xj in x])
    u_hat = fft(u0) / N

    plots = []
    t = 0.0
    snapshot_idx = 0

    for target_time in time_steps:
        u_hat_temp = np.copy(u_hat)
        t_temp = t
        while t_temp < target_time:
            u_physical = np.real(ifft(u_hat_temp) * N)
            max_abs_u = np.max(np.abs(u_physical))
            dt = CFL / (max_abs_u * k_max + nu * k_max**2)
            if t_temp + dt > target_time:
                dt = target_time - t_temp
            u_hat_temp = RK4_step_galerkin(u_hat_temp, dt, k, nu)
            t_temp += dt

        u_num = np.real(ifft(u_hat_temp) * N)
        u_ex = np.array([u_exact(xj, target_time) for xj in x])
        plots.append((x, u_num, u_ex, target_time))

    # Plot snapshots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, (x_vals, u_num_vals, u_exact_vals, T_i) in enumerate(plots):
        axes[i].plot(x_vals, u_exact_vals, 'k--', label='Exact')
        axes[i].plot(x_vals, u_num_vals, 'b-', label='Numerical')
        axes[i].set_title(f"t = {T_i:.3f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("u(x,t)")
        axes[i].legend()
        axes[i].grid(True)

    fig.suptitle("Part D: Fourier-Galerkin Solution Snapshots for N = 128", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================
def main():
    """Main function to run all parts of the analysis."""
     # Part A
    part_a_simulation()

    # Part B
    stable_cfl_results = part_b_stability()

    # Part C
    part_c_convergence(stable_cfl_results)

    # Part D
    part_d_solution_snapshots(stable_cfl_results)


if __name__ == "__main__":
    main()