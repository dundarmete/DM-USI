# main.py

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import pandas as pd

# =============================================================================
# Global Parameters and Core Functions
# =============================================================================

# Physical parameters for the Burgers' equation
nu = 0.1  # Viscosity
c = 4.0   # Wave speed

def phi(a, b, K=100):
    """
    Helper function (periodic heat kernel sum) for the exact solution.
    """
    k = np.arange(-K, K + 1)
    return np.sum(np.exp(-(a - (2 * k + 1) * np.pi)**2 / (4 * nu * b)))

def u_exact(x, t):
    """
    Computes the exact solution of the periodic Burgers' equation using the Cole-Hopf transformation.
    """
    a = x - c * t
    b = t + 1
    # Central difference approximation for the derivative of phi
    delta_phi = (phi(a + 1e-6, b) - phi(a - 1e-6, b)) / (2e-6)
    return c - 2 * nu * delta_phi / phi(a, b)

def fourier_derivative(u, dx):
    """
    Computes the first and second spatial derivatives of a function u
    using the Fourier spectral method.
    """
    N = len(u)
    k = fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers
    u_hat = fft(u)
    du_dx_hat = 1j * k * u_hat
    d2u_dx2_hat = -(k**2) * u_hat
    du_dx = np.real(ifft(du_dx_hat))
    d2u_dx2 = np.real(ifft(d2u_dx2_hat))
    return du_dx, d2u_dx2

def F(u, dx):
    """
    Computes the right-hand side of the semi-discretized Burgers' equation:
    F(u) = -u * u_x + nu * u_xx
    """
    du_dx, d2u_dx2 = fourier_derivative(u, dx)
    return -u * du_dx + nu * d2u_dx2

def RK4_step(u, dt, dx):
    """
    Performs a single time step using a low-storage 4th-order Runge-Kutta method.
    This specific implementation was used across all original files.
    """
    u1 = u + 0.5 * dt * F(u, dx)
    u2 = u + 0.5 * dt * F(u1, dx)
    u3 = u + dt * F(u2, dx)
    return (1/3) * (-u + u1 + 2*u2 + u3 + 0.5 * dt * F(u3, dx))

# =============================================================================
# Part A: Numerical vs. Exact Solution and Error Plot
# =============================================================================

def part_a():
    """
    Handles the tasks from part_a.py:
    1. Runs the simulation for N=128 at the final time T=pi/4.
    2. Compares the numerical and exact solutions in a printed table.
    3. Plots the point-wise error between the two solutions.
    """
    print("--- Running Part A: Solution and Error for N=128 at T=pi/4 ---")
    
    def run_simulation(N, T, CFL):
        """Simulates the Burgers' equation up to time T."""
        x = np.linspace(0, 2*np.pi, N+1)[:-1]
        dx = x[1] - x[0]
        u = np.array([u_exact(xj, 0) for xj in x])
        
        max_u = np.max(np.abs(u))
        dt = CFL * (1 / (max_u / dx + nu / dx**2))

        t = 0.0
        while t < T:
            if t + dt > T:
                dt = T - t
            u = RK4_step(u, dt, dx)
            t += dt
        return x, u

    # Run simulation for N=128
    T_final = np.pi / 4
    x_vals, u_num = run_simulation(N=128, T=T_final, CFL=0.4)
    u_ex = np.array([u_exact(xj, T_final) for xj in x_vals])
    
    # Display numerical vs. exact results in a DataFrame
    df = pd.DataFrame({'x': x_vals, 'Numerical': u_num, 'Exact': u_ex})
    print(df)
    print("\nThe table above compares the numerical solution with the analytical solution")
    print("for Burgers' equation at t=pi/4 for N=128 spatial points.\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, u_num, label='Numerical (Collocation)', lw=2)
    plt.plot(x_vals, u_ex, '--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.title('Part A: Fourier Collocation Simulation at T = π/4')
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# Part B: Stability Analysis
# =============================================================================

def part_b():
    """
    Handles the tasks from part_b.py:
    1. Tests for numerical stability across a range of CFL numbers and grid sizes (N).
    2. Determines and reports the largest stable CFL for each grid size.
    """
    print("\n--- Running Part B: Stability Analysis ---")
    T = np.pi / 4

    def check_stability(N, T, CFL):
        """Runs a simulation and checks for signs of instability."""
        x = np.linspace(0, 2*np.pi, N+1)[:-1]
        dx = x[1] - x[0]
        u = np.array([u_exact(xj, 0) for xj in x])
        
        max_u = np.max(np.abs(u))
        dt = CFL * (1 / (max_u / dx + nu / dx**2))

        t = 0.0
        while t < T:
            if t + dt > T:
                dt = T - t
            u_new = RK4_step(u, dt, dx)
            if np.any(np.isnan(u_new)) or np.any(np.abs(u_new) > 1e6):
                return False  # Unstable
            u = u_new
            t += dt
        return True  # Stable

    N_values = [16, 32, 48, 64, 96, 128, 192, 256]
    CFLs = np.arange(0.05, 1.00, 0.05)
    stable_cfl_map = {}

    print("Testing stability for different N and CFL values...")
    for N in N_values:
        max_stable_cfl = 0.0
        for CFL in CFLs:
            is_stable = check_stability(N, T, CFL)
            if is_stable:
                max_stable_cfl = round(CFL, 2)
            else:
                break  # Stop at the first unstable CFL
        stable_cfl_map[N] = max_stable_cfl

    stable_cfl_df = pd.DataFrame(list(stable_cfl_map.items()), columns=['N', 'Max_Stable_CFL'])
    print("\nThe largest stable CFL number for each grid size N up to T=π/4 is:")
    print(stable_cfl_df)
    print("\nAs expected, the maximum stable CFL tends to decrease for higher resolutions (larger N).")
    
    return stable_cfl_map

# =============================================================================
# Part C & D: Convergence Analysis and Solution Snapshots
# =============================================================================

def part_cd(stable_cfl_map):
    """
    Handles the tasks from part_cd.py.
    
    Part C:
    1. Computes the L-infinity error for different N using the provided stable CFLs.
    2. Estimates the rate of convergence and plots the log-log error graph.
    
    Part D:
    1. Generates and plots solution snapshots at different times for N=128.
    """
    # --- Part C: Convergence Analysis ---
    print("\n--- Running Part C: Convergence Analysis ---")
    
    def compute_l_inf_error(N, CFL, T):
        """Runs a simulation and returns the L-infinity norm of the error."""
        x = np.linspace(0, 2*np.pi, N+1)[:-1]
        dx = x[1] - x[0]
        u_num = np.array([u_exact(xj, 0) for xj in x])
        
        max_u = np.max(np.abs(u_num))
        dt = CFL * (1 / (max_u / dx + nu / dx**2))

        t = 0.0
        while t < T:
            if t + dt > T:
                dt = T - t
            u_num = RK4_step(u_num, dt, dx)
            t += dt
        
        u_ex = np.array([u_exact(xj, T) for xj in x])
        return np.max(np.abs(u_num - u_ex))

    T = np.pi / 4
    N_vals = sorted(stable_cfl_map.keys())
    errors = [compute_l_inf_error(N, stable_cfl_map[N], T) for N in N_vals]

    error_df = pd.DataFrame({'N': N_vals, 'L∞ Error': errors})
    print("\nThe L∞ errors for each grid size N are:")
    print(error_df)

    # Estimate convergence rate from the slope of the log-log plot
    valid_indices = [i for i, err in enumerate(errors) if err > 1e-12] # Avoid log(0)
    log_N = np.log2(np.array(N_vals)[valid_indices])
    log_err = np.log2(np.array(errors)[valid_indices])
    
    if len(log_N) > 1:
        convergence_rate = np.polyfit(log_N, log_err, 1)[0]
        print(f"\nEstimated convergence rate (slope of log-log plot): {convergence_rate:.4f}")
    
        # Plot the L-infinity error vs. N on a log-log scale
        plt.figure(figsize=(10, 6))
        # CORRECTED LINE: Replaced basex/basey with base=2
        plt.loglog(N_vals, errors, 'o-', base=2, label='L∞ Error')
        plt.xlabel('N (Grid Size)')
        plt.ylabel('L∞ Error')
        plt.title('Part C: Convergence of Fourier Spectral Method')
        plt.grid(True, which="both", ls="--")
        plt.xticks(N_vals, labels=N_vals)
        plt.legend()
        plt.show()

    # --- Part D: Solution Snapshots ---
    print("\n--- Running Part D: Solution Snapshots for N=128 ---")

    def run_snapshot_simulation(N, T, CFL):
        """Simulation runner for generating snapshots."""
        x = np.linspace(0, 2*np.pi, N+1)[:-1]
        dx = x[1] - x[0]
        u = np.array([u_exact(xj, 0) for xj in x])
        
        max_u = np.max(np.abs(u))
        dt = CFL * (1 / (max_u / dx + nu / dx**2))
        
        t = 0.0
        while t < T:
            if t + dt > T:
                dt = T - t
            u = RK4_step(u, dt, dx)
            t += dt
        return x, u

    time_steps = [0, np.pi / 8, np.pi / 6, np.pi / 4]
    N_d = 128
    cfl_d = stable_cfl_map.get(N_d, 1.00) # Use the stable CFL for N=128
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, T_i in enumerate(time_steps):
        x_vals_num, u_num_vals = run_snapshot_simulation(N=N_d, T=T_i, CFL=cfl_d)
        x_vals_exact = np.linspace(0, 2*np.pi, 256) # Use more points for a smooth exact curve
        u_exact_vals = np.array([u_exact(xj, T_i) for xj in x_vals_exact])
        
        axes[i].plot(x_vals_exact, u_exact_vals, 'k--', label='Exact')
        axes[i].plot(x_vals_num, u_num_vals, 'b-o', markersize=3, label=f'Numerical (N={N_d})')
        axes[i].set_title(f"Solution at t = {T_i:.4f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("u(x,t)")
        axes[i].legend()
        axes[i].grid(True)

    fig.suptitle(f"Part D: Comparison of Numerical and Exact Solutions (N = {N_d})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================
def main():
    """Main function to run all parts of the analysis."""
    # Part A
    part_a()
    
    # Part B
    stable_cfl_results = part_b()

    # Part C and D
    part_cd(stable_cfl_results)

if __name__ == "__main__":
    main()