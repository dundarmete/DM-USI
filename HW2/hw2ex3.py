import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Spatial Derivative Approximations for the Advection PDE
#     u_t + a u_x = 0, on [0,2pi] with periodic BC.
# --------------------------------------------------------

def initial_condition(x):
    """
    Initial condition: u(x,0) = exp(sin(x))
    """
    return np.exp(np.sin(x))

def exact_solution(x, t, a):
    """
    Exact solution: u(x,t) = exp(sin(x - a*t))
    """
    return np.exp(np.sin(x - a*t))

def compute_RHS_2nd(u, dx, a):
    """
    Second Order Centered Difference:
       u_t = - a * (u(x+dx) - u(x-dx))/(2*dx)
    """
    N = len(u)
    f = a * u
    rhs = np.zeros_like(u)
    for j in range(N):
        jp = (j + 1) % N
        jm = (j - 1) % N
        rhs[j] = - (f[jp] - f[jm]) / (2.0 * dx)
    return rhs

def compute_RHS_4th(u, dx, a):
    """
    Fourth Order Centered Difference:
       u_t = - a * (-u(x+2dx) + 8u(x+dx) - 8u(x-dx) + u(x-2dx))/(12*dx)
    """
    N = len(u)
    f = a * u
    rhs = np.zeros_like(u)
    for j in range(N):
        jp2 = (j + 2) % N
        jp1 = (j + 1) % N
        jm1 = (j - 1) % N
        jm2 = (j - 2) % N
        rhs[j] = - (-f[jp2] + 8*f[jp1] - 8*f[jm1] + f[jm2]) / (12.0 * dx)
    return rhs

# --- MODIFIED FUNCTION ---
def compute_RHS_inf(u, Lx, a):
    """
    Infinite Order (Spectral) Differentiation using Differentiation Matrix:
    Computes u_x using the Fourier differentiation matrix D for a
    periodic function on the domain [0,Lx]. Then computes u_t = -a*u_x.
    This version uses the explicit differentiation matrix instead of FFT.
    """
    N = len(u)
    if N == 0:
        return np.array([]) # Handle empty input case

    # Construct the standard Fourier differentiation matrix for [0, 2*pi]
    D = np.zeros((N, N))
    if N > 1: # Matrix is non-trivial only if N > 1
        j_indices = np.arange(N)
        k_indices = np.arange(N)
        jj, kk = np.meshgrid(j_indices, k_indices, indexing='ij')

        # Difference matrix (j - k)
        diff = jj - kk

        # Mask for off-diagonal elements (j != k)
        mask = (diff != 0)

        # Calculate cotangent term carefully for numerics if needed,
        # but standard formula usually works.
        # Angle = (j - k) * pi / N
        angle = diff[mask] * np.pi / N
        cot_term = 1.0 / np.tan(angle)

        # Fill matrix D for j != k
        D[mask] = 0.5 * (-1.0)**(diff[mask]) * cot_term

    # Scale the standard matrix D (for [0, 2*pi]) to the domain [0, Lx]
    # The derivative operator scales inversely with the domain length.
    # d/dx = (2*pi/Lx) * d/d(theta), where theta = x * (2*pi/Lx) is in [0, 2*pi]
    scale_factor = 2 * np.pi / Lx
    D_scaled = D * scale_factor

    # Compute the spatial derivative u_x = D_scaled @ u
    # Using matrix multiplication: D_scaled acts on the vector u
    du_dx = D_scaled @ u

    # Return the RHS for the advection equation: u_t = -a * u_x
    return -a * du_dx
# --- END OF MODIFIED FUNCTION ---

# --------------------------------------------------------
# 2. RK4 Time Stepping (Generic to Accept a Spatial Derivative)
# --------------------------------------------------------

def rk4_step(u, dt, dx, a, RHS_func, Lx=None):
    """
    One step of 4th-order Runge-Kutta for u_t = RHS(u).
    For spectral differentiation (infinite order), Lx is needed.
    """
    # Check if the RHS function is the spectral one, which needs Lx
    # This check remains valid as the function signature for compute_RHS_inf
    # is unchanged (still takes u, Lx, a).
    if RHS_func == compute_RHS_inf:
        if Lx is None:
            raise ValueError("Lx must be provided for compute_RHS_inf")
        k1 = dt * RHS_func(u, Lx, a)
        k2 = dt * RHS_func(u + 0.5*k1, Lx, a)
        k3 = dt * RHS_func(u + 0.5*k2, Lx, a)
        k4 = dt * RHS_func(u + k3, Lx, a)
    else:
        # Finite difference methods need dx
        k1 = dt * RHS_func(u, dx, a)
        k2 = dt * RHS_func(u + 0.5*k1, dx, a)
        k3 = dt * RHS_func(u + 0.5*k2, dx, a)
        k4 = dt * RHS_func(u + k3, dx, a)
    return u + (k1 + 2*k2 + 2*k3 + k4)/6.0

# --------------------------------------------------------
# 3. Solver for the PDE with a Given Spatial Derivative
# --------------------------------------------------------

def solve_pde(a=1.0, N=128, CFL=0.5, tmax=np.pi, RHS_func=compute_RHS_2nd, record_history=False, Lx=2*np.pi):
    """
    Solves u_t + a u_x = 0 on [0,Lx] with RK4 time integration.
    Returns grid x, u at final time, time levels, and history (if requested).
    """
    x = np.linspace(0, Lx, N, endpoint=False)
    u = initial_condition(x)
    dx = Lx / N
    # Ensure dt calculation is safe if a=0, though not typical for advection
    if abs(a) > 1e-15:
        dt = CFL * dx / abs(a)
    else:
        # If a=0, PDE is u_t=0, solution is constant. Choose arbitrary dt or handle separately.
        # Here, just choose a dt based on tmax.
        dt = tmax / 100 # Or some other reasonable default if tmax is small
        if dt == 0: dt = 0.01

    # Check if tmax requires any steps
    if tmax <= 0:
        times = np.array([0.0])
        history = [u.copy()] if record_history else []
        return x, u, times, history

    Nt = int(np.ceil(tmax/dt))
    # Ensure Nt is at least 1 if tmax > 0
    Nt = max(1, Nt)
    dt = tmax / Nt  # adjust dt so that final time is exactly reached.
    times = np.linspace(0, tmax, Nt+1)

    history = []
    if record_history:
        history.append(u.copy())

    for n in range(Nt):
        # Pass Lx to rk4_step, it will be used if RHS_func is compute_RHS_inf
        u = rk4_step(u, dt, dx, a, RHS_func, Lx=Lx)
        if record_history:
            history.append(u.copy())
            
    return x, u, times, history

# --------------------------------------------------------
# 4. Part (a): Convergence Study with Visual Plotting
# --------------------------------------------------------

def part_a():
    print("=== Part (a): L∞ Error Convergence Study at t = π ===")
    schemes = {
        "2nd Order": compute_RHS_2nd,
        "4th Order": compute_RHS_4th,
        "Infinite Order (Matrix)": compute_RHS_inf # Updated label
    }
    # Reduced N_list for faster testing, especially with matrix construction
    N_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # N_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048] # Original N_list
    results = {name: [] for name in schemes.keys()}
    t_final = np.pi
    a_coeff = 1.0
    domain_length = 2 * np.pi

    # Loop over each scheme and grid resolution to compute the L∞ error.
    for name, RHS in schemes.items():
        print(f"\nScheme: {name}")
        for N in N_list:
            print(f"  Running N = {N:4d}...", end="")
            x, u_num, _, _ = solve_pde(a=a_coeff, N=N, CFL=0.5, tmax=t_final,
                                         RHS_func=RHS, record_history=False, Lx=domain_length)
            u_ex = exact_solution(x, t_final, a_coeff)
            error = np.max(np.abs(u_num - u_ex))
            results[name].append(error)
            print(f" | Error = {error:.4e}")

    # Print computed convergence rates.
    for name in schemes.keys():
        errs = results[name]
        print(f"\nConvergence rates for {name}:")
        if len(N_list) > 1:
          for i in range(1, len(N_list)):
              # Avoid division by zero or log of zero if error is tiny
              if errs[i-1] > 1e-16 and errs[i] > 1e-16:
                  # Rate calculation assumes error scales like C * (1/N)^p
                  # log(E2/E1) = p * log( (1/N2) / (1/N1) ) = p * log(N1/N2)
                  # p = log(E2/E1) / log(N1/N2) = log(E1/E2) / log(N2/N1)
                  # N_list[i]/N_list[i-1] is typically 2
                  rate = np.log(errs[i-1]/errs[i]) / np.log(N_list[i]/N_list[i-1])
                  print(f"  Between N = {N_list[i-1]:4d} and {N_list[i]:4d}: p ~ {rate:.2f}")
              else:
                  print(f"  Between N = {N_list[i-1]:4d} and {N_list[i]:4d}: Rate undefined (error near zero)")


    # Determine target error from second order scheme at highest N tested.
    if results["2nd Order"]:
        target_error = results["2nd Order"][-1]
        print(f"\nTarget error from 2nd order scheme at N = {N_list[-1]}: {target_error:.4e}")

        def find_min_N(RHS, target, init_N=8, max_N=4096):
            N = init_N
            while N <= max_N:
                print(f"  Testing {RHS.__name__} with N = {N}...", end="")
                x, u_num, _, _ = solve_pde(a=a_coeff, N=N, CFL=0.5, tmax=t_final,
                                           RHS_func=RHS, record_history=False, Lx=domain_length)
                error = np.max(np.abs(u_num - exact_solution(x, t_final, a_coeff)))
                print(f" Error = {error:.4e}")
                if error <= target:
                    return N, error
                # Increase N, typically by doubling for convergence studies
                N *= 2
            print(f"Could not reach target error {target:.4e} with N up to {max_N}")
            return None, None

        for name in ["4th Order", "Infinite Order (Matrix)"]:
            print(f"\nFinding minimum N for {name} to match 2nd Order error:")
            N_req, err_val = find_min_N(schemes[name], target_error, init_N=min(N_list), max_N=max(N_list)*2)
            if N_req is not None:
              print(f"  To achieve error <= {target_error:.4e} using {name} scheme, minimal N tested >= {N_req} with error = {err_val:.4e}")
            else:
              print(f"  Could not achieve target error {target_error:.4e} within tested N range for {name}.")

    # ------------------------------
    # Visual Plot: Error vs N on a Log-Log Plot
    # ------------------------------
    plt.figure(figsize=(8,6))
    markers = {"2nd Order": "o", "4th Order": "s", "Infinite Order (Matrix)": "d"}
    colors = {"2nd Order": "red", "4th Order": "green", "Infinite Order (Matrix)": "blue"}

    for name in schemes.keys():
        # Filter out results where N might not have run if list was shortened
        valid_N = N_list[:len(results[name])]
        if valid_N: # Only plot if there are results
           plt.loglog(valid_N, results[name], marker=markers[name], color=colors[name],
                      linestyle='-', label=f"{name}")

    # Add reference lines for expected slopes if desired
    if len(N_list) > 1:
        N_ref = np.array(N_list)
        # Example: Add O(N^-2) line, adjust constant C to match data roughly
        if results["2nd Order"]:
             C2 = results["2nd Order"][0] * (N_ref[0]**2)
             plt.loglog(N_ref, C2 * N_ref**(-2.0), 'r:', label='$O(N^{-2})$')
        # Example: Add O(N^-4) line
        if results["4th Order"]:
             C4 = results["4th Order"][0] * (N_ref[0]**4)
             plt.loglog(N_ref, C4 * N_ref**(-4.0), 'g:', label='$O(N^{-4})$')

    plt.xlabel("Number of grid points, N")
    plt.ylabel("$L_\\infty$ Error at $t = \\pi$") # Use LaTeX for math symbols
    plt.title("Error Convergence for Different Spatial Derivative Schemes")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# --------------------------------------------------------
# 5. Part (b): Long-Time Integration Comparison
# --------------------------------------------------------

def part_b():
    print("\n=== Part (b): Long Time Integration Comparison ===")
    t_save = [0, 100, 200] # Time points to save/plot solution
    t_max_long = 200 # Maximum simulation time
    Lx = 2*np.pi
    a_coeff = 1.0

    # -- Second Order Scheme with N = 200 --
    N2 = 200
    print(f"\nRunning Second Order Scheme (N = {N2}) up to t = {t_max_long}...")
    x2, u2_final, times2, history2 = solve_pde(a=a_coeff, N=N2, CFL=0.5, tmax=t_max_long,
                                               RHS_func=compute_RHS_2nd, record_history=True, Lx=Lx)
    sol2_plot = {}
    for t_target in t_save:
        # Find the index closest to the target time
        idx = np.argmin(np.abs(times2 - t_target))
        print(f"  2nd Order: Target t={t_target}, Actual t={times2[idx]:.2f}, Index={idx}")
        sol2_plot[t_target] = history2[idx]

    # -- Infinite Order Scheme with N = 10 --
    N_inf = 10 # Small N for spectral method
    print(f"\nRunning Infinite Order (Matrix) Scheme (N = {N_inf}) up to t = {t_max_long}...")
    x_inf, u_inf_final, times_inf, history_inf = solve_pde(a=a_coeff, N=N_inf, CFL=0.5, tmax=t_max_long,
                                                           RHS_func=compute_RHS_inf, record_history=True, Lx=Lx)
    sol_inf_plot = {}
    for t_target in t_save:
        # Find the index closest to the target time
        idx = np.argmin(np.abs(times_inf - t_target))
        print(f"  Inf Order: Target t={t_target}, Actual t={times_inf[idx]:.2f}, Index={idx}")
        sol_inf_plot[t_target] = history_inf[idx]

    # Create subplots for comparison.
    fig, axes = plt.subplots(2, len(t_save), figsize=(5*len(t_save), 8), sharey=True)
    fig.suptitle(f"Long Time Integration Comparison (u_t + {a_coeff} u_x = 0)\n"
                 f"Row 1: Second Order (N = {N2}); Row 2: Infinite Order (Matrix, N = {N_inf})",
                 y=0.98) # Adjust title position

    for j, t_target in enumerate(t_save):
        # Plot for Second Order Scheme.
        ax = axes[0, j] if len(t_save) > 1 else axes[0]
        u_ex_2 = exact_solution(x2, t_target, a_coeff)
        ax.plot(x2, sol2_plot[t_target], 'bo-', markersize=4, linewidth=1, label="Computed")
        ax.plot(x2, u_ex_2, 'k--', linewidth=1.5, label="Exact")
        ax.set_title(f"Second Order, $t = {t_target}$") # Use LaTeX
        ax.set_xlabel("$x$")
        if j == 0: ax.set_ylabel("$u(x,t)$")
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.set_xlim(0, Lx) # Ensure x-axis limits are consistent

        # Plot for Infinite Order Scheme.
        ax = axes[1, j] if len(t_save) > 1 else axes[1]
        u_ex_inf = exact_solution(x_inf, t_target, a_coeff)
        ax.plot(x_inf, sol_inf_plot[t_target], 'ro-', markersize=4, linewidth=1, label="Computed")
        ax.plot(x_inf, u_ex_inf, 'k--', linewidth=1.5, label="Exact")
        ax.set_title(f"Infinite Order, $t = {t_target}$") # Use LaTeX
        ax.set_xlabel("$x$")
        if j == 0: ax.set_ylabel("$u(x,t)$")
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.set_xlim(0, Lx) # Ensure x-axis limits are consistent

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

# --------------------------------------------------------
# 6. Main Section: Run Both Parts
# --------------------------------------------------------

if __name__ == "__main__":
    part_a()
    part_b()