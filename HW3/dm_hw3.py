# Full Python Code for Exercises 1, 2, 3, 4

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp

###############################
# Exercise 1 - Fourier Galerkin
###############################

def rhs_ex1(t, u_hat, N):
    k = fftfreq(2*N+1, 1/(2*np.pi))
    u_x_hat = 1j * k * u_hat
    u_x = np.fft.ifft(u_x_hat)

    x = np.linspace(0, 2*np.pi, 2*N+1, endpoint=False)
    sin_x = np.sin(x)
    adv_term = sin_x * u_x
    adv_term_hat = fft(adv_term)

    return -adv_term_hat

def solve_ex1(N=32, t_span=(0, 2), dt=0.01):
    x = np.linspace(0, 2*np.pi, 2*N+1, endpoint=False)
    u0 = np.sin(x) + 0.5*np.sin(2*x)
    u0_hat = fft(u0)
    
    sol = solve_ivp(lambda t, u_hat: rhs_ex1(t, u_hat, N), t_span, u0_hat,
                    t_eval=np.arange(t_span[0], t_span[1], dt), method='RK45')
    return x, sol

def plot_ex1_solution(x, sol):
    plt.figure(figsize=(10, 6))
    times_to_plot = np.linspace(0, sol.t[-1], 5)
    
    for t_plot in times_to_plot:
        idx = np.argmin(np.abs(sol.t - t_plot))
        u_rec = np.real(ifft(sol.y[:, idx]))
        plt.plot(x, u_rec, label=f't = {sol.t[idx]:.2f}')
    
    plt.title('Exercise 1: Variable-Coefficient Advection')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.show()


###############################
# Exercise 2 - Fourier Galerkin with Dirichlet BCs
###############################

def rhs_ex2(t, u_hat, N):
    # Use wave numbers from 0 to N-1 to match u_hat dimensions
    k = np.arange(N) * np.pi / np.pi
    u_x_hat = 1j * k * u_hat

    x = np.linspace(0, np.pi, N, endpoint=False)
    sin_x = np.sin(x)
    u_x = np.fft.ifft(u_x_hat, n=N)

    adv_term = sin_x * u_x
    adv_term_hat = fft(adv_term)

    return -adv_term_hat

def solve_ex2(N=32, t_span=(0, 2), dt=0.01):
    x = np.linspace(0, np.pi, N, endpoint=False)
    u0 = np.sin(x)
    u0_hat = fft(u0)

    sol = solve_ivp(lambda t, u_hat: rhs_ex2(t, u_hat, N), t_span, u0_hat,
                    t_eval=np.arange(t_span[0], t_span[1], dt), method='RK45')
    return x, sol

def plot_ex2_solution(x, sol):
    plt.figure(figsize=(10, 6))
    times_to_plot = np.linspace(0, sol.t[-1], 5)

    for t_plot in times_to_plot:
        idx = np.argmin(np.abs(sol.t - t_plot))
        u_rec = np.real(ifft(sol.y[:, idx], n=len(x)))
        plt.plot(x, u_rec, label=f't = {sol.t[idx]:.2f}')

    plt.title('Exercise 2: Variable-Coefficient Advection with Dirichlet BCs')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.show()


###############################
# Exercise 3 - Tau Method
###############################

def tau_rhs_ex3(t, a_coeffs, N, Nb):
    total_N = N + Nb
    x = np.linspace(0, np.pi, total_N, endpoint=False)

    u = np.zeros_like(x)
    u_x = np.zeros_like(x)
    for n in range(total_N):
        u += a_coeffs[n] * np.cos(n*x)
        u_x += -n * a_coeffs[n] * np.sin(n*x)

    sin_x = np.sin(x)
    adv = sin_x * u_x

    rhs = np.zeros_like(a_coeffs)
    for n in range(total_N):
        rhs[n] = -np.trapz(adv * np.cos(n*x), x) * (2/np.pi)
    return rhs

def solve_ex3(N=10, Nb=2, t_span=(0, 2), dt=0.01):
    total_N = N + Nb
    x = np.linspace(0, np.pi, total_N, endpoint=False)
    u0 = np.sin(x)

    a0 = np.array([2/np.pi * np.trapz(u0 * np.cos(n*x), x) for n in range(total_N)])

    sol = solve_ivp(lambda t, a_coeffs: tau_rhs_ex3(t, a_coeffs, N, Nb), t_span, a0,
                    t_eval=np.arange(t_span[0], t_span[1], dt), method='RK45')
    return x, sol

def plot_ex3_solution(x, sol, N, Nb):
    total_N = N + Nb
    plt.figure(figsize=(10, 6))
    times_to_plot = np.linspace(0, sol.t[-1], 5)

    for t_plot in times_to_plot:
        idx = np.argmin(np.abs(sol.t - t_plot))
        u_rec = np.zeros_like(x)
        for n in range(total_N):
            u_rec += sol.y[n, idx] * np.cos(n*x)
        plt.plot(x, u_rec, label=f't = {sol.t[idx]:.2f}')

    plt.title('Exercise 3: Tau Approximation')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.show()


###############################
# Exercise 4 - Burgers Equation Fourier Collocation
###############################

def rhs_ex4(t, u, N, epsilon=0.01):
    k = fftfreq(2*N+1, 1/(2*np.pi))
    u_hat = fft(u)
    u_x_hat = 1j * k * u_hat

    u_x = np.fft.ifft(u_x_hat)
    nonlinear = u * u_x
    nonlinear_hat = fft(nonlinear)

    u_xx_hat = -(k**2) * u_hat
    diff_term = ifft(u_xx_hat)

    return -np.real(ifft(nonlinear_hat)) + epsilon * np.real(diff_term)

def solve_ex4(N=32, t_span=(0, 2), dt=0.01, epsilon=0.01):
    x = np.linspace(0, 2*np.pi, 2*N+1, endpoint=False)
    u0 = np.sin(x)

    sol = solve_ivp(lambda t, u: rhs_ex4(t, u, N, epsilon), t_span, u0,
                    t_eval=np.arange(t_span[0], t_span[1], dt), method='RK45')
    return x, sol

def plot_ex4_solution(x, sol):
    plt.figure(figsize=(10, 6))
    times_to_plot = np.linspace(0, sol.t[-1], 5)

    for t_plot in times_to_plot:
        idx = np.argmin(np.abs(sol.t - t_plot))
        plt.plot(x, sol.y[:, idx], label=f't = {sol.t[idx]:.2f}')

    plt.title('Exercise 4: Burgers Equation Fourier-Collocation')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.show()


###############################
# Example Execution
###############################

if __name__ == "__main__":
    # Exercise 1
    x1, sol1 = solve_ex1()
    plot_ex1_solution(x1, sol1)

    # Exercise 2
    x2, sol2 = solve_ex2()
    plot_ex2_solution(x2, sol2)

    # Exercise 3
    N, Nb = 10, 2
    x3, sol3 = solve_ex3(N, Nb)
    plot_ex3_solution(x3, sol3, N, Nb)

    # Exercise 4
    x4, sol4 = solve_ex4()
    plot_ex4_solution(x4, sol4)