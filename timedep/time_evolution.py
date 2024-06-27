import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.animation import FuncAnimation, PillowWriter

# constants
B = 1
kN = 100  # dimensionality of potential matrix


# defines state at t = 0
def initial_state(x, y):
    z = x + 1j * y
    return (np.sqrt(2 * np.pi)**(-1)) * np.exp((-B / 4) * np.abs(z)**2)


# defines symmetric basis functions
def symmetric_basis(x, y, m):
    z = x + 1j*y
    return (( np.sqrt(math.factorial(m) * (2**(m+1)) * np.pi)**(-1))
            * (z ** m) * np.exp((-B / 4) * np.abs(z)**2))


# creates potential V matrix by grid of delta potentials and diagonalises it
def diagonalisation(d, n):
    V = np.zeros((kN, kN), dtype=np.complex_)
    delta_X, delta_Y = np.meshgrid(np.linspace(0, d, n, dtype=float),np.linspace(0, d, n, dtype=float))
    for i in range(kN):
        for j in range(kN):
            delta_calc = np.conj(symmetric_basis(delta_X,delta_Y, i)) * symmetric_basis(delta_X,delta_Y, j)
            V[i,j] = -sum(map(sum,delta_calc))

    eigenvalues, eigenvectors = np.linalg.eigh(V)
    return eigenvectors[0:10], eigenvalues[0:10]


# defines orthogonal basis in terms of symmetric basis
def orthogonal_basis(eigenvectors, x, y, index):
    wf = 0 + 0j
    for i in range(10):
        wf += eigenvectors[index,i] * symmetric_basis(x, y, i)
    return wf


# finds expansion coefficients of initial state in terms of orthonormal basis
def expansion_coeff(eigenvectors, initial_fxn, index):
    integrand = lambda x, y: (np.conj(orthogonal_basis(eigenvectors, x, y, index)) *
                              initial_fxn(x, y))

    def real_func(x,y):
        return np.real(integrand(x,y))

    def imag_func(x,y):
        return np.imag(integrand(x,y))

    real_int = integrate.dblquad(real_func, -np.inf, np.inf, -np.inf, np.inf)
    imag_int = integrate.dblquad(imag_func, -np.inf, np.inf, -np.inf, np.inf)

    return real_int[0] + 1j * imag_int[0]


# defines evolution of initial state at a later time T
def time_dependence(eigenvalues, eigenvectors, initial_fxn, x, y, time):
    wf = 0 + 0j
    for i in range(10):
        wf += (orthogonal_basis(eigenvectors, x, y, i) *
               expansion_coeff(eigenvectors, initial_fxn, i)
               * np.exp(-1j * eigenvalues[i] * time))
    return wf


# create meshgrid for x, y
N = 1000  # number of plot points
plot_limit = 8  # boundaries of plot
X, Y = np.meshgrid(np.linspace(-plot_limit, plot_limit, N, dtype=float),
                  np.linspace(-plot_limit, plot_limit, N, dtype=float))

lattice_limit = 6  # boundary of lattice
spacing = 80  # spacing of lattice

initial_fxn = lambda x,y: initial_state(x,y)
eigenvectors, eigenvalues = diagonalisation(lattice_limit, spacing)


for t in range(0, 100, 5):
    wf = time_dependence(eigenvalues, eigenvectors, initial_fxn, X, Y, t)
    probability_density = np.abs(wf) ** 2

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$|\psi|^2$')
    ax.view_init(30, 35)
    ax.contour3D(X, Y, probability_density, 50, cmap='binary')

    plt.savefig("probdensity (test_new, t = %s).png" %t)
    fig.clear()
    plt.close(fig)
