import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

coordN = 1000  # number of coordinate points
X,Y = np.meshgrid(np.linspace(-10,10,coordN, dtype=float),
                  np.linspace(-10,10,coordN,dtype=float))

kN = 1000  # discrete number of quantum number k
ks = np.linspace(-1,1,kN,dtype=float)

# defines basis exp functions with varying quantum number k
def lanbasis(x,k):
    return np.exp(-(1/2) * ((x - k) ** 2))


# defines delta potential V as a matrix in k basis
V = np.zeros((kN, kN))
for i in range(kN):
    for j in range(kN):
        V[i,j] = -np.exp(-(1/2)*(ks[i]**2+ks[j]**2))

# finds eigenvalues and eigenvectors of V
eigvals, eigvecs = np.linalg.eigh(V)
eigenvectors = np.transpose(eigvecs)

# defines basis wavefunction as a sum of exp functions
def wfxn(l, x, y):
    wf = 0 + 0j
    for i in range(kN):
        wf += eigenvectors[l, i] * lanbasis(x, ks[i]) * np.exp(1j*ks[i]*y)
    return wf

n = 6 # number of plots

for i in range(6):
    wf = wfxn(i, X, Y)
    Z = np.real(wf)**2 + np.imag(wf)**2
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('$|\psi|^2$')
    ax.view_init(30, 35)
    plt.title('Basis wavefunction density of LLL (i=%s)'
              'in Landau gauge' %i)
    plt.savefig('landau%s.png' %i)
    fig.clear()
    plt.close(fig)
