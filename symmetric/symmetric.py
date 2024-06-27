import numpy as np
import matplotlib.pyplot as plt

B = 1

# create meshgrid for x and y
N = 1000
X,Y = np.meshgrid(np.linspace(-10,10,N, dtype=float),
                  np.linspace(-10,10,N,dtype=float))

kN = 10  # discrete number of quantum number k

# defines symm exp functions with varying quantum number k
def symmbasis(x,y,m):
    r_sq = (x**2) + (y**2)
    z = x + 1j*y
    return z**m * np.exp((-B/2)*r_sq)

V = np.zeros((kN, kN),dtype=np.complex_)
for i in range(kN):
    for j in range(kN):
        V[i,j] = -symmbasis(0,0,i) * symmbasis(0,0,j)


# finds eigenvalues and eigenvectors of V
eigvals, eigvecs = np.linalg.eigh(V)
eigenvectors = np.transpose(eigvecs)

def wfxn(l, x, y):
    wf = 0 + 0j
    for i in range(kN):
        wf += eigenvectors[l, i] * symmbasis(x, y, i)
    return wf

n = 5 # number of plots

# saves plots for basis wavefunctions of symmetric gauge
for i in range(n):
    wf = wfxn(i,X,Y)
    Z = np.real(wf) ** 2 + np.imag(wf) ** 2
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.contour3D(X,Y,Z, 50, cmap= 'binary')
    ax.set_xlabel('x')
    ax.set_label('y')
    ax.set_zlabel('$|\psi|^2$')
    ax.view_init(30,35)
    plt.title('Basis wavefunction density (m=%s) of LLL in '
              'symmetric gauge' %i)
    plt.savefig('symm%s.png' %i)
    fig.clear()
    plt.close(fig)