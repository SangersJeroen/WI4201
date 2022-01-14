from mpl_toolkits.axes_grid1.axes_size import Fraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import reverse_cuthill_mckee as rcv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse.linalg.dsolve.linsolve import spsolve

from wi4201_lib import force_boundary_matrix, force_boundary_vector, get_elem_mat, build_forcing_vector, force_boundary_matrix, force_boundary_vector


def int_force(x, y):
        return (x**2 + y**2)*np.sin(x*y)

def bound_force(x, y):
        return np.sin(x*y)

p = 4
N = 2**p
h = 1/N
x, y = np.linspace(0, 1, N+1), np.linspace(0, 1, N+1)
grids = [x, y]
ELEMENT_MATRIX = get_elem_mat(N+1, h, "2D")
ELEMENT_VECTOR = build_forcing_vector(grids,
                                          int_force,
                                          bound_force)
P_MATRIX = ELEMENT_MATRIX.copy()
ELEMENT_MATRIX2, P_MATRIX = force_boundary_matrix(grids,
                                                     ELEMENT_MATRIX,
                                                     P_MATRIX)
ELEMENT_VECTOR = force_boundary_vector(grids,
                                           bound_force,
                                           ELEMENT_VECTOR,
                                           P_MATRIX)

permut = rcv(ELEMENT_MATRIX, symmetric_mode=True)
RED_BAND = ELEMENT_MATRIX[permut[:,np.newaxis], permut[np.newaxis,:]]


fig, ax = plt.subplots(1, 3, sharey=True)
plt.gcf().set_size_inches(12,4)
ax[0].imshow(ELEMENT_MATRIX.toarray(), origin='upper')
ax[0].title.set_text(r'a')
im = ax[1].imshow(ELEMENT_MATRIX2.toarray(), origin='upper')
ax[1].title.set_text(r'b')
ax[2].imshow(RED_BAND.toarray(), origin='upper')
ax[2].title.set_text(r'c')
fig.colorbar(im, ax=ax.ravel().tolist())
plt.savefig('plots/matrix_figures.pdf')
plt.show()


SOL = spsolve(ELEMENT_MATRIX2, ELEMENT_VECTOR)

plt.imshow(SOL.reshape((N+1,N+1)), origin='lower')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.title(r"Solution of $A^h$ for $p=4$")
plt.colorbar()
plt.savefig('plots/solution.pdf')
plt.show()