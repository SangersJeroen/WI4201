from cProfile import run
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz


from wi4201_lib import build_forcing_vector, force_boundary_matrix, force_boundary_vector, get_elem_mat, sol_chol_dec

if __name__ == "__main__":
    # Global variables
    powers_2D = [2, 4, 6, 8, 10]
    powers_3D = [2, 4, 6, 8, ]

    def int_force(x, y, z):
        return (x**2 + y**2 + z**2)*np.sin(x*y*z)

    def bound_force(x, y, z):
        return np.sin(x*y*z)

    P = 8
    N = 2**P
    h = 1/N
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    z = np.linspace(0,1,N+1)

    grids = [x, y, z]

    ELEMENT_MATRIX = load_npz('data/elemmat{}-3D.npz'.format(N))
    ELEMENT_VECTOR = load_npz('data/elemvec{}-3D.npz'.format(N))

    ELEMENT_MATRIX = ELEMENT_MATRIX.astype('int')
    ELEMENT_VECTOR = ELEMENT_VECTOR.astype('int')


    SOL, time_dict, factor_fill = sol_chol_dec(ELEMENT_MATRIX, ELEMENT_VECTOR)

    print(time_dict)

    plt.imshow(SOL.reshape((N+1,N+1)))
    plt.colorbar()
    plt.show()
