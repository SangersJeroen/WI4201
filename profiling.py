from cProfile import run
import numpy as np

from wi4201_lib import build_forcing_vector, force_boundary_matrix, force_boundary_vector, get_elem_mat

if __name__ == "__main__":
    # Global variables
    powers_2D = [2, 4, 6, 8, 10]
    powers_3D = [2, 4, 6, 8, ]

    def int_force(x, y, z):
        return (x**2 + y**2 + z**2)*np.sin(x*y*z)

    def bound_force(x, y, z):
        return np.sin(x*y*z)

    P = 6
    N = 2**P
    h = 1/N
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    z = np.linspace(0,1,N+1)

    grids = [x, y, z]

    ELEMENT_MATRIX = get_elem_mat(N+1, h, "3D")


    P_MATRIX = ELEMENT_MATRIX


    run("force_boundary_matrix(grids,ELEMENT_MATRIX,P_MATRIX)")
