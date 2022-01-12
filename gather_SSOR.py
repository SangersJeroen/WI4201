import matplotlib.pyplot as plt
import numpy as np
from time import time, time_ns
from scipy.sparse import save_npz, load_npz
import scipy.sparse as ssp
import os.path

from scipy.sparse.linalg.matfuncs import inv

from wi4201_lib import build_forcing_vector, force_boundary_matrix, force_boundary_vector, get_elem_mat, get_norm, prec_ssor_solve, ssor_solve, triang_components


if __name__ == "__main__":
    f = open('data/SSOR_log.txt', "a")
    powers = [2, 3, 4, 5, 6, 7, 8]

    def int_force(x, y):
        return (x**2 + y**2)*np.sin(x*y)

    def bound_force(x, y):
        return np.sin(x*y)

    for p in powers:
        N = 2**p
        h = 1/N

        x, y = np.linspace(0, 1, N+1), np.linspace(0, 1, N+1)

        grids = [x, y]
        if os.path.isfile('data/elemmat{}-2D.npz'.format(N)) and os.path.isfile('data/elemvec{}-2D.npz'.format(N)):
            ELEMENT_MATRIX = load_npz('data/elemmat{}-2D.npz'.format(N))
            ELEMENT_VECTOR = load_npz('data/elemvec{}-2D.npz'.format(N))
        else:

            # Exercise 1
            print("Building Element matrix of size {}".format(N))
            f.write("Building Element matrix of size {}\n".format(N))
            start = time_ns()

            ELEMENT_MATRIX = get_elem_mat(N+1, h, "2D")
            ELEMENT_VECTOR = build_forcing_vector(grids,
                                                  int_force,
                                                  bound_force)

            time_taken = (time_ns() - start)/1e6

            print("building took: {:3f} ms".format(time_taken))

            P_MATRIX = ELEMENT_MATRIX.copy()
            print("Forcing boundary conditions")
            start = time_ns()

            ELEMENT_MATRIX, P_MATRIX = force_boundary_matrix(grids,
                                                             ELEMENT_MATRIX,
                                                             P_MATRIX)
            ELEMENT_VECTOR = force_boundary_vector(grids,
                                                   bound_force,
                                                   ELEMENT_VECTOR,
                                                   P_MATRIX)

            time_taken = (time_ns()-start)/1e6
            print("Forcing boundaries took: {:3f} ms".format(time_taken))
            f.write("Forcing boundaries took: {:3f} ms\n".format(time_taken))
            save_npz('data/elemmat{}-2D.npz'.format(N), ELEMENT_MATRIX)
            save_npz('data/elemvec{}-2D.npz'.format(N), ELEMENT_VECTOR)

    for p in powers:

        N = 2**p
        h = 1/N

        print('---------------- {} --------------'.format(p))

        x, y = np.linspace(0, 1, N+1), np.linspace(0, 1, N+1)

        grids = [x, y]
        ELEMENT_MATRIX = load_npz('data/elemmat{}-2D.npz'.format(N))
        ELEMENT_VECTOR = load_npz('data/elemvec{}-2D.npz'.format(N))

        E, F, D = triang_components(ELEMENT_MATRIX)

        omega = 1.5
        MINV = omega*(2-omega) * inv(D - omega*E) @ D @ inv(D - omega*F)

        shape = MINV.shape
        ID = ssp.csc_matrix(shape)

        u0 = ssp.csc_matrix(np.zeros(shape[0])).T
        r0 = ssp.csc_matrix(np.ones(shape[0])).T

        f = ELEMENT_VECTOR.copy()
        f = f.T

        tolerance = 1e-10

        start = time_ns()

        SOL, r_norms = ssor_solve(
            ELEMENT_MATRIX,
            MINV,
            ELEMENT_VECTOR,
            u0,
            tolerance
        )

        norm_f = get_norm(ELEMENT_VECTOR)
        r_norms = np.append(r_norms, norm_f)

        elapsed = (time_ns() - start)/1e6

        print('ssor took: {:.2f} ms'.format(elapsed))

        np.save('data/ssor_r_norms{}'.format(p), r_norms)

        u0 = ssp.csc_matrix(np.zeros(shape[0])).T
        r0 = ssp.csc_matrix(np.ones(shape[0])).T

        f = ELEMENT_VECTOR.copy()
        f = f.T

        start = time_ns()

        SOL, r_norms = prec_ssor_solve(
            ELEMENT_MATRIX,
            MINV,
            ELEMENT_VECTOR,
            u0,
            r0,
            tolerance
        )

        r_norms = 0

        elapsed = (time_ns() - start)/1e6
        print("prec. ssor took: {:.2f} ms".format(elapsed))
        r_norms = np.append(r_norms, norm_f)
        np.save('data/prec_ssor_r_norms{}'.format(p), r_norms)
