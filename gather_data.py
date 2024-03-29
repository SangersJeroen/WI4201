import numpy as np
import matplotlib.pyplot as plt
from time import time, time_ns
from scipy.sparse import csgraph, save_npz, load_npz
from scipy.sparse.csgraph import reverse_cuthill_mckee as rcv
import os.path
from scipy.sparse.linalg import spsolve

from wi4201_lib import build_forcing_vector, force_boundary_matrix, force_boundary_vector, get_elem_mat, sol_chol_dec


if __name__ == "__main__":
    f = open('data/rlog.txt', 'a')
    # Global variables
    powers_2D = [2,3,4,5,6,7,8,9]
    powers_3D = [2,3,4,5,6,7]

    def int_force(x, y):
        return (x**2 + y**2)*np.sin(x*y)

    def bound_force(x, y):
        return np.sin(x*y)

    # First all the 2D cases:
    print("-------------- 2D ---------------")
    f.write('-------------- 2D ---------------\n')
    for i in powers_2D:

        P = i
        N = 2**P
        h = 1/N

        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)

        X, Y = np.meshgrid(x,y)

        grids = [x, y]
        print('--- 1 ---')
        print("Starting 2D: 2**{}".format(P))
        f.write("--- 1 ---\n")
        f.write("Starting 2D: 2**{}\n".format(P))

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

        #Exercise 2
        print("--- 2 ---")
        f.write("--- 2 ---\n")
        start = time_ns()
        SOL = spsolve(ELEMENT_MATRIX, ELEMENT_VECTOR)
        taken = time_ns() - start
        print("Direct solver took {:3f} ms".format(taken/1e6))
        f.write("Direct solver took {:3f} ms\n".format(taken/1e6))

        u_ex = np.sin(X*Y)
        vec_u_ex = u_ex.flatten()

        error = np.max(np.abs(vec_u_ex - SOL))
        print("h**2: {:.5f}\t error: {:.7f}".format(h**2, error))
        f.write("h**2: {:.5f}\t error: {:.7f}\n".format(h**2, error))

        SOL = None
        u_ex = None
        vec_u_ex = None

        #Exercise 3
        print('--- 3 ---')
        f.write("--- 3 ---\n")
        SOL, time_dict, factor_fill = sol_chol_dec(ELEMENT_MATRIX, ELEMENT_VECTOR)
        print(time_dict)
        f.write(str(time_dict)+"\n")

        #Exercise 4
        print('--- 4 ---')
        f.write("--- 4 ---\n")
        ELEMENT_MATRIX = load_npz('data/elemmat{}-2D.npz'.format(N))
        fill_in = factor_fill / ELEMENT_MATRIX.count_nonzero()
        print("Fill-in: {:.2f}".format(fill_in))
        f.write("Fill-in: {:.2f}\n".format(fill_in))

        #Exercise 5
        print("--- 5 ---")
        f.write("--- 5 ---\n")

        permut = rcv(ELEMENT_MATRIX, symmetric_mode=True)
        RED_BAND = ELEMENT_MATRIX[permut[:,np.newaxis], permut[np.newaxis,:]]

        ELEMENT_MATRIX = None

        SOL, time_dict, factor_fill = sol_chol_dec(RED_BAND, ELEMENT_VECTOR)
        print(time_dict)
        f.write(str(time_dict)+"\n")
        fill_in = factor_fill / RED_BAND.count_nonzero()
        print("Fill-in: {:.2f}".format(fill_in))
        f.write("Fill-in: {:.2f}\n".format(fill_in))
        print('\n\n')
        f.write('\n\n')

    # All the 3D cases:
    print("-------------- 3D ---------------")
    f.write("-------------- 3D ---------------\n")
    def int_force(x, y, z):
        return (x**2 + y**2 + z**2 )*np.sin(x*y*z)

    def bound_force(x, y, z):
        return np.sin(x*y*z)

    for i in powers_3D:

        P = i
        N = 2**P
        h = 1/N

        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        z = np.linspace(0, 1, N+1)

        X, Y, Z = np.meshgrid(x,y,z)

        grids = [x, y, z]

        # Exercise 1
        print('--- 1 ---')
        print("Starting 3D: 2**{}".format(P))
        f.write('--- 1 ---\n')
        f.write("Starting 3D: 2**{}\n".format(P))
        if os.path.isfile('data/elemmat{}-3D.npz'.format(N)) and os.path.isfile('data/elemvec{}-3D.npz'.format(N)):
            ELEMENT_MATRIX = load_npz('data/elemmat{}-3D.npz'.format(N))
            ELEMENT_VECTOR = load_npz('data/elemvec{}-3D.npz'.format(N))
        else:

            print("Building Element matrix of size {}".format(N))
            f.write("Building Element matrix of size {}\n".format(N))
            start = time_ns()

            ELEMENT_MATRIX = get_elem_mat(N+1, h, "3D")
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
            save_npz('data/elemmat{}-3D.npz'.format(N), ELEMENT_MATRIX)
            save_npz('data/elemvec{}-3D.npz'.format(N), ELEMENT_VECTOR)

        #Exercise 2
        print('--- 2 ---')
        f.write('--- 2 ---\n')
        if P < 7:
            start = time_ns()
            SOL = spsolve(ELEMENT_MATRIX, ELEMENT_VECTOR)
            taken = time_ns() - start
            print("Direct solver took {:3f} ms".format(taken/1e6))
            f.write("Direct solver took {:3f} ms\n".format(taken/1e6))

            u_ex = np.sin(X*Y*Z)
            vec_u_ex = u_ex.flatten()

            error = np.max(np.abs(vec_u_ex - SOL))
            print("h**2: {:.5f}\t error: {:.7f}".format(h**2, error))
            f.write("h**2: {:.5f}\t error: {:.7f}\n".format(h**2, error))

            SOL = None
            u_ex = None
            vec_u_ex = None

        #Exercise 3
        print('--- 3 ---')
        f.write('--- 3 ---\n')
        SOL, time_dict, factor_fill = sol_chol_dec(ELEMENT_MATRIX, ELEMENT_VECTOR)
        print(time_dict)
        f.write(str(time_dict)+"\n")

        #Exercise 4
        print('--- 4 ---')
        f.write('--- 4 ---\n')
        ELEMENT_MATRIX = load_npz('data/elemmat{}-3D.npz'.format(N))
        fill_in = factor_fill / ELEMENT_MATRIX.count_nonzero()
        print("Fill-in: {:.2f}".format(fill_in))
        f.write("Fill-in: {:.2f}\n".format(fill_in))

        #Exercise 5
        print('--- 5 ---')
        f.write('--- 5 ---\n')
        permut = rcv(ELEMENT_MATRIX, symmetric_mode=True)
        RED_BAND = ELEMENT_MATRIX[permut[:,np.newaxis], permut[np.newaxis,:]]

        ELEMENT_MATRIX = None

        SOL, time_dict, factor_fill = sol_chol_dec(RED_BAND, ELEMENT_VECTOR)
        print(time_dict)
        f.write(str(time_dict)+"\n")
        fill_in = factor_fill / RED_BAND.count_nonzero()
        print("Fill-in: {:.2f}".format(fill_in))
        f.write("Fill-in: {:.2f}\n".format(fill_in))
        print('\n\n')
        f.write("\n\n")
    f.write("END OF RUN")
    f.close()