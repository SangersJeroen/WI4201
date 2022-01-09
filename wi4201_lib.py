from __future__ import annotations
from typing import List
import numpy as np
from scipy import sparse as ssp
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
from time import time_ns


def get_elem_mat(N: int, spacing: float, dimension: str) -> ssp.csr_matrix:
    """Creates the element matrix 'Ah' for the 2D or 3D discrete laplaciaan

    Parameters
    ----------
    N : int
        Number of points in a direction of the grid
    dimension : str
        2D/3D for two/three-dimensional discrete laplacian

    Returns
    -------
    ssp.csc_matrix
        The element matrix Ah
    """

    one_dim_diff = 2*np.eye((N))-np.eye((N), k=-1)-np.eye((N), k=1)
    h = spacing

    ONE_DIM_DIFF = ssp.csr_matrix(one_dim_diff.astype(int))
    ID = ssp.csr_matrix(np.eye(N).astype(int))

    x_dim_diff = (1/(h**2) * ssp.kron(ONE_DIM_DIFF, ID))
    y_dim_diff = (1/(h**2) * ssp.kron(ID, ONE_DIM_DIFF))

    X_DIM_DIFF = ssp.csr_matrix(x_dim_diff.astype(int))
    Y_DIM_DIFF = ssp.csr_matrix(y_dim_diff.astype(int))
    TWO_LAPLACE = ssp.csr_matrix(x_dim_diff + y_dim_diff)

    one_dim_diff = None
    x_dim_diff = None
    y_dim_diff = None

    if dimension == "2D":
        return TWO_LAPLACE

    elif dimension == "3D":
        z_dim_diff = (1/(h**2)*(
            ssp.kron(ID,
                     ssp.kron(ID, ONE_DIM_DIFF)
                     )
        )
        )

        Z_DIM_DIFF = ssp.csr_matrix(z_dim_diff.astype(int))
        z_dim_diff = None

        THREE_LAPLACE = (ssp.kron(X_DIM_DIFF, ID)
                         + ssp.kron(Y_DIM_DIFF, ID)
                         + Z_DIM_DIFF)

        return THREE_LAPLACE

    else:
        raise ValueError('string: Dimension, either "2D" or "3D"')


def build_forcing_vector(lin_spaces: list,
                         internal_fun,
                         boundary_fun=None
                         ) -> np.ndarray:
    """Builds the RHS of the linear system for a arbitray-dimensional laplacian

    Parameters
    ----------
    lin_spaces : list
        List of arrays of points that will be used to build the meshgrid
    internal_fun : Callable[list[float]]
        Function that takes n-dimensional coordinate arrays and returns the value of the forcing at that coordinate
    boundary_fun : Callable[list[float]]
        Function that takes n-dimensional coordinate arrays and returns the value of the forcing at that boundary coordinate

    Returns
    -------
    np.ndarray
        The forcing vector
    """

    grids = np.meshgrid(*lin_spaces)
    internal_forcing_array = internal_fun(*grids)
    if boundary_fun == None:
        def boundary_fun(*grids):
            return 0*grids[0]

    boundary_forcing_array = boundary_fun(*grids)

    dims = len(lin_spaces)
    mask = tuple([slice(1, -1)]*dims);
    forcing, forcing[mask] = boundary_forcing_array, internal_forcing_array[mask]
    print(mask)

    forcing_vector = forcing.flatten()
    FORCING_VECTOR = ssp.csc_matrix(forcing_vector).T
    return FORCING_VECTOR


def force_boundary_matrix(lin_spaces: list,
                          SYSTEM_MATRIX: ssp.csc_matrix,
                          P_MATRIX: ssp.csc_matrix) -> ssp.csc_matrix:
    """Forces the boundary points to remain unchanged after solving the system using the SYSTEM_MATRIX and subtracting the element vector with P_MATRIX.dot(vec_boundary)

    Parameters
    ----------
    lin_spaces : list
        numpy linspaces describing the grid_points of the problem
    SYSTEM_MATRIX : ssp.csc_matrix
        sparse matrix describing the element matrix of the system
    P_MATRIX : ssp.csc_matrix
        pre-initialised matrix of size same as SYSTEM_MATRIX which will contain the equations for altering the element vector
    """

    grids = np.meshgrid(*lin_spaces)

    boundary_list = np.asarray([])
    for grid in range(len(grids)):
        grid_vals = grids[grid].ravel()
        idx_list = np.squeeze(
            np.where(
                (grid_vals == lin_spaces[grid][0]) | (grid_vals == lin_spaces[grid][-1])
            )
        );
        boundary_list = np.append(boundary_list, idx_list)
        boundary_list = np.unique(boundary_list)

    shape = len(lin_spaces[0])**len(lin_spaces)

    SPARSE_ID = ssp.eye(shape).tocsr()
    P_MATRIX = ssp.csc_matrix((shape, shape))
    ONES = ssp.lil_matrix((shape, shape))
    ZEROES = ssp.csc_matrix((shape, shape))

    ONES[boundary_list, boundary_list] = 1
    ONES = ONES.tocsc()
    SYSTEM_MATRIX = SYSTEM_MATRIX.tocsr()

    for row in boundary_list:
        SYSTEM_MATRIX[row, :] = SPARSE_ID[row, :]

    SYSTEM_MATRIX.tocsc()

    for column in boundary_list:
        P_MATRIX[:, column] = SYSTEM_MATRIX[:, column]
        SYSTEM_MATRIX[:, column] = ZEROES[:, column]

    SYSTEM_MATRIX = (SYSTEM_MATRIX + ONES)
    P_MATRIX = (P_MATRIX - ONES)

    SPARSE_ID = None
    ONES = None
    ZEROES = None

    return SYSTEM_MATRIX, P_MATRIX


def force_boundary_vector(grids: list,
    bound_force,
    ELEMENT_VECTOR: ssp.csc_matrix,
    P_MATRIX) -> ssp.csc_matrix:

    m_grids = np.meshgrid(*grids)
    VALUES = bound_force(*m_grids)
    VALUES_VEC = ssp.csc_matrix(VALUES.flatten()).T

    ELEMENT_VECTOR = ELEMENT_VECTOR - P_MATRIX.dot(VALUES_VEC)

    return ELEMENT_VECTOR



def decomp_lu(M: ssp.csc_matrix) -> ssp.csc_matrix:
    """Creates the sparse lower matrix of the LU decomposition of M

    Parameters
    ----------
    M : ssp.csc_matrix
        The matrix to LU decompose such that L@U = M

    Returns
    -------
    ssp.csc_matrix
        The sparse lower matrix L
    """
    # In this code block all fully uppercase variables are sparse matrices
    shape_m = M.shape
    ID = ssp.eye(shape_m[0])
    ZEROES = ssp.csc_matrix(shape_m)

    SUM = ID.copy()
    for k in range(0, shape_m[1]):
        Akk = M[k, k]
        VEC = M[:, k] / Akk
        VEC[0: k+1] = ZEROES[0:k+1, 0]

        EK = ZEROES[:, 0].copy()
        EK[k] = 1

        ADD = VEC * EK.T
        SUM += ADD

    L = SUM
    U = spsolve(SUM, M)

    return L, U


def decomp_cholesky(M: ssp.csc_matrix) -> ssp.csc_matrix:
    """Creates the sparse lower triangular matrix C that results from cholesky
    decomposition.

    Parameters
    ----------
    M : ssp.csc_matrix
        The matrix M such that cholesky decomposition yields C, scuh that
        C@C.T == M

    Returns
    -------
    ssp.csc_matrix
        The sparse lower triangular matrix C
    """
    shape_m = M.shape
    ZEROES = ssp.lil_matrix(shape_m)
    C = ssp.lil_matrix(shape_m)
    TMP = M.copy().tolil()

    for col in range(0, shape_m[1]):
        C[col, col] = np.sqrt((TMP[col, col] - (C[col, 0:col].power(2)).sum()))
        TMP[col, col] = C[col, col]

        PROD = TMP[col+1:, 0:col]*TMP[col, 0:col].T

        SUM = PROD.sum(axis=1)

        pref = 1/(C[col, col])

        C[col+1:, col] = pref*(TMP[col+1:, col] - SUM)
        TMP[col+1:, col] = C[col+1:, col]

        C[col, col+1:] = ZEROES[col, col+1:]

    return C


def sol_chol_dec(SYSTEM_MATRIX: ssp.csc_matrix,
                 FORCING_VECTR: ssp.csc_matrix) -> Tuple[ssp.csc_matrix, dict, int]:
    """Solves the system by decomposing the SYSTEM_MATRIX to its cholesky components and then solves using forward and backwards subsitution.

    Parameters
    ----------
    SYSTEM_MATRIX : ssp.csc_matrix
        The element matrix
    FORCING_VECTR : ssp.csc_matrix
        The element vector

    Returns
    -------
    ssp.csc_matrix
        Solution vector of size FORCING_VECTR
    """

    start_time = time_ns()

    sys_mat_fact = cholesky(SYSTEM_MATRIX,
                            beta=0,
                            ordering_method='natural')
    LOWER = sys_mat_fact.L()
    UPPER = LOWER.T

    SYSTEM_MATRIX = None

    fact_time = time_ns() - start_time
    start_time = time_ns()

    TMP = spsolve(LOWER, FORCING_VECTR)

    forw_time = time_ns() - start_time

    SOL = spsolve(UPPER, TMP)

    back_time = time_ns() - start_time - forw_time

    time_dict = {"decomposition": fact_time/1e6,
                 "forward": forw_time/1e6,
                 "backward": back_time/1e6}

    fill_in = LOWER.count_nonzero()

    return SOL, time_dict, fill_in
