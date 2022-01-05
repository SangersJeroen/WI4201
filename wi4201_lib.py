# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as ssp
import scipy.linalg as salg
from scipy.sparse.linalg import spsolve, splu, inv
from collections.abc import Callable
from sksparse.cholmod import cholesky

# %%
#Defining the constants
P = 2		#Power of two
N = 2**P	#Number of subdivisions
U0 = 0.01 	#The border constant we might need to use?
h = 1/N		#Discretisation step

DEBUG = True

# %%

def get_elem_mat(N: int, dimension: str) -> ssp.csr_matrix:
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


	ONE_DIM_DIFF	= ssp.csr_matrix(one_dim_diff.astype(int))
	ID 		= ssp.csr_matrix(np.eye(N).astype(int))

	x_dim_diff = (1/(h**2) * ssp.kron(ONE_DIM_DIFF,ID))
	y_dim_diff = (1/(h**2) * ssp.kron(ID, ONE_DIM_DIFF))

	X_DIM_DIFF 	= ssp.csr_matrix(x_dim_diff.astype(int))
	Y_DIM_DIFF 	= ssp.csr_matrix(y_dim_diff.astype(int))
	TWO_LAPLACE 	= ssp.csr_matrix(x_dim_diff + y_dim_diff)

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

		Z_DIM_DIFF 	= ssp.csr_matrix(z_dim_diff.astype(int))
		z_dim_diff = None

		THREE_LAPLACE = (ssp.kron(X_DIM_DIFF, ID)
				+ssp.kron(Y_DIM_DIFF, ID)
				+Z_DIM_DIFF)

		return THREE_LAPLACE

	else:
		raise ValueError('string: Dimension, either "2D" or "3D"')


# %%
def build_forcing_vector(lin_spaces: list,
			 internal_fun,
			 boundary_fun  = None
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
	mask = tuple([slice(1,-1)]*dims)
	forcing, forcing[mask] = boundary_forcing_array, internal_forcing_array[mask]
	print(mask)

	forcing_vector = forcing.flatten()
	return forcing_vector


# %%
%%time

def int_forc_fun(x,y):
	return (x**2 + y**2)*np.sin(x*y)

def bound_forc_fun(x,y):
	return np.sin(x*y)

x = np.linspace(0,1,N+1)
y = np.linspace(0,1,N+1)

X, Y = np.meshgrid(x,y)

# %%
vecF = build_forcing_vector([x,y], int_forc_fun, bound_forc_fun)

# %%
%%time
TWO_LAPLACE = get_elem_mat(N+1, "2D")

# %%
%%time
#Manipulating the 2D-laplacian and the forcing vector to obey boundary elements
#By selecting the boundary points
Xval, Yval = X.ravel(), Y.ravel()

boundary_list = np.squeeze(
	np.where(
		(Xval==x[0]) | (Xval==x[-1]) | (Yval==y[0]) | (Yval==y[-1])
	)
);

SPARSE_ID = ssp.eye((N+1)**2).tocsr()
for row in boundary_list:
	TWO_LAPLACE[row,:] = SPARSE_ID[row,:]
SPARSE_ID = None



# %%
size = (N+1)**2

TWO_LAPLACE = TWO_LAPLACE.tocsc()
P_MATRIX = ssp.csc_matrix((size,size))


TWO_LAPLACE.shape


# %%
ONES = ssp.csr_matrix((size,size))
ONES[boundary_list,boundary_list] = 1


# %%
%%time

ZEROES = ssp.csc_matrix((size,size))

for column in boundary_list:
	P_MATRIX[:, column] = TWO_LAPLACE[:, column]
	TWO_LAPLACE[:,column] = ZEROES[:,column]

print(type(TWO_LAPLACE))

TWO_LAPLACE = (TWO_LAPLACE + ONES)
P_MATRIX = (P_MATRIX - ONES)

print(type(TWO_LAPLACE))

# %%
plt.imshow(TWO_LAPLACE.toarray())
plt.colorbar()

# %%
pm = P_MATRIX

# %%
P_MATRIX.shape
plt.imshow(TWO_LAPLACE.toarray())
plt.colorbar()

# %%
#Building u0
U0 = np.sin(X*Y)
u0 = U0.flatten()
to_subtract = P_MATRIX.dot(u0)


# %%
vecF = np.subtract(vecF, to_subtract)

# %%
TWO_LAPLACE.shape

# %%


# %%
%%time
u = spsolve(TWO_LAPLACE, vecF.T)

# %%
array_u = u.reshape((N+1,N+1))
plt.imshow(array_u, origin='lower');
plt.colorbar()

# %%
u_ex = np.sin(X*Y)
vec_u_ex = u_ex.flatten()

# %%
error = np.sqrt(h**2 *np.sum((vec_u_ex-u)**2))
print("h**2: {:.5f}\t error: {:.5f}".format(h**2, error))
print('relative error: {:.4f}\t [h**2]'.format(error/(h**2)))

# %%
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
	#In this code block all fully uppercase variables are sparse matrices
	shape_m = M.shape
	ID = ssp.eye(shape_m[0])
	ZEROES = ssp.csc_matrix(shape_m)

	SUM = ID.copy()
	for k in range(0, shape_m[1]):
		Akk = M[k,k]
		VEC = M[:,k] / Akk
		VEC[0: k+1] = ZEROES[0:k+1, 0]

		EK = ZEROES[:,0].copy()
		EK[k] = 1

		ADD = VEC * EK.T
		SUM += ADD

	L = SUM
	U = spsolve(SUM, M)

	return L, U

# %%
from tqdm import tqdm

# %%
##Mocht je hier een error krijgen dan moet je waarschijnlijk even het volgende veranderen
## tqdm(range(...)) -> range(...)


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

	for col in tqdm(range(0, shape_m[1])): #Hierzo
		C[col,col] = np.sqrt((TMP[col,col] - ( C[col,0:col].power(2) ).sum()))
		TMP[col,col] = C[col,col]

		PROD = TMP[col+1:,0:col]*TMP[col,0:col].T

		SUM = PROD.sum(axis=1)

		pref = 1/(C[col,col])

		C[col+1:,col] = pref*(TMP[col+1:,col] - SUM)
		TMP[col+1:,col] = C[col+1:, col]

		C[col,col+1:] = ZEROES[col,col+1:]

	return C


# %%
TWO_LAPLACE.tocsc()

# %%

L, U = decomp_lu(TWO_LAPLACE)
D = ssp.diags(U.diagonal(k=0))

plt.imshow(D.toarray())
plt.colorbar()

sqrtD = D.sqrt()

C = L@sqrtD

# %%
Utest = D@L.T

# %%
np.alltrue(Utest.toarray() == U.toarray())

# %%
LU = L@U

np.alltrue(TWO_LAPLACE.toarray() == LU.toarray())

# %%
np.all(np.linalg.eigvals(TWO_LAPLACE.toarray())>0)

# %%
%%time
##Mocht je een error krijgen, lees dan de comment in de cell hierboven
#C = decomp_cholesky(TWO_LAPLACE)

# %%
%%time

#C_factor = cholesky(TWO_LAPLACE, beta=0, ordering_method='natural')


# %%
#C2 = C_factor.L()

# %%
#np.alltrue(C.toarray() == C2.toarray())

# %% [markdown]
# plt.close()
# plt.imshow(C2.toarray())
# plt.colorbar()

# %%
plt.imshow(C.toarray())
plt.colorbar()

# %%
plt.imshow(TWO_LAPLACE.toarray())
plt.colorbar()

# %%
PLZ_WORK = C@C.T
plt.imshow(PLZ_WORK.toarray())
plt.colorbar()

# %%
np.isclose(TWO_LAPLACE.toarray(), PLZ_WORK.toarray())

# %% [markdown]
# MAYBE = C2@C2.T
# plt.imshow(MAYBE.toarray())
# plt.colorbar()

# %% [markdown]
# U_MAYBE = spsolve(MAYBE, vecF)
#
# u_maybe = (U_MAYBE).reshape((N+1,N+1))
# plt.imshow(u_maybe, origin='lower')
# plt.colorbar()

# %%
plt.imshow(TWO_LAPLACE.toarray())
plt.colorbar()

# %%
from time import time_ns

# %%
def sys_solve_chol(f: ssp.csc_matrix, L: ssp.csc_matrix, U: ssp.csc_matrix) -> ssp.csc_matrix:
	LOWER = L #Forward matrix
	UPPER = U #Backward

	startt = time_ns()
	b = spsolve(LOWER, f)
	forwt = time_ns() - startt
	sol = spsolve(UPPER, b)
	backt = time_ns() - startt - forwt

	statement = "Forward step took: {:.2f} ms, Backward step took: {:.2f} ms".format(forwt/1e6, backt/1e6)

	return sol, statement


# %%
%%time

u, state = sys_solve_chol(vecF, L, U)

print(state)

# %%
array_u = u.reshape((N+1,N+1))

plt.imshow(array_u, origin='lower')
plt.colorbar()

# %%
#nonzero counting

nnzA = TWO_LAPLACE.count_nonzero()
nnzC = C.count_nonzero()

fill_ratio = nnzC / nnzA

print(fill_ratio)

# %%
E = -1*ssp.tril(TWO_LAPLACE, k=-1)
F = -1*ssp.triu(TWO_LAPLACE, k=1)
D = ssp.diags(TWO_LAPLACE.diagonal())

# %%
plt.imshow(E.toarray())
plt.colorbar()

# %%
plt.imshow(F.toarray())
plt.colorbar()

# %%
plt.imshow(D.toarray())
plt.colorbar()

# %%
omega = 1.5

MINV = omega*(2-omega) * inv(D - omega* E) @ D @ inv(D - omega*F)

shape = MINV.shape
ID = ssp.csc_matrix(shape)

# %%
u0 = ssp.csc_matrix(np.zeros(shape[0]))
r0 = ssp.csc_matrix(np.ones(shape[0]))
f = ssp.csc_matrix(vecF)

print(vec_u_ex.shape)
print(u0.shape)

epsilon = 1e-10


# %%
def get_norm(vector: ssp.csc_matrix) -> float:
	norm = np.sqrt((vector.power(2)).sum())
	return norm

# %%
"""norm_f = get_norm(f)
norm_r = get_norm(r0)

u = u0.T
r = f.T


i = 0
for i in range(40000):
	u = u + MINV@r
	r = COMP@r

	norm_r = get_norm(r)

	if norm_r / norm_f < epsilon:
		break

print(norm_r/norm_f)
print(i)
"""

COMP = (ID - TWO_LAPLACE@MINV)

norm_f = get_norm(f)
norm_r = get_norm(r0)

u = u0.T
r = f.T

r_norms = np.asarray([norm_r])

while not norm_r/norm_f < epsilon:
	u = u + MINV@r
	r = COMP@r

	norm_r = get_norm(r)
	r_norms = np.append(r_norms, norm_r)

# %%

len(r_norms)

# %%
plt.plot(np.arange(len(r_norms)), r_norms)
plt.semilogy()
plt.ylabel(r"$\frac{|r_m|_2}{|f^h|_2}$")
plt.xlabel(r"iteration $m$")
plt.show()

# %%
"""
norm_f = get_norm(f)
norm_r = get_norm(r0)

u = u0.T
r = f.T

COMP = (ID - TWO_LAPLACE@MINV)

i = 0
A = TWO_LAPLACE
for iter in range(2000):
	uold = u
	for i in range(shape[0]):
		sub1 = A[i,:i]@u[:i]
		print(sub1.shape)
		sub2 = A[i,i+1:]@u[i+1:]/A[i,i]
		print(sub2.shape)
		u[i] = f[i] - sub1[0,0] - sub2[0,0]
	u = (1-omega)*uold + omega*u

	if norm_r / norm_f < epsilon:
		break
"""

# %%
plt.imshow(vec_u_ex.reshape((N+1,N+1)))
plt.colorbar()

# %%
plt.imshow((u.toarray()).reshape((N+1,N+1)))
plt.colorbar()

# %%
#Preconditioned Conjugate-Gradient
u = u0.T
r = f.T

norm_f = get_norm(f)
norm_r = get_norm(r0)

r_prev = 0
z_prev = 0
p = 0

i = 0
while not norm_r/norm_f < epsilon:
	r_pprev = r_prev
	r_prev = r
	z_pprev = z_prev
	p_prev = p

	z_prev = MINV@r_prev

	if i == 0:
		p = z_prev
	else:
		beta = (r_prev.T @ z_prev)/(r_pprev.T @ z_pprev)
		p = z_prev + beta[0,0]*p_prev

	alpha = (r_prev.T @ z_prev)/(p.T @ TWO_LAPLACE @ p)
	u += alpha[0,0]*p
	r += -alpha[0,0]*(TWO_LAPLACE@p)
	i += 1
	norm_r = get_norm(r)

print(i)

# %%
plt.imshow((u.toarray()).reshape((N+1,N+1)))
plt.colorbar()

# %%



