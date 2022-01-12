import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

powers_2D = [2,3,4,5,6,7,8,9]
powers_3D = [2,3,4,5,6]

direct_solver_time_2D = np.asarray([
	0.325136,
	0.200421,
	0.500151,
	1.859381,
	9.050068,
	259.257212,
	1923.719031,
	7787.146169
	])

direct_solver_time_3d = np.asarray([
	0.192483,
	1.057376,
	1844.805921,
	35203.181124,
	2436528.440830
])

direct_solver_error_2D = np.asarray([
	0.0000437,
	0.0000131,
	0.0000034,
	0.0000009,
	0.0000002,
	0.0000001,
	0.0000000,
	0.0000000
])

direct_solver_decomp_2D = np.asarray([
	2.614613,
	0.250215,
	0.316298,
	0.856236,
	8.649238,
	182.426165,
	1569.302678,
	22302.242172

])

direct_solver_forwrd_2D = np.asarray([
	0.224782,
	0.166152,
	0.132293,
	0.217646,
	0.584191,
	3.520188,
	20.735393,
	146.78445
])

direct_solver_bckwrd_2D = np.asarray([
	0.133961,
	0.074228,
	0.067651,
	0.11173,
	0.38172,
	2.658521,
	18.792996,
	144.507891
])

direct_solver_decomp_3D = np.asarray([
	0.256913,
	1.618198,
	142.444764,
	8446.457395,
	1001853.301098
])

direct_solver_forwrd_3D = np.asarray([
	0.124734,
	0.234375,
	1.669368,
	26.535671,
	725.276941
])

direct_solver_bckwrd_3D = np.asarray([
	0.06231,
	0.139923,
	1.240318,
	30.901285,
	1004.673963
])

direct_solver_decomp_2D_red = np.asarray([
	0.152134,
	0.172268,
	0.240675,
	0.597435,
	5.875274,
	65.153858,
	887.975439,
	12091.6085
])

direct_solver_forwrd_2D_red = np.asarray([
	0.111223,
	0.115807,
	0.132208,
	0.219842,
	0.538714,
	2.464024,
	16.297206,
	131.08241
])

direct_solver_bckwrd_2D_red = np.asarray([
	0.059819,
	0.060718,
	0.072318,
	0.130297,
	0.467551,
	1.954503,
	13.83364,
	115.594178
])

direct_solver_decomp_3D_red = np.asarray([
	0.239088,
	0.540723,
	25.276355,
	2937.07454,
	391262.189
])

direct_solver_forwrd_3D_red = np.asarray([
	0.11997,
	0.192803,
	1.024421,
	16.731293,
	500.168987
])

direct_solver_bckwrd_3D_red = np.asarray([
	0.060879,
	0.081378,
	0.75415,
	18.489834,
	605.985535
])

fill_in = np.asarray([
	0.92,
	1.53,
	3.06,
	6.23,
	12.61,
	25.41,
	51,
	102.2,
	1.32,
	6.19,
	30.10,
	132.93,
	558.05
])

fill_in_red = np.asarray([
	0.90,
	1.31,
	2.33,
	4.45,
	8.71,
	17.24,
	34.30,
	68.43,
	1.19,
	4.25,
	18.25,
	76.53,
	313.81
])

problem_size_2D = 2**np.asarray(powers_2D)
problem_size_3D = 2**np.asarray(powers_3D)

problem_sizes = np.append(problem_size_2D, problem_size_3D)

#CPU time per N for decomp

plt.plot(problem_size_2D, direct_solver_decomp_2D, color='blue', label=r'2D')
plt.plot(problem_size_3D, direct_solver_decomp_3D, color='red', label=r'3D')
plt.title('3')
plt.ylabel(r'factorisation time [$ms$]')
plt.xlabel(r'Problem size N')
plt.semilogy()
plt.semilogx()
plt.legend()
plt.savefig('plots/3a.pdf')
plt.close()

#CPU solving time epr N

plt.plot(problem_size_2D, direct_solver_forwrd_2D, color='darkblue', label=r'2D - fw')
plt.plot(problem_size_2D, direct_solver_bckwrd_2D, color='blue', label=r'2D - bw')
plt.plot(problem_size_3D, direct_solver_forwrd_3D, color='red', label=r'3D - fw')
plt.plot(problem_size_3D, direct_solver_bckwrd_3D, color='darkred', label=r'3D - bw')
plt.title('3')
plt.ylabel(r'solving time [$ms$]')
plt.xlabel(r'Problem size N')
plt.semilogy()
plt.semilogx()
plt.legend()
plt.savefig('plots/3b.pdf')
plt.close()

# NNZ

plt.plot(problem_size_2D, fill_in[:8], color='blue', label='2D')
plt.plot(problem_size_3D, fill_in[8:], color='red', label='3D')
plt.title('4')
plt.ylabel(r'fill-in factor $\frac{nnz(C)}{nnz(A)}$ [$-$]')
plt.xlabel(r'Problem size N')
plt.semilogx()
plt.legend()
plt.savefig('plots/4.pdf')
plt.close()

#After bandwitch reduction
plt.plot(problem_size_2D, direct_solver_decomp_2D, color='deepskyblue', label=r'2D')
plt.plot(problem_size_3D, direct_solver_decomp_3D, color='indianred', label=r'3D')
plt.plot(problem_size_2D, direct_solver_decomp_2D_red, color='blue', label=r'2D - red')
plt.plot(problem_size_3D, direct_solver_decomp_3D_red, color='red', label=r'3D - red')
plt.title('3')
plt.ylabel(r'factorisation time [$ms$]')
plt.xlabel(r'Problem size N')
plt.semilogy()
plt.semilogx()
plt.legend()
plt.savefig('plots/5-3a.pdf')
plt.close()

#CPU solving time epr N

plt.plot(problem_size_2D, direct_solver_forwrd_2D, color='steelblue', label=r'2D - fw')
plt.plot(problem_size_2D, direct_solver_bckwrd_2D, color='deepskyblue', label=r'2D - bw')
plt.plot(problem_size_3D, direct_solver_forwrd_3D, color='indianred', label=r'3D - fw')
plt.plot(problem_size_3D, direct_solver_bckwrd_3D, color='firebrick', label=r'3D - bw')
plt.plot(problem_size_2D, direct_solver_forwrd_2D_red, color='darkblue', label=r'2D - fw -red')
plt.plot(problem_size_2D, direct_solver_bckwrd_2D_red, color='blue', label=r'2D - bw -red')
plt.plot(problem_size_3D, direct_solver_forwrd_3D_red, color='red', label=r'3D - fw- red')
plt.plot(problem_size_3D, direct_solver_bckwrd_3D_red, color='darkred', label=r'3D - bw-red')
plt.title('5 - 3')
plt.ylabel(r'solving time [$ms$]')
plt.xlabel(r'Problem size N')
plt.semilogy()
plt.semilogx()
plt.legend()
plt.savefig('plots/5-3b.pdf')
plt.close()

# NNZ

plt.plot(problem_size_2D, fill_in[:8], color='deepskyblue', label='2D')
plt.plot(problem_size_3D, fill_in[8:], color='indianred', label='3D')
plt.plot(problem_size_2D, fill_in_red[:8], color='blue', label='2D red')
plt.plot(problem_size_3D, fill_in_red[8:], color='red', label='3D red')
plt.title('5 - 4')
plt.ylabel(r'fill-in factor $\frac{nnz(C)}{nnz(A)}$ [$-$]')
plt.xlabel(r'Problem size N')
plt.semilogx()
plt.legend()
plt.savefig('plots/5-4.pdf')
plt.close()