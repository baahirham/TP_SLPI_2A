from mod_solve_triangular import *

# resolution par un systeme triangulaire
def solve_tri(A, b, lower = True):
    if (A.format != 'csr'):
        print("A must be in CSR format")
        return b

    x = b.copy()
    if (lower):
        mod_solve_triangular.solve_triangular_lower(A.data, A.indices, A.indptr, x)
    else:
        mod_solve_triangular.solve_triangular_upper(A.data, A.indices, A.indptr, x)

    return x

