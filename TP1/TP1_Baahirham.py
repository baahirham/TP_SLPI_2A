import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import math as math
from module_solve import *
from time import process_time

n = 50

# Question 1

# Permet de créer la matrice du laplacian en dense de taille n
def LaplacianMatrix(n) :
    Diag_n = np.ones(n**2-1)
    Diag_n[n-1::n] = 0.0
    A = np.diag(4*np.ones(n**2)) - np.diag(np.ones(n**2-n) , n) - np.diag(np.ones(n**2-n),-n) - np.diag(Diag_n, -1) - np.diag(Diag_n, 1)
    A = (n+1)**2*A
    return A

# Question 2

# Permet de définir le second membre
def Source(n) :
    F = np.zeros(n**2)
    F[int((n**2)/2+n/2)] = 1
    return F

# Code ci-dessous pour vérifier
# u = np.linalg.solve(LaplacianMatrix(n) , Source(n))
# plt.figure(1)
# plt.imshow(np.reshape(u,(n,n)),cmap = plt.cm.hot, extent=[x[1], x[-2], x[1], x[-2]])
# plt.colorbar()
# plt.title('Distribution de la température $u$ sur $\Omega$')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.show()

# Question 3

# Permet de créer la matrice du laplacian en csr de taille n
def LaplacianSparseMatrix(n) :
    Diag_n = np.ones(n**2-1)
    Diag_n[n-1::n] = 0.0
    data = np.zeros([5, n**2])
    data[0, :] = 4*np.ones(n**2)
    data[1, :] = -np.concatenate((Diag_n, [0]))
    data[2, :] = -np.concatenate(([0], Diag_n))
    data[3, :] = -np.concatenate((np.ones(n**2-n), np.zeros(n)))
    data[4, :] = -np.concatenate((np.zeros(n), np.ones(n**2-n)))
    A = (n+1)**2*sp.sparse.spdiags(data, [0, -1, 1, -n, n], n**2, n**2).tocsr()
    return A 

# Question 4

# Code ci-dessous pour vérifier
# u = sp.sparse.linalg.spsolve(LaplacianSparseMatrix(n), Source(n))
# plt.figure(2)
# plt.imshow(np.reshape(u,(n,n)),cmap = plt.cm.hot)
# plt.show()

# Question 5

# Préconditionneur identité
class id_precond :
    def apply(self, x) :
        return x

# Gradient conjugué
def conjugate_gradient(A, b, x0, prec, epsilon, Nmax) :
    residu_rel = []
    if (np.linalg.norm(b) == 0) :
        x = 0
        return x
    x = x0
    r = b - A.dot(x0)
    k = 0
    while ((np.linalg.norm(r)/np.linalg.norm(b) > epsilon) and (k < Nmax)) :
        z = prec.apply(r)
        rho = np.dot(r,z)
        if (rho == 0) :
            print("Echec de l'algo")
            return 0
        if (k == 0) :
            p = z 
        else :
            gamma = rho/rho0
            p = gamma*p + z
        q = A.dot(p)
        delta = np.dot(p,q)
        if (delta == 0) :
            print("Echec de l'algo")
            return 0
        residu_rel.append(np.linalg.norm(r)/np.linalg.norm(b))
        alpha = rho/delta
        x = x + alpha*p 
        r = r - alpha*q 
        rho0 = rho 
        k = k + 1
    return x, residu_rel

# Code ci-dessous pour vérifier
# prec_id = id_precond()
# x0 = np.zeros(n**2)
# b = Source(n)
# A = LaplacianSparseMatrix(n)
# epsilon = 1e-6; Nmax = 1000
# t1 = process_time()
# x , residu = conjugate_gradient (A , b , x0 , prec_id , epsilon , Nmax )
# t2 = process_time()
# print('Temps de calcul : ', (t2-t1)*1000., 'ms')

# plt.figure(3)
# plt.imshow(np.reshape(x,(n,n)),cmap = plt.cm.hot)

# plt.figure(4)
# k = np.arange(1,np.size(residu)+1,1)
# plt.plot(k, residu)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\frac{||r||}{||b||}$')
# plt.title(r'Résidu relatif $\frac{||r||}{||b||}$ en fonction du nombre ' 'd\'itérations $k$')
# plt.grid()
# plt.show()

# Question 6

# Factorisation de Cholesky incomplète
def Choleski_Facto(A) :
    n = int(math.sqrt(np.shape(A)[0]))
    data = np.zeros([3, n**2])
    for i in range(n**2) :
        if (i > n-1) :
            data[0,i] = (A[i,i] - data[1,i-1]**2 - data[2,i-n]**2)**0.5
        elif ((i > 0) and (i < n)):
            data[0,i] = (A[i,i] - data[1,i-1]**2)**0.5
        else :
            data[0,i] = A[i,i]**0.5
        if (i < n**2-1) :
            data[1,i] = A[i+1,i]/data[0,i]
        if (i < n**2-n) :
            data[2,i] = A[i+n,i]/data[0,i]
    L = sp.sparse.spdiags(data,[0 , -1 , -n],n**2,n**2).tocsr()
    return L

# Code ci-dessous pour vérifier
# A = LaplacianSparseMatrix(n)
# L = Choleski_Facto(A)
# print(A.todense())
# print("----------------")
# print((L@L.transpose()).todense())

# Question 7

# Préconditionneur de ICF
class cho_precond :
    def apply(self, x) :
        L = Choleski_Facto(A)
        y = solve_tri(L, x, lower=True)
        r = solve_tri(L.T.tocsr(), y, lower=False)
        return r

# Code ci-dessous pour vérifier
# A = LaplacianSparseMatrix(n)
# L = Choleski_Facto(A)
# prec_cho = cho_precond()
# x0 = np.zeros(n**2)
# b = Source(n)
# epsilon = 1e-6; Nmax = 1000
# t1 = process_time()
# x , residu = conjugate_gradient (A , b , x0 , prec_cho, epsilon , Nmax )
# t2 = process_time()
# print('Temps de calcul : ', (t2-t1)*1000., 'ms')

# plt.figure(5)
# plt.imshow(np.reshape(x,(n,n)),cmap = plt.cm.hot)

# plt.figure(6)
# k = np.arange(1,np.size(residu)+1,1)
# plt.plot(k, residu)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\frac{||r||}{||b||}$')
# plt.title(r'Résidu relatif $\frac{||r||}{||b||}$ en fonction du nombre ' 'd\'itérations $k$')
# plt.grid()
# plt.show()

# Question 8

# Méthode de Jacobi
def Jacobi(A, b, x0, epsilon, Nmax) :
    residu_rel = []
    x = x0
    r = b - A.dot(x)
    k = 0
    D = A.diagonal()
    D_1 = np.diag(1/D)
    while ((np.linalg.norm(r)/np.linalg.norm(b) > epsilon) and (k < Nmax)) :
        x = x + np.dot(D_1,r)
        r = b - A.dot(x)
        residu_rel.append(np.linalg.norm(r)/np.linalg.norm(b))
        k = k + 1
    return x ,residu_rel

# Code ci-dessous pour vérifier
# x0 = np.zeros(n**2)
# b = Source(n)
# A = LaplacianSparseMatrix(n)
# epsilon = 1e-6; Nmax = 1000
# t1 = process_time()
# x, residu = Jacobi(A, b, x0, epsilon, Nmax)
# t2 = process_time()
# print('Temps de calcul : ', (t2-t1)*1000., 'ms')


# plt.figure(7)
# plt.imshow(np.reshape(x,(n,n)),cmap = plt.cm.hot)

# plt.figure(8)
# k = np.arange(1,np.size(residu)+1,1)
# plt.plot(k, residu)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\frac{||r||}{||b||}$')
# plt.title(r'Résidu relatif $\frac{||r||}{||b||}$ en fonction du nombre ' 'd\'itérations $k$')
# plt.grid()
# plt.show()

# Question 9


def Direct_Relaxation(A, b, x0, w, epsilon, Nmax) :
    residu_rel = []
    x = x0
    r = b - A.dot(x)
    k = 0
    D = np.diag(A.diagonal())
    M = (1/w)*sp.sparse.csr_matrix(D)+sp.sparse.tril(A, -1)
    M_1 = sp.sparse.linalg.inv(M)
    while ((np.linalg.norm(r) > epsilon) and (k < Nmax)) :
        x = x + M_1.dot(r)
        r = b - A.dot(x)
        residu_rel.append(np.linalg.norm(r)/np.linalg.norm(b))
        k = k + 1
    return x ,residu_rel

# Code ci-dessous pour vérifier
# w = 2/(1+math.sin(math.pi*(1/(n+1))))
# x0 = np.zeros(n**2)
# b = Source(n)
# A = LaplacianSparseMatrix(n)
# epsilon = 1e-6; Nmax = 10000
# t1 = process_time()
# x, residu = Direct_Relaxation(A, b, x0, w, epsilon, Nmax)
# t2 = process_time()
# print('Temps de calcul : ', (t2-t1)*1000., 'ms')

# plt.figure(7)
# plt.imshow(np.reshape(x,(n,n)),cmap = plt.cm.hot)

# plt.figure(8)
# k = np.arange(1,np.size(residu)+1,1)
# plt.plot(k, residu)
# print(np.size(residu))
# plt.xlabel(r'$k$')
# plt.ylabel(r'$\frac{||r||}{||b||}$')
# plt.title(r'Résidu relatif $\frac{||r||}{||b||}$ en fonction du nombre ' 'd\'itérations $k$')
# plt.grid()
# plt.show()








