import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from time import process_time

## Question 1

# Fonction pour la lecture des matrices 
def loadMat(nom_fichier):
    A = np.loadtxt (nom_fichier)
    B = sp.sparse.coo_matrix ((A[:,2] ,( np.array(A[:,0]-1 , dtype = 'int64'), np.array(A[:,1] -1 , dtype = 'int64'))))
    return B.tocsr()

# Verification
# print(loadMat('KhInit.dat'))

## Question 2

# Algo du BiCg 
def BiCg(A, b, x0, eps, Nmax) :
    residu = []
    it = []
    r = b - A.dot(x0)
    x = x0
    r_tilde = r
    p = r
    p_tilde = r 
    k = 1
    while ((np.linalg.norm(r)/np.linalg.norm(b) > eps) and (k <= Nmax)) :
        rho_0 = np.dot(r,r_tilde)
        q = A.dot(p)
        q_tilde = A.transpose().dot(p_tilde)
        alpha = np.dot(r,r_tilde)/np.dot(q,p_tilde)
        x = x + alpha*p
        r = r - alpha*q 
        r_tilde = r_tilde - alpha*q_tilde
        rho = np.dot(r,r_tilde)
        beta = rho/rho_0
        p = r + beta*p 
        p_tilde = r_tilde + beta*p_tilde
        k = k + 2
        residu.append(np.linalg.norm(r)/np.linalg.norm(b))
        it.append(k)
    return x, residu, it

## Question 3

# Verification sur un système aléatoire

# eps = 1e-5
# Nmax = 6000
# n = 50
# A = sp.sparse.csr_matrix(np.random.rand(n,n))
# b = np.ones(n)
# x0 = np.zeros(n)
# x, residu, k = BiCg(A, b, x0, eps, Nmax)
# plt.figure(1)
# plt.plot(np.arange(1,np.size(residu)+1,1), residu)
# plt.show()

# Convergence en 2n

# eps = 1e-5
# Nmax = 6000
# it_t = []
# n_t = []
# for n in range(1,51) :
#     A = sp.sparse.csr_matrix(np.random.rand(n,n))
#     b = np.random.rand(n)
#     x0 = np.zeros(n)
#     x, residu, k = BiCg(A, b, x0, eps, Nmax)
#     n_t.append(n)
#     it_t.append(k[-1])
# plt.figure(11)
# plt.plot(n_t,it_t,label='Itérations BiCg')
# plt.plot(n_t,np.multiply(n_t,2),label='$y=2n$')
# plt.grid()
# plt.legend()
# plt.title('Nombre d\'itérations $k$ en fonction de la taille de la matrice $n$')
# plt.ylabel('$k$')
# plt.xlabel('$n$')
# plt.show()

# Verification sur le vrai système

# eps = 1e-5
# Nmax = 5000
# delta_t = 1

# Un = np.loadtxt('U0_Init.dat')
# x0 = np.zeros(np.size(Un))

# A = loadMat('MhInit.dat') + delta_t/2*loadMat('KhInit.dat')
# b = (loadMat('MhInit.dat') - delta_t/2*loadMat('KhInit.dat')).dot(Un)

# t1 = process_time()
# Un1, residu, it = BiCg(A, b, x0, eps, Nmax)
# t2 = process_time()
# print('Temps de calcul : ', (t2-t1)*1000., 'ms')

# print(it[-1])

# plt.figure(2)
# plt.plot(it, residu)
# # plt.yscale("log")
# plt.grid()
# plt.show()

## Question 4

# Algo Gmres
def Gmres(A, b, x0, eps, Nmax, m) :
    r = b - A.dot(x0)
    x = x0
    n = 1
    while ((np.linalg.norm(r)/np.linalg.norm(b) > eps) and (n <= Nmax)) :
        beta = np.linalg.norm(r)
        v = [r/np.linalg.norm(r)]
        H_m = np.zeros((m+1,m))
        j = 0
        z = np.zeros(m+1)
        z[0] = beta
        s = np.zeros(m)
        c = np.zeros(m)
        while ((j < m) and (np.abs(z[j])/np.linalg.norm(b) > eps)) :
            w = A.dot(v[j])
            for i in range(j+1) :
                H_m[i,j] = np.dot(w,v[i])
                w = w - H_m[i,j]*v[i]
            H_m[i+1,j] = np.linalg.norm(w)
            v.append(w/H_m[j+1,j])
            for k in range(j) :
                L_1 = H_m[k,j]
                L_2 = H_m[k+1,j]
                H_m[k,j] = c[k]*L_1 - s[k]*L_2
                H_m[k+1,j] = s[k]*L_1 + c[k]*L_2
            alpha = np.sqrt(H_m[j,j]**2 + H_m[j+1,j]**2)
            c[j] = H_m[j,j]/alpha
            s[j] = -H_m[j+1,j]/alpha
            H_m[j,j] = alpha
            H_m[j+1,j] = 0
            z[j+1] = z[j]*s[j]
            z[j] = z[j]*c[j]
            j = j + 1
        y = sp.linalg.solve_triangular(H_m[0:j,0:j],z[0:j])
        for l in range(j) :
            x += y[l]*v[l]
        r = b - A.dot(x)
        n = n + j + 1
        residu = np.linalg.norm(r)/np.linalg.norm(b)
    return x, residu, n 


## Question 5

# Verification sur un système aléatoire

# eps = 1e-5
# Nmax = 5000
# m = 3000
# n = 100
# A = sp.sparse.csr_matrix(np.random.rand(n,n))
# b = np.ones(n)
# x0 = np.zeros(n)
# x, residu, it = Gmres(A, b, x0, eps, Nmax, m)
# print(x)
# print(sp.sparse.linalg.gmres(A, b, x0, tol=eps, restart=m, maxiter=Nmax)[0])

# Convergence en n

# eps = 1e-5
# Nmax = 5000
# m = 3000
# it_t = []
# n_t = []
# for n in range(1,51) :
#     print(n)
#     A = sp.sparse.csr_matrix(np.random.rand(n,n))
#     b = np.random.rand(n)
#     x0 = np.zeros(n)
#     x, residu, it = Gmres(A, b, x0, eps, Nmax, m)
#     n_t.append(n)
#     it_t.append(it)
# plt.figure(22)
# plt.plot(n_t,it_t,label='Itérations Gmres')
# plt.plot(n_t,np.multiply(n_t,1),label='$y=n$')
# plt.grid()
# plt.legend()
# plt.title('Nombre d\'itérations $k$ en fonction de la taille de la matrice $n$')
# plt.ylabel('$k$')
# plt.xlabel('$n$')
# plt.show()

# Verification sur le vrai système

# eps = 1e-5
# Nmax = 100000
# delta_t = 1
# m = 10

# Un = np.loadtxt('U0_Init.dat')
# x0 = np.zeros(np.size(Un))

# A = loadMat('MhInit.dat') + delta_t/2*loadMat('KhInit.dat')
# b = (loadMat('MhInit.dat') - delta_t/2*loadMat('KhInit.dat')).dot(Un)

# t1 = process_time()
# Un2, residu, k = Gmres(A, b, x0, eps, Nmax, m)
# t2 = process_time()
# print('Temps de calcul : ', (t2-t1)*1000., 'ms')

# print(k)

