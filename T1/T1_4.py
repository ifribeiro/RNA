"""
Primeiro trabalho: Descida de gradiente (Questão 4)
Disciplina: Redes Neurais Artificiais
Prof.: Thomas Rauber
Grupo:
- Alan Carlos Pereira Pinto
- Alexander de Oliveira da Silva
- Iran Freitas Ribeiro
"""

from numpy.linalg import norm
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

np.random.seed(0)

def f(x1, x2):
    return (4.0 - 2.1 * x1**2 + x1**2/ 3.0 ) * x1**3 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2

def dif_finitas(func, x1_k, x2_k, h):
    """
    Método de diferenças finitas
    """
    gradiente = np.array([0.0,0.0])
    gradiente[0] = (func(x1_k+h, x2_k) - func(x1_k, x2_k))/h
    gradiente[1] = (func(x1_k, x2_k+h) - func(x1_k, x2_k))/h
    return gradiente

def grad_f(f1, f2, x1_k, x2_k):
    gradiente = np.array([f1(x1_k, x2_k),f2(x1_k, x2_k)])
    return gradiente

def gradient_descent(x1_0, x2_0, f, alpha, gmin, kmax, diff_fin=True):
    """
    Descida do gradiente

    Params:

        - x1_0: valor inicial x1
        - x2_0: valor inicial x2
        - f: função (para o método de diferenças finitas) ou lista de derivadas
        - alpha: velocidade da descida
        - gmin: limiar
        - diff_fin: usar diferenças finitas ou não
    """
    k = 0    
    X = np.zeros((kmax+1, 2))
    # print (x1_0, x2_0)
    X[0][0], X[0][1] = [x1_0, x2_0]
    if (diff_fin):
        g = dif_finitas(f, X[k][0],X[k][1], 0.01)
    else:
        # para derivadas, f será um vetor com 2 funções
        g = grad_f(f[0], f[1], X[k][0],X[k][1])
    while (k < kmax) and (norm(g)>gmin):        
        # se o gradiente for obtido por diferenças finitas
        if (diff_fin):
            g = dif_finitas(f, X[k][0],X[k][1], 0.01)
        else:           
            g = grad_f(f[0], f[1], X[k][0],X[k][1])
        X[k+1] = X[k] - alpha*g
        k+=1
    return X[:k]

def get_XYZ_3d(f, limx, limy, N):
    X1 = np.linspace(limx,limy,N)
    X2 = np.linspace(limx,limy,N)
    X1, X2 = np.meshgrid(X1, X2)
    zarray = []
    for i in range(len(X1)):
        zarray_interno = []
        for j in range(len(X1)):
            z = f(X1[i][j],X2[i][j])
            zarray_interno.append(z)
        zarray.append(zarray_interno)
    Z = np.array(zarray)
    return X1, X2, Z


def item_a(diff_fin=True, f=None):
    """
    Realiza a descida do gradiente utilizando difereças finitas
    """
    low=-3
    high=3
    X0 = np.random.uniform(low=low, high=high, size=2)    
    X1, X2, Z = get_XYZ_3d(f, -1.5, 1.2, 500)
    X_gradient = gradient_descent(X0[0], X0[1], f, 0.1, gmin=0.1, kmax=20)
    Y_gradient = [f(x[0], x[1]) for x in X_gradient]    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, alpha=0.4, cmap='jet', edgecolor=None)
    ax.plot(X_gradient[:, 0],X_gradient[:, 1], Y_gradient, label='Desc. Grad.')
    ax.scatter(X_gradient[0, 0],X_gradient[0, 1], Y_gradient[0], c='green', label='Início')
    ax.scatter(X_gradient[-1, 0],X_gradient[-1, 1], Y_gradient[-1], c='red', label='Fim')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    plt.legend()
    plt.title("Descida do Gradiente por diferenças finitas")
    plt.savefig("t1_4_item_A.jpeg")
    plt.show()
    return X_gradient, Y_gradient

def item_b(func):
    # criando os símbolos
    x1, x2 = symbols('x1 x2')
    # definindo a funcao
    f = (4.0 - 2.1 * x1**2 + x1**2 / 3.0 ) * x1**3 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2
    # derivadas parciais
    d1 = f.diff(x1)
    d2 = f.diff(x2) 
    fd1 = lambdify([x1, x2], d1) 
    fd2 = lambdify([x1, x2], d2) 
    X0 = np.random.uniform(low=-3, high=3, size=2)
    # X0 = [-1.0, -.5]
    X_gradient = gradient_descent(X0[0], X0[1], f=[fd1, fd2], alpha=0.1, gmin=0.1, kmax=20, diff_fin=False)
    Y_gradient = [func(x[0], x[1]) for x in X_gradient]

    X1, X2, Z = get_XYZ_3d(func, -1.5, 1.2, 500)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, alpha=0.4, cmap='jet', edgecolor=None)
    ax.plot(X_gradient[:, 0],X_gradient[:, 1], Y_gradient, label='Desc. Grad.')
    ax.scatter(X_gradient[0, 0],X_gradient[0, 1], Y_gradient[0], c='green', label='Início')
    ax.scatter(X_gradient[-1, 0],X_gradient[-1, 1], Y_gradient[-1], c='red', label='Fim')
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    plt.legend()
    plt.title("Descida do Gradiente por derivadas")
    plt.savefig("t1_4_item_B.jpeg")
    plt.show()
xg, yg = item_a(f=f)
item_b(f)