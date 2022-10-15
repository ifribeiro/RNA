import numpy as np
from sympy import *

np.random.seed(18)

def f(x1, x2):
    return (4 - 2.1 * x1**2 + x1**2 / 3 ) * x1**3 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
def dif_finitas(func, x1_k, x2_k, h):
    """
    Método de diferenças finitas
    """
    d = func(x1_k+h, x2_k+h) - f(x1_k, x2_k)
    return d

def gradient_descent(x1_0, x2_0, f, alpha, gmin, kmax, diff_fin=True):
    """
    x1_0: valor inicial x1
    x2_0: valor inicial x2
    f: derivada da função f (lambdify)
    """
    k = 0
    xk = np.array([x1_0, x2_0])
    X = []
    while (k < kmax) and (f(xk[0],xk[1])>gmin):
        X.append(xk)
        # se o gradiente for obtido por diferenças finitas
        if (diff_fin):
            # usar função normal
            g = dif_finitas(f,xk[0], xk[1], 0.01)
        else:   
            # usar função simbolica (com sympy)          
            g = f(xk[0],xk[1])
        xk -= alpha*abs(g)
        k+=1
        print (g)
    return X


def item_a():
    X0 = np.random.uniform(low=-3, high=5, size=2)
    X = gradient_descent(X0[0], X0[1], f, 0.1, gmin=0.1, kmax=20)
    print (X)

def item_b():
    # criando um símbolo
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    # definindo a funcao
    f = (4 - 2.1 * x1**2 + x1**2 / 3 ) * x1**3 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
    f1 = lambdify([x1, x2],f)
    X0 = np.random.uniform(low=-1, high=2, size=2)
    print (X0)
    # X0 = [-1.0, -.5]
    Xgradiente = gradient_descent(X0[0], X0[1], f=f1, alpha=0.1, gmin=0.1, kmax=20, diff_fin=False)
def item_c():
    print ("Item c")


# item_a()
item_b()
# item_c()