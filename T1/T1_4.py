from re import X
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# np.random.seed(18)

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
    xk_1, xk_2 = [x1_0, x2_0]
    X = [[xk_1, xk_2]]
    while (k < kmax):        
        # se o gradiente for obtido por diferenças finitas
        # TODO: Rever lógica de parada para diferencas finitas
        if (diff_fin):
            g = dif_finitas(f,xk_1, xk_2, 0.01)
            print (abs(g), gmin)
            # print (abs(g)<=gmin)
            if abs(g) <= gmin: 
                print ("Parando antes")
                break
            # else: 
            #     # usar função normal
            #     g = dif_finitas(f,xk[0], xk[1], 0.01)
        else:   
            # usar função simbolica (definida com sympy)          
            g = f(xk_1,xk_2)
        xk_1 -= alpha*abs(g) 
        xk_2 -= alpha*abs(g) 
        X.append([xk_1, xk_2])
        k+=1
        # print (g)
    return X


def item_a(diff_fin=True, f=None):
    X0 = np.random.uniform(low=0, high=5.5, size=2)
    # X0 = [-0.99530908  1.30724695]
    # X0 = [0, -2]  
    # X0 = [1.90047238, 3.87944662]
    # X0 = [-0.64231882 , 0.57173843]
    X_gradient = gradient_descent(X0[0], X0[1], f, 0.2, gmin=0.1, kmax=70)
    # if diff_fin
    # TODO: fazer dois tipos
    Y_gradient = [f(x[0], x[1]) for x in X_gradient]
    return X_gradient, Y_gradient

def item_b():
    # criando um símbolo
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    # definindo a funcao
    f = (4 - 2.1 * x1**2 + x1**2 / 3 ) * x1**3 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
    f1 = lambdify([x1, x2],f)
    X0 = np.random.uniform(low=-1, high=1.1, size=2)
    print (X0)
    # X0 = [-1.0, -.5]
    Xgradiente = gradient_descent(X0[0], X0[1], f=f1, alpha=0.1, gmin=0.1, kmax=20, diff_fin=False)

X1 = np.linspace(-.75,.75,1000)
X2 = np.linspace(-.75,.75,1000)

X1, X2 = np.meshgrid(X1, X2)

# Z = [f(Xi[0], Xi[1]) for Xi in zip(X1, X2)]
zarray = []
for i in range(len(X1)):
    zarray_interno = []
    for j in range(len(X1)):
        z = f(X1[i][j],X2[i][j])
        zarray_interno.append(z)
    zarray.append(zarray_interno)
Z = np.array(zarray)


def item_c():
    """
    Plota X1, X2 e o valor da função aplicando o gradiente
    """
    # plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt3d = plt.figure().gca(projection='3d')
    # ax.plot_surface(X1, X2, Z, alpha=0.2)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    # ax.set_ylabel("X2")

    Xg, Yg = item_a()
    Xg = np.array(Xg)
    ax.plot(Xg[:, 0], Xg[:, 1], Yg)
    ax.plot(Xg[:, 0], Xg[:, 1], np.zeros_like(Yg), c='red')
    ax.scatter([Xg[0, 0]], [Xg[0, 1]], c='green')
    ax.scatter([Xg[-1, 0]], [Xg[-1, 1]], c='black')
    
    # def animate(i):
    #     ax.clear()
    #     ax.plot_surface(X1, X2, Z, alpha=0.2)
    #     x1, x2 = Xg[i]
    #     y = Yg[i]
    #     ax.scatter([x1], [x2], [y], c='green')
    #     # flinha = f1(x)
    #     # print ('flinha', flinha)
        
    #     ax.set_title("k:{}".format(i)) 
    
    # ani = animation.FuncAnimation(fig, animate, frames=np.arange(1,len(Xg)), interval=500)

    # Ensure that the next plot doesn't overwrite the first plot  

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot
    # ax.plot(X1, X2, Z)
    plt.show()
    
# item a
# X0 = [-0.78764362 , -0.65387373]
# X0 = [-0.64231882 , 0.57173843]
# X0 = [1.90047238 3.87944662] Descida bonita

# xg, yg = item_a()
# Fx = [f(X[0], X[1]) for X in xg]
# plt.plot(Fx)
# plt.show()

# print (xg)
# item_b()
item_c()