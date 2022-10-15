import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import matplotlib.animation as animation

def gradient_descent(x0, f, alpha, gmin, kmax):
    """
    x0: valor inicial
    f: derivada da função f (lambdify)
    """
    k = 0
    xk = x0
    X = []
    while (k < kmax) and (f(xk)>gmin):
        X.append(xk)
        g = f(xk)
        xk -= alpha*abs(g)
        k+=1
    return X


# criando um símbolo
x = Symbol('x')
# definindo a funcao
f = exp(-x)*x*(x**2-x-1)
f1 = lambdify(x,f)

X = np.linspace(1, 3, 20)
Y = [exp(-xi)*xi*(xi**2-xi-1) for xi in X]


# calculando gradiente
Xgradiente = gradient_descent(x0=3, f=f1, alpha=0.1, gmin=0.1, kmax=20)
Ygradiente = [exp(-xi)*xi*(xi**2-xi-1) for xi in Xgradiente]

fig, ax = plt.subplots()
ax.plot(X, Y)
ax.set_title("$x{}:{:.3f}, f(x{}):{:.3f}, f'(x{}):{:.3f}$".format(0, X[0], 0, Y[0], 0, f1(X[0])))

j = 0
def animate(i):
    # global j    
    ax.clear()
    ax.plot(X, Y)
    x = Xgradiente[i]
    y = Ygradiente[i]
    ax.scatter([x], [y], c='green')
    flinha = f1(x)
    print ('flinha', flinha)
    print ('x', x)
    print ('y', y)
    ax.set_title("$x{}:{:.3f}, f(x{}):{:.3f}, f'(x{}):{:.3f}$".format(i, x, i, y, i, flinha))
   
ani = animation.FuncAnimation(fig, animate, frames=np.arange(1,len(Xgradiente)), interval=1000)
ani.save("gd.gif")
plt.show() 
