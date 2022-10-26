import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
def f(x):
    return np.exp(-x) * x*(x**2-x-1)
def dif_finitas(func, xk, h):
    """
    Método de diferenças finitas
    """
    return (func(xk+h) - f(xk))/h

def gd(xk, alpha, grad):
    """
    Um passo da descida do gradiente
    """
    return xk - (alpha*abs(grad))

X = np.arange(-1,3)
y = [f(xi) for xi in X]

x_0 = 1
gradiente = dif_finitas(f, 1, h=0.01)
x_1 = gd(1, 0.1, gradiente)
fig, ax = plt.subplots()
ax.plot(X,y)
ax.scatter([x_0], [f(x_0)], c='red',label='$f(x_0)={:.3f}$'.format(f(x_0)))
ax.scatter([x_1], [f(x_1)], c='green',label='$f(x_1)={:.3f}$'.format(f(x_1)))
plt.legend()
plt.show()
