"""
Primeiro trabalho: Descida de gradiente (Questão 2)
Disciplina: Redes Neurais Artificiais
Prof.: Thomas Rauber
Grupo:
- Alan Carlos Pereira Pinto
- Alexander de Oliveira da Silva
- Iran Freitas Ribeiro
"""

from sympy import *
import numpy as np

def gd(xk, alpha, grad):
    return xk - (alpha*abs(grad))

# criando um símbolo
x = Symbol('x')
# definindo a funcao
f = exp(-x)*x*(x**2-x-1)
f1 = lambdify(x,f)
x0=1
x1 = gd(x0, 0.1, f1(x0))
print (x1)