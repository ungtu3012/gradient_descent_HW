from __future__ import division, unicode_literals, print_function
import numpy as np

#create data
a0 = 2
b0 = 3
N = 100
noise = np.random.normal(0, 1, size=(100, 1))
X = np.asarray([i for i in range(1,101)]).reshape((100,1))
y = b0 + a0*X + noise
print(np.mean(noise))

# Building Xbar
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

def grad_loss(w):
    return 1/N * Xbar.T.dot(Xbar.dot(w)-y)

def loss(w):
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

def gradient_descent(w_init, eta):
    w = [w_init]
    it = 0
    while np.linalg.norm(w[-1])/len(w) > 1e-5:
        w_new = w[-1] - eta*grad_loss(w[-1])
        w.append(w_new)
        it = it + 1
    return (w, it)

def sgrad(w, i):
    xi = Xbar[i, :]
    yi = y[i]
    return xi.T.dot(xi.dot(w) - yi).reshape(2,1)

def SGD(w_init, eta):
    w = [w_init]
    N = X.shape[0]
    for it in range(10000):
        i = np.random.permutation(N)
        for _ in range(N):
            g = sgrad(w[-1], i)
            w_new = w[-1] - eta*g
            w.append(w_new)
        if np.linalg.norm(w[-1])/len(w) < 1e-5:
            break
    return (w, it)

w_init = np.asarray([[4],[3]])
(w1, it1) = gradient_descent(w_init, 0.0001)
(w2, it2) = SGD(w_init, 0.0001)
print('[b1 a1] = ',w1[-1].T, 'after ',it1,'iteration with loss: = ',loss(w1[-1]))
print('[b1 a1] = ',w2[-1].T, 'after ',it2,'iteration')