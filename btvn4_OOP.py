from __future__ import division, unicode_literals, print_function
import numpy as np

class gradient(object):
    def __init__(self, X, y, N):
        self.X = X
        self.y = y
        self.N = N
        self.setup()

    def setup(self):
        one = np.ones((self.X.shape[0], 1))
        self.Xbar = np.concatenate((one, self.X), axis=1)

    def grad_loss(self, w):
        return 1 / self.N * self.Xbar.T.dot(self.Xbar.dot(w) - self.y)

    def loss(self, w):
        return .5 / self.N * np.linalg.norm(self.y - self.Xbar.dot(w), 2) ** 2

    def run(self, w_init, eta):
        w = [w_init]
        it = 0
        while np.linalg.norm(w[-1]) / len(w) > 1e-5:
            w_new = w[-1] - eta * self.grad_loss(w[-1])
            w.append(w_new)
            it = it + 1
        return (w, it)

if __name__ == '__main__':
    a0 = 2
    b0 = 3
    N = 100
    noise = np.random.normal(0, 1, size=(N, 1))
    X = np.asarray([i for i in range(1, N+1)]).reshape((N, 1))
    y = b0 + a0 * X + noise
    w_init = np.asarray([[5],[1]])
    (w1, it1) = gradient(X, y, N).run(w_init, 0.0001)
    print('[b1 a1] = ',w1[-1].T, 'after ',it1, 'iteration')