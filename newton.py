import numpy as np
from scipy.differentiate import jacobian, hessian


def newton_multivariate(f, x0, tolerance=1e-4, max_iter=1000):
    x = np.asarray(x0, dtype=float)

    for _ in range(max_iter):
        grad = jacobian(f, x).df
        H = hessian(f, x).ddf

        if np.linalg.norm(grad) < tolerance:
            break

        x = x - np.linalg.solve(H, grad)

    return x
