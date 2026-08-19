def derivative(f, x, epsilon=1e-3):
    """Approximate the first derivative of a function at a point.

    Args:
        f (callable): Function to differentiate.
        x (float): Point at which to evaluate the derivative.
        epsilon (float, optional): Small step size used for finite differences.
            Defaults to 1e-3.

    Returns:
        float: Approximate derivative of f at x.
    """
    return (f(x + epsilon) - f(x)) / epsilon


def second_derivative(f, x, epsilon=1e-3):
    """Approximate the second derivative of a function at a point.

    Args:
        f (callable): Function to differentiate twice.
        x (float): Point at which to evaluate the second derivative.
        epsilon (float, optional): Small step size used for finite differences.
            Defaults to 1e-3.

    Returns:
        float: Approximate second derivative of f at x.
    """
    return (derivative(f, x + epsilon) - derivative(f, x)) / epsilon


def optimize(f, x0, epsilon=1e-3, difference=1e-5):
    """Use Newton's method to optimize a function starting from an initial guess.

    Args:
        f (callable): Function to optimize.
        x0 (float): Initial guess for the optimizer.
        epsilon (float, optional): Small step size used for finite-difference
            derivatives. Defaults to 1e-3.
        difference (float, optional): Convergence threshold based on the change
            in x between iterations. Defaults to 1e-5.

    Returns:
        float: Approximate optimizer value after convergence.
    """
    x = x0
    while True:
        first_der = derivative(f, x, epsilon)
        second_der = second_derivative(f, x, epsilon)

        x_new = x - first_der / second_der

        if abs(x_new - x) < difference:
            return x_new

        x = x_new
