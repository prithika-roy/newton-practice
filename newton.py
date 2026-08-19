def derivative(f, x, epsilon=1e-3):
    return ((f(x + epsilon) - f(x)) / epsilon)

def second_derivative(f, x, epsilon=1e-3):
    return (derivative(f, x + epsilon) - derivative(f, x)) / epsilon

def optimize(f, x0, epsilon=1e-3, difference=1e-5):
    x = x0
    while True:
        first_der = derivative(f, x, epsilon)
        second_der = second_derivative(f, x, epsilon)

        x_new = x - first_der / second_der

        if abs(x_new - x) < difference:
            return x_new

        x = x_new