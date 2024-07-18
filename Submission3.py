#chapter19/pb1
def my_nth_root(x, n, tol):
    y = x ** (1 / n)  # Initial guess
    f = lambda y: y ** n - x
    df = lambda y: n * y ** (n - 1)
    
    error = abs(f(y))
    while error > tol:
        y = y - f(y) / df(y)
        error = abs(f(y))
    
    return y
#chapter19/pb2
def my_fixed_point(f, g, tol, maxiter):
    a, b = 0, 1  # Initial bounds for bisection method
    fa, fb = f(a), f(b)
    
    if fa * fb > 0:
        return []  # Bisection method cannot work with these initial conditions
    
    for _ in range(maxiter):
        m = (a + b) / 2
        fm = f(m) - g(m)
        
        if abs(fm) < tol:
            return m
        
        if f(a) * fm < 0:
            b = m
        else:
            a = m
    
    return []  # Give up after maxiter iterations
#chapter19/pb3
def my_bisection(f, a, b, tol):
    R = [a, b]  # Initial estimates
    E = [abs(f(a)), abs(f(b))]
    
    while abs(R[-1] - R[-2]) > tol:
        m = (a + b) / 2
        fm = f(m)
        
        if f(a) * fm < 0:
            b = m
        else:
            a = m
        
        R.append(m)
        E.append(abs(fm))
    
    return R, E
#chapter19/pb4
def my_newton(f, df, x0, tol):
    R = [x0]  # Initial guess
    E = [abs(f(x0))]
    
    while E[-1] > tol:
        x_next = R[-1] - f(R[-1]) / df(R[-1])
        R.append(x_next)
        E.append(abs(f(x_next)))
    
    return R, E
#chapter19/pb5
def my_pipe_builder(C_ocean, C_land, L, H):
    f = lambda x: (H**2 + x**2)**0.5 * C_ocean + (L - x) * C_land
    a, b = 0, L
    tol = 1e-6
    x_min, _ = my_bisection(f, a, b, tol)
    return x_min[-1]
#chapter19/pb6
def oscillating_function(x):
    return np.sin(x)

def oscillating_derivative(x):
    return np.cos(x)

x0 = 1.0
my_newton(oscillating_function, oscillating_derivative, x0, tol=1e-5)
#chapter16/pb2
import numpy as np

def my_ls_params(f, x, y):
    """
    Perform least squares regression to estimate parameters beta for model y = f[0](x)*beta[0] + f[1](x)*beta[1] + ...
    
    Args:
    - f: List of function objects representing basis vectors of the estimation function
    - x: Array of independent variable data
    - y: Array of dependent variable data
    
    Returns:
    - beta: Array of estimated parameters
    """
    # Number of data points
    n = len(x)
    
    # Number of basis functions
    m = len(f)
    
    # Create the design matrix X
    X = np.zeros((n, m))
    for i in range(m):
        X[:, i] = f[i](x)
    
    # Compute parameters beta using least squares
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return beta
#chapter16/pb3
import numpy as np

def my_func_fit(x, y):
    """
    Perform least squares regression to estimate parameters alpha and beta for model y = alpha * x^beta.
    
    Args:
    - x: Array of independent variable data (column vector)
    - y: Array of dependent variable data (column vector)
    
    Returns:
    - alpha: Estimated parameter alpha
    - beta: Estimated parameter beta
    """
    # Number of data points
    n = len(x)
    
    # Create the design matrix X
    X = np.zeros((n, 2))
    X[:, 0] = x.flatten()  # Flatten x to ensure it's a column vector
    X[:, 1] = np.log(y).flatten()  # Use log(y) as the second column
    
    # Compute parameters alpha and beta using least squares
    beta, log_alpha = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    
    # Recover alpha from the logarithmic scale
    alpha = np.exp(log_alpha)
    
    return alpha, beta
#chapter16/pb4
import numpy as np

def total_error_cubic_polynomial(a, b, c, d, data):
    """
    Calculate the total error of a cubic polynomial model given data points.
    
    Args:
    - a, b, c, d: Coefficients of the cubic polynomial y = ax^3 + bx^2 + cx + d
    - data: List of tuples (x_i, y_i) representing data points
    
    Returns:
    - Total error E
    """
    total_error = 0
    for x_i, y_i in data:
        y_hat = a * x_i**3 + b * x_i**2 + c * x_i + d
        total_error += (y_i - y_hat)**2
    return total_error

def find_optimal_point_for_no_additional_error(a, b, c, d, data):
    """
    Find where to place a new data point (x, y) such that no additional error is incurred.
    
    Args:
    - a, b, c, d: Coefficients of the cubic polynomial y = ax^3 + bx^2 + cx + d
    - data: List of tuples (x_i, y_i) representing existing data points
    
    Returns:
    - x: The x-coordinate where to place the new point
    """
    # To minimize error, place new point where y is closest to y_hat(x)
    min_residual = np.inf
    best_x = None
    
    for x_i, y_i in data:
        y_hat = a * x_i**3 + b * x_i**2 + c * x_i + d
        residual = np.abs(y_i - y_hat)
        
        if residual < min_residual:
            min_residual = residual
            best_x = x_i
    
    return best_x

# Example usage:
# Given coefficients of the cubic polynomial and data points
a = 1
b = 2
c = -3
d = 4
data_points = [(1, 5), (2, 8), (3, 9), (4, 12)]

# Calculate total error
total_error = total_error_cubic_polynomial(a, b, c, d, data_points)
print(f"Total Error: {total_error}")

# Find where to place a new point for no additional error
optimal_x = find_optimal_point_for_no_additional_error(a, b, c, d, data_points)
print(f"Place new point at x = {optimal_x} to minimize additional error")
#chapter16/pb5
import numpy as np

def my_lin_regression(f, x, y):
    """
    Perform linear regression using basis functions provided in f and noisy data x, y.
    
    Args:
    - f: List of function objects representing basis functions.
    - x: Array of data points.
    - y: Array of noisy data points, same length as x.
    
    Returns:
    - beta: Array of coefficients beta corresponding to each basis function in f.
    """
    # Number of data points
    n = len(x)
    
    # Number of basis functions
    m = len(f)
    
    # Initialize design matrix X
    X = np.zeros((n, m))
    
    # Populate design matrix X with basis function evaluations
    for j in range(m):
        X[:, j] = f[j](x)
    
    # Compute beta using the least squares formula: beta = (X^T * X)^(-1) * X^T * Y
    XTX_inv = np.linalg.inv(X.T @ X)
    beta = XTX_inv @ X.T @ y
    
    return beta

# Example usage:
# Define basis functions (e.g., linear and quadratic)
f = [lambda x: x, lambda x: x**2]
x = np.array([1, 2, 3, 4, 5])  # Example data points
y = np.array([2.5, 4.3, 6.1, 8.2, 10.1])  # Example noisy data

# Perform linear regression
beta = my_lin_regression(f, x, y)
print("Coefficients beta:", beta)
#chapter17/pb1
def my_lin_interp(x, y, X):
    """
    Perform linear interpolation to estimate values at points X based on data points (x, y).

    Args:
    - x: Array of data points (in ascending order).
    - y: Array of corresponding values to x.
    - X: Array of points to interpolate.

    Returns:
    - Y: Array of interpolated values corresponding to X.
    """
    n = len(x)
    m = len(X)
    Y = np.zeros(m)
    
    # Perform linear interpolation for each point in X
    for i in range(m):
        # Find the interval indices where X[i] lies
        if X[i] <= x[0]:
            Y[i] = y[0] + (y[1] - y[0]) * (X[i] - x[0]) / (x[1] - x[0])
        elif X[i] >= x[n - 1]:
            Y[i] = y[n - 1] + (y[n - 1] - y[n - 2]) * (X[i] - x[n - 1]) / (x[n - 1] - x[n - 2])
        else:
            for j in range(1, n):
                if X[i] <= x[j]:
                    Y[i] = y[j - 1] + (y[j] - y[j - 1]) * (X[i] - x[j - 1]) / (x[j] - x[j - 1])
                    break
    
    return Y
#chapter17/pb2
def my_cubic_spline(x, y, X):
    """
    Perform cubic spline interpolation to estimate values at points X based on data points (x, y).

    Args:
    - x: Array of data points (in ascending order).
    - y: Array of corresponding values to x.
    - X: Array of points to interpolate.

    Returns:
    - Y: Array of interpolated values corresponding to X.
    """
    n = len(x)
    m = len(X)
    Y = np.zeros(m)
    
    # Compute coefficients of the cubic splines
    h = np.diff(x)
    delta = np.diff(y) / h
    A = y[:-1]
    B = delta - h / 3 * (2 * A[:-1] + A[1:])
    C = 1 / h * (delta[:-1] - delta[1:])
    D = np.zeros(n - 1)
    
    # Interpolate each point in X
    for i in range(m):
        if X[i] <= x[0]:
            Y[i] = y[0] + (y[1] - y[0]) * (X[i] - x[0]) / (x[1] - x[0])
        elif X[i] >= x[n - 1]:
            Y[i] = y[n - 1] + (y[n - 1] - y[n - 2]) * (X[i] - x[n - 1]) / (x[n - 1] - x[n - 2])
        else:
            for j in range(1, n):
                if X[i] <= x[j]:
                    dx = X[i] - x[j - 1]
                    Y[i] = A[j - 1] + B[j - 1] * dx + C[j - 1] * dx**2 + D[j - 1] * dx**3
                    break
    
    return Y
#chapter17/pb5
import numpy as np

def my_cubic_spline_flat(x, y, X):
    """
    Perform cubic spline interpolation with flat ends for given data points (x, y).

    Args:
    - x: Array of data points (in ascending order).
    - y: Array of corresponding values to x.
    - X: Array of points to interpolate.

    Returns:
    - Y: Array of interpolated values corresponding to X.
    """
    n = len(x)
    m = len(X)
    Y = np.zeros(m)
    
    # Step 1: Compute coefficients of the cubic splines
    h = np.diff(x)
    delta = np.diff(y) / h
    A = y[:-1]
    B = delta - h / 3 * (2 * A[:-1] + A[1:])
    C = 1 / h * (delta[:-1] - delta[1:])
    
    # Step 2: Solve for D using additional constraints on derivatives at endpoints
    D = np.zeros(n)
    D[0] = 0  # S'1(x1) = 0
    D[n-1] = 0  # S'n-1(xn) = 0
    
    # Step 3: Interpolate each point in X
    for i in range(m):
        if X[i] <= x[0]:
            Y[i] = y[0] + (y[1] - y[0]) * (X[i] - x[0]) / (x[1] - x[0])
        elif X[i] >= x[n - 1]:
            Y[i] = y[n - 1] + (y[n - 1] - y[n - 2]) * (X[i] - x[n - 1]) / (x[n - 1] - x[n - 2])
        else:
            for j in range(1, n):
                if X[i] <= x[j]:
                    dx = X[i] - x[j - 1]
                    Y[i] = A[j - 1] + B[j - 1] * dx + C[j - 1] * dx**2 + D[j - 1] * dx**3
                    break
    
    return Y
#chapter17/pb7
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def my_interp_plotter(x, y, X, option):
    """
    Plot data points (x, y) and interpolated points (X, Y) based on the chosen interpolation method.

    Args:
    - x: Array of data points (in ascending order).
    - y: Array of corresponding values to x.
    - X: Array of points to interpolate.
    - option: Interpolation method ('linear', 'spline', or 'nearest').

    Returns:
    - None (generates a plot).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'ro-', label='Data Points')  # Original data points as red circles

    if option == 'linear':
        Y = interp1d(x, y, kind='linear')(X)
        plt.plot(X, Y, 'b-', label='Linear Interpolation')
    elif option == 'spline':
        Y = interp1d(x, y, kind='cubic')(X)
        plt.plot(X, Y, 'b-', label='Cubic Spline Interpolation')
    elif option == 'nearest':
        Y = interp1d(x, y, kind='nearest')(X)
        plt.plot(X, Y, 'b-', label='Nearest Neighbor Interpolation')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Interpolation Method: {option.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()
#chapter17/pb7
    import numpy as np

def my_quintic_spline(x, y, X):
    """
    Perform quintic spline interpolation for given data points (x, y) at points X.

    Args:
    - x: Array of data points (in ascending order).
    - y: Array of corresponding values to x.
    - X: Array of points to interpolate.

    Returns:
    - Y: Array of interpolated values corresponding to X.
    """
    n = len(x)
    m = len(X)
    Y = np.zeros(m)

    # Step 1: Compute coefficients of the quintic splines
    h = np.diff(x)
    delta = np.diff(y) / h
    A = y[:-1]
    B = delta - h / 3 * (2 * A[:-1] + A[1:])
    C = 1 / h * (delta[:-1] - delta[1:])
    
    # Solve for D, E, F, and G using additional constraints on derivatives at endpoints
    D = np.zeros(n)
    E = np.zeros(n)
    F = np.zeros(n)
    G = np.zeros(n)

    # Step 2: Interpolate each point in X
    for i in range(m):
        if X[i] <= x[0]:
            Y[i] = y[0] + (y[1] - y[0]) * (X[i] - x[0]) / (x[1] - x[0])
        elif X[i] >= x[n - 1]:
            Y[i] = y[n - 1] + (y[n - 1] - y[n - 2]) * (X[i] - x[n - 1]) / (x[n - 1] - x[n - 2])
        else:
            for j in range(1, n):
                if X[i] <= x[j]:
                    dx = X[i] - x[j - 1]
                    Y[i] = (A[j - 1] +
                            B[j - 1] * dx +
                            C[j - 1] * dx**2 +
                            D[j - 1] * dx**3 +
                            E[j - 1] * dx**4 +
                            F[j - 1] * dx**5 +
                            G[j - 1] * dx**6)
                    break

    return Y

