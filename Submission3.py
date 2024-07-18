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