# Chapter1/Pb9
def factorielle(n):
    if n == 0:
        return 1
    else:
        return n * factorielle(n-1)

# Calculer la factorielle de 6
factorielle_de_6 = factorielle(6)
print(f" factorielle of 6 is {factorielle_de_6}")
#Chapter1/Pb10
def is_leap_year(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def count_leap_years(start_year, end_year):
    leap_count = 0
    for year in range(start_year, end_year + 1):
        if is_leap_year(year):
            leap_count += 1
    return leap_count

start_year = 1500
end_year = 2010

num_leap_years = count_leap_years(start_year, end_year)
print(f"The number of leap years between {start_year} and {end_year} is: {num_leap_years}")

#chapter1/Pb11
import math

# Define the Ramanujan series approximation function for pi
def ramanujan_pi(N):
    sum_series = 0
    for k in range(N + 1):
        numerator = math.factorial(4 * k) * (1103 + 26390 * k)
        denominator = (math.factorial(k) ** 4) * (396 ** (4 * k))
        sum_series += numerator / denominator
    factor = (2 * math.sqrt(2)) / 9801
    return 1 / (factor * sum_series)

# Compute the approximations for N=0 and N=1
approx_pi_N0 = ramanujan_pi(0)
approx_pi_N1 = ramanujan_pi(1)

# Compare with Python's stored value for pi
true_pi = math.pi

# Print the results with high precision
print(f"Approximation of pi for N=0: {approx_pi_N0:.15f}")
print(f"Approximation of pi for N=1: {approx_pi_N1:.15f}")
print(f"Python's stored value for pi: {true_pi:.15f}")

# Print the differences
print(f"Difference for N=0: {abs(approx_pi_N0 - true_pi):.15f}")
print(f"Difference for N=1: {abs(approx_pi_N1 - true_pi):.15f}")
#Chapter1/Pb13
import math

# List of values for x
x_values = [math.pi, math.pi/2, math.pi/4, math.pi/6]

# Verify the identity sin^2(x) + cos^2(x) = 1
for x in x_values:
    sin_squared = math.sin(x) ** 2
    cos_squared = math.cos(x) ** 2
    result = sin_squared + cos_squared
    print(f"sin^2({x}) + cos^2({x}) = {result:.15f}")

# Additionally, compare results to 1
    is_identity_holds = math.isclose(result, 1, rel_tol=1e-15)
    print(f"Does sin^2({x}) + cos^2({x}) = 1? {is_identity_holds}")
#Chapter1/Pb14
import math

# Convert degrees to radians
angle_degrees = 87
angle_radians = math.radians(angle_degrees)

# Compute the sine of the angle
sin_value = math.sin(angle_radians)

# Print the result
print(f"sin(87°) = {sin_value:.15f}")
#Chapter1/Pb19
def evaluate_expression(P, Q):
    result = (P and Q) or (P and not Q)
    return result

# Test all possible combinations of P and Q
truth_values = [True, False]

print("P\tQ\tResult")
for P in truth_values:
    for Q in truth_values:
        result = evaluate_expression(P, Q)
        print(f"{P}\t{Q}\t{result}")

# Determine the conditions under which the expression is false
print("\nConditions under which the expression is false:")
for P in truth_values:
    for Q in truth_values:
        result = evaluate_expression(P, Q)
        if not result:
            print(f"P = {P}, Q = {Q}")
#Chapter1/Pb22
def xor(P, Q):
    return (P and not Q) or (not P and Q)

# Test all possible combinations of P and Q
truth_values = [True, False]

print("P\tQ\tP XOR Q")
for P in truth_values:
    for Q in truth_values:
        result = xor(P, Q)
        print(f"{P}\t{Q}\t{result}")
 #Chapter2/pb1          
# Assign values to the variables
x = 2
y = 3

# Print the values of x and y before clearing x
print(f"x before clearing: {x}")
print(f"y before clearing: {y}")

# Clear (delete) the variable x
del x

# Attempt to print x after deleting it to confirm it's cleared
try:
    print(f"x after clearing: {x}")
except NameError:
    print("x has been cleared and is no longer defined.")

# Print the value of y to show it is still defined
print(f"y after clearing x: {y}")
