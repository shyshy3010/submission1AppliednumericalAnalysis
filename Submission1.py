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

