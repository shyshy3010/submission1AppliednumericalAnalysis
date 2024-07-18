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
#Chapter2/pb2
# Assign string '123' to the variable S
S = '123'

# Convert the string into a float and assign it to the variable N
N = float(S)

# Verify the types of S and N using the type function
print(f"S is of type: {type(S)}")  # Should output: <class 'str'>
print(f"N is of type: {type(N)}")  # Should output: <class 'float'>
#Chapter2/Pb3
# Assign the string 'HELLO' to the variable s1
s1 = 'HELLO'

# Assign the string 'hello' to the variable s2
s2 = 'hello'

# Use the == operator to show that they are not equal
not_equal = s1 == s2
print(f"s1 == s2: {not_equal}")  # Should output: False

# Use the == operator to show that s1 and s2 are equal if the lower method is used on s1
equal_lower = s1.lower() == s2
print(f"s1.lower() == s2: {equal_lower}")  # Should output: True

# Use the == operator to show that s1 and s2 are equal if upper method is used on s2
equal_upper = s1 == s2.upper()
print(f"s1 == s2.upper(): {equal_upper}")  # Should output: True
#Chapter2/pb4
# Define the strings and their lengths
word1 = "Engineering"
length1 = len(word1)
word2 = "Book"
length2 = len(word2)

# Generate the required strings using print statements
print(f"The word '{word1}' has {length1} letters.")
print(f"The word '{word2}' has {length2} letters.")
#chapter2/pb5
# Define the string
sentence = 'Python is great!'

# Check if 'Python' is in the string 'Python is great!'
if 'Python' in sentence:
    print("'Python' is present in the string 'Python is great!'")
else:
    print("'Python' is not present in the string 'Python is great!'")
#chapter2/pb6
# Define the string
sentence = 'Python is great!'
# Split the string into words using split()
words = sentence.split()

# Get the last word using indexing
last_word = words[-1]

# Print the last word
print(f"The last word in the string '{sentence}' is: {last_word}")
#chapter2/pb7
# Assign list [1, 8, 9, 15] to a variable list_a
list_a = [1, 8, 9, 15]

# Insert 2 at index 1 using the insert method
list_a.insert(1, 2)

# Append 4 to list_a using the append method
list_a.append(4)

# Print the modified list_a
print("Modified list_a:", list_a)
#chapter2/pb8
# Given list_a from the previous problem
list_a = [1, 2, 8, 9, 15, 4]

# Sort list_a in ascending order
list_a.sort()

# Print the sorted list_a
print("Sorted list_a:", list_a)
#chapter2/pb17
import numpy as np

# Generate an array with size 100 evenly spaced between -10 to 10
array = np.linspace(-10, 10, 100)

# Print the array
print("Array with size 100 evenly spaced between -10 to 10:")
print(array)
#chapter2/pb18
import numpy as np

# Define array_a
array_a = np.array([-1, 0, 1, 2, 0, 3])

# Use logical expression as index to get elements larger than zero
filtered_array = array_a[array_a > 0]

# Print the filtered array
print("Filtered array with elements larger than zero:")
print(filtered_array)
#chapter2/pb19
import numpy as np

# Create the array y
y = np.array([[3, 5, 3], [2, 2, 5], [3, 8, 9]])

# Calculate the transpose of y
y_transpose = np.transpose(y)

# Print the original array y and its transpose
print("Original array y:")
print(y)
print("\nTranspose of array y:")
print(y_transpose)
#chapter2/pb20
import numpy as np

# Create a zero array with size (2, 4)
zero_array = np.zeros((2, 4))

# Print the zero array
print("Zero array with size (2, 4):")
print(zero_array)
#chapter2/pb21
import numpy as np

# Create a zero array with size (2, 4)
zero_array = np.zeros((2, 4))

# Change the 2nd column (index 1) to 1
zero_array[:, 1] = 1

# Print the modified array
print("Modified array with the 2nd column changed to 1:")
print(zero_array)
#chapter2/pb22

