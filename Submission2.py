#chapter3/pb1
import math

def my_sinh(x):
    """
    Compute the hyperbolic sine (sinh) of x.
    
    Parameters:
    x (float): Input value for which sinh is computed.
    
    Returns:
    float: Hyperbolic sine of x.
    """
    return (math.exp(x) - math.exp(-x)) / 2

# Test cases
print("my_sinh(0):", my_sinh(0))     # Expected output: 0
print("my_sinh(1):", my_sinh(1))     # Expected output: 1.1752011936438014
print("my_sinh(2):", my_sinh(2))     # Expected output: 3.6268604078470186
#chapter3/pb2
import numpy as np

def my_checker_board(n):
    """
    Generate an n x n checkerboard pattern.
    
    Parameters:
    n (int): Size of the checkerboard (n x n).
    
    Returns:
    numpy.ndarray: Checkerboard pattern as a NumPy array.
    """
    # Initialize an n x n array filled with zeros
    board = np.zeros((n, n), dtype=int)
    
    # Use slicing to set alternating elements to 1
    board[::2, ::2] = 1  # Set elements at even indices to 1
    board[1::2, 1::2] = 1  # Set elements at odd indices to 1
    
    return board

# Test cases
print("my_checker_board(1):\n", my_checker_board(1))  # Expected output: [[1]]
print("my_checker_board(2):\n", my_checker_board(2))  # Expected output: [[1, 0], [0, 1]]
print("my_checker_board(3):\n", my_checker_board(3))  # Expected output: [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
print("my_checker_board(5):\n", my_checker_board(5))  # Expected output: See provided output
#chapter3/pb3
def my_triangle(b, h):
    """
    Calculate the area of a triangle given its base and height.
    
    Parameters:
    b (float): Base of the triangle.
    h (float): Height of the triangle.
    
    Returns:
    float: Area of the triangle.
    """
    area = 0.5 * b * h
    return area

# Test cases
print("my_triangle(1, 1):", my_triangle(1, 1))    # Expected output: 0.5
print("my_triangle(2, 1):", my_triangle(2, 1))    # Expected output: 1
print("my_triangle(12, 5):", my_triangle(12, 5))  # Expected output: 30
#chapter3/pb4
import numpy as np

def my_split_matrix(m):
    """
    Split the matrix m into two halves: m1 (left half) and m2 (right half).
    If the number of columns in m is odd, the middle column goes to m1.

    Parameters:
    m (numpy.ndarray): Input matrix to split.

    Returns:
    numpy.ndarray, numpy.ndarray: m1 (left half) and m2 (right half) of the matrix m.
    """
    num_rows, num_cols = m.shape
    
    if num_cols % 2 == 0:
        split_idx = num_cols // 2
    else:
        split_idx = num_cols // 2 + 1
    
    m1 = m[:, :split_idx]
    m2 = m[:, split_idx:]
    
    return m1, m2

# Test Case 1
m1, m2 = my_split_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print("m1:\n", m1)
print("m2:\n", m2)

# Test Case 2
m = np.ones((5, 5))
m1, m2 = my_split_matrix(m)
print("\nm1:\n", m1)
print("m2:\n", m2)
#chapter3/pb5
import math

def my_cylinder(r, h):
    """
    Calculate the surface area and volume of a cylinder given its radius and height.

    Parameters:
    r (float): Radius of the cylinder.
    h (float): Height of the cylinder.

    Returns:
    list: A list containing the surface area and volume of the cylinder, [surface_area, volume].
    """
    # Calculate surface area
    s = 2 * math.pi * r**2 + 2 * math.pi * r * h
    
    # Calculate volume
    v = math.pi * r**2 * h
    
    # Return the results as a list
    return [round(s, 4), round(v, 4)]

# Test Cases
print(my_cylinder(1, 5))  # Expected output: [37.6991, 15.708]
print(my_cylinder(2, 4))  # Expected output: [62.8319, 25.1327]
#chapter3/pb6
import numpy as np

def my_n_odds(a):
    """
    Count the number of odd numbers in a one-dimensional array of floats.

    Parameters:
    a (numpy.ndarray): One-dimensional array of floats.

    Returns:
    int: Number of odd numbers in the array.
    """
    # Initialize a counter for odd numbers
    odd_count = 0
    
    # Iterate through each element in the array
    for num in a:
        # Check if the element is odd (not divisible by 2)
        if num % 2 != 0:
            odd_count += 1
    
    # Return the count of odd numbers
    return odd_count

# Test Cases
print(my_n_odds(np.arange(100)))              # Expected output: 50
print(my_n_odds(np.arange(2, 100, 2)))        # Expected output: 0
print(my_n_odds(np.array([1, 2, 3, 4, 5, 6]))) # Expected output: 3
#chapter3/pb7
import numpy as np

def my_twos(m, n):
    """
    Generate an m x n array filled with twos.

    Parameters:
    m (int): Number of rows (strictly positive integer).
    n (int): Number of columns (strictly positive integer).

    Returns:
    numpy.ndarray: m x n array filled with twos.
    """
    # Create an m x n array filled with twos
    twos_array = np.full((m, n), 2)
    
    return twos_array

# Test Cases
print(my_twos(3, 2))
print(my_twos(1, 4))
#chapter3/pb8
# Lambda function for subtraction
subtract_lambda = lambda x, y: x - y

# Function to concatenate two strings
def add_string(s1, s2):
    """
    Concatenate two strings.

    Parameters:
    s1 (str): First string.
    s2 (str): Second string.

    Returns:
    str: Concatenated string of s1 and s2.
    """
    return s1 + s2

# Test Cases
s1 = add_string('Programming', ' ')
s2 = add_string('is', 'fun!')
result = add_string(s1, s2)

print(result)  # Output: 'Programming is fun!'
#chapter3/pb9
# Generating the specified errors
def fun(a):
    pass  # Indented block is now correct

# Define the greeting function
def greeting(name, age):
    """
    Generate a greeting message based on given name and age.

    Parameters:
    name (str): Name of the person.
    age (float): Age of the person.

    Returns:
    str: Greeting message formatted as 'Hi, my name is XXX and I am XXX years old.'
    """
    message = f"Hi, my name is {name} and I am {age} years old."
    return message

# Test Cases
print(greeting('John', 26))  # Output: 'Hi, my name is John and I am 26 years old.'
print(greeting('Kate', 19))  # Output: 'Hi, my name is Kate and I am 19 years old.'
#chapter3/pb10
import numpy as np

def my_donut_area(r1, r2):
    """
    Calculate the area of the donut shape given radii r1 and r2.

    Parameters:
    r1 (numpy.ndarray): Array of radii for the inner circle.
    r2 (numpy.ndarray): Array of radii for the outer circle.

    Returns:
    numpy.ndarray: Array of areas corresponding to the donut shape for each pair (r1, r2).
    """
    # Calculate the area of the outer circle with radius r2
    area_outer = np.pi * r2**2

    # Calculate the area of the inner circle with radius r1
    area_inner = np.pi * r1**2

    # Calculate the donut area (difference between outer and inner circle areas)
    donut_area = area_outer - area_inner

    return donut_area

# Test case
result = my_donut_area(np.arange(1, 4), np.arange(2, 7, 2))
print(result)
#chapter3/pb11
import numpy as np

def my_donut_area(r1, r2):
    """
    Calculate the area of the donut shape given radii r1 and r2.

    Parameters:
    r1 (numpy.ndarray): Array of radii for the inner circle.
    r2 (numpy.ndarray): Array of radii for the outer circle.

    Returns:
    numpy.ndarray: Array of areas corresponding to the donut shape for each pair (r1, r2).
    """
    # Calculate the area of the outer circle with radius r2
    area_outer = np.pi * r2**2

    # Calculate the area of the inner circle with radius r1
    area_inner = np.pi * r1**2

    # Calculate the donut area (difference between outer and inner circle areas)
    donut_area = area_outer - area_inner

    return donut_area

# Test case
result = my_donut_area(np.arange(1, 4), np.arange(2, 7, 2))
print(result)
#chapter3/pb11
import numpy as np

def bounding_array(A, top, bottom):
    """
    Modify array A based on specified conditions:
    - Output is equal to A wherever bottom < A < top,
    - Output is equal to bottom wherever A <= bottom,
    - Output is equal to top wherever A >= top.

    Parameters:
    A (numpy.ndarray): One-dimensional array of floats.
    top (float): Upper boundary value.
    bottom (float): Lower boundary value.

    Returns:
    numpy.ndarray: Modified array based on the specified conditions.
    """
    # Create a new array to store modified values
    modified_A = np.copy(A)
    
    # Apply conditions using NumPy vectorized operations
    modified_A[A < bottom] = bottom
    modified_A[A > top] = top

    return modified_A

# Test case
print(bounding_array(np.arange(-5, 6, 1), 3, -3))
#chapter9/pb1
def my_bin_2_dec(b):
    """
    Convert a binary number represented by a list of ones and zeros to decimal.
    The last element of b represents the coefficient of 2^0, the second-to-last
    element represents the coefficient of 2^1, and so on.

    Parameters:
    b (list): Binary number represented as a list of integers (0s and 1s).

    Returns:
    int: Decimal representation of the binary number.
    """
    d = 0
    for i in range(len(b)):
        d += b[i] * (2 ** i)
    return d

# Test cases
print(my_bin_2_dec([1, 1, 1]))       # Output: 7
print(my_bin_2_dec([1, 0, 1, 0, 1, 0, 1]))   # Output: 85
print(my_bin_2_dec([1]*25))         # Output: 33554431
#chapter9/pb2
def my_dec_2_bin(d):
    """
    Convert a positive integer in decimal to binary represented as a list of ones and zeros.
    The leading term must be a 1 unless the decimal input value is 0.

    Parameters:
    d (int): Positive integer in decimal.

    Returns:
    list: Binary representation of d as a list of integers (0s and 1s).
    """
    if d == 0:
        return [0]
    
    b = []
    while d > 0:
        b.append(d % 2)
        d //= 2
    
    b.reverse()
    return b

# Test cases
print(my_dec_2_bin(0))      # Output: [0]
print(my_dec_2_bin(23))     # Output: [1, 0, 1, 1, 1]
print(my_dec_2_bin(2097))   # Output: [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]
#chapter9/pb3
# Compute d = my_bin_2_dec(my_dec_2_bin(12654))
d = 12654
binary_representation = my_dec_2_bin(d)
computed_d = my_bin_2_dec(binary_representation)

print(f"Original d: {d}")
print(f"Computed d: {computed_d}")
print(f"Are they equal? {'Yes' if d == computed_d else 'No'}")
#chapter9/pb4
def my_bin_adder(b1, b2):
    """
    Add two binary numbers represented as lists of ones and zeros.

    Parameters:
    b1 (list): First binary number represented as a list of integers (0s and 1s).
    b2 (list): Second binary number represented as a list of integers (0s and 1s).

    Returns:
    list: Binary sum of b1 and b2 represented as a list of integers (0s and 1s).
    """
    # Make copies of the input lists to avoid modifying them directly
    b1_copy = b1[:]
    b2_copy = b2[:]
    
    # Pad the shorter list with zeros to make them equal length
    max_len = max(len(b1_copy), len(b2_copy))
    b1_copy = [0] * (max_len - len(b1_copy)) + b1_copy
    b2_copy = [0] * (max_len - len(b2_copy)) + b2_copy
    
    # Initialize the result list
    result = [0] * (max_len + 1)
    
    carry = 0
    for i in range(max_len - 1, -1, -1):
        temp_sum = b1_copy[i] + b2_copy[i] + carry
        result[i + 1] = temp_sum % 2
        carry = temp_sum // 2
    
    result[0] = carry
    
    # Remove leading zeros if any
    while result[0] == 0 and len(result) > 1:
        result = result[1:]
    
    return result

# Test cases
print(my_bin_adder([1, 1, 1, 1, 1], [1]))              # Output: [1, 0, 0, 0, 0, 0]
print(my_bin_adder([1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0]))   # Output: [1, 1, 1, 0, 0, 1, 1]
print(my_bin_adder([1, 1, 0], [1, 0, 1]))              # Output: [1, 0, 1, 1]
