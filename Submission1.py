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

