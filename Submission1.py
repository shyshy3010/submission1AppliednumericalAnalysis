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

