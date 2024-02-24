import random
import math
import numpy as np
import matplotlib.pyplot as plt

v0=50
h=100
g=9.81

def strzal(a):
    alpha=math.radians(a)
    distance = (v0*math.sin(alpha) + math.sqrt((v0*math.sin(alpha))**2 + 2*g*h))*(v0*math.cos(alpha)/g)
    return distance


def wykres(al):
    alpha=math.radians(al)
    a = -g/(2*(v0*math.cos(alpha))**2)
    b = math.sin(alpha)/math.cos(alpha)
    c = h
    x = np.linspace(0, wynik, 10000)
    y = a * x**2 + b * x + c
    plt.figure(figsize=(7,8 ))
    plt.plot(x, y, label='Trajektoria ruchu')
    plt.title('Wykres trajektorii ruchu')
    plt.xlabel('Odległość')
    plt.ylabel('Wysokość')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, max(y) * 1.1)
    # plt.show()
    plt.savefig("trajektoria.png")
    plt.close()


cel = random.randint(50, 340)

inp = input(f"podaj kąt alpha do strzału, cel ma odległość {cel}: ")
try:
    alpha = float(inp)
except ValueError:
    print("Podałeś nie liczbe. Koniec programu")

print(f"Wprowadzony kąt {alpha}")
wynik = strzal(alpha)
print(f"odległość na którą trafiłeś wynosi: {wynik}")
licznik = 1

while True:
    if wynik < cel + 5 and wynik > cel - 5:
        print(f"Cel trafiony, trafiłeś z {licznik} próby")
        wykres(alpha)
        break
    else:
        inp = input(f"podaj kąt alpha do strzału, cel ma odległość {cel}: ")
        try:
            alpha = float(inp)
        except ValueError:
            print("Podałeś nie liczbe. Koniec programu")
        print(f"Wprowadzony kąt {alpha}")
        wynik = strzal(alpha)
        print(f"odległość na którą trafiłeś wynosi: {wynik}")
        licznik += 1