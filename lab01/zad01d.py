import datetime
import math

imie = input("podaj imię: ")
rokUrodzenia = int(input("podaj rok urodzenia: "))
miesiacUrodzenia = int(input("podaj miesiąc urodzenia: "))
dzienUrodzenia = int(input("podaj dzień urodzenia: "))


def fizycznaFala(t):
    return math.sin(2*math.pi*t/23)


def emocjonalnaFala(t):
    return math.sin(2*math.pi*t/28)


def intelektualnaFala(t):
    return math.sin(2*math.pi*t/33)

dzisiaj = datetime.datetime.now()
dataUrodzenia = datetime.datetime(rokUrodzenia, miesiacUrodzenia, dzienUrodzenia)
roznica = int(str(dzisiaj - dataUrodzenia).split(' ')[0])  # ilosc dni
fizFala = fizycznaFala(roznica)
emFala = emocjonalnaFala(roznica)
intFala = intelektualnaFala(roznica)


# a
print(f"witaj {imie}")
print(f'urodziłeś się {dataUrodzenia}')
print(f'dzisiaj {dzisiaj}')
print(f'twoja fizyczna fala {fizFala}')
print(f'twoja emocjonalna fala {emFala}')
print(f'twoja intelektualna fala {intFala}')

# b
if fizFala > 0.5 or emFala > 0.5 or intFala > 0.5:
    print('Gratuluję')
if fizFala < -0.5 or emFala < -0.5 or intFala < -0.5:
    print('Nie sumuj')
    if fizycznaFala(roznica+1) > fizFala:
        print('Nie martw się. Jutro będzie lepiej!')
    elif emocjonalnaFala(roznica+1) > emFala:
        print('Nie martw się. Jutro będzie lepiej!')
    elif intelektualnaFala(roznica+1) > intFala:
        print('Nie martw się. Jutro będzie lepiej!')


#chat nic nie zmienił poza usuięciem spacji