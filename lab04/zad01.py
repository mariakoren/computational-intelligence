import pandas as pd
import math

df = pd.read_csv("siatkowka.csv")

def activate (x):
    return 1 / (1 + math.exp(-x))

def forwardPass(wiek,waga,wzrost):
    hidden1 = wiek * (-0.46122) + waga * (0.97314) + wzrost *(-0.39203) + 0.80109
    hidden1_po_aktywacji = activate(hidden1)
    hidden2 = wiek * (0.78548) + waga * (2.10584) + wzrost *(-0.57847) + 0.43529
    hidden2_po_aktywacji = activate(hidden2)
    output = hidden1_po_aktywacji * (-0.81546) + hidden2_po_aktywacji * (1.03775) -0.2368    
    return output


print(forwardPass(23,75,176))
print(forwardPass(25,67,180))
print(forwardPass(28,120,175))
print(forwardPass(22,65,165))
print(forwardPass(46,70,187))
print(forwardPass(50,68,180))
print(forwardPass(48,97,178))


