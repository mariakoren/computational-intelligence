import numpy as np
import math
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# Definiowanie funkcji endurance
def endurance(x, y, z, u, v, w):
    return -(math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w))

# Funkcja pośrednia, która przekształca rój cząstek
def endurance_swarm(particles):
    results = []
    for particle in particles:
        x, y, z, u, v, w = particle
        result = endurance(x, y, z, u, v, w)
        results.append(result)
    return np.array(results)

# Ustawienia PSO
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
n_particles = 10
dimensions = 6

# Definiowanie limitów dla sześciu zmiennych
x_min = np.zeros(dimensions)
x_max = np.ones(dimensions)
my_bounds = (x_min, x_max)

# Tworzenie optymalizatora PSO
optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=my_bounds)

# Optymalizacja funkcji endurance
optimizer.optimize(endurance_swarm, iters=1000)

# Optymalizacja funkcji endurance
cost, pos = optimizer.optimize(endurance_swarm, iters=1000, verbose=True)

# Rysowanie wykresu kosztu
cost_history = optimizer.cost_history

plt.figure(figsize=(10, 5))
plt.plot(cost_history, label='Wartość funkcji celu')
plt.xlabel('Iteracja')
plt.ylabel('Koszt')
plt.title('Zmiana wartości funkcji celu w trakcie iteracji')
plt.legend()
plt.grid(True)
plt.savefig('plot.png')