import numpy as np
import pygad
import math

# Definicja funkcji do optymalizacji
def endurance(x, y, z, u, v, w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w)

# Definicja funkcji oceny
def fitness_func(model, solution, solution_idx):
    return endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])
# Inicjalizacja konfiguracji algorytmu genetycznego
rozmiar_populacji = 50
liczba_genow = 6
liczba_generacji = 50
p_mutacji = 0.15

initial_population = np.random.rand(rozmiar_populacji, liczba_genow)

initial_population = np.clip(initial_population, 0, 1)

ga = pygad.GA(num_generations=liczba_generacji,
               num_parents_mating=rozmiar_populacji//2,
               fitness_func=fitness_func,
               sol_per_pop=rozmiar_populacji,
               num_genes=liczba_genow,
               gene_type=np.float32,
               parent_selection_type="tournament",
               crossover_type="single_point",
               mutation_type="random",
               mutation_percent_genes=p_mutacji,
               mutation_by_replacement=True,
               random_mutation_min_val=0.0,
               random_mutation_max_val=1.0,
               initial_population=initial_population)

# Uruchomienie algorytmu genetycznego
ga.run()

solution, solution_fitness, solution_idx = ga.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
ga.plot_fitness().savefig("plot.png")