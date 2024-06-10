import numpy as np
import gym
import pygad

# Utwórzenie środowisko FrozenLake8x8
env = gym.make('FrozenLake8x8-v1', is_slippery=False, render_mode="human")

# Parametry algorytmu genetycznego
num_generations = 100
num_parents_mating = 5
sol_per_pop = 20
num_genes = 200  # Liczba kroków w chromosomie

# Funkcja fitness
def fitness_func(model, solution, solution_idx):
    state = env.reset()
    total_reward = 0
    for step in solution:
        state, reward, done, truncated, info = env.step(step)
        total_reward += reward
        if done:
            break
    return total_reward

# Definicja chromosomów - ruchy w przestrzeni [0, 1, 2, 3] (lewo, dół, prawo, góra)
gene_space = [0, 1, 2, 3]

# Inicjalizacja algorytmu genetycznego
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space
)

# Uruchomienie algorytmu
ga_instance.run()

# Najlepsze rozwiązanie
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Najlepsze rozwiązanie: {solution}, z fitness: {solution_fitness}")

# Testowanie najlepszego rozwiązania
state = env.reset()
env.render()
for step in solution:
    state, reward, done, truncated, info = env.step(step)
    env.render()
    if done:
        break
