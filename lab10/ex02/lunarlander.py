import gym
import pygad

# Utwórz środowisko LunarLander
env = gym.make("LunarLander-v2")


# Parametry algorytmu genetycznego
num_generations = 200
num_parents_mating = 10
sol_per_pop = 50
num_genes = 1000  # Liczba akcji w chromosomie

# Funkcja fitness
def fitness_func(model, solution, solution_idx):
    state = env.reset()
    total_reward = 0
    for action in solution:
        action = int(action) 
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

# Definicja chromosomów - akcje w przestrzeni [0, 1, 2, 3]
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
for action in solution:
    action = int(action) 
    state, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        break
