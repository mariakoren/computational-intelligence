import pygad
import numpy as np
import time
import matplotlib.pyplot as plt


labirynt = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 'S', 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 'E', 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

def fitness_func(model, solution, solution_idx):
    x, y = 1, 1
    fitness = 0

    for move in solution:
        if move == 0:
            y -= 1
        elif move == 1:
            y += 1
        elif move == 2:
            x -= 1
        elif move == 3:
            x += 1

        if not (0 <= x < 12 and 0 <= y < 12):
            fitness -= 100000
            break

        if labirynt[y][x] == 1:
            fitness -= 100000

        elif labirynt[y][x] == 'E':
            fitness += 10000
            break


        else:
            distance_to_exit = abs(x - 9) + abs(y - 9)
            fitness += 10000 / (distance_to_exit + 1)

    return fitness


gene_space = [0, 1, 2, 3]
num_genes = 30
sol_per_pop = 10
num_parents_mating = 2
num_generations = 5000
parent_selection_type = "sss" 
keep_parents = -1
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 5
ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    gene_space=gene_space,
    parent_selection_type = parent_selection_type,
    keep_parents = keep_parents,
    crossover_type = crossover_type,
    mutation_type = mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    
)

num_iterations = 10
total_time = 0


# for _ in range(num_iterations):
#     start_time = time.time()
#     ga_instance.run()
#     end_time = time.time()
#     total_time += end_time - start_time

# average_time = total_time / num_iterations
# print("Sredni czas:", average_time)  # 1.4391039609909058


start_time = time.time()
ga_instance.run()
end_time = time.time()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parametry najlepszego rozwiązania : {solution}".format(solution=solution))
print("Fitness najlepszego rozwiązania : {solution_fitness}".format(solution_fitness=solution_fitness))
print("Czas wykonania:", end_time - start_time)

x, y = 1, 1
best_path = [(x, y)] 
for move in solution:
    if move == 0:
        y -= 1
    elif move == 1:
        y += 1
    elif move == 2:
        x -= 1
    elif move == 3:
        x += 1
    best_path.append((x, y))

maze_array = np.array(labirynt)
maze_array[maze_array == 'S'] = 0
maze_array[maze_array == 'E'] = 0 
maze_array = maze_array.astype(int)


plt.figure(figsize=(10, 8))
plt.imshow(maze_array, cmap='binary', origin='upper')

path_x, path_y = zip(*best_path)
plt.plot(path_x, path_y, 'r-', linewidth=2)

plt.title('Maze Solution using Genetic Algorithm')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 12, 1))

plt.savefig("plot.png")