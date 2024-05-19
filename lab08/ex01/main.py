import pygad
import time

items = [
    {"name": "zegar", "weight": 7, "value": 100},
    {"name": "obraz-pejzaz", "weight": 7, "value": 300},
    {"name": "obraz-portret", "weight": 6, "value": 400},
    {"name": "radio", "weight": 2, "value": 40},
    {"name": "laptop", "weight": 5, "value": 500},
    {"name": "lampka-nocna", "weight": 6, "value": 70},
    {"name": "srebrne-sztucce", "weight": 1, "value": 100},
    {"name": "porcelana", "weight": 3, "value": 250},
    {"name": "figura-z-bronzu", "weight": 10, "value": 300},
    {"name": "skorzana-torebka", "weight": 3, "value": 280},
    {"name": "odkurzacz", "weight": 15, "value": 300}
]

max_capacity = 25
target_solution_value = 1630

def fitness_function(model, solution, solution_idx):
    total_weight = sum(item['weight'] for item, selected in zip(items, solution) if selected)
    total_value = sum(item['value'] for item, selected in zip(items, solution) if selected)
    
    if total_weight > max_capacity or total_value > 1630:
        return 0
    return total_value

def on_generation(ga_instance):
    best_solution = ga_instance.best_solution()
    best_solution_fitness = best_solution[1]
    if best_solution_fitness >= target_solution_value:
        ga_instance.keep_solving = False

best_solutions_count = 0
total_successful_time = 0

for i in range(10):
    start_time = time.time()

    ga_instance = pygad.GA(
        fitness_func=fitness_function,
        gene_type=int,
        gene_space=[0, 1],
        num_generations=50,
        num_parents_mating=2,
        sol_per_pop=10,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        num_genes=len(items),
        on_generation=on_generation
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    selected_items = [item['name'] for item, selected in zip(items, solution) if selected]
    total_weight = sum(item['weight'] for item, selected in zip(items, solution) if selected)
    total_value = sum(item['value'] for item, selected in zip(items, solution) if selected)

    print(f"Final-Solution: {selected_items}")
    print(f"Total-Value: {total_value}")
    print(f"Total-Weight: {total_weight}")
    total_time = time.time() - start_time

    if solution_fitness >= target_solution_value:
        best_solutions_count += 1
        total_successful_time += total_time

    ga_instance.plot_fitness().savefig(f"plot{i}.png")

print(f"Percentage of best solutions found: {best_solutions_count / 10 * 100}%")
if best_solutions_count > 0:
    print(f"Average time of successful solution: {total_successful_time / best_solutions_count} seconds")
else:
    print("No successful solutions found.")
