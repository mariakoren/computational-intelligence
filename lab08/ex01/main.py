import pygad

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

def fitness_function(model, solution, solution_idx):
    total_weight = sum(item['weight'] for item, selected in zip(items, solution) if selected)
    total_value = sum(item['value'] for item, selected in zip(items, solution) if selected)
    
    if total_weight > max_capacity:
        return 0
    return total_value

if __name__ == "__main__":
    num_genes = len(items)

    ga = pygad.GA(
        fitness_func=fitness_function,
        gene_type=int,
        gene_space=[0, 1],
        num_generations=50,
        num_parents_mating=2,
        sol_per_pop=10,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        num_genes=num_genes
    )
    ga.run()

    solution, solution_fitness, solution_idx = ga.best_solution()
    
    selected_items = [item['name'] for item, selected in zip(items, solution) if selected]
    total_weight = sum(item['weight'] for item, selected in zip(items, solution) if selected)
    total_value = sum(item['value'] for item, selected in zip(items, solution) if selected)

    print(f"Final-Solution: {selected_items}")
    print(f"Total-Value: {total_value}")
    print(f"Total-Weight: {total_weight}")
