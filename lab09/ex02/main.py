import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")

COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (45, 76),
    (35, 40),
    (60, 30),
    (10, 20),
    (55, 65),
    (80, 45),
    (15, 70),
    (50, 50),
    (25, 85),
    (68, 72),
    (33, 55),
    (90, 80),
    (70, 30),
    (40, 60),
    (22, 43),
    (63, 75),
    (55, 25),
    (17, 62),
    (38, 48),
    (83, 60),
    (27, 75),
    (75, 45),
    (49, 30),
    (20, 10),
    (60, 80),
    (35, 50),
    (80, 20),
    (28, 65),
    (72, 55),
    (45, 35),
    (10, 50),
    (50, 90),
    (30, 25),
    (65, 40),
)



def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS, ant_count=3000, alpha=0.5, 
                   beta=1.2, pheromone_evaporation_rate=0.40, 
                   pheromone_constant=10.0,iterations=30)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.savefig("plot4.png")