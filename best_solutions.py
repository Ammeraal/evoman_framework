import numpy as np
from game_setup_solution import GameManager
from demo_controller import player_controller

def load_genomes(num, i, type):
    genome = np.load(f"{i}_final/{num}_{type}type/autosave_best_individual.npy", allow_pickle=True)
    game = GameManager(controller=player_controller(0))
    for g in genome:
        g.fitness = 0.0
        g.fitness, p, e, t = game.play(pcont=g.value)
        individual_gain = p - e
    return str(individual_gain)

for i in [1, 3, 4]:
    for type in ["pheno", "geno"]:
        f = open(f"{i}_final/individual_gain_{type}type.csv", "w")
        for j in range(10):
            f.write(load_genomes(j, i, type) + "\n")