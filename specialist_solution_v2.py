import numpy as np
import random
from numpy.random import default_rng
from Genome import Genome
from game_setup_solution import GameManager
from demo_controller import player_controller


def init_population(pop_size, _n_hidden):
    # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
    seed = 42
    num_inputs = 20
    num_output = 5
    init_bias = 0.0

    # num neurons
    sum = 0
    if _n_hidden > 0:
        #           weights layer1      bias1           w2                      b2
        sum = num_inputs * _n_hidden + _n_hidden + _n_hidden * num_output + num_output
    else:
        # no hidden layer
        sum = num_inputs * num_output + num_output

    # TODO do this more efficient!
    pop = []
    for i in range(pop_size):
        g = Genome(default_rng(seed).random(sum) + init_bias)
        pop.append(g)
        seed += 1

    return np.array(pop)

def mutate(pop):
    mut_rate = 0.2

    pop_offspring = []
    lower = np.min(pop)
    upper = np.max(pop)
    for genome in pop:
        offspring = []
        for gene in genome:
            mutate = np.random.uniform(0, 1)
            if mutate <= mut_rate:
                offspring.append(random.uniform(lower, upper))
            else:
                offspring.append(gene)
        pop_offspring.append(offspring)
    return np.array(pop_offspring)

    # draw probability from distribution, if prob <= mutation threshold, mutate gene
    # mutate gene by changing it into random value between lower and upper domain

def evaluate_fitness_factory(game):
    def evaluate_fitness(pop):
        # TODO trigger game with all net configs
        # TODO parallelize
        for g in pop:
            g.fitness = 0.0
            g.fitness, p, e, t = game.play(pcont=g.value)

    return evaluate_fitness

def selection(pop):
    # TODO return subset of pop
    return np.array([])

def crossover(parents_list):
    # TODO return list of the new offspring
    return np.array([])

if __name__=="__main__":
    pop_size = 20
    generations = 10
    n_hidden = 8

    game = GameManager(controller=player_controller(n_hidden))
    evaluate_fitness = evaluate_fitness_factory(game)

    pop = init_population(pop_size=pop_size, _n_hidden=n_hidden)
    # TODO evaluation
    for i in range(generations):
        evaluate_fitness(pop)
        selected_parents = selection(pop)
        offspring = crossover(selected_parents)
        mutate(offspring)


