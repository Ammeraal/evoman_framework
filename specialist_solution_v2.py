import numpy as np
import random
from numpy.random import default_rng

mut_rate = 0.2

def init_population(pop_size, _n_hidden):
    # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
    seed = 42
    num_inputs = 20
    num_output = 5
    init_bias = 0.0

    # num neurons
    sum = num_inputs * _n_hidden[0] + _n_hidden[-1] * num_output
    for i in range(1, len(_n_hidden)):
        sum += _n_hidden[i - 1] * _n_hidden[i]

    # TODO do this more efficient!
    pop = []
    for i in range(pop_size):
        pop.append(default_rng(seed).random(sum) + init_bias)
        seed += 1

    return np.array(pop)

def mutate(pop):
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
test_pop = init_population(2, np.array([2, 2]))
print(test_pop)
print(len(test_pop[0]))
print(mutate(test_pop))