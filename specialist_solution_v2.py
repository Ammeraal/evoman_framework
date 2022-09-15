import numpy as np
from numpy.random import default_rng


def init_population(num_offspring, _n_hidden):
    # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
    seed = 42
    num_inputs = 8
    num_output = 20
    init_bias = 0.0

    # num neurons
    sum = num_inputs * _n_hidden[0] + _n_hidden[-1] * num_output
    for i in range(1, len(_n_hidden)):
        sum += _n_hidden[i - 1] * _n_hidden[i]

    # TODO do this more efficient!
    pop = []
    for i in range(num_offspring):
        pop.append(default_rng(seed).random(sum) + init_bias)
        seed += 1

    return np.array(pop)


print(init_population(2, np.array([2, 2])))
