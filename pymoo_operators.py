import numpy as np
import random

from pymoo.core.mutation import Mutation
#from sympy import Mul
from ea_lib import SelfAdaptiveMutation
from ea_lib import GaussianMutation
from pymoo.core.crossover import Crossover
from ea_lib import MultiParentCrossover

from Genome import Genome

import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_random_init


# ---------------------------------------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------------------------------------


def mut_gauss(X, xl, xu, sigma, prob):
    n, n_var = X.shape
    assert len(sigma) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = np.random.random(X.shape) < prob[:, None]

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]
    sigma = sigma[:, None].repeat(n_var, axis=1)[mut]

    Xp[mut] = np.random.normal(X[mut], sigma)

    Xp = repair_random_init(Xp, X, xl, xu)

    return Xp


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class GaussianMutationPymoo(Mutation):

    def __init__(self, sigma=0.1, **kwargs):
        super().__init__(**kwargs)
        self.sigma = Real(sigma, bounds=(0.01, 0.25), strict=(0.0, 1.0))

    def _do(self, problem, X, **kwargs):
        X = X.astype(float)

        sigma = get(self.sigma, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_gauss(X, problem.xl, problem.xu, sigma, prob_var)

        return Xp

class NSGA2Mutation(Mutation):
    def __init__(self, tau, eps, mean=0):
        self.tau = tau
        self. eps = eps
        self.mean = mean

    def _do(self, problem, X, **kwargs):
        # TODO implement selfadaptive mutation: uncomment
        # probability to mutate set as last element of X
        # prob_var = X[-1]
        prob_var = self.get_prob_var(problem, size=(len(X)))        # the probability for each gene to mutate
        #mutation = SelfAdaptiveMutation(self.tau, self.eps, prob_var, self.mean)
        mutation = GaussianMutation(stdv=0.25, mutation_rate=prob_var)
        # convert X to Genome style object
        genome = Genome(X) #Genome(X[:-2],X[-1])

        for idx, g in genome.value:
            genome.value[idx] = mutation.mutate_gene(g)

        # mutation.mutate_sigma(genome)

        return [genome.value] # , genome.sigma]

class NSGA2Crossover(Crossover):

    def __init__(self, nr_parents, nr_offspring, **kwargs):
        super().__init__(nr_parents, nr_offspring, **kwargs)
        self.nr_parents = nr_parents

    def _do(self, _, X, **kwargs):
        # TODO implement selfadaptive mutation: uncomment
        parents=[]  
        for parent in X:
            parents.append(Genome(parent))

        crossover = MultiParentCrossover()
        genome = crossover.cross(parents)

        return [genome.value] # , genome.sigma]
