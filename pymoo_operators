import numpy as np
import random

from pymoo.core.mutation import Mutation
from sympy import Mul
from Untitled.ea_lib import SelfAdaptiveMutation
from ea_lib import GaussianMutation
from pymoo.core.crossover import Crossover
from ea_lib import MultiParentCrossover

from Genome import Genome

class NSGA2Mutation(Mutation):
    def __init__(self, tau, eps, mean=0):
        self.tau = tau
        self. eps = eps
        self.mean = mean

    def _do(self, problem, X, **kwargs):
        # TODO implement selfadaptive mutation: uncomment
        # probability to mutate set as last element of X
        # prob_var = X[-1]
        prob_var = self.get_prob_var(problem, size=(len(X)))
        mutation = SelfAdaptiveMutation(self.tau, self.eps, prob_var, self.mean)
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
