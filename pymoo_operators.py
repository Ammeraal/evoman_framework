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

    Xp[mut] = X[mut] + np.random.normal(0, sigma, size=len(X[mut]))

    Xp = repair_random_init(Xp, X, xl, xu)

    return Xp


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class GaussianMutationPymoo(Mutation):

    def __init__(self, sigma_max=0.25, sigma_min=0.1, gen_max=40, adaptive=True, **kwargs):
        """
        If adaptive is false, sigma_max will be used as sigma.
        Otherwise, sigma will be decreased linearly until it reaches sigma_min at generation gen_max. After that it stays sigma_min
        """
        super().__init__(**kwargs)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.gen_max = gen_max
        self.adaptive = adaptive

        self.sigma = self.get_sigma(0)

    def _do(self, problem, X, **kwargs):
        X = X.astype(float)

        sigma = np.array([self.get_sigma(problem.data["n_gen"] -1) for i in range(len(X))])
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_gauss(X, problem.xl, problem.xu, sigma, prob_var)

        return Xp
    
    def get_sigma(self, n_gen):
        if not self.adaptive:
            return self.sigma_max

        if n_gen > self.gen_max:
            return self.sigma_min

        return (-(self.sigma_max - self.sigma_min)/(self.gen_max-1)) * n_gen + self.sigma_max

class NSGA2Mutation(Mutation):
    def __init__(self, tau, eps, mean=0):
        self.tau = tau
        self.eps = eps
        self.mean = mean

    def _do(self, problem, X, **kwargs):
        # TODO implement selfadaptive mutation: uncomment
        # probability to mutate set as last element of X
        # prob_var = X[-1]
        prob_var = self.get_prob_var(problem, size=(len(X)))        # the probability for each gene to mutate
        #mutation = SelfAdaptiveMutation(self.tau, self.eps, prob_var, self.mean)
        mutation = GaussianMutation(stdv=0.25, mutation_rate=prob_var)
        # convert X to Genome style object
        genome = Genome(X) 

        for idx, g in genome.value:
            genome.value[idx] = mutation.mutate_gene(g)

       

        return [genome.value] 

class NSGA2Crossover(Crossover):

    def __init__(self, nr_parents, nr_offspring, **kwargs):
        super().__init__(nr_parents, nr_offspring, **kwargs)
        self.nr_parents = nr_parents

    def _do(self, _, X, **kwargs):
        # TODO implement selfadaptive mutation
        _, n_matings, _ = X.shape
        parents = []
        children = []
        crossover = MultiParentCrossover(nr_parents=self.nr_parents)
        
        for i in range(n_matings):
            [parents.append(Genome(X[j][i])) for j in range(self.n_parents)]
            children.append(crossover.cross(parents, pop_size=n_matings))
            parents = []
        Xp = []
        for child in children:
            Xp.append(child.value)
        Xp = np.array(Xp)

        Xp = np.expand_dims(Xp, axis=0)  # extra dimension to fool the NSGA2 algorithm
        return Xp