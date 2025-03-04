import random
from Genome import Genome
import numpy as np
import copy


class Crossover:
    members = []

    def __init__(self, **kwargs):
        for i in self.members:
            if i not in kwargs:
                raise AssertionError(
                    "Missing Named Argument %s To Class %s" % (i, type(self).__name__))
        for key, val in kwargs.items():
            setattr(self, key, val)

    def cross(self, parent_list, pop_size, incest_thresh):
        raise NotImplementedError("Base Crossover is a Pure Virtual Class")


class UniformCrossover(Crossover):
    members = []

    def cross(self, parents_list, pop_size):
        while True:
            parent1_idx = random.randint(0, len(parents_list) - 1)
            parent2_idx = random.randint(0, len(parents_list) - 1)
            # hotfix by paddy: I guess the idea is to choose two different parents
            if parent1_idx != parent2_idx:
                break

        parent1 = parents_list[parent1_idx].value
        parent2 = parents_list[parent2_idx].value
        child = []
        for i in range(len(parent1)):
            bool = random.getrandbits(1)
            if bool == 1:
                child.append(parent1[i])
            else:
                child.append(parent2[i])

        new_genome = Genome(child)
        return new_genome


class MatrixCrossover(Crossover):
    members = ["nr_parents"]

    def _genome_to_matrix(self, value):
        num_input = 20
        num_output = 5
        # TODO implement for hidden layer as well
        bias = value[:num_output]
        weights = value[num_output:].reshape((num_input, num_output))

        return weights, bias

    def _matrix_to_genome(self, weights, bias):
        # TODO implement for hidden layer as well
        value = np.concatenate((bias, weights.flatten()))

        return value

    def cross(self, parent_list, pop_size):
        group_input_weights = True
        # TODO implement for multiple parents
        # create matrix
        parents = np.random.choice(parent_list, size=self.nr_parents, replace=False)
        biases = []
        weights = []
        for p in parents:
            weight, bias = self._genome_to_matrix(p.value)
            biases.append(bias)
            weights.append(weight)

        # change rows
        offspring_weight = np.ones_like(weights[0])
        offspring_bias = np.ones_like(biases[0])

        # loop over rows
        if group_input_weights:
            for i in range(np.shape(offspring_weight)[0]):
                parent_idx = np.random.randint(0, len(parents))
                offspring_weight[i, :] = np.copy(
                    weights[parent_idx][i, :])  # transfer the i'th column of the sampled parent
        else:
            # loop over cols
            for i in range(np.shape(offspring_weight)[1]):
                parent_idx = np.random.randint(0, len(parents))
                offspring_weight[:, i] = np.copy(
                    weights[parent_idx][:, i])  # transfer the i'th row of the sampled parent

        # loop over colls for bias
        for i in range(len(offspring_bias)):
            parent_idx = np.random.randint(0, len(parents))
            offspring_bias[i] = np.copy(biases[parent_idx][i])

        # flatten matrix to vector
        vector = self._matrix_to_genome(offspring_weight, offspring_bias)
        return Genome(vector)


class FractionalCrossover(Crossover):
    member = []

    def cross(self, parents_list):
        while True:
            parent1_idx = random.randint(0, len(parents_list) - 1)
            parent2_idx = random.randint(0, len(parents_list) - 1)
            if parent1_idx != parent2_idx:
                break

        parent1 = parents_list[parent1_idx].value
        parent2 = parents_list[parent2_idx].value

        fraction = random.uniform(0, 1)
        child = fraction * parent1 + (1 - fraction) * parent2

        return Genome(child)


class OnePointCrossover(Crossover):
    # Member - elitism
    members = []

    def cross(self, parents_list, pop_size):
        while True:
            parent1_idx = random.randint(0, len(parents_list) - 1)
            parent2_idx = random.randint(0, len(parents_list) - 1)
            if parent1_idx != parent2_idx:
                break

        parent1 = parents_list[parent1_idx].value
        parent2 = parents_list[parent2_idx].value

        point = random.randrange(1, len(parents_list[0].value) - 1)

        child = np.concatenate((parent1[:point], parent2[point:]))
        new_genome = Genome(child)
        return new_genome


class MultiParentCrossover(Crossover):
    # Member - nr_parents
    members = ["nr_parents"]

    def diversity(self, pop):
        """
        Returns a scalar that indicates the diversity of a population.
        The higher this value the higher the diversity.
        This heuristic is normalized among the pop size but not the genome length.
        """

        similar_sum = 0
        for i in range(len(pop)):
            for k in range(i+1, len(pop)):
                similar_sum += sum((pop[i].value - pop[k].value) ** 2)

        # normalize by amount of individual sums
        return similar_sum / ((len(pop)**2 - len(pop)) / 2.)

    def cross(self, parents_list, pop_size, incest_thresh=0):
        nr_parents = self.nr_parents
        parents_idx = set()
        parents = []
        points = []
        # print('parents_list is long: ',len(parents_list))
        # select list of crossing parents
        crossing_pool = np.random.choice(parents_list, nr_parents, replace=False)
        distance = self.diversity(crossing_pool)
        running = True
        run = 1
        while running and run < 4:
            if distance <= incest_thresh:
                crossing_pool = np.random.choice(parents_list, nr_parents, replace=False)
            else:
                running = False
            run += 1
        # while True:
        #    parents_idx.add(random.randint(0, len(parents_list) - 1))
        #    if len(parents_idx) == nr_parents:
        #        break
        # print(parents_idx)
        # for idx in parents_idx:
        #    parents.append(parents_list[idx].value)

        parents = [p.value for p in crossing_pool]
        # print(len(parents))
        while True:
            point = random.randrange(1, len(parents_list[0].value) - 1)
            if point not in points:
                points.append(point)
            if len(points) == (nr_parents - 1):
                break
        points.sort()
        for k in range(nr_parents - 1):
            if k == 0:
                child = np.concatenate(
                    (np.copy(parents[k][:points[k]]), np.copy(parents[k + 1][points[k]:points[k + 1]])))
            elif k == (nr_parents - 2):
                child = np.concatenate((child, np.copy(parents[k + 1][points[k]:])))
            else:
                child = np.concatenate(
                    (child, np.copy(parents[k + 1][points[k]:points[k + 1]])))

        new_genome = Genome(child)

        # crossover sigma (chose sigma of one parent)
        if parents_list[0].sigma:
            new_genome.sigma = copy.copy(np.random.choice(crossing_pool).sigma)

        return new_genome


class Mutation:
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, population):
        pop_offspring = []
        for individual in population:
            # modify sigma first and then mutate the offspring
            self.mutate_sigma(individual)

            genome = individual.value
            offspring = []
            for gene in genome:
                # draw random probability for mutation
                mutate = np.random.uniform(0, 1)
                # if mutation prob is below mutation rate, mutate gene in genome by adding random number
                if mutate <= self.mutation_rate:
                    w = self.mutate_gene(gene)
                    offspring.append(w)
                else:
                    offspring.append(gene)

            individual.value = np.array(offspring)

    def mutate_gene(self, gene):
        return gene

    def mutate_sigma(self, individual):
        pass

    def set_mutation(self, rate):
        self.mutation_rate = rate


class SelfAdaptiveMutation(Mutation):
    def __init__(self, tau, eps, mutation_rate=0.2, mean=0):
        super(SelfAdaptiveMutation, self).__init__(mutation_rate=mutation_rate)
        # tau := step size (hyperparam) (eg. 1 / log(pop_size))
        self.tau = tau
        self.mean = mean
        self.eps = eps

    def mutate_sigma(self, individual: Genome):
        # self adaption
        # update: sigma * exp(tau * norm(0, 1))
        new_sigma = individual.sigma * np.exp(self.tau * np.random.normal(0, 1))
        # build a boundary (sigma < eps) to overcome vanishing
        if new_sigma < self.eps:
            new_sigma = self.eps

        individual.sigma = new_sigma
        self.sigma = individual.sigma  # this is temporarily overridden for each individual

    def mutate_gene(self, gene):
        w = gene + np.random.normal(self.mean, self.sigma)

        # crop weight to [0, 1]
        # overcome the problem of a higher probability for weights at the boundary
        if w > 1:
            w += 1 - w
        elif w < -1:
            w += -1 - w

        return w

    @staticmethod
    def init_sigma(pop, mean=0.25):
        for p in pop:
            p.sigma = np.clip(np.random.normal(mean, 0.05), a_min=0.01, a_max=None)


class GaussianMutation(Mutation):
    def __init__(self, mean=0, stdv=0.25, mutation_rate=0.2):
        super().__init__(mutation_rate=mutation_rate)
        self.mean = mean
        self.stdv = stdv

    def mutate_gene(self, gene):
        w = gene + np.random.normal(self.mean, self.stdv)
        if w > 1:
            w = 1
        elif w < -1:
            w = -1
        return w


class UniformMutation(Mutation):
    def __init__(self, mutation_rate=0.2):
        super().__init__(mutation_rate=mutation_rate)

    def mutate_gene(self, gene):
        return np.random.uniform(-1, 1)


class Selection():
    pass


class RankingSelection(Selection):
    def __init__(self, s=2.0):
        self.s = s

    def select(self, pop):
        # sort population by their fitness
        sorted_pop = np.array(sorted(pop, key=lambda p: p.fitness))
        mu = len(sorted_pop)  # round(len(sorted_pop) / 4)     # number of parents (as fraction of the population)

        # generate p_s
        p = []  # each element of the list is the probability for an element in sorted_pop to be selected
        for i in range(len(sorted_pop)):
            p.append((2 - self.s) / mu + (2 * i * (self.s - 1)) / (mu * (mu - 1)))

        # select pool based on p_s
        mating_pool = np.random.choice(sorted_pop, mu, replace=True, p=np.array(p))

        return mating_pool


class NaiveSelection(Selection):
    def __init__(self, s=2.0):
        self.s = 2.0

    def select(self, pop):
        p = []
        mating_pool = []
        fitness = []
        # assess the probability of offspring for each individual (5.2.2 Ranking selection)
        z = round(len(pop) / 4)  # number of parents
        for g in pop:
            fitness.append(g.fitness)
        order = np.argsort(fitness)
        ranks = np.argsort(order)
        for i in ranks:
            p.append((2 - self.s) / z + (2 * i * (self.s - 1)) / (z * (z - 1)))
        order = np.argsort(p)
        p = sorted(p / max(p))
        # select parents according to offspring probability (5.2.3 Implementing selection probabilities)
        current_member = 1
        i = 0
        r = np.random.uniform(0, 1 / z)
        while current_member <= z:
            if i > len(p):
                mating_pool.append(pop[order[i]])
            else:
                while r <= p[i]:
                    mating_pool.append(pop[order[i]])
                    r = r + 1 / z
                    current_member += 1
            i += 1

        return np.array(mating_pool)
