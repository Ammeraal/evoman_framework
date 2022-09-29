import matplotlib.pyplot as plt
import pandas as pd
from calendar import c
import numpy as np
import random
from numpy.random import default_rng
from Genome import Genome
from game_setup_solution import GameManager
from demo_controller import player_controller
import os
import time
import concurrent.futures
import multiprocessing
import concurrent.futures
from tkinter import E

start = time.perf_counter()


class Crossover():
    members = []

    def __init__(self, **kwargs):
        for i in self.members:
            if i not in kwargs:
                raise AssertionError(
                    "Missing Named Argument %s To Class %s" % (i, type(self).__name__))
        for key, val in kwargs.items():
            setattr(self, key, val)

    def cross(self, parent_list, pop_size):
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
                offspring_weight[i,:] = np.copy(weights[parent_idx][i,:])      # transfer the i'th column of the sampled parent
        else:
            # loop over cols
            for i in range(np.shape(offspring_weight)[1]):
                parent_idx = np.random.randint(0, len(parents))
                offspring_weight[:,i] = np.copy(weights[parent_idx][:,i])      # transfer the i'th row of the sampled parent

        # loop over colls for bias
        for i in range(len(offspring_bias)):
            parent_idx = np.random.randint(0, len(parents))
            offspring_bias[i] = np.copy(biases[parent_idx][i])

        # flatten matrix to vector
        vector = self._matrix_to_genome(offspring_weight, offspring_bias)
        return Genome(vector)

class FractionalCrossover(Crossover):
    member=[]
    def cross(self,parents_list):
        while True:
            parent1_idx = random.randint(0, len(parents_list) - 1)
            parent2_idx = random.randint(0, len(parents_list) - 1)
            if parent1_idx != parent2_idx:
                break

        parent1 = parents_list[parent1_idx].value
        parent2 = parents_list[parent2_idx].value

        fraction = random.uniform(0,1)
        child = fraction * parent1 + (1-fraction) * parent2

        return Genome(child)


class OnePointCrossover(Crossover):
    #Member - elitism
    members = []
    def cross(self, parents_list, pop_size):
        while True:
            parent1_idx = random.randint(0, len(parents_list) - 1)
            parent2_idx = random.randint(0, len(parents_list) - 1)
            if parent1_idx != parent2_idx:
                break

        parent1 = parents_list[parent1_idx].value
        parent2 = parents_list[parent2_idx].value

        point = random.randrange(1, len(parents_list[0].value)-1)

        child = np.concatenate((parent1[:point], parent2[point:]))
        new_genome = Genome(child)
        return new_genome



class MultiParentCrossover(Crossover):
    #Member - nr_parents
    members = ["nr_parents"]

    def cross(self, parents_list, pop_size):
        nr_parents = self.nr_parents
        parents_idx = set()
        parents = []
        points = []
        # print('parents_list is long: ',len(parents_list))
        # select list of crossing parents
        crossing_pool = np.random.choice(parents_list, nr_parents, replace=False)

        #while True:
        #    parents_idx.add(random.randint(0, len(parents_list) - 1))
        #    if len(parents_idx) == nr_parents:
        #        break
        # print(parents_idx)
        #for idx in parents_idx:
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
        for k in range(nr_parents-1):
            if k == 0:
                child = np.concatenate(
                    (parents[k][:points[k]], parents[k+1][points[k]:points[k+1]]))
            elif k == (nr_parents - 2):
                child = np.concatenate((child, parents[k+1][points[k]:]))
            else:
                child = np.concatenate(
                    (child, parents[k+1][points[k]:points[k+1]]))
        new_genome = Genome(child)
        return new_genome


class Mutation():
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, population):
        pop_offspring = []
        for individual in population:
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
            pop_offspring.append(Genome(np.array(offspring)))
        return np.array(pop_offspring)

    def mutate_gene(self, gene):
        return gene

    def set_mutation(self, rate):
        self.mutation_rate = rate


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
        mu = len(sorted_pop)        #round(len(sorted_pop) / 4)     # number of parents (as fraction of the population)

        # generate p_s
        p = []              # each element of the list is the probability for an element in sorted_pop to be selected
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
            p.append((2-self.s)/z + (2*i*(self.s-1))/(z*(z-1)))
        order = np.argsort(p)
        p = sorted(p/max(p))
        # select parents according to offspring probability (5.2.3 Implementing selection probabilities)
        current_member = 1
        i = 0
        r = np.random.uniform(0, 1/z)
        while current_member <= z:
            if i > len(p):
                mating_pool.append(pop[order[i]])
            else:
                while r <= p[i]:
                    mating_pool.append(pop[order[i]])
                    r = r+1/z
                    current_member += 1
            i += 1

        return np.array(mating_pool)


class SpecialistSolutionV2():
    def __init__(self, nr_parents=3, mutation_rate=0.2, s=2.0, n_hidden=0, elitism=4):
        self.current_generation = 0
        self.cross_algorithm = MatrixCrossover(nr_parents=nr_parents)      #MultiParentCrossover(nr_parents=3)
        self.mutation_algorithm = GaussianMutation(mutation_rate=mutation_rate)# UniformMutation(mutation_rate=0.03) #
        self.selection_algorithm = RankingSelection(s=s)
        self.save_interval = 10
        self.load_pop = False
        self.load_generation = 20
        self.n_hidden = n_hidden
        self.elitism = elitism

        self.best_fitness = -10
        self.best_individual = None

    def init_population(self, pop_size, _n_hidden):
        # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
        seed = 42
        num_inputs = 20
        num_output = 5
        init_bias = -1.0

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
            g = Genome(default_rng(seed).random(sum) * 2 + init_bias)
            pop.append(g)
            seed += 1

        return np.array(pop)

    def threaded_evaluation(self, population, n_hidden):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # We submit the list of the seconds we want to have.
            # TODO insert list of all genomes

            results = executor.map(self.threaded_evaluation_fittness, population, [
                                   n_hidden for i in range(len(population))])
            return list(results)

    # TODO pass n_hidden to this function in threated_evaluation
    def threaded_evaluation_fittness(self, g, n_hidden=0):
        game = GameManager(controller=player_controller(n_hidden))
        g.fitness = 0.0
        g.fitness, p, e, t = game.play(pcont=g.value)
        return g

    def evaluate_fitness_factory(self, game):
        def evaluate_fitness(pop):
            # TODO trigger game with all net configs
            # TODO parallelize
            for g in pop:
                g.fitness = 0.0
                g.fitness, p, e, t = game.play(pcont=g.value)

        return evaluate_fitness
    def save_fitness(self, file_handle, fitness_values):
        print("saving pop to file")

        np.savetxt(file_handle, np.array(fitness_values), newline=" ")
        file_handle.write("\n")

    def save_diversity(self, file_handle, div):
        print("saving pop diversity to file")

        file_handle.write(str(div) + "\n")

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

    def save_population(self, path, pop, generation, best_fitness, best_individual):
        print("saving population at {}".format(path))
        np.save(f"{path}.npy", pop)
        np.save(f"{path}_best_individual.npy", np.array([best_individual]))
        np.savetxt(f"{path}_generation", np.array([generation, best_fitness]))

    def load_population(self, path):
        print("loading initial population for {}".format(path))
        pop = np.load(f"{path}.npy", allow_pickle=True)
        additional_data = np.loadtxt(f"{path}_generation")
        best_individual = np.load(f"{path}_best_individual.npy", allow_pickle=True)[0]
        last_gen = int(additional_data[0])
        best_fitness = float(additional_data[1])

        return pop, last_gen, best_fitness, best_individual

    def fitness_boxplot(self, file, generations):
        print(file)
        df = pd.read_csv(file, header=None, sep=" ").iloc[:, :-1]
        na = np.array(df)
        r = [(gix,gval) for gix,gen in enumerate(na) for gval in gen]
        x = [left for left,right in r]
        y = [right for left,right in r]
        plt.figure(figsize=(8,4)), plt.scatter(x,y,s = 0.2)
        plt.boxplot(na.transpose())
        plt.plot(list(map(np.max,na)))
        plt.plot(list(map(np.mean,na)))
        plt.plot(list(map(np.std,na)))
        print(df)
        return

    def visualize(self, file, generations):
        # make plot of mean fitness over generations, with standard deviation
        # TODO mean should average over 10 runs!! for now this is just one run
        # TODO include max fitness with standard deviation over 10 runs
        # TODO make separate plots for enemies
        df = pd.read_csv(file, header=None, sep=" ").iloc[:, :-1]
        df_avg = df.mean(axis=1)
        df_std = df.std(axis=1)
        df_max = df.max(axis=1)

        # make plot
        plt.plot(df_avg)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.fill_between(range(generations), df_avg -
                         df_std, df_avg + df_std, alpha=.3)
        plt.ylim(-6)
        plt.savefig(f"{self.save_dir}avg_lineplot.png")
        plt.clf()

    def diversity_plot(self, file):
        df = pd.read_csv(file, header=None)
        plt.plot(df)
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.savefig(f"{self.save_dir}diversity_plot.png")
        plt.clf()

    def initialize_run(self, pop_size, auto_load=True):
        # TODO load last executed state
        if os.path.exists(f"{self.save_dir}autosave.npy") and auto_load:
            self.load_pop = True

        # initialization
        if not self.load_pop:
            pop = self.init_population(pop_size=pop_size, _n_hidden=self.n_hidden)
            file_open_mode = "w"
            self.load_generation = -1
        else:
            #pop = self.load_population(f"{self.save_dir}pop_{self.load_generation}.npy")
            pop, self.load_generation, self.best_fitness, self.best_individual = self.load_population(f"{self.save_dir}autosave")
            file_open_mode = "a"

        save_txt_handle = open(f"{self.save_dir}fitness.csv", file_open_mode)
        div_file = open(f"{self.save_dir}diversity.csv", file_open_mode)
        self.pop = pop

        return save_txt_handle, div_file

    def update_algorithms(self):
        pass

    def next_generation(self,pop_size):
        offspring = []

        # standard crossover
        selected_parents = self.selection_algorithm.select(self.pop)
        for i in range(pop_size - self.elitism):
            new_genome = self.cross_algorithm.cross(selected_parents, pop_size=pop_size)
            offspring.append(new_genome)
        offspring = np.array(offspring)

        # mutation
        offspring = self.mutation_algorithm.mutate(offspring)

        # elitism
        elite_parents = sorted(self.pop, key=lambda x: x.fitness)[(self.elitism+1)*-1:-1]
        self.pop = np.append(offspring, elite_parents)

        self.current_generation += 1
        self.update_algorithms()

    def run(self, generations, pop_size, save_txt_handle, div_file):
        for i in range(self.load_generation + 1, generations):
            div = self.diversity(self.pop)
            print("**** Starting with evaluation of generation {}. Diversity: {}".format(i,
                                                                                         self.diversity(self.pop)))
            #if div < 30:
            #    self.mutation_algorithm.set_mutation(0.1)
            #else:
            #    self.mutation_algorithm.set_mutation(0.02)
            start_t = time.perf_counter()
            self.pop = self.threaded_evaluation(self.pop, self.n_hidden)

            # save best fitness
            local_fitness = np.array([g.fitness for g in self.pop])
            local_max = np.max(local_fitness)
            if local_max > self.best_fitness:
                self.best_fitness = local_max
                self.best_individual = self.pop[np.argmax(local_fitness)]

            fitness_values = [p.fitness for p in self.pop]      # store them but save them right before the backup
            self.next_generation(pop_size)

            # saving system
            #if i % self.save_interval == 0:
            self.save_fitness(save_txt_handle, fitness_values)
            self.save_diversity(div_file, self.diversity(self.pop))
            self.save_population(f"{self.save_dir}autosave", self.pop, i, self.best_fitness, self.best_individual)

            end = time.perf_counter()
            print("execution for one generation took: {} sec".format(end-start_t))
        return self.best_fitness

    def start(self, generations=30, pop_size=20,
              experiment_name="test", generate_plots=True, auto_load=True, evaluate_best=False):
        # TODO set all hyper params
        # Hyper params

        self.save_dir = f"specialist_solution_v2/{experiment_name}/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        save_txt_handle, div_file = self.initialize_run(pop_size, auto_load)

        # evaluation
        if not evaluate_best:
            # the loaded generation should be processed by the EA algorithm so we start directly with evaluation
            max_fitness = self.run(generations, pop_size,
                                   save_txt_handle, div_file)
        else:
            self.threaded_evaluation(np.array([self.best_individual]), self.n_hidden)

        # TODO return best fitness
        # TODO implement early stopping
        save_txt_handle.close()
        div_file.close()
        print("all done!")

        if generate_plots:
            print("visualizing and saving results...")
            self.visualize(f"{self.save_dir}fitness.csv", generations)
            self.diversity_plot(f"{self.save_dir}diversity.csv")

        return max_fitness



if __name__ == "__main__":
    ea_instance = SpecialistSolutionV2(mutation_rate=0.16, s=1.95, nr_parents=3)
    best_fitness = ea_instance.start(generations=80, pop_size=40, experiment_name="4_test/0_phenotype", evaluate_best=False)
    print("best_fitness: {}".format(best_fitness))
    