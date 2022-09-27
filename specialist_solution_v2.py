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

import pandas as pd
import matplotlib.pyplot as plt

class SpecialistSolutionV2():
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

    def mutate(self, pop,mut_rate,mean,sigma):
        pop_offspring = []
        for individual in pop:
            genome = individual.value
            offspring = []
            for gene in genome:
                # draw random probability for mutation
                mutate = np.random.uniform(0, 1)
                # if mutation prob is below mutation rate, mutate gene in genome by adding random number
                if mutate <= mut_rate:
                    w = gene + np.random.normal(mean, sigma)
                    if w > 1:
                        w = 1
                    elif w < -1:
                        w = -1
                    offspring.append(w)
                else:
                    offspring.append(gene)
            pop_offspring.append(Genome(np.array(offspring)))
        return np.array(pop_offspring)

    def threaded_evaluation(self, population, n_hidden):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # We submit the list of the seconds we want to have.
            # TODO insert list of all genomes

            results = executor.map(self.threaded_evaluation_fittness, population, [n_hidden for i in range(len(population))])
            return list(results)


    def threaded_evaluation_fittness(self, g, n_hidden=0): # TODO pass n_hidden to this function in threated_evaluation
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

    def selection(self, pop, s):
        p = []
        mating_pool=[]
        fitness=[]
        
        # assess the probability of offspring for each individual (5.2.2 Ranking selection)
        z=round(len(pop)/4) # number of parents
        for g in pop:
            fitness.append(g.fitness)
        order = np.argsort(fitness)
        ranks = np.argsort(order)
        for i in ranks:
            p.append((2-s)/z + (2*i*(s-1))/(z*(z-1)))
        order=np.argsort(p)
        p=sorted(p/max(p))
        # select parents according to offspring probability (5.2.3 Implementing selection probabilities)
        current_member = 1
        i = 0
        r=np.random.uniform(0,1/z)
        while current_member<=z:
            while r<=p[i]:
                mating_pool.append(pop[order[i]])
                r=r+1/z
                current_member+=1
                i+=1
            i+=1
        return np.array(mating_pool)

    def uniform_crossover(self, parents_list, pop_size):
        # TODO return list of the new offspring
        children = []
        for z in range(pop_size):
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
            children.append(new_genome)
        children = np.array(children)

        return children

    def one_point_crossover(self, parents_list, pop_size, elitism):
        children = []
        for i in range(pop_size - elitism):
            while True:
                parent1_idx = random.randint(0, len(parents_list) - 1)
                parent2_idx = random.randint(0, len(parents_list) - 1)
                if parent1_idx != parent2_idx:
                    break

            parent1 = parents_list[parent1_idx].value
            parent2 = parents_list[parent2_idx].value

            point = random.randrange(1,len(parents_list[0].value)-1)

            child = np.concatenate((parent1[:point],parent2[point:]))
            new_genome = Genome(child)
            children.append(new_genome)

        children = np.array(children)

        return children

    def multi_parent_crossover(self, parents_list, pop_size, nr_parents):
        children = []

        for z in range(pop_size):
            parents_idx = set()
            parents = []
            points = []
            # print('parents_list is long: ',len(parents_list))
            while True:
                parents_idx.add(random.randint(0,len(parents_list) - 1))
                if len(parents_idx) == nr_parents:
                    break
            # print(parents_idx)
            for idx in parents_idx:
                parents.append(parents_list[idx].value)
            # print(len(parents))
            while True:
                point = random.randrange(1,len(parents_list[0].value) -1)
                if point not in points:
                    points.append(point)
                if len(points) == (nr_parents - 1):
                    break

            points.sort()

            for k in range(nr_parents-1):
                # print(points[k])
                # print(points[k+1])
                # print(parents[0][:points[k]])
                # print()
                # print(parents[points[k]:points[k+1]])
                if k == 0:
                    child = np.concatenate((parents[k][:points[k]],parents[k+1][points[k]:points[k+1]]))
                elif k == (nr_parents - 2):
                    child = np.concatenate((child,parents[k+1][points[k]:]))
                else:
                    child = np.concatenate((child,parents[k+1][points[k]:points[k+1]]))
                # print('length of child is: ',len(child))
            new_genome = Genome(child)
            # print(new_genome.value)
            children.append(new_genome)

        children = np.array(children)
        # print(children)
        # children = np.squeeze(children)
        # print()
        # print(children)
        return children

    def save_fitness(self, file_handle, pop):
        print("saving pop to file")

        fitness_values = [p.fitness for p in pop]
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
        return similar_sum / ( (len(pop)**2 - len(pop)) / 2. )



    def save_population(self, path, pop):
        print("saving population at {}".format(path))
        np.save(path, pop)


    def load_population(self, path):
        print("loading initial population for {}".format(path))
        return np.load(path, allow_pickle=True)


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
        plt.fill_between(range(generations), df_avg - df_std, df_avg + df_std, alpha=.3)
        plt.xlim(-6)
        plt.savefig(f"{self.save_dir}avg_lineplot.png")
        plt.clf()

    def diversity_plot(self, file):
        df = pd.read_csv(file, header=None)
        plt.plot(df)
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.savefig(f"{self.save_dir}diversity_plot.png")
        plt.clf()

    def start(self, mut_rate=0.2, s=2.0, n_hidden=0, generations=10, pop_size=20, sigma=0.25, experiment_name="test",
              generate_plots=True):
        # TODO set all hyper params
        # Hyper params
        mean = 0

        # additional params
        save_interval = 10                  # there will be a save of the population every x generations
        load_pop = False                    # if true the state stored in the generation of <load_generation> be used as initial population
        load_generation = 4


        self.save_dir = f"specialist_solution_v2/{experiment_name}/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        game = GameManager(controller=player_controller(n_hidden))
        div_file = open(f"{self.save_dir}diversity.csv", "w")

        # initialization
        if not load_pop:
            pop = self.init_population(pop_size=pop_size, _n_hidden=n_hidden)
            save_txt_handle = open(f"{self.save_dir}fitness.csv", "w")
            load_generation = -1
        else:
            pop = self.load_population(f"{self.save_dir}pop_{load_generation}.npy")
            save_txt_handle = open(f"{self.save_dir}fitness.csv", "a")

        # evaluation
        # the loaded generation should be processed by the EA algorithm so we start directly with evaluation
        max_fitness = -10
        for i in range(load_generation + 1, generations):
            div = self.diversity(pop)
            self.save_diversity(div_file, div)
            print("**** Starting with evaluation of generation {}. Diversity: {}".format(i, self.diversity(pop)))
            start_t = time.perf_counter()
            pop = self.threaded_evaluation(pop, n_hidden)

            # save best fitness
            local_max = np.max(np.array([g.fitness for g in pop]))
            if local_max > max_fitness:
                max_fitness = local_max

            self.save_fitness(save_txt_handle, pop)

            selected_parents = self.selection(pop,s)
            offspring = self.multi_parent_crossover(selected_parents, pop_size=pop_size, nr_parents=3)
            pop = self.mutate(offspring, mut_rate, mean, sigma)

            # saving system
            if i % save_interval == 0:
                self.save_population(f"{self.save_dir}pop_{i}", pop)

            end = time.perf_counter()
            print("execution for one generation took: {} sec".format(end-start_t))

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


if __name__=="__main__":
    ea_instance = SpecialistSolutionV2()
    best_fitness = ea_instance.start()
    print("best_fitness: {}".format(best_fitness))
