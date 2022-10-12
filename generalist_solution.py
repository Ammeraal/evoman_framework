import sys
import os
from numpy.random import default_rng
import time
import concurrent.futures
import copy
import json

sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from game_setup_solution import GameManager
from ea_lib import *
from visualize import visualize_fitness, diversity_plot

class SpecialistSolutionV2:
    def __init__(self, nr_parents=3, mutation_rate=0.2, s=2.0, n_hidden=0, elitism=4, pop_size=40, incest_thresh=100, enemy_numbers=[4]):
        self.current_generation = 0

        self.cross_algorithm = MultiParentCrossover(nr_parents=nr_parents)

        self.mutation_algorithm = SelfAdaptiveMutation(tau=1/np.log(pop_size), eps=0.01, mutation_rate=0.16)#GaussianMutation(mutation_rate=mutation_rate)# UniformMutation(mutation_rate=0.03) #
        self.selection_algorithm = RankingSelection(s=s)
        self.save_interval = 10
        self.load_pop = False
        self.n_hidden = n_hidden
        self.elitism = elitism
        self.pop_size = pop_size
        self.incest_thresh = incest_thresh

        self.enemy_numbers = enemy_numbers
        self.best_individual = None

        self.save_file_names = ["m_fitness.txt", "m_p.txt", "m_e.txt"]

        # initial values for meta_data. Will be overridden if data is loaded
        self.meta_data =   {"n_gen": -1,
                            "pop_size": self.pop_size,
                            "n_enemies": len(self.enemy_numbers),
                            "best_fitness": -100
                            }

    def compact_data(self, pop):
        return [[p.m_fitness for p in pop],
                [p.m_p for p in pop],
                [p.m_e for p in pop]
                ]

    def init_population(self, pop_size, _n_hidden):
        # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
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

        pop = []
        for i in range(pop_size):
            g = Genome(default_rng().random(sum) * 2 + init_bias)
            pop.append(g)

        if type(self.mutation_algorithm) == SelfAdaptiveMutation:
            SelfAdaptiveMutation.init_sigma(pop, mean=0.25)

        return np.array(pop)

    def threaded_evaluation(self, population, n_hidden):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # insert list of all genomes
            results = executor.map(self.threaded_evaluation_fittness, population,
                                   [n_hidden for _ in range(len(population))])
            return list(results)

    def threaded_evaluation_fittness(self, g, n_hidden=0):
        game = GameManager(controller=player_controller(n_hidden), enemy_numbers=self.enemy_numbers, multi_fitness=True)
        g.fitness = 0.0
        g.m_fitness, g.m_p, g.m_e, t = game.play(pcont=g.value)
        g.fitness = game.cons_multi_old(g.m_fitness)
        g.gain = g.m_p.sum() - g.m_e.sum()

        return g

    def save_vector(self, file_handle, fitness_values):
        print("saving vector to {}".format(file_handle.name))

        np.savetxt(file_handle, np.array(fitness_values), newline=" ")
        file_handle.write("\n")

    def save_scalar(self, file_handle, div):
        print("saving scalar to {}".format(file_handle.name))

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

    def save_population(self, path, pop, best_individual, old_pop):
        print("saving population at {}".format(path))
        np.save(f"{path}.npy", pop)
        np.save(f"{path}_best_individual.npy", np.array([best_individual]))

        # save additional data for plotting
        for data, name in zip(self.compact_data(old_pop), self.save_file_names):
            with open(f"{self.save_dir}{name}", "a") as f:
                np.savetxt(f, np.array(data))
                f.write("\n")

        # save meta_data
        with open(f"{self.save_dir}meta_data.json", "w") as f:
            json.dump(self.meta_data, f)

    def load_population(self, path):
        print("loading initial population for {}".format(path))
        pop = np.load(f"{path}.npy", allow_pickle=True)

        with open(f"{self.save_dir}meta_data.json", "r") as f:
            self.meta_data = json.load(f)

        best_individual = np.load(f"{path}_best_individual.npy", allow_pickle=True)[0]

        return pop, best_individual
    def initialize_run(self, pop_size, auto_load=True):
        if os.path.exists(f"{self.save_dir}autosave.npy") and auto_load:
            self.load_pop = True

        # initialization
        if not self.load_pop:
            self.pop = self.init_population(pop_size=pop_size, _n_hidden=self.n_hidden)
            file_open_mode = "w"
            # reset plot files
            for name in zip(self.save_file_names, ["meta_data.json"]):
                if os.path.exists(f"{self.save_dir}{name}"):
                    os.remove(f"{self.save_dir}{name}")

            self.meta_data["n_gen"] = 0
        else:
            self.pop, self.best_individual = self.load_population(f"{self.save_dir}autosave")
            file_open_mode = "a"

        save_txt_handle = open(f"{self.save_dir}fitness.csv", file_open_mode)
        div_file = open(f"{self.save_dir}diversity.csv", file_open_mode)
        sigma_handle = open(f"{self.save_dir}sigma.csv", file_open_mode)

        return save_txt_handle, div_file, sigma_handle

    def update_algorithms(self):
        pass

    def next_generation(self, pop_size, incest_thresh):
        offspring = []

        # standard crossover
        selected_parents = self.selection_algorithm.select(self.pop)
        for i in range(pop_size - self.elitism):
            new_genome = self.cross_algorithm.cross(selected_parents, pop_size=pop_size, incest_thresh=incest_thresh)
            offspring.append(new_genome)
        offspring = np.array(offspring)

        # mutation
        self.mutation_algorithm.mutate(offspring)

        # elitism
        elite_parents = copy.deepcopy(sorted(self.pop, key=lambda x: x.fitness)[-self.elitism:])
        self.pop = np.append(offspring, elite_parents)

        self.current_generation += 1
        self.update_algorithms()

    def run(self, generations, pop_size, fitness_handle, div_file, sigma_handle, incest_thresh):
        for i in range(self.meta_data["n_gen"], generations):
            print("**** Starting with evaluation of generation {}. Diversity: {}".format(i, self.diversity(self.pop)))
            start_t = time.perf_counter()

            # *** evaluate
            self.pop = self.threaded_evaluation(self.pop, self.n_hidden)
            # save pop for later plotting
            old_pop = copy.deepcopy(self.pop)

            # save best fitness
            local_fitness = np.array([g.fitness for g in self.pop])
            local_max = np.max(local_fitness)
            if local_max > self.meta_data["best_fitness"]:
                self.meta_data["best_fitness"] = copy.copy(local_max)
                self.best_individual = copy.deepcopy(self.pop[np.argmax(local_fitness)])
                self.meta_data["best_gain"] = copy.copy(self.pop[np.argmax(local_fitness)].gain)
                print("new best individual with fitness: {} and gain: {}".format(self.meta_data["best_fitness"], self.meta_data["best_gain"]))
            else:
                self.incest_thresh -= 5
            print(f"prevention rate: {self.incest_thresh}")
            fitness_values = [p.fitness for p in self.pop]      # store them but save them right before the backup

            # *** update pop
            self.next_generation(pop_size, incest_thresh)
            self.meta_data["n_gen"] = i + 1

            # saving system
            self.save_vector(fitness_handle, fitness_values)
            self.save_scalar(div_file, self.diversity(self.pop))
            self.save_population(f"{self.save_dir}autosave", self.pop, self.best_individual, old_pop)
            if type(self.mutation_algorithm) == SelfAdaptiveMutation:
                self.save_vector(sigma_handle, [p.sigma for p in self.pop])

            end = time.perf_counter()
            print("execution for one generation took: {} sec".format(end-start_t))

        return self.meta_data["best_fitness"]

    def start(self, generations=30, experiment_name="test", generate_plots=True, auto_load=True, evaluate_best=False):
        # TODO set all hyper params
        # Hyper params
        self.enemy_numbers = enemy_numbers

        self.save_dir = f"generalist_solution/{experiment_name}/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        fitness_handle, div_file, sigma_handle = self.initialize_run(self.pop_size, auto_load)

        # evaluation
        if not evaluate_best:
            # the loaded generation should be processed by the EA algorithm so we start directly with evaluation
            max_fitness = self.run(generations, self.pop_size, fitness_handle, div_file, sigma_handle, self.incest_thresh)
        else:
            self.threaded_evaluation(np.array([self.best_individual]), self.n_hidden)

        # TODO return best fitness
        # TODO implement early stopping
        fitness_handle.close()
        div_file.close()
        sigma_handle.close()
        print("all done!")

        if generate_plots:
            print("visualizing and saving results...")
            visualize_fitness(f"{self.save_dir}fitness.csv", generations, save_dir=self.save_dir)
            diversity_plot(f"{self.save_dir}diversity.csv", save_dir=self.save_dir)

        return max_fitness



if __name__ == "__main__":
    experiment_name = f"save_test"
    enemy_numbers = [1, 2]
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        if len(sys.argv) > 2:
            enemy_numbers = sys.argv[2]

    print("enemy_numbers: {} ********************".format(enemy_numbers))

    ea_instance = SpecialistSolutionV2(mutation_rate=0.16, s=1.95, nr_parents=3, n_hidden=10, pop_size=5, incest_thresh=140, enemy_numbers=enemy_numbers)
    best_fitness = ea_instance.start(generations=10, experiment_name=experiment_name, evaluate_best=False, generate_plots=False)
    print("best_fitness: {}".format(best_fitness))
    sys.exit(0)
    