import sys
import os
from numpy.random import default_rng
import time
import concurrent.futures
import copy
import numpy as np
from Genome import Genome
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from game_setup_solution import GameManager
from visualize import visualize_fitness, diversity_plot


class EvoProblem(Problem):
    def threaded_evaluation(self, population, n_hidden):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # insert list of all genomes
            results = executor.map(self.threaded_evaluation_fittness, population,
                                   [n_hidden for _ in range(len(population))])
            return list(results)

    def threaded_evaluation_fittness(self, value, n_hidden=0):
        enemy_numbers = [1, 8]

        game = GameManager(controller=player_controller(n_hidden), enemy_numbers=enemy_numbers)
        fitness, p, e, t = game.play(pcont=value)
        for f in fitness:
            f *= -1

        return fitness

    def _evaluate(self, designs, out, *args, **kwargs):
        # evaluate designs
        out['F'] = self.threaded_evaluation(designs, 10)


class SpecialistSolutionV2:
    def __init__(self, nr_parents=3, mutation_rate=0.2, s=2.0, n_hidden=0, elitism=4, pop_size=40):
        self.current_generation = 0

        self.save_interval = 10
        self.n_hidden = n_hidden
        self.pop_size = pop_size

        self.best_fitness = -100
        self.best_individual = None
        self.best_gain = None

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

        #if type(self.mutation_algorithm) == SelfAdaptiveMutation:
        #    SelfAdaptiveMutation.init_sigma(pop, mean=0.25)

        return np.array(pop)

    def save_vector(self, file_handle, fitness_values):
        print("saving vector to {}".format(file_handle.name))

        np.savetxt(file_handle, np.array(fitness_values), newline=" ")
        file_handle.write("\n")

    def save_scalar(self, file_handle, div):
        print("saving scalar to {}".format(file_handle.name))

        file_handle.write(str(div) + "\n")

    def save_population(self, path, pop, generation, best_fitness, best_individual, best_gain):
        print("saving population at {}".format(path))
        np.save(f"{path}.npy", pop)
        np.save(f"{path}_best_individual.npy", np.array([best_individual]))

        # container for some meta variables
        np.savetxt(f"{path}_generation", np.array([generation, best_fitness, best_gain]))

    def load_population(self, path):
        print("loading initial population for {}".format(path))
        pop = np.load(f"{path}.npy", allow_pickle=True)
        additional_data = np.loadtxt(f"{path}_generation")
        best_individual = np.load(f"{path}_best_individual.npy", allow_pickle=True)[0]
        last_gen = int(additional_data[0])
        best_fitness = float(additional_data[1])
        best_gain = float(additional_data[2])

        return pop, last_gen, best_fitness, best_individual, best_gain

    def initialize_run(self, pop_size, auto_load=True):
        # TODO load last executed state
        load_pop = False
        # todo load is disabled
        #if os.path.exists(f"{self.save_dir}autosave.npy") and auto_load:
        #    load_pop = True

        # initialization
        if not load_pop:
            self.pop = self.init_population(pop_size=pop_size, _n_hidden=self.n_hidden)
            file_open_mode = "w"
            self.load_generation = -1
        else:
            # pop = self.load_population(f"{self.save_dir}pop_{self.load_generation}.npy")
            self.pop, self.load_generation, self.best_fitness, self.best_individual, self.best_gain = self.load_population(
                f"{self.save_dir}autosave")
            file_open_mode = "a"

        save_txt_handle = open(f"{self.save_dir}fitness.csv", file_open_mode)
        div_file = open(f"{self.save_dir}diversity.csv", file_open_mode)
        sigma_handle = open(f"{self.save_dir}sigma.csv", file_open_mode)

        return save_txt_handle, div_file, sigma_handle


    def run(self, generations, pop_size, fitness_handle, div_file, sigma_handle):
        values = [p.value for p in self.pop]
        n_var = len(values[0])

        algorithm = NSGA2(pop_size=20,
                          sampling=FloatRandomSampling())
        problem = EvoProblem(n_var=n_var, n_obj=2, xl=[-1 for _ in range(n_var)], xu=[1 for _ in range(n_var)])

        # todo randomize seed
        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', generations),
                       seed=1,
                       verbose=True)

        print(res.F)
        return






        for i in range(self.load_generation + 1, generations):
            print("**** Starting with evaluation of generation {}. Diversity: {}".format(i, self.diversity(self.pop)))
            start_t = time.perf_counter()

            # *** evaluate
            self.pop = self.threaded_evaluation(self.pop, self.n_hidden)

            # save best fitness
            local_fitness = np.array([g.fitness for g in self.pop])
            local_max = np.max(local_fitness)
            if local_max > self.best_fitness:
                self.best_fitness = copy.copy(local_max)
                self.best_individual = copy.deepcopy(self.pop[np.argmax(local_fitness)])
                self.best_gain = copy.copy(self.pop[np.argmax(local_fitness)].gain)
                print("new best individual with fitness: {} and gain: {}".format(self.best_fitness, self.best_gain))

            fitness_values = [p.fitness for p in self.pop]  # store them but save them right before the backup

            # *** update pop
            self.next_generation(pop_size)

            # saving system
            self.save_vector(fitness_handle, fitness_values)
            self.save_scalar(div_file, self.diversity(self.pop))
            self.save_population(f"{self.save_dir}autosave", self.pop, i, self.best_fitness, self.best_individual,
                                 self.best_gain)
            if type(self.mutation_algorithm) == SelfAdaptiveMutation:
                self.save_vector(sigma_handle, [p.sigma for p in self.pop])

            end = time.perf_counter()
            print("execution for one generation took: {} sec".format(end - start_t))

        return self.best_fitness

    def start(self, generations=30, experiment_name="test", generate_plots=True, auto_load=True, evaluate_best=False,
              enemy_numbers=[4]):
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
            max_fitness = self.run(generations, self.pop_size, fitness_handle, div_file, sigma_handle)
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
    experiment_name = f"pymoo"
    enemy_numbers = [1, 8]
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        if len(sys.argv) > 2:
            enemy_numbers = sys.argv[2]

    print("enemy_numbers: {} ********************".format(enemy_numbers))

    ea_instance = SpecialistSolutionV2(mutation_rate=0.16, s=1.95, nr_parents=3, n_hidden=10, pop_size=40)
    best_fitness = ea_instance.start(generations=10, experiment_name=experiment_name, evaluate_best=False,
                                     enemy_numbers=enemy_numbers, generate_plots=True)
    print("best_fitness: {}".format(best_fitness))

    sys.exit(0)

