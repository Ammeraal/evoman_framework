import sys
import os
from numpy.random import default_rng
import time
import concurrent.futures
import copy

sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from game_setup_solution import GameManager
from ea_lib import *
from visualize import visualize_fitness, diversity_plot

class SpecialistSolutionV2:
    def __init__(self, nr_parents=3, mutation_rate=0.2, s=2.0, n_hidden=0, elitism=4):
        self.current_generation = 0

        self.cross_algorithm = MultiParentCrossover(nr_parents=nr_parents)

        self.mutation_algorithm = GaussianMutation(mutation_rate=mutation_rate)# UniformMutation(mutation_rate=0.03) #
        self.selection_algorithm = RankingSelection(s=s)
        self.save_interval = 10
        self.load_pop = False
        self.load_generation = 20
        self.n_hidden = n_hidden
        self.elitism = elitism

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

        return np.array(pop)

    def threaded_evaluation(self, population, n_hidden):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # insert list of all genomes
            results = executor.map(self.threaded_evaluation_fittness, population,
                                   [n_hidden for _ in range(len(population))])
            return list(results)

    def threaded_evaluation_fittness(self, g, n_hidden=0):
        game = GameManager(controller=player_controller(n_hidden), enemy_numbers=self.enemy_numbers)
        g.fitness = 0.0
        g.fitness, p, e, t = game.play(pcont=g.value)
        g.gain = p - e
        return g

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

    def save_population(self, path, pop, generation, best_fitness, best_individual, best_gain):
        print("saving population at {}".format(path))
        np.save(f"{path}.npy", pop)
        np.save(f"{path}_best_individual.npy", np.array([best_individual]))
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
        if os.path.exists(f"{self.save_dir}autosave.npy") and auto_load:
            self.load_pop = True

        # initialization
        if not self.load_pop:
            pop = self.init_population(pop_size=pop_size, _n_hidden=self.n_hidden)
            file_open_mode = "w"
            self.load_generation = -1
        else:
            #pop = self.load_population(f"{self.save_dir}pop_{self.load_generation}.npy")
            pop, self.load_generation, self.best_fitness, self.best_individual, self.best_gain = self.load_population(f"{self.save_dir}autosave")
            file_open_mode = "a"

        save_txt_handle = open(f"{self.save_dir}fitness.csv", file_open_mode)
        div_file = open(f"{self.save_dir}diversity.csv", file_open_mode)
        self.pop = pop

        return save_txt_handle, div_file

    def update_algorithms(self):
        pass

    def next_generation(self, pop_size):
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
        elite_parents = copy.deepcopy(sorted(self.pop, key=lambda x: x.fitness)[-self.elitism:])
        self.pop = np.append(offspring, elite_parents)

        self.current_generation += 1
        self.update_algorithms()

    def run(self, generations, pop_size, save_txt_handle, div_file):
        for i in range(self.load_generation + 1, generations):
            print("**** Starting with evaluation of generation {}. Diversity: {}".format(i, self.diversity(self.pop)))
            start_t = time.perf_counter()

            self.pop = self.threaded_evaluation(self.pop, self.n_hidden)

            # save best fitness
            local_fitness = np.array([g.fitness for g in self.pop])
            local_max = np.max(local_fitness)
            if local_max > self.best_fitness:
                self.best_fitness = copy.copy(local_max)
                self.best_individual = copy.deepcopy(self.pop[np.argmax(local_fitness)])
                self.best_gain = copy.copy(self.pop[np.argmax(local_fitness)].gain)
                print("new best individual with fitness: {} and gain: {}".format(self.best_fitness, self.best_gain))

            fitness_values = [p.fitness for p in self.pop]      # store them but save them right before the backup
            self.next_generation(pop_size)

            # saving system
            #if i % self.save_interval == 0:
            self.save_fitness(save_txt_handle, fitness_values)
            self.save_diversity(div_file, self.diversity(self.pop))
            self.save_population(f"{self.save_dir}autosave", self.pop, i, self.best_fitness, self.best_individual, self.best_gain)

            end = time.perf_counter()
            print("execution for one generation took: {} sec".format(end-start_t))
        return self.best_fitness

    def start(self, generations=30, pop_size=20,
              experiment_name="test", generate_plots=True, auto_load=True, evaluate_best=False, enemy_numbers=[4]):
        # TODO set all hyper params
        # Hyper params
        self.enemy_numbers = enemy_numbers

        self.save_dir = f"generalist_solution/{experiment_name}/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        save_txt_handle, div_file = self.initialize_run(pop_size, auto_load)

        # evaluation
        if not evaluate_best:
            # the loaded generation should be processed by the EA algorithm so we start directly with evaluation
            max_fitness = self.run(generations, pop_size, save_txt_handle, div_file)
        else:
            self.threaded_evaluation(np.array([self.best_individual]), self.n_hidden)

        # TODO return best fitness
        # TODO implement early stopping
        save_txt_handle.close()
        div_file.close()
        print("all done!")

        if generate_plots:
            print("visualizing and saving results...")
            visualize_fitness(f"{self.save_dir}fitness.csv", generations, save_dir=self.save_dir)
            diversity_plot(f"{self.save_dir}diversity.csv", save_dir=self.save_dir)

        return max_fitness



if __name__ == "__main__":
    experiment_name = f"test"
    enemy_numbers = [1, 8]
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        if len(sys.argv) > 2:
            enemy_numbers = sys.argv[3]

    print("enemy_numbers: {} ********************".format(enemy_numbers))

    ea_instance = SpecialistSolutionV2(mutation_rate=0.16, s=1.95, nr_parents=3, n_hidden=10)
    best_fitness = ea_instance.start(generations=42, pop_size=40, experiment_name=experiment_name, evaluate_best=False, enemy_numbers=enemy_numbers, generate_plots=True)
    print("best_fitness: {}".format(best_fitness))

    sys.exit(0)
    