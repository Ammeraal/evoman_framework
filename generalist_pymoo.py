import sys
import os
from numpy.random import default_rng
import time
import concurrent.futures
import copy
import numpy as np
from Genome import Genome
import dill
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.callback import Callback
import json

sys.path.insert(0, 'evoman')
from demo_controller import player_controller
from game_setup_solution import GameManager
from visualize import visualize_fitness, diversity_plot

class SaveCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

    def notify(self, algorithm):
        # after loading the evaluation jumps immediately in here, so we skip the first
        if algorithm.data["just_loaded"]:
            algorithm.data["just_loaded"] = False
            return

        algorithm.data["n_gen"] += 1
        print("Saving loop num {} to {} *******************************************".
              format(algorithm.data["n_gen"], self.save_dir))

        with open(f"{self.save_dir}checkpoint", "wb") as f:
            dill.dump(algorithm.pop, f)

        # save fitness
        fitness = []
        for p in algorithm.pop:
            fitness.append(p.F)
        with open(f"{self.save_dir}fitness.txt", "a") as f:
            np.savetxt(f, np.array(fitness))
            f.write("\n")

        with open(f"{self.save_dir}data.json", "w") as f:
            metadata = algorithm.data
            json.dump(metadata, f)


class EvoProblem(Problem):
    def __init__(self, n_hidden, enemy_numbers, **kwargs):
        super().__init__(**kwargs)

        self.n_hidden = n_hidden
        self.enemy_numbers = enemy_numbers

    def threaded_evaluation(self, population, n_hidden):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # insert list of all genomes
            results = executor.map(self.threaded_evaluation_fittness, population,
                                   [n_hidden for _ in range(len(population))])
            return list(results)

    def threaded_evaluation_fittness(self, value, n_hidden=0):
        game = GameManager(controller=player_controller(n_hidden), enemy_numbers=self.enemy_numbers, multi_fitness=True)
        fitness, p, e, t = game.play(pcont=value)
        for i in range(len(fitness)):
            fitness[i] *= -1

        return fitness

    def _evaluate(self, designs, out, *args, **kwargs):
        # evaluate designs
        out['F'] = self.threaded_evaluation(designs, self.n_hidden)


class SpecialistSolutionV2:
    def __init__(self, n_hidden=0, elitism=4, pop_size=40,
                 experiment_name="test", generations=20, enemy_numbers=[1, 8]):
        self.n_hidden = n_hidden
        self.pop_size = pop_size
        self.generations = generations
        self.enemy_numbers = enemy_numbers

        self.save_dir = f"generalist_solution/{experiment_name}/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        num_inputs = 20
        num_output = 5
        # num neurons
        self.n_var = 0
        if self.n_hidden > 0:
            #           weights layer1      bias1           w2                      b2
            self.n_var = num_inputs * self.n_hidden + self.n_hidden + self.n_hidden * num_output + num_output
        else:
            # no hidden layer
            self.n_var = num_inputs * num_output + num_output

    def init_algorithm(self, pop_size, _n_hidden):
        if os.path.exists(f"{self.save_dir}fitness.txt"):
            os.remove(f"{self.save_dir}fitness.txt")
        if os.path.exists(f"{self.save_dir}data.json"):
            os.remove(f"{self.save_dir}data.json")


        # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
        init_bias = -1.0
        pop = []
        for i in range(pop_size):
            g = default_rng().random(self.n_var) * 2 + init_bias
            pop.append(list(g))

        algorithm = NSGA2(pop_size=pop_size,
                          sampling=np.array(pop),
        )
        # init custom metadata dict
        algorithm.data = {"n_gen": 0,
                          "pop_size": self.pop_size,
                          "n_enemies": len(self.enemy_numbers),
                          "just_loaded": False}

        return algorithm

    def load_algorithm(self, path):
        print("loading initial population for {}".format(path))

        with open(f"{self.save_dir}data.json", "r") as f:
            data = json.load(f)

        with open(f"{self.save_dir}checkpoint", "rb") as f:
            pop = dill.load(f)

        algorithm = NSGA2(pop_size=self.pop_size,
                          sampling=pop,
                          )
        # load custom metadata dict
        data["just_loaded"] = True
        algorithm.data = data

        return algorithm

    def initialize_run(self, pop_size, auto_load=True):
        # load last executed state
        if os.path.exists(f"{self.save_dir}checkpoint") and auto_load:
            algorithm = self.load_algorithm(self.save_dir)
            # todo delete fitness
            # todo save meta data
        else:
            algorithm = self.init_algorithm(pop_size=pop_size, _n_hidden=self.n_hidden)

        return algorithm

    def run(self, algorithm):
        problem = EvoProblem(n_hidden=10,
                             enemy_numbers=self.enemy_numbers,
                             n_var=self.n_var,
                             n_obj=len(enemy_numbers),
                             xl=[-1 for _ in range(self.n_var)],
                             xu=[1 for _ in range(self.n_var)]
                             )

        # todo randomize seed
        gen_offset = 1 if algorithm.data["just_loaded"] else 0
        res = minimize(problem,
                       algorithm,
                       callback=SaveCallback(save_dir=self.save_dir),
                       termination=('n_gen', self.generations - algorithm.data["n_gen"] + gen_offset),
                       seed=1,
                       copy_algorithm=False,
                       verbose=True
                       )

        print(res.F)
        return

    def start(self, generate_plots=True, auto_load=True, evaluate_best=False):
        algorithm = self.initialize_run(self.pop_size, auto_load)

        # evaluation
        if not evaluate_best:
            # the loaded generation should be processed by the EA algorithm so we start directly with evaluation
            self.run(algorithm)
        else:
            self.threaded_evaluation(np.array([self.best_individual]), self.n_hidden)

        # TODO return best fitness
        # TODO implement early stopping
        print("all done!")

        #if generate_plots:
        #    print("visualizing and saving results...")
        #    visualize_fitness(f"{self.save_dir}fitness.csv", generations, save_dir=self.save_dir)
        #    diversity_plot(f"{self.save_dir}diversity.csv", save_dir=self.save_dir)



if __name__ == "__main__":
    experiment_name = f"pymoo"
    enemy_numbers = [1, 8]
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        if len(sys.argv) > 2:
            enemy_numbers = sys.argv[2]

    print("enemy_numbers: {} ********************".format(enemy_numbers))

    ea_instance = SpecialistSolutionV2(n_hidden=10, pop_size=40, generations=50,
                                       experiment_name=experiment_name, enemy_numbers=enemy_numbers)
    ea_instance.start(auto_load=True, evaluate_best=False, generate_plots=True)

    sys.exit(0)

