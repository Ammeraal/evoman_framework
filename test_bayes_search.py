import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

import numpy as np
import random
from numpy.random import default_rng
from Genome import Genome
from game_setup_solution import GameManager
from demo_controller import player_controller
import os
from specialist_solution_v2 import SpecialistSolutionV2

import time
import concurrent.futures
import multiprocessing
import concurrent.futures
from tkinter import E
import pandas as pd


class EvoEAEstimator(BaseEstimator):
    def __init__(self, n_hidden, s, mut_rate, elitism):
        self.n_hidden = n_hidden
        self.s = s
        self.mut_rate = mut_rate
        self.elitism = elitism

    def fit(self, X, y):
        # constants
        generations = 10
        pop_size = 20
        sigma = 0.25      # this is the std deviations used for sampling mutation steps
        experiment_name = "bayes_search"

        ea_instance = SpecialistSolutionV2()
        self.best_fitness = ea_instance.start(mut_rate=self.mut_rate, s=self.s, n_hidden=self.n_hidden,
                                              generations=generations, pop_size=pop_size, sigma=sigma,
                                              experiment_name=experiment_name, generate_plots=False)

        return self

    def score(self, X, y):
        return self.best_fitness

    def get_params(self, deep=True):
        # this returns a dict of all input constants of this estimator
        return {"n_hidden": self.n_hidden, "s": self.s, "mut_rate": self.mut_rate, "elitism": self.elitism}

    def set_params(self, **params):
        # TODO implement (this sets the given params as estimator constants)
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

if __name__ == "__main__":
    # check_estimator(EvoEAEstimator(10, 2.0, 0.2))

    opt = BayesSearchCV(EvoEAEstimator(5, 2.0, 0.2, 2),
                        {"n_hidden": (0, 5, 10, 15),
                        "s": (1.4, 1.6, 1.8, 2.0),
                        "mut_rate": (0.01, 0.025, 0.05, 0.075),
                        "elitism": (1,2,3,4)
                        },
                        n_iter=2,
                        verbose=3,
                        #cv=3,
                        n_jobs=1
                        )
    opt.fit(np.zeros((100, 100)), np.zeros(100))


    print("best score: %s" % opt.best_score_)
    print("best params: {}".format(opt.best_params_))
