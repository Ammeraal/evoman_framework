# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:24:32 2022

@author: Agustin Guevara C
"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

class GameManager():
    def __init__(self,controller = None,config = None):
        if config is not None:
            raise NotImplementedError("Custom config not implemented yet")
        if controller is None:
            raise NotImplementedError("Empty controller not implemented yet")
        # choose this for not using visuals and thus making experiments faster
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        experiment_name = 'individual_specialist_solution'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                          enemies=[2],
                          playermode="ai",
                          player_controller=controller,
                          enemymode="static",
                          level=2,
                          speed="fastest")
        # default environment fitness is assumed for experiment
        env.state_to_log() # checks environment state
        ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
        ini = time.time()  # sets time marker
        self.environment = env
        self.controller = controller
    #def __repr__(self):
    #def export_config(self):
    def play(self,*args,**kwargs):
        return self.environment.play(*args,**kwargs)