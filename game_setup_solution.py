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
        experiment_name = 'individual_demo'
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
























    # initializes population loading old solutions or generating new ones
    if not os.path.exists(experiment_name+'/evoman_solstate'):

        print( '\nNEW EVOLUTION\n')

        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        fit_pop = evaluate(pop)
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        ini_g = 0
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)

    else:

        print( '\nCONTINUING EVOLUTION\n')

        env.load_state()
        pop = env.solutions[0]
        fit_pop = env.solutions[1]

        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)

        # finds last generation number
        file_aux  = open(experiment_name+'/gen.txt','r')
        ini_g = int(file_aux.readline())
        file_aux.close()




    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()


    # evolution

    last_sol = fit_pop[best]
    notimproved = 0

    for i in range(ini_g+1, gens):

        offspring = crossover(pop)  # crossover
        fit_offspring = evaluate(offspring)   # evaluation
        pop = np.vstack((pop,offspring))
        fit_pop = np.append(fit_pop,fit_offspring)

        best = np.argmax(fit_pop) #best solution in generation
        fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
        best_sol = fit_pop[best]

        # selection
        fit_pop_cp = fit_pop
        fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fit_pop_norm)/(fit_pop_norm).sum()
        chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
        chosen = np.append(chosen[1:],best)
        pop = pop[chosen]
        fit_pop = fit_pop[chosen]


        # searching new areas

        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        if notimproved >= 15:

            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\ndoomsday')
            file_aux.close()

            pop, fit_pop = doomsday(pop,fit_pop)
            notimproved = 0

        best = np.argmax(fit_pop)
        std  =  np.std(fit_pop)
        mean = np.mean(fit_pop)


        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt',pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()




    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()


    env.state_to_log() # checks environment state