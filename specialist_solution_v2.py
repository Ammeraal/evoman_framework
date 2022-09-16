from calendar import c
import numpy as np
import random
from numpy.random import default_rng
from Genome import Genome
from game_setup_solution import GameManager
from demo_controller import player_controller


def init_population(pop_size, _n_hidden):
    # each offspring has a list of weights with size sum_i(size(l_i-1) * size(l_i))
    seed = 42
    num_inputs = 20
    num_output = 5
    init_bias = 0.0

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
        g = Genome(default_rng(seed).random(sum) + init_bias)
        pop.append(g)
        seed += 1

    return np.array(pop)

def mutate(pop):
    mut_rate = 0.2

    pop_offspring = []
    lower = np.min(pop)
    upper = np.max(pop)
    for genome in pop:
        offspring = []
        for gene in genome:
            mutate = np.random.uniform(0, 1)
            if mutate <= mut_rate:
                offspring.append(random.uniform(lower, upper))
            else:
                offspring.append(gene)
        pop_offspring.append(offspring)
    return np.array(pop_offspring)

    # draw probability from distribution, if prob <= mutation threshold, mutate gene
    # mutate gene by changing it into random value between lower and upper domain

def evaluate_fitness_factory(game):
    def evaluate_fitness(pop):
        # TODO trigger game with all net configs
        # TODO parallelize
        for g in pop:
            g.fitness = 0.0
            g.fitness, p, e, t = game.play(pcont=g.value)

    return evaluate_fitness

def selection(pop):
    p=[]
    mating_pool=[]
    fitness=[]
    
    # assess the probability of offspring for each individual (5.2.2 Ranking selection)
    z=round(len(pop)/4) # number of parents
    for g in pop:
        fitness.append(np.random.uniform(0,20))
    order=np.argsort(fitness)
    s=2
    for i in order:
        p.append((2-s)/z + (2*i*(s-1))/(z*(z-1)))
    
    # select parents according to offspring probability (5.2.3 Implementing selection probabilities)
    current_member=i=1
    r=np.random.uniform(0,1/z)
    while current_member<=z:
        while r<=p[i]:
            mating_pool.append(pop[i])
            r=r+1/z
            current_member+=1
            break
        i+=1
        
    return np.array([mating_pool])

<<<<<<< Updated upstream
def crossover(parents_list):
=======
def crossover(parents_list, pop_size):
    # TODO return list of the new offspring
>>>>>>> Stashed changes
    children = []
    for z in range(pop_size):
        while True:
            parent1 = random.choice(parents_list)
            parent2 = random.choice(parents_list)
            if parent1 != parent2:
                break

        child = []
        for i in range(len(parent1)):
            bool = random.getrandbits(1)
            if bool == 1:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        children.append(child)
    children = np.array(children)

    return children

if __name__=="__main__":
    pop_size = 20
    generations = 10
    n_hidden = 8

    game = GameManager(controller=player_controller(n_hidden))
    evaluate_fitness = evaluate_fitness_factory(game)

    pop = init_population(pop_size=pop_size, _n_hidden=n_hidden)
    # TODO evaluation
    for i in range(generations):
        evaluate_fitness(pop)
        selected_parents = selection(pop)
        offspring = crossover(selected_parents)
        mutate(offspring)


