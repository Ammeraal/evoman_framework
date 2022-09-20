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

def mutate(pop):
    # define mutation rate
    mut_rate = 0.2
    mean = 0
    sigma = 0.25
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

def evaluate_fitness_factory(game):
    def evaluate_fitness(pop):
        # TODO trigger game with all net configs
        # TODO parallelize
        for g in pop:
            g.fitness = 0.0
            g.fitness, p, e, t = game.play(pcont=g.value)

    return evaluate_fitness

def selection(pop,s):
    p = []
    mating_pool=[]
    fitness=[]
    
    # assess the probability of offspring for each individual (5.2.2 Ranking selection)
    z=round(len(pop)/4) # number of parents
    for g in pop:
        fitness.append(g.fitness)
    order=np.argsort(fitness)
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
        
    return np.array(mating_pool)

def crossover(parents_list, pop_size):
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

if __name__=="__main__":
    pop_size = 20
    generations = 100
    n_hidden = 0
    s = 2               # used in formula to allocate selection probabilities 

    game = GameManager(controller=player_controller(n_hidden))
    evaluate_fitness = evaluate_fitness_factory(game)

    pop = init_population(pop_size=pop_size, _n_hidden=n_hidden)
    for p in pop:
        print(p.value)
    # TODO evaluation
    for i in range(generations):
        print("Starting with evaluation of generation {} ...".format(i))
        evaluate_fitness(pop)

        selected_parents = selection(pop,s)
        offspring = crossover(selected_parents, pop_size=pop_size)
        pop = mutate(offspring)

    # TODO print best fitness
    # TODO implement early stopping
    # TODO somehow save the weights
    print("all done!")


