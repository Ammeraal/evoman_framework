import os, sys
import random

import neat
import visualize

sys.path.insert(0, 'evoman')
from specialist_controller_solution import player_controller
from game_setup_solution import GameManager


def eval_genomes_factory(game):
    def eval_genomes(genomes,config):
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness,p,e,t = game.play(pcont = net)
    return eval_genomes


def run(config_file,generation_count = 100):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="./specialist_solution/neat-checkpoint-"))

    # create environment
    game = GameManager(player_controller())

    # Run for up to 300 generations.
    eval_genomes = eval_genomes_factory(game=game)
    winner = p.run(eval_genomes, generation_count)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    visualize.draw_net(config, winner, True)#, node_names=node_names)
    #visualize.draw_net(config, winner, True, prune_unused=True)#, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # TODO checkpoint stuff
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')

    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    #local_dir = os.path.dirname(__file__)
    config_path = os.path.join(f"./specialist_solution", 'neat_config')
    run(config_path, 5)
