import os, sys
import random

import neat
import visualize

sys.path.insert(0, 'evoman')
from specialist_controller_solution import player_controller
from game_setup_solution import GameManager


def eval_genomes_factory(game):
    def eval_genomes(genomes,config):
        # play the game for each individual
        for genome_id, genome in genomes:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness,p,e,t = game.play(pcont = net)
    return eval_genomes


def run(config_file,generation_count=100, output_path="./specialist_solution/"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # uncomment this to load a checkpoint
    #p = neat.Checkpointer.restore_checkpoint(f"{output_path}neat-checkpoint-199")

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=f"{output_path}neat-checkpoint-"))

    # create environment (check GameManager object for additional settings)
    game = GameManager(player_controller(), experiment_name=output_path)

    # Run for up to generation_count generations.
    eval_genomes = eval_genomes_factory(game=game)
    winner = p.run(eval_genomes, generation_count)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


    # draw visualisations
    node_names = {-1: "x", -2: "y", -3: "p_d", -4: "e_d", -5: "b0_x", -6: "b0_y", -7: "b1_x", -8: "b1_y", -9: "b2_x", -10: "b2_y", -11: "b3_x", -12: "b3_y", -13: "b4_x", -14: "b4_y", -15: "b5_x", -16: "b5_y", -17: "b6_x", -18: "b6_y", -19: "b7_x", -20: "b7_y",
        0: "left", 1: "right", 2: "jump", 3: "shoot", 4: "release"}
    # dotted means deactivated
    # red means pos. weight (but aren't they dependent on the input? So only the weights in the last frame??)
    visualize.draw_net(config, winner, True, filename=f"{output_path}net.dot", node_names=node_names)
    #visualize.draw_net(config, winner, True, prune_unused=True)#, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True, filename=f"{output_path}stats.svg")
    visualize.plot_species(stats, view=True, filename=f"{output_path}species.svg")


if __name__ == '__main__':
    config_path = os.path.join(f"./specialist_solution", 'neat_config')
    run(config_path, generation_count=20, output_path="./specialist_solution/")
