import numpy as np
from game_setup_solution import GameManager
from demo_controller import player_controller

def load_solution(path):
    return np.loadtxt(path)

def run(solution):
    global_gain = 0
    n_def_enemies = 0
    for i in range(1, 9):
        game = GameManager(controller=player_controller(10), enemy_numbers=[i])
        if i == 5:
            attempts = 5
        else:
            attempts = 1

        avg_gain = 0
        for k in range(attempts):
            f, p, e, t = game.play(pcont=solution)
            avg_gain += p - e

        avg_gain /= attempts
        if p > 0 and e == 0:
            n_def_enemies += 1

        global_gain += avg_gain

    return global_gain, n_def_enemies

if __name__=="__main__":
    prefix = "final_generalist/"
    paths = [f"{prefix}final_3_4v2/", f"{prefix}final_6_8v2/"]
    #strategies = ["pymoo", "incest"]
    strategies = ["incest"]

    for p in paths:
        for s in strategies:
            results = []
            for i in range(10):
                gain, n_def_enemies = run(load_solution(f"{p}{i}_{s}/solution.txt"))
                print(f"{p}{i}_{s}/solution.txt gain: {gain}, enemies: {n_def_enemies}")
                results.append(gain)
            # todo save results
            np.savetxt(f"{p}{s}_gains.txt", np.array(results))
