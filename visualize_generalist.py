import numpy as np
import matplotlib.pyplot as plt
import json

def load_data(path):
    with open(f"{path}data.json", "r") as f:
        data = json.load(f)
    with open(f"{path}fitness.txt", "r") as f:
        values = np.loadtxt(f)

    values = values.reshape((data["n_gen"], data["pop_size"], data["n_enemies"]))

    return values, data

def plot_fitness(values, path):
    avg = np.average(values, axis=1)
    maxes = np.min(values, axis=1)

    plt.plot(avg)
    plt.plot(maxes)

    plt.savefig(f"{path}/fitness.svg")
    plt.show()

def plot_pareto_front(values):
    last_gen = values[-1,:,:]
    plt.scatter(x=last_gen[:,0], y=last_gen[:,1])
    plt.show()
    pass

if __name__=="__main__":
    path = "generalist_solution/pymoo_first_long_run/"
    values, data = load_data(path)
    plot_fitness(values, path)
    plot_pareto_front(values)


