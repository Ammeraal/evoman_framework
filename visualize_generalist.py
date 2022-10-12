import numpy as np
import matplotlib.pyplot as plt
import json


def load_data(path, convert_to_max=False):
    """
    convert_to_max: Converts the loaded fitness from minimization to maximisation (by changing the sign)
    """
    with open(f"{path}meta_data.json", "r") as f:
        data = json.load(f)

    data_names = ["m_fitness.txt", "m_p.txt", "m_e.txt"]
    ret = []
    for name in data_names:

        with open(f"{path}{name}", "r") as f:
            values = np.loadtxt(f)
        if convert_to_max and name == "m_fitness.txt":
            values *= -1
        ret.append(values.reshape((data["n_gen"], data["pop_size"], data["n_enemies"])))


    return data, ret


def plot_fitness(values, path, n_enemies, name=""):
    std = np.std(values, axis=1)
    avg = np.mean(values, axis=1)
    maxes = np.max(values, axis=1)

    labels = ["enemy{}".format(chr(ord("A") + i)) for i in range(np.shape(std)[-1])]
    plt.plot(avg, label=labels)
    for i in range(np.shape(avg)[-1]):
        plt.fill_between(np.arange(len(avg[:,i])), avg[:,i] - std[:,i], avg[:,i] + std[:,i], alpha=.3)

    plt.plot(maxes, linestyle="dashed", label=labels)

    plt.legend()
    plt.title("fitness on {}".format(name))
    plt.savefig(f"{path}/fitness.svg")
    plt.show()


def calc_gain_t(p, e):
    gain = np.sum(p, axis=2) - np.sum(e, axis=2)
    avg_gain = np.average(gain, axis=1)
    max_gain = np.max(gain, axis=1)
    return avg_gain, max_gain

def plot_gain_t(avg_gain, max_gain, name=""):
    plt.plot(avg_gain, label="avg")
    plt.plot(max_gain, label="max", linestyle="dashed")
    plt.legend()
    plt.title("Gain over time on {}".format(name))
    plt.xlabel("time")
    plt.ylabel("gain")
    plt.show()

def plot_pareto_front(values, name=""):
    last_gen = values[-1, :, :]
    plt.scatter(x=last_gen[:, 0], y=last_gen[:, 1])
    plt.title("Fittness of two Enemies in last gen on {}".format(name))
    plt.xlabel("enemy A")
    plt.ylabel("enemy B")
    plt.show()
    pass


if __name__ == "__main__":
    experiment_name = "pymoo_1_2_adaptive_more_mut"
    path = f"generalist_solution/{experiment_name}/"
    # always work with a fitness maximisation function. If the fitness is minimized in the algorithm use convert_to_max=True
    data, (fitness, p, e) = load_data(path, convert_to_max=True)
    plot_fitness(fitness, path, data["n_enemies"], name=experiment_name)
    plot_pareto_front(fitness, name=experiment_name)
    avg_gain, max_gain = calc_gain_t(p, e)
    plot_gain_t(avg_gain, max_gain, name=experiment_name)
