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


def plot_line(mean, std, linestyle, label):
    plt.plot(mean, linestyle=linestyle, label=label)  # todo add label
    plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=.3)


def final_plot_fitness(prefix, experiments, enemies):
    for en in enemies:
        for ex in experiments:
            name = ex
            if ex == "incest":
                convert = False
            else:
                name = "NSGA-II"
                convert = True

            # get data of all 10 samples
            f = []
            for i in range(10):
                data, (m_f, m_p, m_e) = load_data(f"{prefix}final_{en[0]}_{en[1]}v2/{i}_{ex}/", convert_to_max=convert)
                m_f = m_f[:80,:]
                # make multi_fitness
                f.append(np.mean(m_f, axis=2) - np.std(m_f, axis=2))

            f = np.array(f)
            # get the mean fitness of the populations on each timestep for each training sample
            f_max = np.max(f, axis=2)
            f = np.mean(f, axis=2)
            # plot mean and std over the training samples
            f_mean = np.mean(f, axis=0)
            f_std = np.std(f, axis=0)
            plot_line(f_mean, f_std, linestyle="solid", label="{} mean".format(name))

            # plot max
            f_mean_max = np.mean(f_max, axis=0)
            f_std_max = np.std(f_max, axis=0)
            plot_line(f_mean_max, f_std_max, linestyle="dashed", label="{} max".format(name))

        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.title("enemy set [{},{}]".format(en[0],en[1]))
        plt.legend()
        plt.savefig("generalist_fig/fitness_{}_{}.svg".format(en[0], en[1]))
        plt.show()
        pass

            # todo average multi_fitness


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

def boxplot(prefix, enemies, experiements):
    fig, ax1 = plt.subplots(figsize=(6, 6))

    space = 0.15
    box_param = dict(patch_artist=True)
    colors = ["g", "b"]
    boxes = []
    for ex, c in zip(experiements, colors):
        gains = []
        if ex == "incest":
            suffix = "v2"
        else:
            suffix = ""
        for en in enemies:
            gains.append(np.loadtxt(f"{prefix}final_{en[0]}_{en[1]}{suffix}/{ex}_gains.txt"))

        boxes.append( ax1.boxplot(np.array(gains).T, positions=np.arange(2)-space,
                           boxprops=dict(facecolor=c),
                           **box_param) )
        space *= -1

    #plt.xticks([1, 3], ["test1", "test3"])
    ax1.legend([boxes[0]["boxes"][0], boxes[1]["boxes"][0]], ["NSGA-II", "incest"])
    ax1.set_xticks(np.arange(2))
    ax1.set_xticklabels(["[{},{}]".format(enemies[0][0], enemies[0][1]),"[{},{}]".format(enemies[1][0], enemies[1][1])])
    ax1.set_ylabel("Gain")
    ax1.set_xlabel("Enemy Group")

    plt.title("Gain over 8 Enemies")
    plt.savefig("generalist_fig/boxplot.svg")
    plt.show()

def plot_final():
    prefix = "final_generalist/"
    experiments = ["pymoo", "incest"]
    enemies = [(3, 4), (6, 8)]

    final_plot_fitness(prefix, experiments, enemies)
    boxplot(prefix, enemies, experiments)

if __name__ == "__main__":
    plot_final()
    exit()

    experiment_name = "final_6_8/0_incest"
    path = f"generalist_solution/{experiment_name}/"
    #path = ~"final_generalist/{experiment_name}/"
    # always work with a fitness maximisation function. If the fitness is minimized in the algorithm use convert_to_max=True
    data, (fitness, p, e) = load_data(path, convert_to_max=False)
    plot_fitness(fitness, path, data["n_enemies"], name=experiment_name)
    plot_pareto_front(fitness, name=experiment_name)
    avg_gain, max_gain = calc_gain_t(p, e)
    plot_gain_t(avg_gain, max_gain, name=experiment_name)
