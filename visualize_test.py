import pandas as pd
import matplotlib.pyplot as plt

def visualize(file, generations):
    # make plot of mean fitness over generations, with standard deviation
    df_geno = pd.DataFrame()
    df_pheno = pd.DataFrame()
    df_geno_max = pd.DataFrame()
    df_pheno_max = pd.DataFrame()
    for i in range(10):
        df = pd.read_csv(file + str(i) + "_genotype/fitness.csv", header=None, sep=" ").iloc[:, :-1]
        df_avg = df.mean(axis=1)
        df_max = df.max(axis=1)
        df_geno[i] = df_avg
        df_geno_max[i] = df_max
    for i in range(10):
        df = pd.read_csv(file + str(i) + "_phenotype/fitness.csv", header=None, sep=" ").iloc[:, :-1]
        df_avg = df.mean(axis=1)
        df_max = df.max(axis=1)
        df_pheno[i] = df_avg
        df_pheno_max[i] = df_max

    df_geno_mean = df_geno.mean(axis=1)
    df_geno_mean_std = df_geno.std(axis=1)
    df_pheno_mean = df_pheno.mean(axis=1)
    df_pheno_mean_std = df_pheno.std(axis=1)

    df_geno_max = df_geno_max.mean(axis=1)
    df_geno_max_std = df_geno.std(axis=1)
    df_pheno_max = df_pheno_max.mean(axis=1)
    df_pheno_max_std = df_pheno.std(axis=1)

    # make plot
    plt.plot(df_geno_mean, label="Mean of genotype")
    plt.plot(df_pheno_mean, label="Mean of phenotype")
    plt.plot(df_geno_max, label="Maximum of genotype")
    plt.plot(df_pheno_max, label="Maximum of phenotype")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xlim(-1,80)
    plt.legend(loc="lower right")
    plt.fill_between(range(generations), df_geno_mean -
                     df_geno_mean_std, df_geno_mean + df_geno_mean_std, alpha=.3)
    plt.fill_between(range(generations), df_pheno_mean -
                     df_pheno_mean_std, df_pheno_mean + df_pheno_mean_std, alpha=.3)
    plt.fill_between(range(generations), df_geno_max -
                     df_geno_max_std, df_geno_max + df_geno_max_std, alpha=.3)
    plt.fill_between(range(generations), df_pheno_max -
                     df_pheno_max_std, df_pheno_max + df_pheno_max_std, alpha=.3)
    # plt.ylim(-6)
    plt.savefig(f"{file}avg_lineplot.png")
    plt.clf()

visualize("4_final/", 80)