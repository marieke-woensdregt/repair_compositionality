import pandas as pd
from evolution_compositionality_under_noise import plot_graph



initial = [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')]  # this is data that would be produced by a holistic language
gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for "Learnability Only" condition, and gamma = 2 for both "Expressivity Only" and "Learnability and Expressivity" conditions
turnover = True  # determines whether new individuals enter the population or not
b = 20  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training (Kirby et al. used a bottleneck of 20 in the body of the paper.
rounds = 1*b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply a pair of agents.
runs = 10  # the number of independent simulation runs (Kirby et al., 2015 used 100)
gens = 10  # the number of generations (Kirby et al., 2015 used 100)
noise = False  # parameter that determines whether environmental noise is on or off
noise_prob = 0.1  # the probability of environmental noise masking part of an utterance
proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.



pickle_file_title = "Pickle_n_runs_"+str(runs)+"_n_gens_"+str(gens)+"_b_"+str(b)+"_rounds_"+str(rounds)+"_gamma_" + str(gamma) + "_turnover_" + str(turnover)+"_noise_"+str(noise)+"_noise_prob_"+str(noise_prob)+"_prop_measure_"+proportion_measure


lang_class_prop_over_gen_df = pd.read_pickle(pickle_file_title+".pkl")

fig_file_title = "Plot_n_runs_" + str(runs) + "_n_gens_" + str(gens) + "_b_" + str(b) + "_rounds_" + str(
    rounds) + "_gamma_" + str(gamma) + "_turnover_" + str(turnover) + "_noise_" + str(noise) + "_noise_prob_" + str(
    noise_prob) + "_prop_measure_" + proportion_measure
if gamma == 0 and turnover == True:
    plot_title = "Learnability only"
elif gamma > 0 and turnover == False:
    plot_title = "Expressivity only"
elif gamma > 0 and turnover == True:
    plot_title = "Learnability and expressivity"
if noise:
    plot_title = plot_title + " Plus Noise"
plot_graph(lang_class_prop_over_gen_df, plot_title, fig_file_title)
