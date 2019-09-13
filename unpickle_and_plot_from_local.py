import pandas as pd
import numpy as np
from evolution_compositionality_under_noise import plot_timecourse, plot_barplot, classify_all_languages, create_all_possible_languages, dataframe_to_results, results_to_dataframe




# PARAMETER SETTINGS:

meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = ['aa', 'ab', 'ba', 'bb']  # all possible forms, excluding their possible 'noisy variants'
noisy_forms = ['a_', 'b_', '_a', '_b']  # all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise+noisy_forms  # all possible forms, including both complete
# forms and noisy variants
error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

turnover = True  # determines whether new individuals enter the population or not
b = 20  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training (Kirby et al.
        # used a bottleneck of 20 in the body of the paper.
rounds = 2*b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
popsize = 10  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
            # a pair of agents.
runs = 20  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 100  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

noise = True  # parameter that determines whether environmental noise is on or off
noise_prob = 0.9  # the probability of environmental noise masking part of an utterance
# proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.
production = 'my_code'  # can be set to 'simlang' or 'my_code'
mutual_understanding = True
if mutual_understanding:
    gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
else:
    gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
minimal_effort = True
cost_vector = [0.0, 0.15, 0.45]  # costs of no repair, restricted request, and open request, respectively
compressibility_bias = False  # determines whether agents have a prior that favours compressibility, or a flat prior
observed_meaning = 'intended'  # determines which meaning the learner observes when receiving a meaning-form pair; can
# be set to either 'intended', where the learner has direct access to the speaker's intended meaning, or 'inferred',
# where the learner has access to the hearer's interpretation.

gen_start = int(generations/2)

batches = 3


if batches > 1:
    all_results = []
    for i in range(batches):
        pickle_file_title = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+ "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+str(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_noise_prob_" + str(noise_prob)+"_"+production+"_observed_m_"+observed_meaning+"_"+str(i)

        lang_class_prop_over_gen_df = pd.read_pickle(pickle_file_title+".pkl")

        results = dataframe_to_results(lang_class_prop_over_gen_df, runs, generations)

        for j in range(len(results)):
            all_results.append(results[j])

    print('')
    print("len(all_results) are:")
    print(len(all_results))
    print("len(all_results[0]) are:")
    print(len(all_results[0]))
    print("len(all_results[0][0]) are:")
    print(len(all_results[0][0]))

    lang_class_prop_over_gen_df = results_to_dataframe(all_results, runs * batches, generations)
    print('')
    print('')
    print("lang_class_prop_over_gen_df is:")
    print(lang_class_prop_over_gen_df)

    fig_file_title = "r_" + str(runs*batches) + "_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(
        rounds) + "_pop_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(
        gamma) + "_minimal_e_" + str(minimal_effort) + "_c_" + str(cost_vector) + "_turnover_" + str(
        turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(
        noise) + "_noise_prob_" + str(noise_prob) + "_" + production + "_observed_m_" + observed_meaning



else:
    pickle_file_title = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+ "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+str(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_noise_prob_" + str(noise_prob)+"_"+production+"_observed_m_"+observed_meaning

    lang_class_prop_over_gen_df = pd.read_pickle(pickle_file_title+".pkl")

    fig_file_title = "r_" + str(runs) + "_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(
        rounds) + "_pop_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(
        gamma) + "_minimal_e_" + str(minimal_effort) + "_c_" + str(cost_vector) + "_turnover_" + str(
        turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(
        noise) + "_noise_prob_" + str(noise_prob) + "_" + production + "_observed_m_" + observed_meaning


if mutual_understanding == False and minimal_effort == False:
    if gamma == 0 and turnover == True:
        plot_title = "Learnability only"
    elif gamma > 0 and turnover == False:
        plot_title = "Expressivity only"
    elif gamma > 0 and turnover == True:
        plot_title = "Learnability and expressivity"
    if noise:
        plot_title = plot_title + " Plus Noise"
else:
    if mutual_understanding == True and minimal_effort == False:
        plot_title = "Mutual Understanding Only"
    elif mutual_understanding == False and minimal_effort == True:
        plot_title = "Minimal Effort Only"
    elif mutual_understanding == True and minimal_effort == True:
        plot_title = "Mutual Understanding and Minimal Effort"

plot_timecourse(lang_class_prop_over_gen_df, plot_title, fig_file_title)


all_possible_languages = create_all_possible_languages(meanings, forms_without_noise)
print("number of possible languages is:")
print(len(all_possible_languages))

class_per_lang = classify_all_languages(all_possible_languages)
print('')
print('')
no_of_each_class = np.bincount(class_per_lang.astype(int))
print('')
print("no_of_each_class is:")
print(no_of_each_class)

baseline_proportions = np.divide(no_of_each_class, len(all_possible_languages))
print('')
print('')
print("baseline_proportions are:")
print(baseline_proportions)

plot_barplot(lang_class_prop_over_gen_df, plot_title, fig_file_title, runs*batches, generations, gen_start,
             baseline_proportions)

