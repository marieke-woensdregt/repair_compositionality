import pickle
from evolution_compositionality_under_noise import plot_timecourse, plot_barplot, classify_all_languages, create_all_possible_languages, results_to_dataframe, convert_float_value_to_string, convert_array_to_string
import numpy as np



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
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
            # a pair of agents.
runs = 50  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 100  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

noise = True  # parameter that determines whether environmental noise is on or off
noise_prob = 0.9  # Setting the 'noise_prob' parameter based on the command-line input #NOTE: first argument in sys.argv list is always the name of the script  # the probability of environmental noise masking part of an utterance
# proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.
production = 'my_code'  # can be set to 'simlang' or 'my_code'
mutual_understanding = False  # Setting the 'mutual_understanding' parameter based on the command-line input #NOTE: first argument in sys.argv list is always the name of the script
if mutual_understanding:
    gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
else:
    gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
minimal_effort = True  # Setting the 'minimal_effort' parameter based on the command-line input #NOTE: first argument in sys.argv list is always the name of the script
cost_vector = np.array([0.0, 0.15, 0.45])  # costs of no repair, restricted request, and open request, respectively
compressibility_bias = False  # determines whether agents have a prior that favours compressibility, or a flat prior
observed_meaning = 'intended'  # determines which meaning the learner observes when receiving a meaning-form pair; can
# be set to either 'intended', where the learner has direct access to the speaker's intended meaning, or 'inferred',
# where the learner has access to the hearer's interpretation.
interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

gen_start = 70  # the burn-in period that is excluded when calculating the mean distribution over languages after convergence

n_lang_classes = 5  # the number of language classes that are distinguished (int). This should be 4 if the old code was
# used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
# languages, and 5 if the new code was used which does make this distinction.


batches = 2

pickle_file_path = "pickles/"

fig_file_path = "plots/"




all_results = []
for i in range(batches):
    pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+ "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+convert_array_to_string(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_noise_prob_" + convert_float_value_to_string(noise_prob)+"_"+production+"_observed_m_"+observed_meaning+"_n_lang_classes_"+str(n_lang_classes)+"_"+str(i)

    results = pickle.load(open(pickle_file_path+pickle_file_name+".p", "rb"))

    for j in range(len(results)):
        all_results.append(results[j])


print('')
print("len(all_results) are:")
print(len(all_results))
print("len(all_results[0]) are:")
print(len(all_results[0]))
print("len(all_results[0][0]) are:")
print(len(all_results[0][0]))


lang_class_prop_over_gen_df = results_to_dataframe(all_results, runs*batches, generations, n_lang_classes)
print('')
print('')
print("lang_class_prop_over_gen_df is:")
print(lang_class_prop_over_gen_df)



fig_file_name = "r_" + str(runs*batches) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+  "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+convert_array_to_string(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_" + str(noise) + "_noise_prob_" + convert_float_value_to_string(noise_prob)+"_"+production+"_observed_m_"+observed_meaning+"_n_lang_classes_"+str(n_lang_classes)


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
        plot_title = "Mutual Understanding & Minimal Effort"

plot_timecourse(lang_class_prop_over_gen_df, plot_title, fig_file_path, fig_file_name, n_lang_classes)


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

plot_barplot(lang_class_prop_over_gen_df, plot_title, fig_file_path, fig_file_name, runs*batches, generations, gen_start, n_lang_classes, baseline_proportions)

