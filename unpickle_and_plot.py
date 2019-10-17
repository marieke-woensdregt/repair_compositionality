import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evolution_compositionality_under_noise import create_all_possible_languages, classify_all_languages, convert_float_value_to_string, convert_array_to_string


###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

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
runs = 100  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 150  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

production = 'my_code'  # can be set to 'simlang' or 'my_code'

cost_vector = np.array([0.0, 0.15, 0.45])  # costs of no repair, restricted request, and open request, respectively
compressibility_bias = False  # determines whether agents have a prior that favours compressibility, or a flat prior
observed_meaning = 'intended'  # determines which meaning the learner observes when receiving a meaning-form pair; can
# be set to either 'intended', where the learner has direct access to the speaker's intended meaning, or 'inferred',
# where the learner has access to the hearer's interpretation.
interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

# proportion_measure = 'posterior'  # the way in which the proportion of language classes present in the population is
# measured. Can be set to either 'posterior' (where we directly measure the total amount of posterior probability
# assigned to each language class), or 'sampled' (where at each generation we make all agents in the population pick a
# language and we count the resulting proportions.

burn_in = 100  # the burn-in period that is excluded when calculating the mean distribution over languages after convergence

n_lang_classes = 5  # the number of language classes that are distinguished (int). This should be 4 if the old code was
# used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
# languages, and 5 if the new code was used which does make this distinction.

noise = True  # parameter that determines whether environmental noise is on or off
noise_prob = 0.9  # the probability of environmental noise masking part of an utterance

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

communicative_success = False  # determines whether there is a pressure for communicative success or not
communicative_success_pressure_strength = (2./3.)  # determines how much more likely a <meaning, form> pair from a
# successful interaction is to enter the data set that is passed on to the next generation, compared to a
# <meaning, form> pair from a unsuccessful interaction.

pickle_file_path = "pickles/"

fig_file_path = "plots/"


batches = 1


###################################################################################################################
# ALL FUNCTION DEFINITIONS GO HERE:

# FIRST WE NEED A FUNCTION THAT CONVERTS A DATA LIST INTO A PANDAS DATA FRAME FOR PLOTTING:
def language_stats_to_dataframe(results, n_runs, n_gens, n_language_classes):
    """
    Takes a results list and puts it in a pandas dataframe together with other relevant variables (runs, generations,
    and language class)

    :param results: a list containing proportions for each of the 5 language classes, for each generation, for each run
    :param n_runs: the number of runs (int); corresponds to global variable 'runs'
    :param n_gens: the number of generations (int); corresponds to global variable 'generations'
    :param n_language_classes: the number of language classes that are distinguished (int); corresponds to global
    variable 'n_lang_classes'. This should be 4 if the old code was used (from before 13 September 2019, 1:30 pm), which
    did not yet distinguish between 'holistic' and 'hybrid' languages, and 5 if the new code was used which does make
    this distinction
    :return: a pandas dataframe containing four columns: 'run', 'generation', 'proportion' and 'class'
    """
    column_proportion = np.array(results)
    column_proportion = column_proportion.flatten()

    column_runs = []
    for i in range(n_runs):
        for j in range(n_gens):
            for k in range(n_language_classes):
                column_runs.append(i)
    column_runs = np.array(column_runs)

    column_generation = []
    for i in range(n_runs):
        for j in range(n_gens):
            for k in range(n_language_classes):
                column_generation.append(j)
    column_generation = np.array(column_generation)

    column_type = []
    for i in range(n_runs):
        for j in range(n_gens):
            if n_language_classes == 4:
                column_type.append('degenerate')
                column_type.append('holistic')
                column_type.append('other')
                column_type.append('compositional')
            elif n_language_classes == 5:
                column_type.append('degenerate')
                column_type.append('holistic')
                column_type.append('hybrid')
                column_type.append('compositional')
                column_type.append('other')

    data = {'run': column_runs,
            'generation': column_generation,
            'proportion': column_proportion,
            'class': column_type}

    lang_class_prop_over_gen_df = pd.DataFrame(data)

    return lang_class_prop_over_gen_df


# And this function turns a pandas dataframe back into a list of lists of lists of language stats, just in case
# that's needed:
def dataframe_to_language_stats(dataframe, n_runs, n_gens, n_language_classes):
    """
    Takes a pandas dataframe of results and turns it back into a simple results array, which only contains the
    populations' posterior probability distributions over generations.

    :param dataframe: a pandas dataframe which contains at least a column named "proportions", which contains the
    proportions of the different language classes over generations over runs.
    :param n_runs: number of runs (int)
    :param n_gens: number of generations (int)
    :param n_language_classes: the number of language classes that are distinguished (int); corresponds to global
    variable 'n_lang_classes'. This should be 4 if the old code was used (from before 13 September 2019, 1:30 pm), which
    did not yet distinguish between 'holistic' and 'hybrid' languages, and 5 if the new code was used which does make
    this distinction
    :return: a numpy array containing the proportions of the different language classes for each generation for each run
    """
    proportion_column = np.array(dataframe['proportion'])
    proportion_column_as_results = proportion_column.reshape((n_runs, n_gens, n_language_classes))
    return proportion_column_as_results


# AND NOW FOR THE PLOTTING FUNCTIONS:

def plot_timecourse(lang_class_prop_over_gen_df, title, file_path, file_name, n_language_classes):
    """
    Takes a pandas dataframe which contains the proportions of language classes over generations and plots timecourses

    :param lang_class_prop_over_gen_df: a pandas dataframe containing four columns: 'run', 'generation', 'proportion'
    and 'class'
    :param title: The title of the condition that should be on the plot (string)
    :param file_path: path to folder in which the figure file should be saved
    :param file_name: The file name that the plot should be saved under
    :param n_language_classes: the number of language classes that are distinguished (int). This should be 4 if the old code
    was used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
    languages, and 5 if the new code was used which does make this distinction.
    :return: Nothing. Just saves the plot and then shows it.
    """
    sns.set_style("whitegrid")
    sns.set_context("talk")

    fig, ax = plt.subplots()

    if n_language_classes == 4:
        palette = sns.color_palette(["black", "red", "grey", "green"])
    elif n_language_classes == 5:
        palette = sns.color_palette(["black", "red", "magenta", "green", "grey"])

    sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette)
    # sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette, ci=95, err_style="bars")

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim(-0.05, 1.05)
    plt.title(title, fontsize=22)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel('Mean proportion', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(file_path + "Timecourse_plot_" + file_name + ".pdf")
    plt.show()


def plot_barplot(lang_class_prop_over_gen_df, title, file_path, file_name, n_runs, n_gens, gen_start, n_language_classes, lang_class_baselines_all, lang_class_baselines_fully_expressive):
    """
    Takes a pandas dataframe which contains the proportions of language classes over generations and generates a barplot
    (excluding the burn-in period)

    :param lang_class_prop_over_gen_df: a pandas dataframe containing four columns: 'run', 'generation', 'proportion'
    and 'class'
    :param title: The title of the condition that should be on the plot (string)
    :param file_path: path to folder in which the figure file should be saved
    :param file_name: The file name that the plot should be saved under
    :param n_runs: the number of runs (int); corresponds to global variable 'runs'
    :param n_gens: the number of generations (int); corresponds to global variable 'generations'
    :param gen_start: the burn-in period that is excluded when calculating the means and confidence intervals
    :param n_language_classes: the number of language classes that are distinguished (int). This should be 4 if the old code
    was used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
    languages, and 5 if the new code was used which does make this distinction.
    :param lang_class_baselines_all: The baseline proportion for each language class, where the ordering depends on the code that was
    used, as described above.
    :param lang_class_baselines_fully_expressive: The baseline proportion for only the fully expressive language classes
    (i.e. 'holistic', 'hybrid', and 'compositional')
    :return: Nothing. Just saves the plot and then shows it.
    """

    sns.set_style("whitegrid")
    sns.set_context("talk")

    proportion_column_as_results = dataframe_to_language_stats(lang_class_prop_over_gen_df, n_runs, n_gens, n_language_classes)

    proportion_column_from_start_gen = proportion_column_as_results[:, gen_start:]

    proportion_column_from_start_gen = proportion_column_from_start_gen.flatten()

    runs_column_from_start_gen = []
    for i in range(n_runs):
        for j in range(gen_start, n_gens):
            for k in range(n_language_classes):
                runs_column_from_start_gen.append(i)
    runs_column_from_start_gen = np.array(runs_column_from_start_gen)

    generation_column_from_start_gen = []
    for i in range(n_runs):
        for j in range(gen_start, n_gens):
            for k in range(n_language_classes):
                generation_column_from_start_gen.append(j)
    generation_column_from_start_gen = np.array(generation_column_from_start_gen)

    class_column_from_start_gen = []
    for i in range(n_runs):
        for j in range(gen_start, n_gens):
            if n_language_classes == 4:
                class_column_from_start_gen.append('degenerate')
                class_column_from_start_gen.append('holistic')
                class_column_from_start_gen.append('other')
                class_column_from_start_gen.append('compositional')
            elif n_language_classes == 5:
                class_column_from_start_gen.append('degen.')
                class_column_from_start_gen.append('holistic')
                class_column_from_start_gen.append('hybrid')
                class_column_from_start_gen.append('comp.')
                class_column_from_start_gen.append('other')


    new_data_dict = {'run': runs_column_from_start_gen,
            'generation': generation_column_from_start_gen,
            'proportion': proportion_column_from_start_gen,
            'class': class_column_from_start_gen}

    lang_class_prop_over_gen_df_from_starting_gen = pd.DataFrame(new_data_dict)

    if n_language_classes == 4:
        color_palette = sns.color_palette(["black", "red", "grey", "green"])
    elif n_language_classes == 5:
        color_palette = sns.color_palette(["black", "red", "magenta", "green", "grey"])

    sns.barplot(x="class", y="proportion", data=lang_class_prop_over_gen_df_from_starting_gen, palette=color_palette)

    if n_language_classes == 4:
        plt.axhline(y=lang_class_baselines_all[0], xmin=0.0, xmax=0.25, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[1], xmin=0.25, xmax=0.5, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[2], xmin=0.5, xmax=0.75, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[3], xmin=0.75, xmax=1.0, color='k', linestyle='--', linewidth=2)
    elif n_language_classes == 5:
        plt.axhline(y=lang_class_baselines_all[0], xmin=0.0, xmax=0.2, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[1], xmin=0.2, xmax=0.4, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[2], xmin=0.4, xmax=0.6, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[3], xmin=0.6, xmax=0.8, color='k', linestyle='--', linewidth=2)
        plt.axhline(y=lang_class_baselines_all[4], xmin=0.8, xmax=1.0, color='k', linestyle='--', linewidth=2)

        if title == 'Mutual Understanding Only' or title == 'Minimal Effort & Mutual Understanding':
            plt.axhline(y=lang_class_baselines_fully_expressive[0], xmin=0.2, xmax=0.4, color='0.6', linestyle='--', linewidth=2)
            plt.axhline(y=lang_class_baselines_fully_expressive[1], xmin=0.4, xmax=0.6, color='0.6', linestyle='--', linewidth=2)
            plt.axhline(y=lang_class_baselines_fully_expressive[2], xmin=0.6, xmax=0.8, color='0.6', linestyle='--', linewidth=2)


    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tick_params(axis='both', which='minor', labelsize=20)
    plt.ylim(-0.05, 1.05)
    plt.title(title, fontsize=22)
    # plt.xlabel('Language class')
    plt.xlabel('', fontsize=20)
    plt.ylabel('Mean proportion', fontsize=20)
    plt.tight_layout()

    plt.savefig(file_path + "Barplot_" + file_name + "_burn_in_" + str(gen_start) + ".pdf")
    plt.show()


###################################################################################################################
# THE UNPICKLING HAPPENS HERE:

all_results = []
for i in range(batches):

    pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(gamma) +"_minimal_e_" + str(minimal_effort) + "_c_" + convert_array_to_string(cost_vector) + "_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type[:5] + "_noise_" + str(noise) + "_" + convert_float_value_to_string(noise_prob) +"_observed_m_" + observed_meaning +"_n_l_classes_" + str(n_lang_classes) +"_CS_" + str(communicative_success) + "_" + convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2))

    language_stats_over_gens_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_lang_stats" + ".p", "rb"))
    data_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_data"+".p", "rb"))
    final_pop_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_final_pop" + ".p", "rb"))

    for j in range(len(language_stats_over_gens_per_run)):
        all_results.append(language_stats_over_gens_per_run[j])

print('')
print("len(all_results) are:")
print(len(all_results))
print("len(all_results[0]) are:")
print(len(all_results[0]))
print("len(all_results[0][0]) are:")
print(len(all_results[0][0]))

lang_class_prop_over_gen_df = language_stats_to_dataframe(all_results, runs*batches, generations, n_lang_classes)
print('')
print('')
print("lang_class_prop_over_gen_df is:")
print(lang_class_prop_over_gen_df)


###################################################################################################################
# THE PLOTTING HAPPENS HERE:

fig_file_name = "r_" + str(runs*batches) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_mutual_u_"+str(mutual_understanding)+  "_gamma_" + str(gamma) +"_minimal_e_"+str(minimal_effort)+ "_c_"+convert_array_to_string(cost_vector)+ "_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type[:5] + "_noise_" + str(noise) + "_noise_prob_" + convert_float_value_to_string(noise_prob)+"_"+production+"_obs_m_"+observed_meaning+"_n_lang_classes_"+str(n_lang_classes)+"_CS_"+str(communicative_success)+"_"+convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2))


if mutual_understanding is False and minimal_effort is False:
    if gamma == 0 and turnover is True:
        plot_title = "Learnability only"
    elif gamma > 0 and turnover is False:
        plot_title = "Expressivity only"
    elif gamma > 0 and turnover is True:
        plot_title = "Learnability and expressivity"
    if noise:
        plot_title = plot_title + " Plus Noise"
else:
    if mutual_understanding is True and minimal_effort is False:
        plot_title = "Mutual Understanding Only"
    elif mutual_understanding is False and minimal_effort is True:
        plot_title = "Minimal Effort Only"
    elif mutual_understanding is True and minimal_effort is True:
        plot_title = "Minimal Effort & Mutual Understanding"


plot_timecourse(lang_class_prop_over_gen_df, plot_title, fig_file_path, fig_file_name, n_lang_classes)


all_possible_languages = create_all_possible_languages(meanings, forms_without_noise)
print("number of possible languages is:")
print(len(all_possible_languages))

class_per_lang = classify_all_languages(all_possible_languages, forms_without_noise, meanings)
print('')
print('')
no_of_each_class = np.bincount(class_per_lang.astype(int))
print('')
print("no_of_each_class is:")
print(no_of_each_class)
print("np.sum(no_of_each_class) is:")
print(np.sum(no_of_each_class))


baseline_proportions_all = np.divide(no_of_each_class, len(all_possible_languages))  # 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other
print('')
print('')
print("baseline_proportions_all are:")
print(baseline_proportions_all)


baseline_proportions_fully_expressive = np.divide(no_of_each_class[1:4], np.sum(no_of_each_class[1:4]))  # 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other
print('')
print('')
print("baseline_proportions_fully_expressive are:")
print(baseline_proportions_fully_expressive)


plot_barplot(lang_class_prop_over_gen_df, plot_title, fig_file_path, fig_file_name, runs, generations, burn_in, n_lang_classes, baseline_proportions_all, baseline_proportions_fully_expressive)
