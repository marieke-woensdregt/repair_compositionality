import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evolution_compositionality_under_noise import *


###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
possible_form_lengths = np.array([2, 4])  # all possible form lengths
forms_without_noise = create_all_possible_forms(2, possible_form_lengths)  # all possible forms, excluding their
# possible 'noisy variants'
noisy_forms = create_all_possible_noisy_forms(forms_without_noise)
# all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise + noisy_forms  # all possible forms, including both
# complete forms and noisy variants

error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

turnover = True  # determines whether new individuals enter the population or not
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
# a pair of agents.
runs = 1  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 5  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

burn_in = round(generations / 2)  # the burn-in period that is excluded when calculating the mean distribution over
# languages after convergence

b = 20 # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training
    # (Kirby et al. used a bottleneck of 20 in the body of the paper.
rounds = 2 * b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
print('')
print("b is:")
print(b)

compressibility_bias = True  # Setting the 'compressibility_bias' parameter based on the
# command-line input #NOTE: first argument in sys.argv list is always the name of the script; Determines whether
# agents have a prior that favours compressibility, or a flat prior
print('')
print("compressibility_bias (i.e. learnability pressure) is:")
print(compressibility_bias)

noise_prob = 0.5  # Setting the 'noise_prob' parameter based on the command-line input #NOTE: first
# argument in sys.argv list is always the name of the script  # the probability of environmental noise obscuring
# part of an utterance
print('')
print("noise_prob is:")
print(noise_prob)

gamma = 2.0 # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used
# gamma = 0 for "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and
# Expressivity" conditions
print('')
print("gamma is:")
print(gamma)

delta = 2.0 # parameter that determines strength of effort penalty (i.e. how strongly speaker
# tries to avoid using long utterances)
print('')
print("delta is:")
print(delta)

pickle_file_path = "pickles/"

fig_file_path = "plots/"


batches = 1


holistic_without_partial_meaning = True

###################################################################################################################
# ALL FUNCTION DEFINITIONS GO HERE:

# FIRST WE NEED A FUNCTION THAT CONVERTS A DATA LIST INTO A PANDAS DATA FRAME FOR PLOTTING:
def language_stats_to_dataframe(results, n_runs, n_gens, possible_form_lengths):
    """
    Takes a results list and puts it in a pandas dataframe together with other relevant variables (runs, generations,
    and language class)

    :param results: a list containing proportions for each of the 5 language classes, for each generation, for each run
    :param n_runs: the number of runs (int); corresponds to global variable 'runs'
    :param n_gens: the number of generations (int); corresponds to global variable 'generations'
    :param possible_form_lengths: all possible form lengths (global parameter)
    :return: a pandas dataframe containing four columns: 'run', 'generation', 'proportion' and 'class'
    """

    if len(possible_form_lengths) == 1:
        n_language_classes = 4
    else:
        n_language_classes = 7  #TODO: or should this be 6 (i.e. collapsing the two different reduplication strategies?)

    column_proportion = np.array(results)

    if n_language_classes == 4 and column_proportion.shape[2] > n_language_classes:
        column_proportion_compositional_summed = np.zeros((n_runs, n_gens, n_language_classes))
        for r in range(len(column_proportion_compositional_summed)):
            for g in range(len(column_proportion_compositional_summed[0])):
                column_proportion_compositional_summed[r][g] = np.array([column_proportion[r][g][0], column_proportion[r][g][1], column_proportion[r][g][2]+column_proportion[r][g][3], column_proportion[r][g][4]])
        column_proportion = column_proportion_compositional_summed.flatten()

    else:
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
            if len(possible_form_lengths) == 1:
                column_type.append('degenerate')
                column_type.append('holistic')
                column_type.append('compositional')
                column_type.append('other')
            else:
                column_type.append('degenerate')
                column_type.append('holistic')
                column_type.append('holistic_diversify_signal')
                column_type.append('compositional')
                column_type.append('compositional_reduplicate_segments')
                column_type.append('compositional_reduplicate_whole_signal')
                column_type.append('other')

    data = {'run': column_runs,
            'generation': column_generation,
            'proportion': column_proportion,
            'class': column_type}

    lang_class_prop_over_gen_df = pd.DataFrame(data)

    return lang_class_prop_over_gen_df


# And this function turns a pandas dataframe back into a list of lists of lists of language stats, just in case
# that's needed:
def dataframe_to_language_stats(dataframe, n_runs, n_batches, n_gens, possible_form_lengths):
    """
    Takes a pandas dataframe of results and turns it back into a simple results array, which only contains the
    populations' posterior probability distributions over generations.

    :param dataframe: a pandas dataframe which contains at least a column named "proportions", which contains the
    proportions of the different language classes over generations over runs.
    :param n_runs: number of runs (int)
    :param n_gens: number of generations (int)
    :param possible_form_lengths: all possible form lengths (global parameter)
    :return: a numpy array containing the proportions of the different language classes for each generation for each run
    """
    if len(possible_form_lengths) == 1:
        n_language_classes = 4
    else:
        n_language_classes = 7  #TODO: or should this be 6 (i.e. collapsing the two different reduplication strategies?)
    proportion_column = np.array(dataframe['proportion'])
    proportion_column_as_results = proportion_column.reshape((n_runs*n_batches, n_gens, n_language_classes))
    return proportion_column_as_results


# AND NOW FOR THE PLOTTING FUNCTIONS:

def plot_timecourse(lang_class_prop_over_gen_df, title, file_path, file_name):
    """
    Takes a pandas dataframe which contains the proportions of language classes over generations and plots timecourses

    :param lang_class_prop_over_gen_df: a pandas dataframe containing four columns: 'run', 'generation', 'proportion'
    and 'class'
    :param title: The title of the condition that should be on the plot (string)
    :param file_path: path to folder in which the figure file should be saved
    :param file_name: The file name that the plot should be saved under
    :return: Nothing. Just saves the plot and then shows it.
    """
    sns.set_style("darkgrid")
    sns.set_context("talk")

    fig, ax = plt.subplots()

    if len(possible_form_lengths) == 1:
        palette = sns.color_palette(["black", "red", "green", "grey"])
    else:
        palette = sns.color_palette(["black",
                                     sns.color_palette("colorblind")[3],
                                     sns.color_palette("colorblind")[1],
                                     sns.color_palette("colorblind")[2],
                                     sns.color_palette("colorblind")[9],
                                     sns.color_palette("colorblind")[0],
                                     sns.color_palette("colorblind")[7]])

    sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette)
    # sns.lineplot(x="generation", y="proportion", hue="class", data=lang_class_prop_over_gen_df, palette=palette, ci=95, err_style="bars")

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.ylim(-0.05, 1.05)
    plt.title(title, fontsize=22)
    plt.xlabel('Generation', fontsize=20)
    plt.ylabel('Mean proportion', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()

    labels = ['D', 'H', 'H+Div.', 'C', 'C+Red.-part', 'C+Red.-whole', 'O']

    # ax.legend(handles=handles[1:], labels=labels[1:])
    ax.legend(handles=handles, labels=labels)
    plt.tight_layout()
    plt.savefig(file_path + "Timecourse_plot_" + file_name + ".png")
    plt.show()


def plot_barplot(lang_class_prop_over_gen_df, title, file_path, file_name, n_runs, n_batches, n_gens, gen_start, lang_class_baselines_all, lang_class_baselines_fully_expressive, possible_form_lengths):
    """
    Takes a pandas dataframe which contains the proportions of language classes over generations and generates a
    barplot (excluding the burn-in period)

    :param lang_class_prop_over_gen_df: a pandas dataframe containing four columns: 'run', 'generation', 'proportion'
    and 'class'
    :param title: The title of the condition that should be on the plot (string)
    :param file_path: path to folder in which the figure file should be saved
    :param file_name: The file name that the plot should be saved under
    :param n_runs: the number of runs (int); corresponds to global variable 'runs'
    :param n_gens: the number of generations (int); corresponds to global variable 'generations'
    :param gen_start: the burn-in period that is excluded when calculating the means and confidence intervals
    :param lang_class_baselines_all: The baseline proportion for each language class, where the ordering depends on the
    code that was used, as described above.
    :param lang_class_baselines_fully_expressive: The baseline proportion for only the fully expressive language classes
    (i.e. 'holistic', and 'compositional')
    :param possible_form_lengths: all possible form lengths (global parameter)
    :return: Nothing. Just saves the plot and then shows it.
    """

    sns.set_style("darkgrid")
    sns.set_context("talk")

    if len(possible_form_lengths) == 1:
        n_language_classes = 4
    else:
        n_language_classes = 7  #TODO: or should this be 6 (i.e. collapsing the two different reduplication strategies?)

    proportion_column_as_results = dataframe_to_language_stats(lang_class_prop_over_gen_df, n_runs, n_batches, n_gens, possible_form_lengths)

    proportion_column_from_start_gen = proportion_column_as_results[:, gen_start:]

    proportion_column_from_start_gen = proportion_column_from_start_gen.flatten()

    runs_column_from_start_gen = []
    for i in range(n_runs*n_batches):
        for j in range(gen_start, n_gens):
            for k in range(n_language_classes):
                runs_column_from_start_gen.append(i)
    runs_column_from_start_gen = np.array(runs_column_from_start_gen)

    generation_column_from_start_gen = []
    for i in range(n_runs*n_batches):
        for j in range(gen_start, n_gens):
            for k in range(n_language_classes):
                generation_column_from_start_gen.append(j)
    generation_column_from_start_gen = np.array(generation_column_from_start_gen)

    class_column_from_start_gen = []
    for i in range(n_runs*n_batches):
        for j in range(gen_start, n_gens):
            if n_language_classes == 4:
                class_column_from_start_gen.append('degenerate')
                class_column_from_start_gen.append('holistic')
                class_column_from_start_gen.append('compositional')
                class_column_from_start_gen.append('other')
            elif n_language_classes == 7:
                class_column_from_start_gen.append('D')
                class_column_from_start_gen.append('H')
                class_column_from_start_gen.append('H+Div.')
                class_column_from_start_gen.append('C')
                class_column_from_start_gen.append('C+Red.-part')
                class_column_from_start_gen.append('C+Red.-whole')
                class_column_from_start_gen.append('O')

    new_data_dict = {'run': runs_column_from_start_gen,
            'generation': generation_column_from_start_gen,
            'proportion': proportion_column_from_start_gen,
            'class': class_column_from_start_gen}

    lang_class_prop_over_gen_df_from_starting_gen = pd.DataFrame(new_data_dict)

    if len(possible_form_lengths) == 1:
        palette = sns.color_palette(["black", "red", "green", "grey"])
    else:
        palette = sns.color_palette(["black",
                                     sns.color_palette("colorblind")[3],
                                     sns.color_palette("colorblind")[1],
                                     sns.color_palette("colorblind")[2],
                                     sns.color_palette("colorblind")[9],
                                     sns.color_palette("colorblind")[0],
                                     sns.color_palette("colorblind")[7]])

    sns.barplot(x="class", y="proportion", data=lang_class_prop_over_gen_df_from_starting_gen, palette=palette)

    # plt.axhline(y=lang_class_baselines_all[0], xmin=0.0, xmax=0.25, color='k', linestyle='--', linewidth=2)
    # plt.axhline(y=lang_class_baselines_all[1], xmin=0.25, xmax=0.5, color='k', linestyle='--', linewidth=2)
    # plt.axhline(y=lang_class_baselines_all[2], xmin=0.5, xmax=0.75, color='k', linestyle='--', linewidth=2)
    # plt.axhline(y=lang_class_baselines_all[3], xmin=0.75, xmax=1.0, color='k', linestyle='--', linewidth=2)
    #
    # if title == 'Mutual Understanding Only' or title == 'Minimal Effort & Mutual Understanding':
    #     plt.axhline(y=lang_class_baselines_fully_expressive[0], xmin=0.25, xmax=0.5, color='0.6', linestyle='--', linewidth=2)
    #     plt.axhline(y=lang_class_baselines_fully_expressive[1], xmin=0.5, xmax=0.75, color='0.6', linestyle='--', linewidth=2)

    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.ylim(-0.05, 1.05)
    plt.title(title, fontsize=22)
    # plt.xlabel('Language class')
    plt.xlabel('', fontsize=20)
    plt.ylabel('Mean proportion', fontsize=20)
    plt.tight_layout()

    if holistic_without_partial_meaning is True:
        plt.savefig(file_path + "Barplot_" + file_name + "_burn_in_" + str(gen_start) + ".png")
    else:
        plt.savefig(file_path + "Barplot_" + file_name + "_burn_in_" + str(gen_start) + "_NEW.png")
    plt.show()


###################################################################################################################
# THE UNPICKLING HAPPENS HERE:

pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_form_lengths_"+convert_array_to_string(possible_form_lengths)+"_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_gamma_" + convert_float_value_to_string(gamma) + "_delta_"+convert_float_value_to_string(delta)+"_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type + "_noise_prob_" + convert_float_value_to_string(noise_prob)

all_results = []
all_repair_counts = []
if batches > 1:
    for i in range(batches):
        if holistic_without_partial_meaning is True:
            language_stats_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_lang_stats_"+str(i)+".p", "rb"))
            data_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_data_"+str(i)+".p", "rb"))
            repair_counts_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_repairs_"+str(i)+".p", "rb"))
            final_pop_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_final_pop_"+str(i)+".p", "rb"))
        else:
            language_stats_over_gens_per_run = pickle.load(
                open(pickle_file_path + pickle_file_name + "_lang_stats_" + str(i) + "_NEW.p", "rb"))
            data_over_gens_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_data_" + str(i) + "_NEW.p", "rb"))
            repair_counts_over_gens_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_repairs_" + str(i) + "_NEW.p", "rb"))
            final_pop_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_final_pop_" + str(i) + "_NEW.p", "rb"))
        for j in range(len(language_stats_over_gens_per_run)):
            all_results.append(language_stats_over_gens_per_run[j])
            all_repair_counts.append(repair_counts_over_gens_per_run[j])


else:
    if holistic_without_partial_meaning is True:
        language_stats_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_lang_stats"+".p", "rb"))
        data_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_data"+".p", "rb"))
        repair_counts_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_repairs"+".p", "rb"))
        final_pop_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_final_pop"+".p", "rb"))
    else:
        language_stats_over_gens_per_run = pickle.load(
            open(pickle_file_path + pickle_file_name + "_lang_stats" + "_NEW.p", "rb"))
        data_over_gens_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_data" + "_NEW.p", "rb"))
        repair_counts_over_gens_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_repairs" + "_NEW.p", "rb"))
        final_pop_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_final_pop" + "_NEW.p", "rb"))
    for j in range(len(language_stats_over_gens_per_run)):
        all_results.append(language_stats_over_gens_per_run[j])
        all_repair_counts.append(repair_counts_over_gens_per_run[j])

print('')
print("len(all_results) are:")
print(len(all_results))
print("len(all_results[0]) are:")
print(len(all_results[0]))
print("len(all_results[0][0]) are:")
print(len(all_results[0][0]))

print('')
print("len(all_repair_counts) are:")
print(len(all_repair_counts))
print("len(all_repair_counts[0]) are:")
print(len(all_repair_counts[0]))

lang_class_prop_over_gen_df = language_stats_to_dataframe(all_results, runs*batches, generations, possible_form_lengths)
print('')
print('')
print("lang_class_prop_over_gen_df is:")
print(lang_class_prop_over_gen_df)


###################################################################################################################
# THE PLOTTING HAPPENS HERE:

fig_file_name = "Repair_vs_Redundancy_r_" + str(runs*batches) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_gamma_" + convert_float_value_to_string(gamma) + "_delta_"+convert_float_value_to_string(delta)+"_turnover_" + str(turnover) + "_bias_" +str(compressibility_bias) + "_init_" + initial_language_type + "_noise_prob_" + convert_float_value_to_string(noise_prob)

if delta == 0.0 and compressibility_bias is False:
    plot_title = "Neither Pressure"
elif delta > 0.0 and compressibility_bias is False:
    plot_title = "Minimal Effort Only"
elif delta == 0.0 and compressibility_bias is True:
    plot_title = "Learnability Only"
elif delta > 0.0 and compressibility_bias is True:
    plot_title = "Learnability & Minimal Effort"

plot_timecourse(lang_class_prop_over_gen_df, plot_title, fig_file_path, fig_file_name)


all_possible_languages = create_all_possible_languages(meanings, forms_without_noise)
print("number of possible languages is:")
print(len(all_possible_languages))

class_per_lang = classify_all_languages(all_possible_languages, forms_without_noise, meanings)
print('')
print('')

if len(possible_form_lengths) == 1:
    n_language_classes = 4
else:
    n_language_classes = 7  # TODO: or should this be 6 (i.e. collapsing the two different reduplication strategies?)

no_of_each_class = np.bincount(class_per_lang.astype(int))
if n_language_classes == 4 and len(no_of_each_class) > n_language_classes:
    no_of_each_class_compositional_summed = np.array([no_of_each_class[0], no_of_each_class[1], no_of_each_class[2]+no_of_each_class[3], no_of_each_class[4]])
    no_of_each_class = no_of_each_class_compositional_summed

print('')
print("no_of_each_class is:")
print(no_of_each_class)
print("np.sum(no_of_each_class) is:")
print(np.sum(no_of_each_class))


baseline_proportions_all = np.divide(no_of_each_class, len(all_possible_languages))  # 0 = degenerate, 1 = holistic, 2 = compositional, 3 = other
print('')
print('')
print("baseline_proportions_all are:")
print(baseline_proportions_all)


baseline_proportions_fully_expressive = np.divide(no_of_each_class[1:3], np.sum(no_of_each_class[1:3]))  # 0 = degenerate, 1 = holistic, 2 = compositional, 3 = other
print('')
print('')
print("baseline_proportions_fully_expressive are:")
print(baseline_proportions_fully_expressive)


plot_barplot(lang_class_prop_over_gen_df, plot_title, fig_file_path, fig_file_name, runs, batches, generations, burn_in, baseline_proportions_all, baseline_proportions_fully_expressive, possible_form_lengths)
