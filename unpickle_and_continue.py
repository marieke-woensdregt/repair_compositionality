from evolution_compositionality_under_noise import *


###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = create_all_possible_forms(2, [2])  # all possible forms, excluding their possible
# 'noisy variants'
noisy_forms = create_all_possible_noisy_forms(forms_without_noise)
# all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise + noisy_forms  # all possible forms, including both
# complete forms and noisy variants
error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

turnover = True  # determines whether new individuals enter the population or not

popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
            # a pair of agents.
runs = 100  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 800  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

production = 'my_code'  # can be set to 'simlang' or 'my_code'

cost_vector = np.array([0.0, 0.2, 0.4])  # costs of no repair, restricted request, and open request, respectively

observed_meaning = 'intended'  # determines which meaning the learner observes when receiving a meaning-form pair; can
# be set to either 'intended', where the learner has direct access to the speaker's intended meaning, or 'inferred',
# where the learner has access to the hearer's interpretation.
interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

communicative_success = False  # determines whether there is a pressure for communicative success or not
communicative_success_pressure_strength = (2./3.)  # determines how much more likely a <meaning, form> pair from a
# successful interaction is to enter the data set that is passed on to the next generation, compared to a
# <meaning, form> pair from a unsuccessful interaction.

pickle_file_path = "pickles/"

extra_gens = 200


# THE FOLLOWING PARAMETERS SHOULD ONLY BE SET IF __name__ == '__main__', BECAUSE THEY ARE RETRIEVED FROM THE INPUT
# ARGUMENTS GIVEN TO THE PYTHON SCRIPT WHEN RUN FROM THE TERMINAL OR FROM A .SH SCRIPT:
if __name__ == '__main__':

    b = int(sys.argv[1])  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training
    # (Kirby et al. used a bottleneck of 20 in the body of the paper.
    rounds = 2 * b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
    print('')
    print("b is:")
    print(b)

    compressibility_bias = str_to_bool(sys.argv[2])  # Setting the 'compressibility_bias' parameter based on the
    # command-line input #NOTE: first argument in sys.argv list is always the name of the script; Determines whether
    # agents have a prior that favours compressibility, or a flat prior
    print('')
    print("compressibility_bias (i.e. learnability pressure) is:")
    print(compressibility_bias)

    noise_prob = float(sys.argv[3])  # Setting the 'noise_prob' parameter based on the command-line input #NOTE: first
    # argument in sys.argv list is always the name of the script  # the probability of environmental noise obscuring
    # part of an utterance
    print('')
    print("noise_prob is:")
    print(noise_prob)

    mutual_understanding = str_to_bool(sys.argv[4])  # Setting the 'mutual_understanding' parameter based on the
    # command-line input #NOTE: first argument in sys.argv list is always the name of the script
    print('')
    print("mutual_understanding is:")
    print(mutual_understanding)
    if mutual_understanding:
        gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
        # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
        # conditions
    else:
        gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
        # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
        # conditions

    minimal_effort = str_to_bool(sys.argv[5])  # Setting the 'minimal_effort' parameter based on the command-line input
    # #NOTE: first argument in sys.argv list is always the name of the script
    print('')
    print("minimal_effort is:")
    print(minimal_effort)

    batch_number = int(sys.argv[6])  # Setting the 'batch_number' parameter based on the command-line input
    # #NOTE: first argument in sys.argv list is always the name of the script
    print('')
    print("batch_number is:")
    print(batch_number)


###################################################################################################################
# THE UNPICKLING HAPPENS HERE:



pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(gamma) +"_minimal_e_" + str(minimal_effort) + "_c_" + convert_array_to_string(cost_vector) + "_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type + "_noise_prob_" + convert_float_value_to_string(noise_prob) +"_observed_m_" + observed_meaning +"_CS_" + str(communicative_success) + "_" + convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2))

language_stats_over_gens_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_lang_stats_"+str(batch_number)+".p", "rb"))
data_over_gens_per_run = pickle.load(open(pickle_file_path+pickle_file_name+"_data_"+str(batch_number)+".p", "rb"))
final_pop_per_run = pickle.load(open(pickle_file_path + pickle_file_name + "_final_pop_"+str(batch_number)+".p", "rb"))

if type(language_stats_over_gens_per_run) == list:
    language_stats_over_gens_per_run = np.array(language_stats_over_gens_per_run)
if type(final_pop_per_run) == list:
    final_pop_per_run = np.array(final_pop_per_run)

###################################################################################################################
# NOW LET'S RUN THE ACTUAL SIMULATION:

t0 = time.process_time()

hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)

class_per_lang = classify_all_languages(hypothesis_space, forms_without_noise, meanings)

if compressibility_bias:
    priors = prior(hypothesis_space, forms_without_noise, meanings, possible_form_lengths)
else:
    priors = np.ones(len(hypothesis_space))
    priors = np.divide(priors, np.sum(priors))
    priors = np.log(priors)

language_stats_over_gens_per_run_new = np.zeros((runs, generations+extra_gens, int(max(class_per_lang)+1)))
data_over_gens_per_run_new = []
final_pop_per_run_new = np.zeros((runs, popsize, len(hypothesis_space)))
for r in range(runs):

    print('')
    print('This is run:')
    print(r)

    final_pop = final_pop_per_run[r]

    initial_dataset = data_over_gens_per_run[r][-1]

    language_stats_over_gens, data_over_gens, final_pop = simulation(final_pop, extra_gens, rounds, b, popsize, meanings, possible_form_lengths, hypothesis_space, class_per_lang, priors, initial_dataset, interaction, production, gamma, noise_prob, all_forms_including_noisy_variants, mutual_understanding, minimal_effort, communicative_success)

    language_stats_over_gens_per_run_new[r] = np.concatenate((language_stats_over_gens_per_run[r], language_stats_over_gens))
    data_over_gens_per_run_new.append(data_over_gens_per_run[r] + data_over_gens)
    final_pop_per_run_new[r] = final_pop

timestr = time.strftime("%Y%m%d-%H%M%S")

pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations+extra_gens) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(gamma) +"_minimal_e_" + str(minimal_effort) + "_c_" + convert_array_to_string(cost_vector) + "_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type + "_noise_prob_" + convert_float_value_to_string(noise_prob) +"_observed_m_" + observed_meaning +"_CS_" + str(communicative_success) + "_" + convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2)) + "_" + timestr

pickle.dump(language_stats_over_gens_per_run_new, open(pickle_file_path + pickle_file_name + "_lang_stats_"+str(batch_number)+".p", "wb"))
pickle.dump(data_over_gens_per_run, open(pickle_file_path+pickle_file_name+"_data_"+str(batch_number)+".p", "wb"))
pickle.dump(final_pop_per_run_new, open(pickle_file_path + pickle_file_name + "_final_pop_"+str(batch_number)+".p", "wb"))

t1 = time.process_time()

print('')
print("number of minutes it took to run simulation:")
print(round((t1-t0)/60., ndigits=2))

print('')
print('results were saved in folder:')
print(pickle_file_path)

print('')
print('using filename:')
print(pickle_file_name)

