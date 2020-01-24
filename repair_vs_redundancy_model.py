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
generations = 10  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'holistic'  # set the language class that the first generation is trained on

interaction = 'taking_turns'  # can be set to either 'random' or 'taking_turns'. The latter is what Kirby et al. (2015)
# used, but NOTE that it only works with a popsize of 2!
n_parents = 'single'  # determines whether each generation of learners receives data from a single agent from the
# previous generation, or from multiple (can be set to either 'single' or 'multiple').

burn_in = round(generations / 2)  # the burn-in period that is excluded when calculating the mean distribution over
# languages after convergence

pickle_file_path = "pickles/"


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

    gamma = float(sys.argv[4])  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used
    # gamma = 0 for "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and
    # Expressivity" conditions
    print('')
    print("gamma is:")
    print(gamma)

    delta = float(sys.argv[5])  # parameter that determines strength of effort penalty (i.e. how strongly speaker
    # tries to avoid using long utterances)
    print('')
    print("delta is:")
    print(delta)


###################################################################################################################
# NOW SOME FUNCTIONS THAT HANDLE PRODUCTION, NOISY PRODUCTION, AND RECEPTION WITH AND WITHOUT REPAIR:


def production_likelihoods_with_noise_and_minimal_effort(language, topic, meaning_list, forms, noisy_variants, ambiguity_penalty, effort_penalty, error_prob, prob_of_noise):
    """
    Calculates the production probabilities for each of the possible forms (including both forms without noise and all
    possible noisy variants) given a language and topic, and the probability of environmental noise

    :param language: list of forms that has same length as list of meanings (global variable), where each form is
    mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :param forms: list of all possible forms *excluding* their noisy variants
    :param noisy_variants: list of all possible noisy variants of forms
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
    :param error_prob: the probability of making an error in production
    :param prob_of_noise: the probability of environmental noise masking part of the utterance
    :return: 1D numpy array containing a production probability for each possible form (where the index of the
    probability corresponds to the index of the form in the global variable "all_forms_including_noisy_variants")
    """
    all_possible_forms = forms + noisy_variants
    for m in range(len(meaning_list)):
        if meaning_list[m] == topic:
            topic_index = m
    correct_form = language[topic_index]
    error_forms = list(forms)  # This may seem a bit weird, but a speaker should be able to produce *any*
    # form as an error form right? Not limited to only the other forms that exist within their language? (Otherwise a
    # speaker with a degenerate language could never make a production error).
    error_forms = remove_all_instances(error_forms, correct_form)
    if len(error_forms) == 0:  # if the list of error_forms is empty because the language is degenerate
        error_forms = language  # simply choose an error_form from the whole language
    noisy_variants_correct_form = create_noisy_variants(correct_form)
    noisy_variants_error_forms = []
    for error_form in error_forms:
        noisy_variants = create_noisy_variants(error_form)
        noisy_variants_error_forms = noisy_variants_error_forms+noisy_variants
    ambiguity = 0
    for f in language:
        if f == correct_form:
            ambiguity += 1
    # prop_to_prob_correct_form_complete = ((1./ambiguity) ** ambiguity_penalty) * (1. - error_prob) * (1 - prob_of_noise)
    # prop_to_prob_error_form_complete = error_prob / (len(forms) - 1) * (1 - prob_of_noise)
    # prop_to_prob_correct_form_noisy = ((1. / ambiguity) ** ambiguity_penalty) * (1. - error_prob) * (prob_of_noise / len(noisy_variants))
    # prop_to_prob_error_form_noisy = error_prob / (len(forms) - 1) * (1 - prob_of_noise) * (prob_of_noise / len(noisy_variants))
    prop_to_prob_per_form_array = np.zeros(len(all_possible_forms))
    for i in range(len(all_possible_forms)):
        possible_form = all_possible_forms[i]
        if possible_form == correct_form:
            prop_to_prob_per_form_array[i] = ((1./ambiguity) ** ambiguity_penalty) * ((1./len(possible_form)) ** effort_penalty) * (1. - error_prob) * (1 - prob_of_noise)
        elif possible_form in noisy_variants_correct_form:
            prop_to_prob_per_form_array[i] = ((1. / ambiguity) ** ambiguity_penalty) * ((1./len(possible_form)) ** effort_penalty) * (1. - error_prob) * (prob_of_noise / len(noisy_variants))
        elif possible_form in noisy_variants_error_forms:
            prop_to_prob_per_form_array[i] = error_prob / (len(forms) - 1) * (1 - prob_of_noise) * (prob_of_noise / len(noisy_variants))
        else:
            prop_to_prob_per_form_array[i] = error_prob / (len(forms) - 1) * (1 - prob_of_noise)
    return prop_to_prob_per_form_array


# And finally, let's write a function that actually produces an utterance, given a language and a topic:
def produce_with_minimal_effort(language, topic, ambiguity_penalty, effort_penalty, error_prob, prob_of_noise):
    """
    Produces an actual utterance, given a language and a topic

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
    :param error_prob: the probability of making an error in production
    :param prob_of_noise: the probability of noise happening (only relevant when noise_switch is set to True);
    corresponds to global variable 'noise_prob'
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    prop_to_prob_per_form_array = production_likelihoods_with_noise_and_minimal_effort(language, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, effort_penalty, error_prob, prob_of_noise)
    prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
    utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    return utterance


def receive_with_repair_open_only(language, utterance):
    """
    Receives and utterance and gives a response, which can either be an interpretation or a repair initiator. How likely
    these two response types are to happen depends on the settings of the paremeters 'mutual_understanding' and
    'minimal_effort' (and, if minimal_effort is set to True, the parameter 'cost_vector'). These three parameters are
    all assumed to be global variables.

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param utterance: an utterance (string)
    :return: a response, which can either be an interpretation (i.e. meaning) or a repair initiator ('??'). The listener
    initiates repair whenever there is any ambiguity about how to interpret the utterance according to the language
    (either caused by noise or just by ambiguity in the language itself).
    """
    if '_' in utterance:
        compatible_forms = noisy_to_complete_forms(utterance, forms_without_noise)
        possible_interpretations = find_possible_interpretations(language, compatible_forms)
    else:
        possible_interpretations = find_possible_interpretations(language, [utterance])
    if len(possible_interpretations) == 0:
        possible_interpretations = meanings
    if len(possible_interpretations) == 1:
        response = possible_interpretations[0]
    else:
        response = '??'
    return response


# AND NOW FOR THE FUNCTIONS THAT DO THE BAYESIAN LEARNING:

def update_posterior_from_cache(log_posterior, log_likelihood_cache, topic, utterance, meaning_list, all_possible_forms):
    """
    Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability
    distribution accordingly

    :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
    :param log_likelihood_cache: 3D numpy array with axis 0 = meanings, axis 1 = all possible forms, and axis 2 = LOG
    likelihood of corresponding <meaning, form> pair for each hypothesis
    :param topic: a topic (string from the global variable meanings)
    :param utterance: an utterance (string from the global variable forms (can be a noisy form if parameter noise is
    True)
    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
    'all_forms_including_noisy_variants'
    :return: the updated (and normalized) log_posterior (1D numpy array)
    """
    # First let's find out what the index of the meaning is:
    for i in range(len(meanings)):
        if meaning_list[i] == topic:
            meaning_index = i
    # Then, let's find out what the index of the utterance is in the list of all possible forms (including the noisy
    # variants):
    for i in range(len(all_possible_forms)):
        if all_possible_forms[i] == utterance:
            utterance_index = i
    # Now, let's retrieve the corresponding log_likelihood values for this particular <meaning, form> pair form the
    # log_likelihood_cache, and update the posterior accordingly (addition in logspace == multiplication in probability
    # space).
    new_log_posterior_new_method = np.add(log_posterior, log_likelihood_cache[meaning_index][utterance_index])
    new_log_posterior_normalized_new_method = np.subtract(new_log_posterior_new_method,
                                                          scipy.special.logsumexp(new_log_posterior_new_method))
    return new_log_posterior_normalized_new_method


def population_communication_mutual_understanding(population, n_rounds, ambiguity_penalty, effort_penalty, prob_of_noise, hypotheses, log_likelihood_cache):
    """
    Takes a population, makes it communicate for a number of rounds (where agents' posterior probability distribution
    is updated every time the agent gets assigned the role of hearer)

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :param n_rounds: the number of rounds for which the population should communicate; corresponds to global variable
    'rounds'
    :param ambiguity_penalty: parameter which determines the extent to which the speaker tries to avoid ambiguity;
    corresponds to global variable 'gamma'
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
    :param prob_of_noise: the probability of noise happening (only relevant when noise_switch == True); corresponds to
    global variable 'noise_prob'
    :param hypotheses: list of all possible languages; corresponds to global parameter 'hypothesis_space'
    :param log_likelihood_cache: 3D numpy array with axis 0 = meanings, axis 1 = all possible forms, and axis 2 = LOG
    likelihood of corresponding <meaning, form> pair for each hypothesis
    :return: the data that was produced during the communication rounds, as a list of (topic, utterance) tuples
    """
    if n_parents == 'single':
        if len(population) != 2 or interaction != 'taking_turns':
            raise ValueError("OOPS! n_parents = 'single' only works if popsize = 2 and interaction = 'taking_turns'.")
        random_parent_index = np.random.choice(np.arange(len(population)))  # this determines which agent's productions
        # will form the input for the next generation
    data = []
    for i in range(n_rounds):
        if interaction == 'taking_turns':
            if len(population) != 2:
                raise ValueError("OOPS! interaction = 'taking_turns' only works if popsize = 2.")
            if i % 2 == 0:
                speaker_index = 0
                hearer_index = 1
            else:
                speaker_index = 1
                hearer_index = 0
        else:
            pair_indices = np.random.choice(np.arange(len(population)), size=2, replace=False)
            speaker_index = pair_indices[0]
            hearer_index = pair_indices[1]
        topic = random.choice(meanings)
        # whenever a speaker is called upon to produce a utterance, they first sample a language from their
        # posterior probability distribution. So each agent keeps updating their language according to the data
        # received from their communication partner.
        speaker_language = sample(hypotheses, population[speaker_index])
        hearer_language = sample(hypotheses, population[hearer_index])
        utterance = produce_with_minimal_effort(speaker_language, topic, ambiguity_penalty, effort_penalty, error, prob_of_noise)
        listener_response = receive_with_repair_open_only(hearer_language, utterance)
        counter = 0
        while '?' in listener_response:
            if counter == 3:  # After 3 attempts, the listener stops trying to do repair
                break
            utterance = produce_with_minimal_effort(speaker_language, topic, ambiguity_penalty, effort_penalty, error, prob_of_noise)
            listener_response = receive_with_repair_open_only(hearer_language, utterance)
            counter += 1

        population[hearer_index] = update_posterior_from_cache(population[hearer_index], log_likelihood_cache, topic, utterance, meanings, all_forms_including_noisy_variants)

        if n_parents == 'single':
            if speaker_index == random_parent_index:
                data.append((topic, utterance))
        elif n_parents == 'multiple':
            data.append((topic, utterance))

    return data


# AND NOW FINALLY FOR THE FUNCTION THAT RUNS THE ACTUAL SIMULATION:

def simulation_repair_vs_redundancy(population, n_gens, n_rounds, bottleneck, pop_size, hypotheses, log_likelihood_cache, class_per_language, log_priors, data, interaction_order, ambiguity_penalty, effort_penalty, prob_of_noise, all_possible_forms):
    """
    Runs the full simulation and returns the total amount of posterior probability that is assigned to each language
    class over generations (language_stats_over_gens) as well as the data that each generation produced (data)

    :param population: the population at generation 0
    :param n_gens: the desired number of generations (int); corresponds to global variable 'generations'
    :param n_rounds: the desired number of communication rounds *within* each generation; corresponds to global variable
    'rounds'
    :param bottleneck: the amount of data (<meaning, form> pairs) that each learner receives
    :param pop_size: the desired size of the population (int); corresponds to global variable 'popsize'
    :param hypotheses: list of all possible languages; corresponds to global variable 'hypothesis_space'
    :param log_likelihood_cache: 3D numpy array with axis 0 = meanings, axis 1 = all possible forms, and axis 2 = LOG
    likelihood of corresponding <meaning, form> pair for each hypothesis
    :param class_per_language: list specifiying the class for each corresponding language in the variable 'hypotheses'
    :param log_priors: the LOG prior probability distribution that each agent should be initialised with
    :param data: the initial data that generation 0 learns from
    :param interaction_order: the order in which agents take turns in interaction (can be set to either 'taking_turns'
    or 'random')
    :param ambiguity_penalty: parameter that determines the extent to which the speaker tries to avoid ambiguity;
    corresponds to global variable 'gamma'
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
    :param prob_of_noise: probability of noise (only relevant when noise_switch == True); corresponds to global variable
    'noise_prob'
    :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
    'all_forms_including_noisy_variants'
    :return: language_stats_over_gens (which contains language stats over generations over runs), data (which contains
    data over generations over runs), and the final population
    """
    language_stats_over_gens = np.zeros((n_gens, int(max(class_per_language)+1)))
    data_over_gens = []
    for i in range(n_gens):

        print('gen: '+str(i))

        for j in range(pop_size):
            for k in range(bottleneck):
                if interaction_order == 'taking_turns':
                    if len(population) != 2:
                        raise ValueError(
                            "OOPS! interaction = 'taking_turns' only works if popsize = 2.")
                    if bottleneck != len(data):
                        raise ValueError(
                            "UH-OH! data should have the same size as the bottleneck b")
                    meaning, signal = data[k]
                else:
                    meaning, signal = random.choice(data)
                population[j] = update_posterior_from_cache(population[j], log_likelihood_cache, meaning, signal, meanings, all_possible_forms)
        data = population_communication_mutual_understanding(population, n_rounds, ambiguity_penalty, effort_penalty, prob_of_noise, hypotheses, log_likelihood_cache)
        language_stats_over_gens[i] = language_stats(population, possible_form_lengths, class_per_language)
        data_over_gens.append(data)
        if i == n_gens-1:
            final_pop = population
        if turnover:
            population = new_population(pop_size, log_priors)
    return language_stats_over_gens, data_over_gens, final_pop


###################################################################################################################
if __name__ == '__main__':

    ###################################################################################################################
    # FIRST LET'S RETRIEVE THE RELEVANT LOG LIKELIHOOD CACHE:

    log_likelihood_cache = pickle.load(open("pickles/log_likelihood_cache_form_lengths_"+convert_array_to_string(possible_form_lengths)+"_noise_prob_"+convert_float_value_to_string(noise_prob)+"_gamma_"+convert_float_value_to_string(gamma)+"_delta_"+convert_float_value_to_string(delta)+"_error_"+convert_float_value_to_string(error)+".p", "rb"))
    print('')
    print("log_likelihood_cache.shape is:")
    print(log_likelihood_cache.shape)

    ###################################################################################################################
    # NOW LET'S RUN THE ACTUAL SIMULATION:

    t0 = time.process_time()

    hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
    print("number of possible languages is:")
    print(len(hypothesis_space))
    t1 = time.process_time()
    print('')
    print("number of minutes it took to create all languages (i.e. the hypothesis space):")
    print(round((t1-t0)/60., ndigits=2))

    class_per_lang = classify_all_languages(hypothesis_space, forms_without_noise, meanings)
    print("class_per_lang.shape is:")
    print(class_per_lang.shape)

    t2 = time.process_time()
    print('')
    print("number of minutes it took to classify languages:")
    print(round((t2-t1)/60., ndigits=2))

    if compressibility_bias:
        priors = prior(hypothesis_space, forms_without_noise, meanings)
    else:
        priors = np.ones(len(hypothesis_space))
        priors = np.divide(priors, np.sum(priors))
        priors = np.log(priors)

    t3 = time.process_time()
    print('')
    print("number of minutes it took to create prior:")
    print(round((t3-t2)/60., ndigits=2))

    initial_dataset = create_initial_dataset(initial_language_type, b, hypothesis_space, class_per_lang, meanings, possible_form_lengths)  # the data that the first generation learns from
    print('')
    print("initial_dataset is:")
    print(initial_dataset)

    language_stats_over_gens_per_run = np.zeros((runs, generations, int(max(class_per_lang)+1)))
    data_over_gens_per_run = []
    final_pop_per_run = np.zeros((runs, popsize, len(hypothesis_space)))
    for r in range(runs):

        print('')
        print("r is:")
        print(r)

        population = new_population(popsize, priors)

        language_stats_over_gens, data_over_gens, final_pop = simulation_repair_vs_redundancy(population, generations, rounds, b, popsize, hypothesis_space, log_likelihood_cache, class_per_lang, priors, initial_dataset, interaction, gamma, delta, noise_prob, all_forms_including_noisy_variants)

        language_stats_over_gens_per_run[r] = language_stats_over_gens
        data_over_gens_per_run.append(data_over_gens)
        final_pop_per_run[r] = final_pop

    timestr = time.strftime("%Y%m%d-%H%M%S")

    pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_form_lengths_"+convert_array_to_string(possible_form_lengths)+"_b_" + str(b) + "_rounds_" + str(rounds) + "_pop_size_" + str(popsize) + "_gamma_" + convert_float_value_to_string(gamma) + "_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type + "_noise_prob_" + convert_float_value_to_string(noise_prob) + "_" + timestr
    pickle.dump(language_stats_over_gens_per_run, open(pickle_file_path + pickle_file_name + "_lang_stats" + ".p", "wb"))
    pickle.dump(data_over_gens_per_run, open(pickle_file_path+pickle_file_name+"_data"+".p", "wb"))
    pickle.dump(final_pop_per_run, open(pickle_file_path + pickle_file_name + "_final_pop" + ".p", "wb"))

    t4 = time.process_time()

    print('')
    print("number of minutes it took to run simulation:")
    print(round((t4-t3)/60., ndigits=2))

    print('')
    print('results were saved in folder:')
    print(pickle_file_path)

    print('')
    print('using filename:')
    print(pickle_file_name)

