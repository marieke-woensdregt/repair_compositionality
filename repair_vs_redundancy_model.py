import pickle

from evolution_compositionality_under_noise import *

###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = create_all_possible_forms(2, [2, 4])  # all possible forms, excluding their 'noisy variants'
print('')
print('')
print("forms_without_noise are:")
print(forms_without_noise)
print("len(forms_without_noise) are:")
print(len(forms_without_noise))
noisy_forms = create_all_possible_noisy_forms(forms_without_noise)
print('')
print("noisy_forms are:")
print(noisy_forms)
print("len(noisy_forms) are:")
print(len(noisy_forms))
# all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise + noisy_forms  # all possible forms, including both complete
print('')
print("len(all_forms_including_noisy_variants) are:")
print(len(all_forms_including_noisy_variants))

hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
print('')
print('')
print("len(hypothesis_space) is:")
print(len(hypothesis_space))

mutual_understanding = True  # Setting the 'mutual_understanding' parameter based on the command-line input #NOTE:
# first argument in sys.argv list is always the name of the script
if mutual_understanding:
    gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
else:
    gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions

error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

noise_prob = 0.5  # the probability of environmental noise obscuring part of an utterance

###################################################################################################################




###################################################################################################################
# NOW SOME FUNCTIONS THAT HANDLE PRODUCTION, NOISY PRODUCTION, AND RECEPTION WITH AND WITHOUT REPAIR:


def production_likelihoods_with_noise(language, topic, meaning_list, forms, noisy_variants, ambiguity_penalty, error_prob, prob_of_noise):
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
    prop_to_prob_correct_form_complete = ((1./ambiguity) ** ambiguity_penalty) * (1. - error_prob) * (1 - prob_of_noise)
    prop_to_prob_error_form_complete = error_prob / (len(forms) - 1) * (1 - prob_of_noise)
    prop_to_prob_correct_form_noisy = ((1. / ambiguity) ** ambiguity_penalty) * (1. - error_prob) * (prob_of_noise / len(noisy_variants))
    prop_to_prob_error_form_noisy = error_prob / (len(forms) - 1) * (1 - prob_of_noise) * (prob_of_noise / len(noisy_variants))
    prop_to_prob_per_form_array = np.zeros(len(all_possible_forms))
    for i in range(len(all_possible_forms)):
        if all_possible_forms[i] == correct_form:
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form_complete
        elif all_possible_forms[i] in noisy_variants_correct_form:
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form_noisy
        elif all_possible_forms[i] in noisy_variants_error_forms:
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form_noisy
        else:
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form_complete
    return prop_to_prob_per_form_array


# And finally, let's write a function that actually produces an utterance, given a language and a topic:
def produce(language, topic, ambiguity_penalty, error_prob, noise_switch, prob_of_noise):
    """
    Produces an actual utterance, given a language and a topic

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param error_prob: the probability of making an error in production
    :param noise_switch: turns noise on when set to True, and off when set to False
    :param prob_of_noise: the probability of noise happening (only relevant when noise_switch is set to True);
    corresponds to global variable 'noise_prob'
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    if noise_switch:
        prop_to_prob_per_form_array = production_likelihoods_with_noise(language, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, error_prob, prob_of_noise)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_likelihoods_kirby_et_al(language, topic, meanings, ambiguity_penalty, error_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(forms_without_noise, p=prob_per_form_array)
    return utterance



#TODO: This has turned into a bit of a monster function. Maybe shorten it by pulling out the code that calculates the
# probabilities for the different response options and putting that in a separate function?
def receive_with_repair(language, utterance, mutual_understanding_pressure, minimal_effort_pressure):
    """
    Receives and utterance and gives a response, which can either be an interpretation or a repair initiator. How likely
    these two response types are to happen depends on the settings of the paremeters 'mutual_understanding' and
    'minimal_effort' (and, if minimal_effort is set to True, the parameter 'cost_vector'). These three parameters are
    all assumed to be global variables.

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param utterance: an utterance (string)
    :param mutual_understanding_pressure: determines whether the pressure for mutual understanding is switched on or off
    (i.e. set to True or False); corresponds to global variable 'mutual_understanding'
    :param minimal_effort_pressure: determines whether the pressure for minimal effort is switched on or off
    (i.e. set to True or False); corresponds to global variable 'minimal_effort'
    :return: a response, which can either be an interpretation (i.e. meaning) or a repair initiator. A repair initiator
    can be of two types: if the listener has grasped part of the meaning, it will be a restricted request, which is a
    string containing the partial meaning that the listener did grasp, followed by a question mark. If the listener did
    not grasp any of the meaning, it will be an open request, which is simply '??'
    """
    if not mutual_understanding_pressure and not minimal_effort_pressure:
        raise ValueError(
            "Sorry, this function has only been implemented for at least one of either mutual_understanding or minimal_"
            "effort being True")
    if '_' in utterance:
        compatible_forms = noisy_to_complete_forms(utterance, forms_without_noise)
        possible_interpretations = find_possible_interpretations(language, compatible_forms)
        if len(possible_interpretations) == 0:
            possible_interpretations = meanings
        partial_meaning = find_partial_meaning(language, utterance)
        if mutual_understanding_pressure and minimal_effort_pressure:
            prop_to_prob_no_repair = (1./len(possible_interpretations))-cost_vector[0]
            if len(partial_meaning) == 1:
                prop_to_prob_repair = (1.-(1./len(possible_interpretations)))-cost_vector[1]
                repair_initiator = str(partial_meaning[0])+'?'
            elif len(partial_meaning) == 0:
                prop_to_prob_repair = (1.-(1./len(possible_interpretations)))-cost_vector[2]
                repair_initiator = '??'
        elif mutual_understanding_pressure and not minimal_effort_pressure:
            if len(possible_interpretations) > 1:
                prop_to_prob_no_repair = 0.
                prop_to_prob_repair = 1.
                if len(partial_meaning) == 1:
                    repair_initiator = str(partial_meaning[0])+'?'
                elif len(partial_meaning) == 0:
                    repair_initiator = '??'
            elif len(possible_interpretations) == 1:
                prop_to_prob_no_repair = 1.
                prop_to_prob_repair = 0.
        elif not mutual_understanding_pressure and minimal_effort_pressure:
            prop_to_prob_no_repair = 1.
            prop_to_prob_repair = 0.
            if len(partial_meaning) == 1:
                repair_initiator = str(partial_meaning[0])+'?'
            elif len(partial_meaning) == 0:
                repair_initiator = '??'
        prop_to_prob_per_response = np.array([prop_to_prob_no_repair, prop_to_prob_repair])
        for i in range(len(prop_to_prob_per_response)):
            if prop_to_prob_per_response[i] < 0.0:
                prop_to_prob_per_response[i] = 0.0
        normalized_response_probs = np.divide(prop_to_prob_per_response, np.sum(prop_to_prob_per_response))
        selected_response = np.random.choice(np.arange(2), p=normalized_response_probs)
        if selected_response == 0:
            response = random.choice(possible_interpretations)
        elif selected_response == 1:
            response = repair_initiator
    else:
        response = receive_without_repair(language, utterance)
    return response



# AND NOW FOR THE FUNCTIONS THAT DO THE BAYESIAN LEARNING:

def update_posterior(log_posterior, hypotheses, topic, utterance, ambiguity_penalty, noise_switch, prob_of_noise, all_possible_forms):
    """
    Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability
    distribution accordingly

    :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
    :param hypotheses: list of all possible languages
    :param topic: a topic (string from the global variable meanings)
    :param utterance: an utterance (string from the global variable forms (can be a noisy form if parameter noise is
    True)
    :param ambiguity_penalty: parameter that determines extent to which speaker tries to avoid ambiguity; corresponds
    to global variable 'gamma'
    :param noise_switch: determines whether noise is on or off (set to either True or False); corresponds to global
    variable 'noise'
    :param prob_of_noise: the probability of noise (only relevant when noise_switch == True); corresponds to global
    variable 'noise_prob'
    :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
    'all_forms_including_noisy_variants'
    :return: the updated (and normalized) log_posterior (1D numpy array)
    """
    # First, let's find out what the index of the utterance is in the list of all possible forms (including the noisy
    # variants):
    for i in range(len(all_possible_forms)):
        if all_possible_forms[i] == utterance:
            utterance_index = i
    # Now, let's go through each hypothesis (i.e. language), and update its posterior probability given the
    # <topic, utterance> pair that was given as input:
    new_log_posterior = []
    for j in range(len(log_posterior)):
        hypothesis = hypotheses[j]
        if noise_switch:
            likelihood_per_form_array = production_likelihoods_with_noise(hypothesis, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, error, prob_of_noise)
        else:
            likelihood_per_form_array = production_likelihoods_kirby_et_al(hypothesis, topic, meanings, ambiguity_penalty, error)
        log_likelihood_per_form_array = np.log(likelihood_per_form_array)
        new_log_posterior.append(log_posterior[j] + log_likelihood_per_form_array[utterance_index])

    new_log_posterior_normalized = np.subtract(new_log_posterior, scipy.special.logsumexp(new_log_posterior))

    return new_log_posterior_normalized




# TODO: population_communication() has become a bit of a monster function; see if I can take some parts out to be
#  separate functions (e.g. for the different possible settings of the mutual_understanding and minimal_effort
#  parameters. Also, there has to be a way to not have exactly the same lines of code for doing the communicative
#  success pressure stuff in there twice (should probably be a separate function)!
def population_communication(population, n_rounds, mutual_understanding_pressure, minimal_effort_pressure, ambiguity_penalty, noise_switch, prob_of_noise, communicative_success_pressure, hypotheses):
    """
    Takes a population, makes it communicate for a number of rounds (where agents' posterior probability distribution
    is updated every time the agent gets assigned the role of hearer)

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :param n_rounds: the number of rounds for which the population should communicate; corresponds to global variable
    'rounds'
    :param mutual_understanding_pressure: turns mutual understanding on or off (set to either True or False);
    corresponds to global variable 'mutual_understanding'
    :param ambiguity_penalty: parameter which determines the extent to which the speaker tries to avoid ambiguity;
    corresponds to global variable 'gamma'
    :param noise_switch: determines whether noise is on or off (i.e. set to True or False); corresponds to global
    variable 'noise'
    :param prob_of_noise: the probability of noise happening (only relevant when noise_switch == True); corresponds to
    global variable 'noise_prob'
    :param communicative_success_pressure: determines whether pressure for communicative success is switched on or off
    (i.e. set to True or False); corresponds to global variable 'communicative_success'
    :param hypotheses: list of all possible languages; corresponds to global parameter 'hypothesis_space'
    :return: the data that was produced during the communication rounds, as a list of (topic, utterance) tuples
    """
    if n_parents == 'single':
        if len(population) != 2 or interaction != 'taking_turns':
            raise ValueError(
                "OOPS! n_parents = 'single' only works if popsize = 2 and interaction = 'taking_turns'.")
        random_parent_index = np.random.choice(np.arange(len(population)))
    data = []
    data_for_just_in_case = []
    for i in range(n_rounds):
        if interaction == 'taking_turns':
            if len(population) != 2:
                raise ValueError(
                "OOPS! interaction = 'taking_turns' only works if popsize = 2.")
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
        speaker_language = sample(hypotheses, population[speaker_index])
        hearer_language = sample(hypotheses, population[hearer_index])
        if mutual_understanding_pressure is True:
            if production == 'simlang':
                utterance = produce_simlang(speaker_language, topic)
            else:
                utterance = produce(speaker_language, topic, ambiguity_penalty, error, noise_switch, prob_of_noise)
                # whenever a speaker is called upon to produce a utterance, they first sample a language from their
                # posterior probability distribution. So each agent keeps updating their language according to the data
                # received from their communication partner.
            listener_response = receive_with_repair(hearer_language, utterance, mutual_understanding_pressure, minimal_effort_pressure)
            counter = 0
            while '?' in listener_response:
                if counter == 3:  # After 3 attempts, the listener stops trying to do repair
                    break
                if production == 'simlang':
                    utterance = produce_simlang(speaker_language, topic)
                else:
                    utterance = produce(speaker_language, topic, ambiguity_penalty, error, noise_switch=False, prob_of_noise=0.0)
                    # For now, we assume that the speaker's response to a repair initiator always comes through without
                    # noise.
                listener_response = receive_with_repair(hearer_language, utterance, mutual_understanding_pressure, minimal_effort_pressure)
                counter += 1
            if production == 'simlang':
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], hypotheses, topic,
                                                                    utterance)  # (Thus, in this simplified version of
                # the model, agents are still able to "track changes in their partners' linguistic behaviour over time
                elif observed_meaning == 'inferred':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], hypotheses, listener_response,
                                                                        utterance)  # (Thus, in this simplified version
                    # of the model, agents are still able to "track changes in their partners' linguistic behaviour over
                    # time
            else:
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior(population[hearer_index], hypotheses, topic, utterance, ambiguity_penalty, noise_switch, prob_of_noise, all_forms_including_noisy_variants)
                elif observed_meaning == 'inferred':
                    population[hearer_index] = update_posterior(population[hearer_index], hypotheses, listener_response, utterance, ambiguity_penalty, noise_switch, prob_of_noise, all_forms_including_noisy_variants)

        elif mutual_understanding_pressure is False:
            if production == 'simlang':
                utterance = produce_simlang(speaker_language, topic)
            else:
                utterance = produce(speaker_language, topic, ambiguity_penalty, error, noise_switch, prob_of_noise)
                # whenever a speaker is called upon to produce a utterance, they first sample a language from their
                # posterior probability distribution. So each agent keeps updating their language according to the data
                # they receive from their communication partner.
            if production == 'simlang':
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], hypotheses, topic, utterance)
                    # Thus, in this simplified version of the model, agents are still able to "track changes in their
                    # partners' linguistic behaviour over time
                elif observed_meaning == 'inferred':
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior_simlang(population[hearer_index], hypotheses, inferred_meaning,
                                                                        utterance)  # Thus, in this simplified version
                    # of the model, agents are still able to "track changes in their partners' linguistic behaviour over
                    # time
            else:
                if observed_meaning == 'intended':
                    population[hearer_index] = update_posterior(population[hearer_index], hypotheses, topic, utterance, ambiguity_penalty, noise_switch, prob_of_noise, all_forms_including_noisy_variants)
                elif observed_meaning == 'inferred':
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior(population[hearer_index], hypotheses, inferred_meaning, utterance, ambiguity_penalty, noise_switch, prob_of_noise, all_forms_including_noisy_variants)

        if n_parents == 'single':

            if speaker_index == random_parent_index:
                if observed_meaning == 'intended':
                    meaning_observed = topic
                    inferred_meaning = receive_without_repair(hearer_language, utterance)

                elif observed_meaning == 'inferred':
                    if mutual_understanding:
                        meaning_observed = listener_response
                    else:
                        meaning_observed = inferred_meaning

                if communicative_success_pressure is True:
                    if mutual_understanding:
                        inferred_meaning = listener_response
                    success = False
                    if inferred_meaning == topic:
                        success = True
                    random_float = np.random.uniform()
                    if random_float < communicative_success_pressure_strength:  # if our random_float falls within
                        # the range of our communicative_success_pressure_strength parameter, the <meaning, form>
                        # pair is added to the dataset only if the interaction was successful
                        if success:
                            data.append((meaning_observed, utterance))
                    else:  # if our random_float falls above the range of our
                        # communicative_success_pressure_strength parameter, the <meaning, form>
                        # pair is added to the dataset no matter whether the interaction was successful or not
                        data.append((meaning_observed, utterance))
                elif communicative_success_pressure is False:
                    data.append((meaning_observed, utterance))

                data_for_just_in_case.append((meaning_observed, utterance))  # this is being recorded just in case the
                # dataset otherwise ends up being empty, as a result of the pressure for communicative success being
                # too strong.


        elif n_parents == 'multiple':
            if observed_meaning == 'intended':
                meaning_observed = topic
                inferred_meaning = receive_without_repair(hearer_language, utterance)

            elif observed_meaning == 'inferred':
                if mutual_understanding:
                    meaning_observed = listener_response
                else:
                    meaning_observed = inferred_meaning

            if communicative_success_pressure is True:
                if mutual_understanding:
                    inferred_meaning = listener_response
                success = False
                if inferred_meaning == topic:
                    success = True
                random_float = np.random.uniform()
                if random_float < communicative_success_pressure_strength:  # if our random_float falls within
                    # the range of our communicative_success_pressure_strength parameter, the <meaning, form>
                    # pair is added to the dataset only if the interaction was successful
                    if success:
                        data.append((meaning_observed, utterance))
                else:  # if our random_float falls above the range of our
                    # communicative_success_pressure_strength parameter, the <meaning, form>
                    # pair is added to the dataset no matter whether the interaction was successful or not
                    data.append((meaning_observed, utterance))
            elif communicative_success_pressure is False:
                data.append((meaning_observed, utterance))

            data_for_just_in_case.append((meaning_observed, utterance))  # this is being recorded just in case the
                # dataset otherwise ends up being empty, as a result of the pressure for communicative success being
                # too strong.

    if len(data) == 0:  # In case the data set is empty (which might happen if the pressure for communicative success
        # is too strong; especially if the "n_parents" parameter has been set to 'single'), we just use all the
        # <meaning, form> pairs from all the interactions as the data set, no matter whether they were successful or
        # not.
        data = data_for_just_in_case

    # I ADDED THE BIT BELOW TO CHECK HOW OFTEN SELECTING FOR COMMUNICATIVE SUCCESS LEADS TO DATA SETS THAT ARE SMALLER
    # THAN THE BOTTLENECK:
    if communicative_success_pressure:
        if len(data) < b:
            print('')
            print('')
            print("UH-OH! len(data) < b!")
            print("len(data) is:")
            print(len(data))

    return data




# AND NOW FINALLY FOR THE FUNCTION THAT RUNS THE ACTUAL SIMULATION:

def simulation(population, n_gens, n_rounds, bottleneck, pop_size, hypotheses, class_per_language, log_priors, data, interaction_order, production_implementation, ambiguity_penalty, noise_switch, prob_of_noise, all_possible_forms, mutual_understanding_pressure, minimal_effort_pressure, communicative_success_pressure):
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
    :param class_per_language: list specifiying the class for each corresponding language in the variable 'hypotheses'
    :param log_priors: the LOG prior probability distribution that each agent should be initialised with
    :param data: the initial data that generation 0 learns from
    :param interaction_order: the order in which agents take turns in interaction (can be set to either 'taking_turns'
    or 'random')
    :param production_implementation: can be set to either 'my_code' or 'simlang'
    :param ambiguity_penalty: parameter that determines the extent to which the speaker tries to avoid ambiguity;
    corresponds to global variable 'gamma'
    :param noise_switch: determines whether noise is on or off (set to either True or False); corresponds to global
    variable 'noise'
    :param prob_of_noise: probability of noise (only relevant when noise_switch == True); corresponds to global variable
    'noise_prob'
    :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
    'all_forms_including_noisy_variants'
    :param mutual_understanding_pressure: determines whether the pressure for mutual understanding is switched on or off
    (set to either True or False); corresponds to global variable 'mutual_understanding'
    :param minimal_effort_pressure: determines whether the pressure for minimal effort is switched on or off
    (set to either True or False); corresponds to global variable 'minimal_effort'
    :param communicative_success_pressure: determines whether pressure for communicative success is turned on or off
    (i.e. set to True or False); corresponds to global variable 'communicative_succes'
    :return: language_stats_over_gens (which contains language stats over generations over runs), data (which contains
    data over generations over runs), and the final population
    """
    language_stats_over_gens = np.zeros((n_gens, int(max(class_per_language)+1)))
    data_over_gens = []
    for i in range(n_gens):
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
                if production_implementation == 'simlang':
                    population[j] = update_posterior_simlang(population[j], hypotheses, meaning, signal)
                else:
                    population[j] = update_posterior(population[j], hypotheses, meaning, signal, ambiguity_penalty, noise_switch, prob_of_noise, all_possible_forms)
        data = population_communication(population, n_rounds, mutual_understanding_pressure, minimal_effort_pressure, ambiguity_penalty, noise_switch, prob_of_noise, communicative_success_pressure, hypotheses)
        language_stats_over_gens[i] = language_stats(population, possible_form_lengths, class_per_language)
        data_over_gens.append(data)
        if i == n_gens-1:
            final_pop = population
        if turnover:
            population = new_population(pop_size, log_priors)
    return language_stats_over_gens, data_over_gens, final_pop



###################################################################################################################
if __name__ == '__main__':




    likelihood_cache = pickle.load(open(
        "pickles/likelihood_cache_noise_" + str(noise) + "_" + convert_float_value_to_string(
            noise_prob) + "_gamma_" + str(gamma) + "_error_" + convert_float_value_to_string(error) + ".p", "rb"))
    print('')
    print("likelihood_cache.shape is:")
    print(likelihood_cache.shape)





    ###################################################################################################################
    # NOW LET'S RUN THE ACTUAL SIMULATION:

    t0 = time.process_time()

    hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
    print("number of possible languages is:")
    print(len(hypothesis_space))

    t1 = time.process_time()
    print('')
    print("number of minutes it took to create all languages (i.e. the hypothesis space:")
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

    initial_dataset = create_initial_dataset(initial_language_type, b, hypothesis_space, class_per_lang, meanings)  # the data that the first generation learns from

    language_stats_over_gens_per_run = np.zeros((runs, generations, int(max(class_per_lang)+1)))
    data_over_gens_per_run = []
    final_pop_per_run = np.zeros((runs, popsize, len(hypothesis_space)))
    for r in range(runs):

        print('')
        print("r is:")
        print(r)

        population = new_population(popsize, priors)

        language_stats_over_gens, data_over_gens, final_pop = simulation(population, generations, rounds, b, popsize, hypothesis_space, class_per_lang, priors, initial_dataset, interaction, production, gamma, noise, noise_prob, all_forms_including_noisy_variants, mutual_understanding, minimal_effort, communicative_success)

        language_stats_over_gens_per_run[r] = language_stats_over_gens
        data_over_gens_per_run.append(data_over_gens)
        final_pop_per_run[r] = final_pop

    timestr = time.strftime("%Y%m%d-%H%M%S")

    pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(gamma) +"_minimal_e_" + str(minimal_effort) + "_c_" + convert_array_to_string(cost_vector) + "_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type[:5] + "_noise_" + str(noise) + "_" + convert_float_value_to_string(noise_prob) +"_observed_m_" + observed_meaning +"_n_l_classes_" + str(n_lang_classes) +"_CS_" + str(communicative_success) + "_" + convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2)) + "_" + timestr
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

