import pickle

from evolution_compositionality_under_noise import *

###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
possible_form_lengths = [2, 4]  # all possible form lengths
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

production = 'my_code'  # can be set to 'simlang' or 'my_code'

cost_vector = np.array([0.0, 0.2, 0.4])  # costs of no repair, restricted request, and open request, respectively
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

noise = True  # parameter that determines whether environmental noise is on or off

communicative_success = False  # determines whether there is a pressure for communicative success or not
communicative_success_pressure_strength = (2./3.)  # determines how much more likely a <meaning, form> pair from a
# successful interaction is to enter the data set that is passed on to the next generation, compared to a
# <meaning, form> pair from a unsuccessful interaction.

burn_in = round(generations / 2)  # the burn-in period that is excluded when calculating the mean distribution over
# languages after convergence

n_lang_classes = 5  # the number of language classes that are distinguished (int). This should be 4 if the old code was
# used (from before 13 September 2019, 1:30 pm), which did not yet distinguish between 'holistic' and 'hybrid'
# languages, and 5 if the new code was used which does make this distinction.

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
    if minimal_effort:
        delta = 2  # parameter that determines strength of effort penalty (i.e. how strongly speaker tries to avoid
        # using long utterances)
    else:
        delta = 0  # parameter that determines strength of effort penalty (i.e. how strongly speaker tries to avoid
        # using long utterances)

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
def produce_with_minimal_effort(language, topic, ambiguity_penalty, effort_penalty, error_prob, noise_switch, prob_of_noise):
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
    :param noise_switch: turns noise on when set to True, and off when set to False
    :param prob_of_noise: the probability of noise happening (only relevant when noise_switch is set to True);
    corresponds to global variable 'noise_prob'
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise is False) or the global variable "all_forms_including_noisy_variants" (if noise is True).
        """
    if noise_switch:
        prop_to_prob_per_form_array = production_likelihoods_with_noise_and_minimal_effort(language, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, effort_penalty, error_prob, prob_of_noise)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_likelihoods_kirby_et_al(language, topic, meanings, ambiguity_penalty, error_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(forms_without_noise, p=prob_per_form_array)
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

def update_posterior_from_cache(log_posterior, likelihood_cache, hypotheses, topic, utterance, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, meaning_list, all_possible_forms):
    """
    Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability
    distribution accordingly

    :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
    :param likelihood_cache: 3D numpy array with axis 0 = meanings, axis 1 = all possible forms, and axis 2 = likelihood
    of corresponding <meaning, form> pair for each hypothesis
    :param hypotheses: list of all possible languages
    :param topic: a topic (string from the global variable meanings)
    :param utterance: an utterance (string from the global variable forms (can be a noisy form if parameter noise is
    True)
    :param ambiguity_penalty: parameter that determines extent to which speaker tries to avoid ambiguity; corresponds
    to global variable 'gamma'
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
    :param noise_switch: determines whether noise is on or off (set to either True or False); corresponds to global
    variable 'noise'
    :param prob_of_noise: the probability of noise (only relevant when noise_switch == True); corresponds to global
    variable 'noise_prob'
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

    # Now, let's go through each hypothesis (i.e. language), and update its posterior probability given the
    # <topic, utterance> pair that was given as input:
    new_log_posterior = []
    for j in range(len(log_posterior)):
        hypothesis = hypotheses[j]
        if noise_switch:
            likelihood_per_form_array = production_likelihoods_with_noise_and_minimal_effort(hypothesis, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, effort_penalty, error, prob_of_noise)
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
def population_communication(population, n_rounds, mutual_understanding_pressure, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, communicative_success_pressure, hypotheses):
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
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
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
                utterance = produce_with_minimal_effort(speaker_language, topic, ambiguity_penalty, effort_penalty, error, noise_switch, prob_of_noise)
                # whenever a speaker is called upon to produce a utterance, they first sample a language from their
                # posterior probability distribution. So each agent keeps updating their language according to the data
                # received from their communication partner.
            listener_response = receive_with_repair_open_only(hearer_language, utterance)
            counter = 0
            while '?' in listener_response:
                if counter == 3:  # After 3 attempts, the listener stops trying to do repair
                    break
                if production == 'simlang':
                    utterance = produce_simlang(speaker_language, topic)
                else:
                    utterance = produce_with_minimal_effort(speaker_language, topic, ambiguity_penalty, effort_penalty, error, noise_switch=False, prob_of_noise=0.0)
                    # For now, we assume that the speaker's response to a repair initiator always comes through without
                    # noise.
                listener_response = receive_with_repair_open_only(hearer_language, utterance)
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
                    population[hearer_index] = update_posterior_from_cache(population[hearer_index], likelihood_cache, hypotheses, topic, utterance, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, meanings, all_forms_including_noisy_variants)
                elif observed_meaning == 'inferred':
                    population[hearer_index] = update_posterior_from_cache(population[hearer_index], likelihood_cache, hypotheses, listener_response, utterance, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, meanings, all_forms_including_noisy_variants)

        elif mutual_understanding_pressure is False:
            if production == 'simlang':
                utterance = produce_simlang(speaker_language, topic)
            else:
                utterance = produce_with_minimal_effort(speaker_language, topic, ambiguity_penalty, effort_penalty, error, noise_switch, prob_of_noise)
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
                    population[hearer_index] = update_posterior_from_cache(population[hearer_index], likelihood_cache, hypotheses, topic, utterance, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, meanings, all_forms_including_noisy_variants)
                elif observed_meaning == 'inferred':
                    inferred_meaning = receive_without_repair(hearer_language, utterance)
                    population[hearer_index] = update_posterior_from_cache(population[hearer_index], likelihood_cache, hypotheses, inferred_meaning, utterance, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, meanings, all_forms_including_noisy_variants)

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

def simulation(population, n_gens, n_rounds, bottleneck, pop_size, hypotheses, class_per_language, log_priors, data, interaction_order, production_implementation, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, all_possible_forms, mutual_understanding_pressure, communicative_success_pressure):
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
    :param effort_penalty: parameter that determines the strength of the penalty on speaker effort (i.e. utterance
    length)
    :param noise_switch: determines whether noise is on or off (set to either True or False); corresponds to global
    variable 'noise'
    :param prob_of_noise: probability of noise (only relevant when noise_switch == True); corresponds to global variable
    'noise_prob'
    :param all_possible_forms: list of all possible forms INCLUDING noisy variants; corresponds to global variable
    'all_forms_including_noisy_variants'
    :param mutual_understanding_pressure: determines whether the pressure for mutual understanding is switched on or off
    (set to either True or False); corresponds to global variable 'mutual_understanding'
    :param communicative_success_pressure: determines whether pressure for communicative success is turned on or off
    (i.e. set to True or False); corresponds to global variable 'communicative_succes'
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
                if production_implementation == 'simlang':
                    population[j] = update_posterior_simlang(population[j], hypotheses, meaning, signal)
                else:
                    population[j] = update_posterior_from_cache(population[j], likelihood_cache, hypotheses, meaning, signal, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, meanings, all_possible_forms)
        data = population_communication(population, n_rounds, mutual_understanding_pressure, ambiguity_penalty, effort_penalty, noise_switch, prob_of_noise, communicative_success_pressure, hypotheses)
        language_stats_over_gens[i] = language_stats(population, possible_form_lengths, class_per_language)
        data_over_gens.append(data)
        if i == n_gens-1:
            final_pop = population
        if turnover:
            population = new_population(pop_size, log_priors)
    return language_stats_over_gens, data_over_gens, final_pop



###################################################################################################################
if __name__ == '__main__':




    likelihood_cache = pickle.load(open("pickles/likelihood_cache_noise_prob_"+convert_float_value_to_string(noise_prob)+"_gamma_"+str(gamma)+"_delta_"+str(delta)+"_error_"+convert_float_value_to_string(error)+".p", "rb"))
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

        language_stats_over_gens, data_over_gens, final_pop = simulation(population, generations, rounds, b, popsize, hypothesis_space, class_per_lang, priors, initial_dataset, interaction, production, gamma, delta, noise, noise_prob, all_forms_including_noisy_variants, mutual_understanding, communicative_success)

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

