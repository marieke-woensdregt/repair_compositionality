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


likelihood_cache = pickle.load(open("pickles/likelihood_cache_noise_"+str(noise)+"_"+convert_float_value_to_string(noise_prob)+"_gamma_"+str(gamma)+"_error_"+convert_float_value_to_string(error)+".p", "rb"))
print('')
print("likelihood_cache.shape is:")
print(likelihood_cache.shape)




def receive_with_repair_open_only(language, utterance, mutual_understanding_pressure, minimal_effort_pressure):
    """
    Receives and utterance and gives a response, which can either be an interpretation or a repair initiator. How likely
    these two response types are to happen depends on the settings of the paremeters 'mutual_understanding' and
    'minimal_effort'. These three parameters are
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
