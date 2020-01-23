from evolution_compositionality_under_noise import *
from repair_vs_redundancy_model import production_likelihoods_with_noise_and_minimal_effort
import numpy as np
import time
import pickle


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

mutual_understanding = False  # Setting the 'mutual_understanding' parameter based on the command-line input #NOTE:
# first argument in sys.argv list is always the name of the script
if mutual_understanding:
    gamma = 2  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions
else:
    gamma = 0  # parameter that determines strength of ambiguity penalty (Kirby et al., 2015 used gamma = 0 for
    # "Learnability Only" condition, and gamma = 2 for both "Expressivity Only", and "Learnability and Expressivity"
    # conditions

minimal_effort = False
if minimal_effort:
    delta = 2  # parameter that determines strength of effort penalty (i.e. how strongly speaker tries to avoid
    # using long utterances)
else:
    delta = 0  # parameter that determines strength of effort penalty (i.e. how strongly speaker tries to avoid
    # using long utterances)

error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

noise_prob = 0.5  # the probability of environmental noise obscuring part of an utterance

###################################################################################################################


###################################################################################################################
# FUNCTION DEFINITIONS:



def cache_likelihoods_per_datapoint(meaning_list, forms_including_noisy_variants, hypotheses):
    """
    Calculates the likelihood of every possible datapoint (i.e. <meaning, form> pair) for every hypothesis, and saves
    them all in one big 3D numpy matrix.

    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :param forms_including_noisy_variants: list of all possible forms INCLUDING noisy variants; corresponds to global
    variable 'all_forms_including_noisy_variants'
    :param hypotheses: list of all possible languages; corresponds to global parameter 'hypothesis_space'
    :return: 3D numpy matrix with axis 0 = meanings, axis 1 = forms (incl. noisy variants), and axis 2 = hypotheses
    """
    likelihood_cache_matrix = np.zeros((len(meaning_list), len(forms_including_noisy_variants), len(hypotheses)))
    for m in range(len(meaning_list)):
        topic = meaning_list[m]
        for h in range(len(hypotheses)):
            prop_to_prob_per_form_array = production_likelihoods_with_noise_and_minimal_effort(hypotheses[h], topic, meanings,
                                                                             forms_without_noise, noisy_forms, gamma, delta,
                                                                             error, noise_prob)
            for f in range(len(prop_to_prob_per_form_array)):
                likelihood_cache_matrix[m][f][h] = prop_to_prob_per_form_array[f]
    return likelihood_cache_matrix



###################################################################################################################
# MEMOISING LIKELIHOODS FOR ALL POSSIBLE <MEANING, FORM> COMBINATIONS AND SAVING RESULTING MATRIX AS PICKLE FILE:

t0 = time.process_time()

likelihood_cache = cache_likelihoods_per_datapoint(meanings, all_forms_including_noisy_variants, hypothesis_space)

t1 = time.process_time()

print('')
print('')
print('min. it took to fill out likelihood cache:')
print((t1-t0)/60)

print('')
print('')
# print("likelihood_cache is:")
# print(likelihood_cache)
print('')
print('')
print("np.exp(likelihood_cache) is:")
print(np.exp(likelihood_cache))
print("likelihood_cache.shape is:")
print(likelihood_cache.shape)


t2 = time.process_time()

pickle.dump(likelihood_cache, open("pickles/likelihood_cache_noise_"+str(noise)+"_"+convert_float_value_to_string(noise_prob)+"_gamma_"+str(gamma)+"_delta_"+str(delta)+"_error_"+convert_float_value_to_string(error)+".p", "wb"))

t3 = time.process_time()
print('')
print('')
print('min. it took to pickle likelihood cache:')
print((t3-t2)/60)