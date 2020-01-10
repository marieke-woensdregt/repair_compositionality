from evolution_compositionality_under_noise import *
import numpy as np
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
# FUNCTION DEFINITIONS:


def enumerate_all_possible_datapoints(meaning_list, all_possible_forms):
    """
    Enumerates all possible <meaning, form> pairs and saves them as tuples in a list

    :param meaning_list: list of strings corresponding to all possible meanings
    :param all_possible_forms: list of strings corresponding to all possible forms, including their noisy variants
    :return: list of <meaning, form> tuples
    """
    mf_pair_list = []
    for m in meaning_list:
        for f in all_possible_forms:
            mf_pair = (m, f)
            mf_pair_list.append(mf_pair)
    return mf_pair_list



def cache_likelihoods_per_datapoint(all_datapoints, hypotheses):

    likelihood_cache_matrix = np.zeros((len(all_datapoints), len(hypotheses)))
    print("likelihood_cache_matrix.shape is:")
    print(likelihood_cache_matrix.shape)
    for d in range(len(all_datapoints)):
        counter = 0
        print('')
        print("d is:")
        print(d)
        topic = all_datapoints[d][0]
        print("topic is:")
        print(topic)
        form = all_datapoints[d][1]
        print("form is:")
        print(form)
        for h in range(len(hypotheses)):
            print('')
            print("h is:")
            print(h)
            prop_to_prob_per_form_array = production_likelihoods_with_noise(hypotheses[h], topic, meanings,
                                                                             forms_without_noise, noisy_forms, gamma,
                                                                             error, noise_prob)
            print("len(prop_to_prob_per_form_array) is:")
            print(len(prop_to_prob_per_form_array))
            print("len(all_forms_including_noisy_variants) is:")
            print(len(all_forms_including_noisy_variants))
            print("range(d, d+len(prop_to_prob_per_form_array)) is:")
            print(range(d, d+len(prop_to_prob_per_form_array)))
            for i in range(d, d+len(prop_to_prob_per_form_array)):
                print('')
                print("i is:")
                print(i)
                likelihood_cache_matrix[i][h] = prop_to_prob_per_form_array[counter]
                counter += 1

    return likelihood_cache_matrix



def cache_likelihoods_per_datapoint_new(meaning_list, forms_including_noisy_variants, hypotheses):
    print('')
    print('')
    print("This is the cache_likelihoods_per_datapoint_new() function:")
    likelihood_cache_matrix = np.zeros((len(meaning_list), len(forms_including_noisy_variants), len(hypotheses)))
    print("likelihood_cache_matrix.shape is:")
    print(likelihood_cache_matrix.shape)
    for m in range(len(meaning_list)):
        for f in range(len(forms_including_noisy_variants)):
            print('')
            topic = meaning_list[m]
            print("topic is:")
            print(topic)
            form = forms_including_noisy_variants[f]
            print("form is:")
            print(form)
            for h in range(len(hypotheses)):
                print('')
                print("h is:")
                print(h)
                prop_to_prob_per_form_array = production_likelihoods_with_noise(hypotheses[h], topic, meanings,
                                                                                 forms_without_noise, noisy_forms, gamma,
                                                                                 error, noise_prob)
                print("len(prop_to_prob_per_form_array) is:")
                print(len(prop_to_prob_per_form_array))
                for i in range(len(prop_to_prob_per_form_array)):
                    likelihood_cache_matrix[m][i][h] = prop_to_prob_per_form_array[i]
        print('')
        print("likelihood_cache_matrix is:")
        print(likelihood_cache_matrix)
    return likelihood_cache_matrix



###################################################################################################################
# MEMOISING LIKELIHOODS FOR ALL POSSIBLE <MEANING, FORM> COMBINATIONS AND SAVING RESULTING MATRIX AS PICKLE FILE:


all_possible_datapoints = enumerate_all_possible_datapoints(meanings, all_forms_including_noisy_variants)
print('')
print('')
print("all_possible_datapoints is:")
print(all_possible_datapoints)
print("len(all_possible_datapoints) is:")
print(len(all_possible_datapoints))

#
# likelihood_cache = cache_likelihoods_per_datapoint(all_possible_datapoints, hypothesis_space[0:10])
# print('')
# print('')
# print("likelihood_cache is:")
# print(likelihood_cache)
# print('')
# print('')
# print("np.exp(likelihood_cache) is:")
# print(np.exp(likelihood_cache))
# print("likelihood_cache.shape is:")
# print(likelihood_cache.shape)



likelihood_cache = cache_likelihoods_per_datapoint_new(meanings, all_forms_including_noisy_variants, hypothesis_space[0:6])
print('')
print('')
print("likelihood_cache is:")
print(likelihood_cache)
print('')
print('')
print("np.exp(likelihood_cache) is:")
print(np.exp(likelihood_cache))
print("likelihood_cache.shape is:")
print(likelihood_cache.shape)

