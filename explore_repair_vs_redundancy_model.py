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