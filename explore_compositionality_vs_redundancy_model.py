from evolution_compositionality_under_noise import create_all_possible_forms, create_all_possible_noisy_forms, create_all_possible_languages


###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = create_all_possible_forms(2, [2, 4])  # all possible forms, excluding their possible 'noisy
# variants'
print('')
print('')
print("forms_without_noise are:")
print(forms_without_noise)
print("len(forms_without_noise) are:")
print(len(forms_without_noise))
noisy_forms = create_all_possible_noisy_forms(forms_without_noise)  # all possible noisy variants of the forms above
print('')
print('')
print("noisy_forms are:")
print(noisy_forms)
print("len((noisy_forms) are:")
print(len(noisy_forms))
all_forms_including_noisy_variants = forms_without_noise+noisy_forms  # all possible forms, including both complete
# forms and noisy variants




hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
print('')
print('')
print("number of possible languages is:")
print(len(hypothesis_space))
print("len(hypothesis_space)/256 is:")
print(len(hypothesis_space)/256)


L = ((2**30)-1)**8

print("L is:")
print(L)
print("{:.2e}".format(L))

import numpy as np

h_space_size = np.power((np.power(2, 30)-1), 8)
print(h_space_size)