from evolution_compositionality_under_noise import *



###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
possible_form_lengths = np.array([2])  # all possible form lengths
forms = create_all_possible_forms(2, possible_form_lengths)  # all possible forms


###################################################################################################################
if __name__ == '__main__':

    print('')
    print("meanings are:")
    print(meanings)
    print('')
    print("forms are:")
    print(forms)

    hypothesis_space = create_all_possible_languages(meanings, forms)
    print('')
    print("number of possible languages is:")
    print(len(hypothesis_space))

    # COPIED FROM SIMLANG LAB 21 (2019):
    languages_simlang = [[('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')],
                         [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')]]

    types_simlang = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2,
                     2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                     2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

    priors_simlang = [-0.9178860550328204, -10.749415928290118, -10.749415928290118, -11.272664072079987,
                      -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075,
                      -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075,
                      -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987,
                      -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075,
                      -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075,
                      -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543,
                      -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075,
                      -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075,
                      -16.95425710594061, -16.95425710594061, -16.95425710594061, -12.460704095246543,
                      -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075,
                      -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075,
                      -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987,
                      -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075,
                      -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075,
                      -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987,
                      -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075,
                      -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075,
                      -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749,
                      -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075,
                      -10.749415928290118, -10.749415928290118, -16.95425710594061, -17.294055179550075,
                      -10.749415928290118, -2.304180416152711, -11.272664072079987, -10.749415928290118,
                      -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061,
                      -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118,
                      -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749,
                      -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061,
                      -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061,
                      -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061,
                      -17.294055179550075, -17.294055179550075, -12.460704095246543, -17.294055179550075,
                      -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118,
                      -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061,
                      -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118,
                      -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075,
                      -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749,
                      -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075,
                      -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075,
                      -16.95425710594061, -16.95425710594061, -16.95425710594061, -20.83821243446749,
                      -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061,
                      -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061,
                      -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061,
                      -10.749415928290118, -16.95425710594061, -10.749415928290118, -17.294055179550075,
                      -16.95425710594061, -11.272664072079987, -11.272664072079987, -16.95425710594061,
                      -10.749415928290118, -11.272664072079987, -2.304180416152711, -10.749415928290118,
                      -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118,
                      -17.294055179550075, -12.460704095246543, -17.294055179550075, -17.294055179550075,
                      -20.83821243446749, -16.95425710594061, -16.95425710594061, -16.95425710594061,
                      -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118,
                      -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118,
                      -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987,
                      -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075,
                      -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075,
                      -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987,
                      -17.294055179550075, -17.294055179550075, -20.83821243446749, -17.294055179550075,
                      -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118,
                      -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061,
                      -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118,
                      -17.294055179550075, -20.83821243446749, -17.294055179550075, -17.294055179550075,
                      -12.460704095246543, -16.95425710594061, -16.95425710594061, -16.95425710594061,
                      -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118,
                      -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118,
                      -11.272664072079987, -17.294055179550075, -17.294055179550075, -11.272664072079987,
                      -17.294055179550075, -10.749415928290118, -16.95425710594061, -10.749415928290118,
                      -17.294055179550075, -16.95425710594061, -10.749415928290118, -10.749415928290118,
                      -11.272664072079987, -10.749415928290118, -10.749415928290118, -0.9178860550328204]

    ###################################################################################################################

    # FIRSTLY, Let's check whether the functions in this cell work correctly by comparing the number of languages of
    # each type we get with the SimLang lab 21:

    types_simlang = np.array(types_simlang)
    no_of_each_type = np.bincount(types_simlang)
    print('')
    print("no_of_each_type ACCORDING TO SIMLANG CODE, where 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional is:")
    print(no_of_each_type)

    class_per_lang = classify_all_languages(hypothesis_space, forms, meanings)
    no_of_each_class = np.bincount(class_per_lang.astype(int))
    no_of_each_class_simlang_order = np.array([no_of_each_class[0], no_of_each_class[1], no_of_each_class[4], no_of_each_class[2]+no_of_each_class[3]])
    print('')
    print("no_of_each_class_simlang_order ACCORDING TO MY CODE, where 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional is:")
    print(no_of_each_class_simlang_order)

    # Now, let's check that the lanuages that are classified as holistic and compositional are the same in my code as
    # they are in the simlang code:

    # First, we need to change the way we represent the languages to match that of the simlang code:
    all_langs_as_in_simlang = transform_all_languages_to_simlang_format(hypothesis_space, meanings)
    print('')
    print('')
    print(np.array(all_langs_as_in_simlang).shape)

    # Then, let's check whether the resulting list is the same (and in the same order) as the list of languages
    # provided in SimLang Lab 21 (if the order is already the same, this would suggest that Kirby et al. (2015) also
    # used the itertools.product() function (or an equivalent procedure) to produce it.
    languages_same = np.array_equal(np.array(all_langs_as_in_simlang), np.array(languages_simlang))
    print('')
    print("languages_same is:")
    print(languages_same)

    # Ok, this shows that the list of languages generated by my code is exactly the same as the list of languages
    # provided in SimLang Lab 21 (except for the difference in representation format). Now, let's have a look at which
    # languages are classified as holistic and compositional

    # First, let's have a look at which languages it classifies as holistic:

    holistic_langs_indices_my_code = np.where(class_per_lang==1)[0]
    print('')
    print('')
    print("holistic_langs_indices_my_code MY CODE are:")
    print(holistic_langs_indices_my_code)
    print("len(holistic_langs_indices_my_code) MY CODE are:")
    print(len(holistic_langs_indices_my_code))

    # And now let's do the same for the languages from SimLang Lab 21:

    holistic_langs_indices_simlang = np.where(np.array(types_simlang)==1)[0]
    print('')
    print("holistic_langs_indices_simlang SIMLANG CODE are:")
    print(holistic_langs_indices_simlang)
    print("len(holistic_langs_indices_simlang) SIMLANG CODE are:")
    print(len(holistic_langs_indices_simlang))

    same_holistic_indices = np.array_equal(holistic_langs_indices_my_code, holistic_langs_indices_simlang)
    print('')
    print("same_holistic_indices is:")
    print(same_holistic_indices)

    holistic_langs_my_code = np.array(hypothesis_space)[holistic_langs_indices_my_code]
    print('')
    print('')
    print("holistic_langs MY CODE are:")
    print(holistic_langs_my_code)

    holistic_langs_simlang_code = np.array(hypothesis_space)[holistic_langs_indices_simlang]
    print('')
    print('')
    print("holistic_langs SIMLANG CODE are:")
    print(holistic_langs_simlang_code)

    # Just to double-check...
    same_holistic_languages = np.array_equal(holistic_langs_my_code, holistic_langs_simlang_code)
    print('')
    print("same_holistic_languages is:")
    print(same_holistic_languages)

    # Now, let's have a look at which languages are classified as compositional:

    compositional_langs_indices_my_code = np.sort(np.concatenate((np.where(class_per_lang==2)[0], np.where(class_per_lang==3)[0])))
    print('')
    print('')
    print('')
    print('')
    print("compositional_langs_indices_my_code MY CODE are:")
    print(compositional_langs_indices_my_code)
    print("len(compositional_langs_indices_my_code) MY CODE are:")
    print(len(compositional_langs_indices_my_code))

    # And the same for the languages from SimLang Lab 21:

    compositional_langs_indices_simlang = np.where(np.array(types_simlang)==3)[0]
    print('')
    print("compositional_langs_indices_simlang SIMLANG CODE are:")
    print(compositional_langs_indices_simlang)
    print("len(compositional_langs_indices_simlang) SIMLANG CODE are:")
    print(len(compositional_langs_indices_simlang))

    same_compositional_indices = np.array_equal(compositional_langs_indices_my_code, compositional_langs_indices_simlang)
    print('')
    print("same_compositional_indices is:")
    print(same_compositional_indices)

    compositional_langs_my_code = np.array(hypothesis_space)[compositional_langs_indices_my_code]
    print('')
    print('')
    print("compositional_langs MY CODE are:")
    print(compositional_langs_my_code)

    compositional_langs_simlang_code = np.array(hypothesis_space)[compositional_langs_indices_simlang]
    print('')
    print('')
    print("compositional_langs SIMLANG CODE are:")
    print(compositional_langs_simlang_code)

    # Just to double-check...
    same_compositional_languages = np.array_equal(compositional_langs_my_code, compositional_langs_simlang_code)
    print('')
    print("same_compositional_languages is:")
    print(same_compositional_languages)

    ###################################################################################################################
    # SECONDLY, LET'S CHECK WHETHER MY FUNCTIONS FOR GENERATING/CALCULATING THE LANGAUGES' REWRITE RULES, MINIMALLY
    # REDUNDANT FORMS, AND CODING LENGTHS GIVE THE SAME RESULTS AS SHOWN FOR THE EXAMPLE LANGUAGES IN Kirby et al.
    # (2015) (in the table on p. 92):

    # First, the example languages from Kirby et al. :
    example_languages = [['aa', 'aa', 'aa', 'aa'],
                         ['ab', 'ab', 'ab', 'ab'],
                         ['aa', 'aa', 'aa', 'ab'],
                         ['aa', 'aa', 'aa', 'bb'],
                         ['aa', 'ab', 'ba', 'bb'],
                         ['aa', 'aa', 'ab', 'ba'],
                         ['aa', 'aa', 'ab', 'bb'],
                         ['aa', 'ab', 'bb', 'ba']]

    # And now let's calculate their coding lengths:
    lang_classes_text = ['degenerate', 'holistic', 'compositional', 'compositional_reverse', 'other']
    print('')
    print('')
    print('')
    print('Checking coding lengths of example languages from Kirby et al. (2015), table on p. 92:')
    for i in range(len(example_languages)):
        lang = example_languages[i]
        print('')
        print(i)
        lang_class = classify_language_four_forms(lang, forms, meanings)
        lang_class_text = lang_classes_text[lang_class]
        print("lang_class_text is:")
        print(lang_class_text)
        print("lang is:")
        print(lang)
        mrf_string = minimally_redundant_form_four_forms(lang, forms, meanings)
        print("mrf_string is:")
        print(mrf_string)
        coding_len = coding_length(mrf_string)
        print("coding_len is:")
        print(round(coding_len, ndigits=2))
        lang_prior = prior_single_lang(lang, forms, meanings)
        print("prior this lang is:")
        print(lang_prior)

    print('')
    print('')
    print('')
    print('PRIORS FOR COMPOSITIONAL LANGUAGES ACCORDING TO SIMLANG CODE:')
    for i in range(len(compositional_langs_simlang_code)):
        lang = compositional_langs_simlang_code[i]
        print('')
        print(i)
        lang_class = classify_language_four_forms(lang, forms, meanings)
        lang_class_text = lang_classes_text[lang_class]
        print("lang_class_text is:")
        print(lang_class_text)
        print("lang is:")
        print(lang)
        mrf_string = minimally_redundant_form_four_forms(lang, forms, meanings)
        print("mrf_string is:")
        print(mrf_string)
        coding_len = coding_length(mrf_string)
        print("coding_len is:")
        print(round(coding_len, ndigits=2))
        lang_prior = prior_single_lang(lang, forms, meanings)
        print("prior this lang is:")
        print(lang_prior)

    ###################################################################################################################
    # FINALLY, LET'S CHECK MY PRIOR PROBABILITY DISTRIBUTION AGAINST THE SIMLANG ONE:

    my_log_prior = prior(hypothesis_space, forms, meanings)
    print('')
    print('')
    print("my_log_prior is:")
    print(list(my_log_prior))
    print("my_log_prior.shape is:")
    print(my_log_prior.shape)
    print("np.exp(scipy.special.logsumexp(my_log_prior)) is:")
    print(np.exp(scipy.special.logsumexp(my_log_prior)))

    print('')
    # print("np.array(priors_simlang) is:")
    # print(np.array(priors_simlang))
    print("np.array(priors_simlang).shape is:")
    print(np.array(priors_simlang).shape)
    print("np.exp(scipy.special.logsumexp(priors_simlang)) is:")
    print(np.exp(scipy.special.logsumexp(priors_simlang)))

    log_prior_checks_per_lang = np.zeros(len(my_log_prior))
    prob_prior_checks_per_lang = np.zeros(len(my_log_prior))
    for i in range(len(my_log_prior)):
        if np.round(my_log_prior[i], decimals=3) == np.round(priors_simlang[i], decimals=3):
            log_prior_checks_per_lang[i] = 1.
        if np.round(np.exp(my_log_prior[i]), decimals=4) == np.round(np.exp(priors_simlang[i]), decimals=4):
            prob_prior_checks_per_lang[i] = 1.

    print('')
    print('')
    print("log_prior_checks_per_lang is:")
    print(log_prior_checks_per_lang)
    print("np.sum(log_prior_checks_per_lang) is:")
    print(np.sum(log_prior_checks_per_lang))
    print('')
    print("prob_prior_checks_per_lang is:")
    print(prob_prior_checks_per_lang)
    print("np.sum(prob_prior_checks_per_lang) is:")
    print(np.sum(prob_prior_checks_per_lang))

    # Hmm, we can see that there are some languages for which these log_prior values aren't exactly the same.
    # Let's see which ones they are and how large the differences are:

    diff_value_indices = np.argwhere(log_prior_checks_per_lang==0)
    diff_value_indices_flattened = diff_value_indices.flatten()
    print('')
    print('')
    print("diff_value_indices_flattened are:")
    print(diff_value_indices_flattened)

    # --> Hey, what's striking here is that this list of indices is EXACTLY the same as the list of indices of
    # languages that are classified as compositional. Let's have a closer look at these languages and the
    # priors that are assigned to them:

    # (Also note that if numbers in the log_prior and probability version of the prior are allowed to have more decimals
    # (than 3 and 4 respectively) the numbers start to diverge for more than just these 8 languages.)

    print('')
    print('')
    for index in diff_value_indices_flattened:
        print('')
        print("index is:")
        print(index)
        print("hypothesis_space[index] is:")
        print(hypothesis_space[index])
        print("my_log_prior[index] is:")
        print(my_log_prior[index])
        print("priors_simlang[index] is:")
        print(priors_simlang[index])

    # My own code consistently assigns slightly lower prior probabilities to all compositional languages compared to
    # the simlang code. This is in accordance with the fact that my code consistently assigns slightly longer coding
    # length to those same languages.

    log_priors_my_code = np.array([my_log_prior[i] for i in diff_value_indices_flattened])
    print('')
    print('')
    # print("log_priors_my_code is:")
    # print(log_priors_my_code)

    log_priors_simlang_code = np.array([priors_simlang[i] for i in diff_value_indices_flattened])
    print('')
    # print("log_priors_simlang_code is:")
    # print(log_priors_simlang_code)

    sum_my_code = scipy.special.logsumexp(log_priors_my_code)
    print('')
    print("np.exp(sum_my_code) is:")
    print(np.exp(sum_my_code))

    sum_simlang_code = scipy.special.logsumexp(log_priors_simlang_code)
    print('')
    print("np.exp(sum_simlang_code) is:")
    print(np.exp(sum_simlang_code))

    difference_selected_langs = np.exp(sum_simlang_code)-np.exp(sum_my_code)
    print('')
    print("difference_selected_langs is:")
    print(difference_selected_langs)

    my_log_prior_masked = my_log_prior
    my_log_prior_masked[diff_value_indices_flattened] = 0
    priors_simlang_masked = np.array(priors_simlang)
    priors_simlang_masked[diff_value_indices_flattened] = 0
    difference = np.subtract(np.exp(my_log_prior_masked), np.exp(priors_simlang_masked))
    print('')
    print('')
    # print("difference is:")
    # print(difference)
    print("np.sum(difference) is:")
    print(np.sum(difference))

    # Yep, the resulting difference values are pretty much the same.
