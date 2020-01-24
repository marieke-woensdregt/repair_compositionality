import string
import itertools
import numpy as np
from math import log2
from scipy.special import logsumexp


def create_all_possible_forms(n_characters, form_length_list):
    """
    Takes a number of characters and a list of allowed form lengths, and creates a list of all possible complete forms
    based on that.

    :param n_characters: the number of different characters that may be used
    :param form_length_list: a list of all form lengths that are allowed
    :return: a list of all possible complete forms
    """
    alphabet = string.ascii_lowercase
    form_alphabet = alphabet[:n_characters]
    all_forms = []
    for length in form_length_list:
        all_forms = all_forms+list(itertools.product(form_alphabet, repeat=length))
    all_forms_as_strings = []
    for form in all_forms:
        string_form = ''
        for i in range(len(form)):
            string_form = string_form+form[i]
        all_forms_as_strings.append(string_form)
    return all_forms_as_strings

###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
possible_form_lengths = np.array([2])  # all possible form lengths
forms = create_all_possible_forms(2, possible_form_lengths)  # all possible forms, excluding their possible
# 'noisy variants'
print('')
print('')
print("forms are:")
print(forms)

###################################################################################################################

def create_all_possible_languages(meaning_list, forms):
    """Creates all possible languages

    :param meaning_list: list of strings corresponding to all possible meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :returns: list of tuples which represent languages, where each tuple consists of forms_without_noisy_variants and
    has length len(meanings)
    """
    all_possible_languages = list(itertools.product(forms, repeat=len(meaning_list)))
    return all_possible_languages


# In case it's relevant for checking my implementation against the simlang one, just as a sanity check:
def transform_all_languages_to_simlang_format(language_list, meaning_list):
    """
    Takes a list of languages as represented by me (with only the forms_without_noisy_variants listed
    for each language, assuming the meaning for each form is specified by the
    form's index), and turning it into a list of languages as represented in
    SimLang lab 21 (which in turn is based on Kirby et al., 2015), in which a
    <meaning, form> pair forms_without_noisy_variants a tuple, and four of those tuples in a list form
    a language

    :param language_list: list of all languages
    :param meaning_list: list of all possible meanings; corresponds to global variable 'meanings'
    :returns: list of the input languages in the format of SimLang lab 21
    """
    all_langs_as_in_simlang = []
    for l in range(len(language_list)):
        lang_as_in_simlang = [(meaning_list[x], language_list[l][x]) for x in range(len(meaning_list))]
        all_langs_as_in_simlang.append(lang_as_in_simlang)
    return all_langs_as_in_simlang


def check_all_forms_unique(lang):
    """
    Takes a language and checks whether each form used by the language is unique; i.e. whether the language is fully
    expressive.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :return: True if all forms in the language are unique, False otherwise.
    """
    form_counts = np.unique(np.array(lang), return_counts=True)
    if np.any(form_counts[1] != 1):  # if any form occurs more than once in the language, all_forms_unique = False
        return False
    else:
        return True


def classify_language_four_forms_debugged(lang, forms, meaning_list):
    """
    Classify one particular language as either 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional,
    4 = compositional_reverse, and 5 = other (Kirby et al., 2015). NOTE that this function is specific to classifying
    languages that consist of exactly 4 forms, where each form consists of exactly 2 characters. For a more general
    version of this function, see classify_language_general() below.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :param meaning_list: list of strings corresponding to all possible meanings
    :returns: integer corresponding to category that language belongs to:
    0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = compositional_reverse, and 5 = other
    (Kirby et al., 2015).
    """
    class_degenerate = 0
    class_holistic = 1
    class_hybrid = 2  # this is a hybrid between a holistic and a compositional language; where *half* of the partial
    # forms is mapped consistently to partial meanings (instead of that being the case for *all* partial forms)
    class_compositional = 3
    class_compositional_reverse = 4
    class_other = 5

    # First check whether some conditions are met, bc this function hasn't been coded up in the most general way yet:
    if len(forms) != 4:
        raise ValueError(
            "This function only works for a world in which there are 4 possible forms_without_noisy_variants"
        )
    if len(forms[0]) != 2:
        raise ValueError(
            "This function only works when each form consists of 2 elements")
    if len(lang) != len(meaning_list):
        raise ValueError("Lang should have same length as meanings")

    # lang is degenerate if it uses the same form for every meaning:
    if lang[0] == lang[1] and lang[1] == lang[2] and lang[2] == lang[3]:
        return class_degenerate

    # If each form is unique, the language is either COMPOSITIONAL or HOLISTIC:
    all_forms_unique = check_all_forms_unique(lang)
    if all_forms_unique is True:
        # lang is compositional if each form element maps to the same meaning element for each form:
        if lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0] and lang[0][1] == lang[2][1] and lang[1][1] == lang[3][1]:
            return class_compositional
        elif lang[0][0] == lang[2][0] and lang[1][0] == lang[3][0] and lang[0][1] == lang[1][1] and lang[2][1] == lang[3][1]:
            return class_compositional_reverse

        # lang is holistic if it is *not* compositional, but *does* make use of all possible forms_without_noisy_variants:
        else:
            # within holistic languages, we can distinguish between those in which at least one part form is mapped
            # consistently onto one part meaning. This class we will call 'hybrid' (because for the purposes of repair, it
            # is a hybrid between a holistic and a compositional language, because for half of the possible noisy forms that
            # a listener could receive it allows the listener to figure out *part* of the meaning, and therefore use a
            # restricted request for repair instead of an open request.
            if lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0]:
                return class_hybrid
            elif lang[0][1] == lang[2][1] and lang[1][1] == lang[3][1]:
                return class_hybrid
            elif lang[0][0] == lang[2][0] and lang[1][0] == lang[3][0]:
                return class_hybrid
            elif lang[0][1] == lang[1][1] and lang[2][1] == lang[3][1]:
                return class_hybrid
            else:
                return class_holistic

    # In all other cases, a language belongs to the 'other' category:
    else:
        return class_other


def classify_all_languages_debugged(language_list, complete_forms, meaning_list):
    """
    Classify all languages as either 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional,
    4 = compositional_reverse, and 5 = other (Kirby et al., 2015).

    :param language_list: list of all languages
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meanings: list of all possible meanings; corresponds to global variable 'meanings'
    :returns: 1D numpy array containing integer corresponding to category of corresponding
    language index as hardcoded in classify_language function: 0 = degenerate, 1 = holistic, 2 = hybrid,
    3 = compositional, 4 = compositional_reverse, and 5 = other (Kirby et al., 2015).
    """
    class_per_lang = np.zeros(len(language_list))
    for l in range(len(language_list)):
        class_per_lang[l] = classify_language_four_forms_debugged(language_list[l], complete_forms, meaning_list)
    return class_per_lang

###################################################################################################################
# FIRST, LET'S DEFINE SOME FUNCTIONS TO CHECK MY CODE FOR CREATING AND CLASSIFYING ALL LANGUAGES AGAINST THE LISTS OF
# LANGUAGES AND TYPES THAT WERE COPIED INTO LAB 21 OF THE SIMLANG COURSE 2019:


def check_language_lists_same_order(languages_my_code, languages_simlang_code):
    """
    Simply checks whether each element of two lists is the same. Returns True if so, and False otherwise.

    :param languages_my_code: languages generated by my own code, but reformatted into the simlang representation format
    :param languages_simlang_code: languages as provided in SimLang lab 21
    :return: True if the two lists are exactly the same (and in the same order), and False otherwise.
    """
    same_order = True
    for l in range(len(languages_my_code)):
        if languages_my_code[l] != languages_simlang_code[l]:
            same_order = False
    return same_order



###################################################################################################################
# THEN SOME FUNCTIONS FOR CALCULATING THE SIMPLICITY PRIOR (BASED ON THE COMPRESSIBILITY OF THE LANGUAGES):

def mrf_degenerate(lang, meaning_list):
    """
    Takes a degenerate language and returns a minimally redundant form description of the language's context free
    grammar.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param meaning_list: list of strings corresponding to all possible meanings
    :return: minimally redundant form description of the language's context free grammar (string)
    """
    mrf_string = 'S'
    for i in range(len(meaning_list)):
        meaning = meaning_list[i]
        if i != len(meaning_list) - 1:
            mrf_string += str(meaning) + ','
        else:
            mrf_string += str(meaning)
    mrf_string += lang[0]
    return mrf_string


def mrf_holistic(lang, meaning_list):
    """
    Takes a holistic OR hybrid language and returns a minimally redundant form description of the language's context
    free grammar.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param meaning_list: list of strings corresponding to all possible meanings
    :return: minimally redundant form description of the language's context free grammar (string)
    """
    mrf_string = ''
    for i in range(len(meaning_list)):
        meaning = meaning_list[i]
        form = lang[i]
        if i != len(meaning_list) - 1:
            mrf_string += 'S' + meaning + form + '.'
        else:
            mrf_string += 'S' + meaning + form
    return mrf_string


def mrf_compositional(lang, meaning_list, reverse_meanings):
    """
    Takes a compositional language and returns a minimally redundant form description of the language's context free
    grammar.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param meaning_list: list of strings corresponding to all possible meanings
    :param reverse_meanings: Boolean: True if the compositional mappings are to the meaning elements in reverse order
    :return: minimally redundant form description of the language's context free grammar (string)
    """
    if reverse_meanings:
        meaning_list_reversed = [meaning[::-1] for meaning in meaning_list]
        meaning_list = meaning_list_reversed
    n_features = len(meaning_list[0])
    non_terminals = string.ascii_uppercase[:n_features]
    mrf_string = 'S' + non_terminals
    for i in range(len(non_terminals)):
        non_terminal_symbol = non_terminals[i]
        feature_values = []
        feature_value_segments = []
        for j in range(len(meaning_list)):
            if meaning_list[j][i] not in feature_values:
                feature_values.append(meaning_list[j][i])
                feature_value_segments.append(lang[j][i])
        for k in range(len(feature_values)):
            value = feature_values[k]
            segment = feature_value_segments[k]
            mrf_string += "." + non_terminal_symbol + value + segment
    return mrf_string


def mrf_other(lang, meaning_list):
    """
    Takes a language of the 'other' category and returns a minimally redundant form description of the language's
    context free grammar.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param meaning_list: list of strings corresponding to all possible meanings
    :return: minimally redundant form description of the language's context free grammar (string)
    """
    mapping_dict = {}
    for i in range(len(lang)):
        mapping_dict.setdefault(lang[i], []).append(meaning_list[i])
    mrf_string = 'S'
    counter = 0
    for form in mapping_dict.keys():
        for k in range(len(mapping_dict[form])):
            meaning = mapping_dict[form][k]
            if k != len(mapping_dict[form]) - 1:
                mrf_string += meaning + ','
            else:
                mrf_string += meaning
        if counter != len(mapping_dict.keys()) - 1:
            mrf_string += form + '.S'
        else:
            mrf_string += form
        counter += 1
    return mrf_string


def minimally_redundant_form_four_forms(lang, complete_forms, meaning_list):
    """
    Takes a language of any class and returns a minimally redundant form description of its context free grammar.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meaning_list: list of strings corresponding to all possible meanings
    :return: minimally redundant form description of the language's context free grammar (string)
    """
    if len(complete_forms) != 4 or len(complete_forms[0]) != 2:
        raise ValueError("This function only works for forms of length 2")
    lang_class = classify_language_four_forms_debugged(lang, complete_forms, meaning_list)
    if lang_class == 0:  # the language is 'degenerate'
        mrf_string = mrf_degenerate(lang, meaning_list)
    elif lang_class == 1 or lang_class == 2:  # the language is 'holistic' or 'hybrid'
        mrf_string = mrf_holistic(lang, meaning_list)
    elif lang_class == 3:  # the language is 'compositional'
        mrf_string = mrf_compositional(lang, meaning_list, reverse_meanings=False)
    elif lang_class == 4:  # the language is of the 'compositional_reverse' category
        mrf_string = mrf_compositional(lang, meaning_list, reverse_meanings=True)
    elif lang_class == 5:  # the language is of the 'other' category
        mrf_string = mrf_other(lang, meaning_list)
    return mrf_string


def character_probs(mrf_string):
    """
    Takes a string in minimally redundant form and generates a dictionary specifying the probability of each of the
    symbols used in the string

    :param mrf_string: a string in minimally redundant form
    :return: a dictionary with the symbols as keys and their corresponding probabilities as values
    """
    count_dict = {}
    for character in mrf_string:
        if character in count_dict.keys():
            count_dict[character] += 1
        else:
            count_dict[character] = 1
    prob_dict = {}
    for character in count_dict.keys():
        char_prob = count_dict[character] / len(mrf_string)
        prob_dict[character] = char_prob
    return prob_dict


def coding_length(mrf_string):
    """
    Takes a string in minimally redundant form and returns its coding length in bits

    :param mrf_string: a string in minimally redundant form
    :return: coding length in bits
    """
    char_prob_dict = character_probs(mrf_string)
    coding_len = 0
    for character in mrf_string:
        coding_len += log2(char_prob_dict[character])
    return -coding_len


def prior_single_lang(lang, complete_forms, meaning_list):
    """
    Takes a language and returns its PROPORTIONAL prior probability; this still needs to be normalized over all
    languages in order to give the real prior probability.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meaning_list: list of strings corresponding to all possible meanings
    :return: PROPORTIONAL prior probability (float)
    """
    if len(complete_forms) == 4 and len(complete_forms[0]) == 2:
        mrf_string = minimally_redundant_form_four_forms(lang, complete_forms, meaning_list)
    # else:
    #     mrf_string = minimally_redundant_form_multiple_forms(lang, complete_forms, meaning_list)
    coding_len = coding_length(mrf_string)
    prior = 2 ** -coding_len
    return prior


def prior(hypothesis_space, complete_forms, meaning_list):
    logpriors = np.zeros(len(hypothesis_space))
    for i in range(len(hypothesis_space)):
        lang_prior = prior_single_lang(hypothesis_space[i], complete_forms, meaning_list)
        logpriors[i] = np.log(lang_prior)
    logpriors_normalized = np.subtract(logpriors, logsumexp(logpriors))
    return logpriors_normalized




###################################################################################################################
if __name__ == '__main__':

    hypothesis_space = create_all_possible_languages(meanings, forms)
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


    # Let's check whether the functions in this cell work correctly by comparing the number of languages of each type we
    # get with the SimLang lab 21:

    types_simlang = np.array(types_simlang)
    no_of_each_type = np.bincount(types_simlang)
    print('')
    print("no_of_each_type ACCORDING TO SIMLANG CODE, where 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional is:")
    print(no_of_each_type)


    class_per_lang = classify_all_languages_debugged(hypothesis_space, forms, meanings)
    # print('')
    # print('')
    # print("class_per_lang is:")
    # print(class_per_lang)
    no_of_each_class = np.bincount(class_per_lang.astype(int))
    no_of_each_class_simlang_order = np.array([no_of_each_class[0], no_of_each_class[1] + no_of_each_class[2], no_of_each_class[5], no_of_each_class[3]+no_of_each_class[4]])
    print('')
    print("no_of_each_class_simlang_order ACCORDING TO MY CODE, where 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional is:")
    print(no_of_each_class_simlang_order)

    # Hmmm, that gives us slightly different numbers! Is that caused by a problem in my
    # create_all_languages() function, or in my classify_lang() function?
    # To find out, let's compare my list of all languages to that from SimLang lab 21:

    # First, we need to change the way we represent the list of all languages to match
    # that of lab 21:
    all_langs_as_in_simlang = transform_all_languages_to_simlang_format(hypothesis_space, meanings)
    print('')
    print('')
    # print("all_langs_as_in_simlang is:")
    # print(all_langs_as_in_simlang)
    print("len(all_langs_as_in_simlang) is:")
    print(len(all_langs_as_in_simlang))
    print("len(all_langs_as_in_simlang[0]) is:")
    print(len(all_langs_as_in_simlang[0]))
    print("len(all_langs_as_in_simlang[0][0]) is:")
    print(len(all_langs_as_in_simlang[0][0]))

    # Then, let's check whether the resulting list is the same (and in the same order) as the list of languages
    # provided in SimLang Lab 21 (if the order is already the same, this would suggest that Kirby et al. (2015) also
    # used the itertools.product() function (or an equivalent procedure) to produce it.
    order_same = check_language_lists_same_order(all_langs_as_in_simlang, languages_simlang)
    print('')
    print('')
    print("order_same is:")
    print(order_same)

    # Ok, this shows that the list of languages generated by my code is exactly the same as the list of languages
    # provided in SimLang Lab 21 (except for the difference in representation format), so instead there must be
    # something wrong with the way I categorise the languages. Firstly, it looks like my classify_language() function
    # underestimates the number of compositional languages. So let's first have a look at which languages it classifies
    # as compositional:

    compositional_langs_indices_my_code = np.concatenate((np.where(class_per_lang==3)[0], np.where(class_per_lang==4)[0]))
    print('')
    print('')
    print("compositional_langs_indices_my_code MY CODE are:")
    print(compositional_langs_indices_my_code)
    print("len(compositional_langs_indices_my_code) MY CODE are:")
    print(len(compositional_langs_indices_my_code))

    # for index in compositional_langs_indices_my_code:
    #     print('')
    #     print("index MY CODE is:")
    #     print(index)
    #     print("hypothesis_space[index] MY CODE is:")
    #     print(hypothesis_space[index])

    # And now let's do the same for the languages from SimLang Lab 21:

    compositional_langs_indices_simlang = np.where(np.array(types_simlang)==3)[0]
    print('')
    print('')
    print("compositional_langs_indices_simlang SIMLANG CODE are:")
    print(compositional_langs_indices_simlang)
    print("len(compositional_langs_indices_simlang) SIMLANG CODE are:")
    print(len(compositional_langs_indices_simlang))

    # for index in compositional_langs_indices_simlang:
    #     print('')
    #     print("index SIMLANG CODE is:")
    #     print(index)
    #     print("languages_simlang[index] SIMLANG CODE is:")
    #     print(languages_simlang[index])

    # # Hmm, so it looks like instead of there being a bug in my code, there might actually be a bug in the SimLang lab 21
    # # code (or rather, in the code that generated the list of types that was copied into SimLang lab 21)
    # # Let's check whether maybe the holistic languages that are miscategorised as compositional in the SimLang code
    # # happen to be the ones I identified as "hybrids" (i.e. kind of in between holistic and compositional) above:
    #
    # hybrid_langs_indices_my_code = np.where(class_per_lang==2)[0]
    # print('')
    # print('')
    # print("hybrid_langs_indices_my_code MY CODE are:")
    # print(hybrid_langs_indices_my_code)
    # print("len(hybrid_langs_indices_my_code) MY CODE are:")
    # print(len(hybrid_langs_indices_my_code))
    #
    # for index in hybrid_langs_indices_my_code:
    #     print('')
    #     print("index MY CODE is:")
    #     print(index)
    #     print("hypothesis_space[index] MY CODE is:")
    #     print(hypothesis_space[index])
    #
    # # Nope, that isn't the case.

    ###################################################################################################################
    # SECONDLY, LET'S CHECK WHETHER MY FUNCTIONS FOR GENERATING/CALCULATING THE LANGAUGES' REWRITE RULES, MINIMALLY
    # REDUNDANT FORMS, AND CODING LENGTHS GIVE THE SAME RESULTS AS SHOWN FOR THE EXAMPLE LANGUAGES IN Kirby et al.
    # (2015) (in the table on p. 92):

    # First some parameter settings:
    meanings = ['02', '03', '12', '13']
    forms_without_noisy_variants = ['aa', 'ab', 'ba', 'bb']

    # Then let's specify the example languages from Kirby et al. (2015)
    example_languages = [['aa', 'aa', 'aa', 'aa'],
                         ['ab', 'ab', 'ab', 'ab'],
                         ['aa', 'aa', 'aa', 'ab'],
                         ['aa', 'aa', 'aa', 'bb'],
                         ['aa', 'ab', 'ba', 'bb'],
                         ['aa', 'aa', 'ab', 'ba'],
                         ['aa', 'aa', 'ab', 'bb'],
                         ['aa', 'ab', 'bb', 'ba']]


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

    # And now let's calculate their coding lengths:
    lang_classes_text = ['degenerate', 'holistic', 'hybrid', 'compositional', 'compositional_reverse', 'other']
    for i in range(len(example_languages)):
        lang = example_languages[i]
        print('')
        print(i)
        lang_class = classify_language_four_forms_debugged(lang, forms_without_noisy_variants, meanings)
        lang_class_text = lang_classes_text[lang_class]
        print("lang_class_text is:")
        print(lang_class_text)
        print("lang is:")
        print(lang)
        mrf_string = minimally_redundant_form_four_forms(lang, forms_without_noisy_variants, meanings)
        print("mrf_string is:")
        print(mrf_string)
        coding_len = coding_length(mrf_string)
        print("coding_len is:")
        print(round(coding_len, ndigits=2))
        lang_prior = prior_single_lang(lang, forms_without_noisy_variants, meanings)
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
        lang_class = classify_language_four_forms_debugged(lang, forms_without_noisy_variants, meanings)
        lang_class_text = lang_classes_text[lang_class]
        print("lang_class_text is:")
        print(lang_class_text)
        print("lang is:")
        print(lang)
        mrf_string = minimally_redundant_form_four_forms(lang, forms_without_noisy_variants, meanings)
        print("mrf_string is:")
        print(mrf_string)
        coding_len = coding_length(mrf_string)
        print("coding_len is:")
        print(round(coding_len, ndigits=2))
        lang_prior = prior_single_lang(lang, forms_without_noisy_variants, meanings)
        print("prior this lang is:")
        print(lang_prior)


    ###################################################################################################################
    # FINALLY, LET'S CHECK MY PRIOR PROBABILITY DISTRIBUTION AGAINST THE SIMLANG ONE:

    my_log_prior = prior(hypothesis_space, forms, meanings)
    print('')
    print('')
    # print("my_log_prior is:")
    # print(my_log_prior)
    print("my_log_prior.shape is:")
    print(my_log_prior.shape)
    print("np.exp(logsumexp(my_log_prior)) is:")
    print(np.exp(logsumexp(my_log_prior)))

    print('')
    # print("np.array(priors_simlang) is:")
    # print(np.array(priors_simlang))
    print("np.array(priors_simlang).shape is:")
    print(np.array(priors_simlang).shape)
    print("np.exp(logsumexp(priors_simlang)) is:")
    print(np.exp(logsumexp(priors_simlang)))

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
    # languages that are classified as compositional according to the simlang code (four of which, 39, 114, 141, and
    # 216 are instead classified as holistic in my own code). Let's have a closer look at these languages and the
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

    # My own code assigns slightly lower prior probabilities to those languages which are *also* compositional
    # according to my own code than the simlang code does, and assigns *much* lower prior probability to those languages
    # which are classed as holistic instead of compositional according to my code (compared to the prior probability
    # they are assigned in the simlang code). As we'd expect, all languages that are classed as compositional are
    # assigned the same prior probability in the simlang code, whereas in my code all languages that are classed as
    # compositional are assigned the same prior probability, and all languages that are classed as holistic are too.

    # To conclude: the differences we find between the two prior probability distributions actually make a lot of sense
    # given the differences we found in the classification of the languages.

    # The only thing that still seems a bit puzzling at first sight is the fact that the languages that are classed as
    # compositional according to my code, receive *less* prior probability than the languages that are classed as
    # compositional according to the simlang code. This is surprising because the languages that my code classes as
    # holistic rather than compositional (as opposed to the simlang code) receive *way* lower prior probability, and we
    # would therefore expect the prior probability that remains from this to be redistributed over the languages that
    # *are* classed as compositional. However, this difference of course causes a redistribution of the remaining prior
    # probability over the *whole* hypothesis space, and not just over the four compositional languages.

    # In order to check whether that might indeed be the explanation, I'm just checking that the difference in the sum
    # prior probability that is assigned to these 8 languages between my code and the simlang code is indeed equal to
    # the sum difference in prior probability over the remaining languages.

    log_priors_my_code = np.array([my_log_prior[i] for i in diff_value_indices_flattened])
    print('')
    print('')
    # print("log_priors_my_code is:")
    # print(log_priors_my_code)

    log_priors_simlang_code = np.array([priors_simlang[i] for i in diff_value_indices_flattened])
    print('')
    # print("log_priors_simlang_code is:")
    # print(log_priors_simlang_code)

    sum_my_code = logsumexp(log_priors_my_code)
    print('')
    print("np.exp(sum_my_code) is:")
    print(np.exp(sum_my_code))

    sum_simlang_code = logsumexp(log_priors_simlang_code)
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
