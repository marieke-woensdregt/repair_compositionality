import string
import itertools
import numpy as np


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


def classify_language_four_forms(lang, forms, meaning_list):
    """
    Classify one particular language as either 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other
    (Kirby et al., 2015). NOTE that this function is specific to classifying languages that consist of exactly 4 forms,
    where each form consists of exactly 2 characters. For a more general version of this function, see
    classify_language_general() below.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :param meaning_list: list of strings corresponding to all possible meanings
    :returns: integer corresponding to category that language belongs to:
    0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, 4 = other (here I'm following the
    ordering used in the Kirby et al., 2015 paper; NOT the ordering from SimLang lab 21)
    """
    class_degenerate = 0
    class_holistic = 1
    class_hybrid = 2  # this is a hybrid between a holistic and a compositional language; where *half* of the partial
    # forms is mapped consistently to partial meanings (instead of that being the case for *all* partial forms)
    class_compositional = 3
    class_other = 4

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

    # lang is compositional if it makes use of all possible forms_without_noisy_variants, *and* each form element maps
    # to the same meaning element for each form:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[
        3] in lang and lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0] and lang[0][
        1] == lang[2][1] and lang[1][1] == lang[3][1]:
        return class_compositional

    # lang is holistic if it is *not* compositional, but *does* make use of all possible forms_without_noisy_variants:
    elif forms[0] in lang and forms[1] in lang and forms[2] in lang and forms[3] in lang:
        # within holistic languages, we can distinguish between those in which at least one part form is mapped
        # consistently onto one part meaning. This class we will call 'hybrid' (because for the purposes of repair, it
        # is a hybrid between a holistic and a compositional language, because for half of the possible noisy forms that
        # a listener could receive it allows the listener to figure out *part* of the meaning, and therefore use a
        # restricted request for repair instead of an open request.
        if lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0]:
            return class_hybrid
        elif lang[0][1] == lang[2][1] and lang[1][1] == lang[3][1]:
            return class_hybrid
        else:
            return class_holistic

    # In all other cases, a language belongs to the 'other' category:
    else:
        return class_other


def classify_all_languages(language_list, complete_forms, meaning_list):
    """
    Classify all languages as either 'degenerate' (0), 'holistic' (1), 'other' (2) or 'compositional' (3)
    (Kirby et al., 2015)

    :param language_list: list of all languages
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meanings: list of all possible meanings; corresponds to global variable 'meanings'
    :returns: 1D numpy array containing integer corresponding to category of corresponding
    language index as hardcoded in classify_language function: 0 = degenerate, 1 = holistic, 2 = hybrid,
    3 = compositional, 4 = other (here I'm following the ordering used in the Kirby et al., 2015 paper; NOT the ordering
    from SimLang lab 21)
    """
    class_per_lang = np.zeros(len(language_list))
    for l in range(len(language_list)):
        class_per_lang[l] = classify_language_four_forms(language_list[l], complete_forms, meaning_list)
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
if __name__ == '__main__':

    ###################################################################################################################
    # FIRST LET'S CHECK MY LANGUAGES AND THE CLASSIFICATION OF THEM AGAINST THE SIMLANG CODE, JUST AS A SANITY CHECK:

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

    meanings = ['02', '03', '12', '13']  # all possible meanings
    forms = create_all_possible_forms(2, [2])  # all possible forms, excluding their possible
    # 'noisy variants'
    print('')
    print('')
    print("forms are:")
    print(forms)

    hypothesis_space = create_all_possible_languages(meanings, forms)
    print("number of possible languages is:")
    print(len(hypothesis_space))

    # Let's check whether the functions in this cell work correctly by comparing the number of languages of each type we
    # get with the SimLang lab 21:

    types_simlang = np.array(types_simlang)
    no_of_each_type = np.bincount(types_simlang)
    print('')
    print("no_of_each_type ACCORDING TO SIMLANG CODE, where 0 = degenerate, 1 = holistic, 2 = other, 3 = compositional is:")
    print(no_of_each_type)


    class_per_lang = classify_all_languages(hypothesis_space, forms, meanings)
    print('')
    print('')
    print("class_per_lang is:")
    print(class_per_lang)
    no_of_each_class = np.bincount(class_per_lang.astype(int))
    print('')
    print("no_of_each_class ACCORDING TO MY CODE, where 0 = degenerate, 1 = holistic, 2 = hybrid, 3 = compositional, "
          "4 = other is:")
    print(no_of_each_class)

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

    compositional_langs_indices_my_code = np.where(class_per_lang==3)[0]
    print('')
    print('')
    print("compositional_langs_indices_my_code MY CODE are:")
    print(compositional_langs_indices_my_code)
    print("len(compositional_langs_indices_my_code) MY CODE are:")
    print(len(compositional_langs_indices_my_code))

    for index in compositional_langs_indices_my_code:
        print('')
        print("index MY CODE is:")
        print(index)
        print("hypothesis_space[index] MY CODE is:")
        print(hypothesis_space[index])

    # And now let's do the same for the languages from SimLang Lab 21:

    compositional_langs_indices_simlang = np.where(np.array(types_simlang)==3)[0]
    print('')
    print('')
    print("compositional_langs_indices_simlang SIMLANG CODE are:")
    print(compositional_langs_indices_simlang)
    print("len(compositional_langs_indices_simlang) SIMLANG CODE are:")
    print(len(compositional_langs_indices_simlang))

    for index in compositional_langs_indices_simlang:
        print('')
        print("index SIMLANG CODE is:")
        print(index)
        print("languages_simlang[index] SIMLANG CODE is:")
        print(languages_simlang[index])

    # Hmm, so it looks like instead of there being a bug in my code, there might actually be a bug in the SimLang lab 21
    # code (or rather, in the code that generated the list of types that was copied into SimLang lab 21)
    # Let's check whether maybe the holistic languages that are miscategorised as compositional in the SimLang code
    # happen to be the ones I identified as "hybrids" (i.e. kind of in between holistic and compositional) above:

    hybrid_langs_indices_my_code = np.where(class_per_lang==2)[0]
    print('')
    print('')
    print("hybrid_langs_indices_my_code MY CODE are:")
    print(hybrid_langs_indices_my_code)
    print("len(hybrid_langs_indices_my_code) MY CODE are:")
    print(len(hybrid_langs_indices_my_code))

    for index in hybrid_langs_indices_my_code:
        print('')
        print("index MY CODE is:")
        print(index)
        print("hypothesis_space[index] MY CODE is:")
        print(hypothesis_space[index])

    # Nope, that isn't the case.

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

    # And now let's calculate their coding lengths:
    lang_classes_text = ['degenerate', 'holistic', 'hybrid', 'compositional', 'other']
    for i in range(len(example_languages)):
        lang = example_languages[i]
        print('')
        print(i)
        lang_class = classify_language_four_forms(lang, forms_without_noisy_variants, meanings)
        lang_class_text = lang_classes_text[lang_class]
        print("lang_class_text is:")
        print(lang_class_text)
        print("lang is:")
        print(lang)
        mrf_string = minimally_redundant_form(lang, forms_without_noisy_variants, meanings)
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
