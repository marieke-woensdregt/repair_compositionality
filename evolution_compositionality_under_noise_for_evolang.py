__author__ = "Marieke Woensdregt"

import sys
import string
import numpy as np
import itertools
import random
from math import log2
import scipy.special
import pickle
import time


###################################################################################################################
# THE FUNCTIONS BELOW HAVE TO BE DEFINED BEFORE SETTING THE PARAMETERS BECAUSE THEY ARE NEEDED TO DO SO.
def str_to_bool(s):
    """
    Takes a string which is either 'True' or '1' or 'False' or '0' and turns it into the corresponding boolean.

    :param s: a string (only accepts 'True', 'true', '1', 'False', 'false' and '0'
    :return: a boolean
    """
    if s == 'True' or s == 'true' or s == 1:
        return True
    elif s == 'False' or s == 'false' or s == 0:
        return False
    else:
        raise ValueError("string does not seem to correspond to a boolean")


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


def create_noisy_variants(form):
    """
    Takes a form and generates all its possible noisy variants. NOTE however that in its current form, this function
    only creates noisy variants in which only a single element of the original form is replaced with a blank! (So it
    creates for instance 'a_' and '_b', but not '__'.)

    :param form: a form (string)
    :return: a list of possible noisy variants of that form
    """
    noisy_variant_list = []
    for i in range(len(form)):
        noisy_variant = form[:i] + '_' + form[i+1:]
        # Instead of string slicing, another way of doing this would be to convert the string into a list, replace the
        # element at the ith index, and then convert it back into a string using the 'join' method,
        # see: https://www.quora.com/How-do-you-change-one-character-in-a-string-in-Python
        noisy_variant_list.append(noisy_variant)
    return noisy_variant_list


def create_all_possible_noisy_forms(all_complete_forms):
    """
    Takes a list of all possible complete forms and derives a list of all possible noisy forms.

    :param all_complete_forms: list of all possible complete forms
    :return: list of all possible noisy forms
    """
    noisy_forms = []
    for form in all_complete_forms:
        noisy_variants = create_noisy_variants(form)
        for variant in noisy_variants:
            if variant not in noisy_forms:
                noisy_forms.append(variant)
    return noisy_forms


###################################################################################################################
# ALL PARAMETER SETTINGS GO HERE:

meanings = ['02', '03', '12', '13']  # all possible meanings
forms_without_noise = create_all_possible_forms(2, [2])  # all possible forms, excluding their possible
# 'noisy variants'
noisy_forms = create_all_possible_noisy_forms(forms_without_noise)
# all possible noisy variants of the forms above
all_forms_including_noisy_variants = forms_without_noise + noisy_forms  # all possible forms, including both
# complete forms and noisy variants

error = 0.05  # the probability of making a production error (Kirby et al., 2015 use 0.05)

turnover = True  # determines whether new individuals enter the population or not
popsize = 2  # If I understand it correctly, Kirby et al. (2015) used a population size of 2: each generation is simply
# a pair of agents.
runs = 10  # the number of independent simulation runs (Kirby et al., 2015 used 100)
generations = 50  # the number of generations (Kirby et al., 2015 used 100)
initial_language_type = 'degenerate'  # set the language class that the first generation is trained on

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

communicative_success = False  # determines whether there is a pressure for communicative success or not
communicative_success_pressure_strength = (2./3.)  # determines how much more likely a <meaning, form> pair from a
# successful interaction is to enter the data set that is passed on to the next generation, compared to a
# <meaning, form> pair from a unsuccessful interaction.

burn_in = generations/2  # the burn-in period that is excluded when calculating the mean distribution over
# languages after convergence

n_lang_classes = 4  # the number of language classes that are distinguished (int).

pickle_file_path = ""  # Use this to specify a path to a folder where you want the pickle (result) files to be stored.


# THE FOLLOWING PARAMETERS SHOULD ONLY BE SET IF __name__ == '__main__', BECAUSE THEY ARE RETRIEVED FROM THE INPUT
# ARGUMENTS GIVEN TO THE PYTHON SCRIPT WHEN RUN FROM THE TERMINAL OR FROM AN .SH SCRIPT:
if __name__ == '__main__':

    b = int(sys.argv[1])  # the bottleneck (i.e. number of meaning-form pairs the each pair gets to see during training
    # (Kirby et al. used a bottleneck of 20 in the body of the paper.
    rounds = 2 * b  # Kirby et al. (2015) used rounds = 2*b, but SimLang lab 21 uses 1*b
    print('')
    print("bottleneck b is:")
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


###################################################################################################################
# FIRST SOME FUNCTIONS TO CREATE ALL POSSIBLE LANGUAGES AND CLASSIFY THEM:


def create_all_possible_languages(meaning_list, forms):
    """Creates all possible languages

    :param meaning_list: list of strings corresponding to all possible meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :returns: list of tuples which represent languages, where each tuple consists of forms_without_noisy_variants and
    has length len(meanings)
    """
    all_possible_languages = list(itertools.product(forms, repeat=len(meaning_list)))
    return all_possible_languages


def classify_language_four_forms(lang, forms, meaning_list):
    """
    Classify one particular language as either 0 = degenerate, 1 = holistic, 2 = compositional, and 3 = other
    (Kirby et al., 2015). NOTE that this function is specific to classifying languages that consist of exactly 4 forms,
    where each form consists of exactly 2 characters.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param forms: list of strings corresponding to all possible forms_without_noisy_variants
    :param meaning_list: list of strings corresponding to all possible meanings
    :returns: integer corresponding to category that language belongs to: 0 = degenerate, 1 = holistic,
    2 = compositional, and 3 = other (Kirby et al., 2015).
    """
    class_degenerate = 0
    class_holistic = 1
    class_compositional = 2
    class_other = 3

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

    # The language is DEGENERATE if it uses the same form for each meaning:
    if lang.count(lang[0]) == len(lang):
        return class_degenerate

    # If each form is unique, the language is either COMPOSITIONAL or HOLISTIC:
    all_forms_unique = check_all_forms_unique(lang)
    if all_forms_unique is True:
        # lang is compositional if each form element maps to the same meaning element for each form:
        if lang[0][0] == lang[1][0] and lang[2][0] == lang[3][0] and lang[0][1] == lang[2][1] and lang[1][1] == lang[3][1]:
            return class_compositional
        elif lang[0][0] == lang[2][0] and lang[1][0] == lang[3][0] and lang[0][1] == lang[1][1] and lang[2][1] == lang[3][1]:
            return class_compositional
        # lang is holistic if it is *not* compositional, but *does* make use of all possible complete forms:
        else:
            return class_holistic

    # In all other cases, a language belongs to the 'other' category:
    else:
        return class_other


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


def classify_all_languages(language_list, complete_forms, meaning_list):
    """
    Classify all languages as either 'degenerate' (0), 'holistic' (1), 'compositional' (2) or 'other' (3)
    (Kirby et al., 2015)

    :param language_list: list of all languages
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meanings: list of all possible meanings; corresponds to global variable 'meanings'
    :returns: 1D numpy array containing integer corresponding to category of corresponding
    language index as hardcoded in classify_language_four_forms function: 0 = degenerate, 1 = holistic,
    2 = compositional, 3 = other
    """
    class_per_lang = np.zeros(len(language_list))
    for l in range(len(language_list)):
        class_per_lang[l] = classify_language_four_forms(language_list[l], complete_forms, meaning_list)
    return class_per_lang


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
    Takes a holistic language and returns a minimally redundant form description of the language's context free grammar.

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


def minimally_redundant_form(lang, complete_forms, meaning_list):
    """
    Takes a language of any class and returns a minimally redundant form description of its context free grammar.

    :param lang: a language; represented as a tuple of forms_without_noisy_variants, where each form index maps to same
    index in meanings
    :param complete_forms: list containing all possible complete forms; corresponds to global variable
    'forms_without_noise'
    :param meaning_list: list of strings corresponding to all possible meanings
    :return: minimally redundant form description of the language's context free grammar (string)
    """
    lang_class = classify_language_four_forms(lang, complete_forms, meaning_list)
    if lang_class == 0:  # the language is 'degenerate'
        mrf_string = mrf_degenerate(lang, meaning_list)
    elif lang_class == 1:  # the language is 'holistic'
        mrf_string = mrf_holistic(lang, meaning_list)
    elif lang_class == 2:  # the language is 'compositional'
        mrf_string = mrf_compositional(lang, meaning_list)
    elif lang_class == 3:  # the language is of the 'other' category
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
    mrf_string = minimally_redundant_form(lang, complete_forms, meaning_list)
    coding_len = coding_length(mrf_string)
    prior = 2 ** -coding_len
    return prior


def prior(hypotheses, complete_forms, meaning_list):
    """
    Calculates the LOG prior over the full hypothesis space

    :param hypotheses: list of all possible languages
    :param complete_forms: The full set of possible complete forms (corresponds to global parameter
    'forms_without_noise')
    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :return: 1D numpy array containing the LOG prior probability for each hypothesis
    """
    logpriors = np.zeros(len(hypotheses))
    for i in range(len(hypotheses)):
        lang_prior = prior_single_lang(hypotheses[i], complete_forms, meaning_list)
        logpriors[i] = np.log(lang_prior)
    logpriors_normalized = np.subtract(logpriors, scipy.special.logsumexp(logpriors))
    return logpriors_normalized


###################################################################################################################
# NOW SOME FUNCTIONS THAT HANDLE PRODUCTION, NOISY PRODUCTION, AND RECEPTION WITH AND WITHOUT REPAIR:

# A reproduction of the production function of Kirby et al. (2015):
def production_likelihoods_kirby_et_al(language, topic, meaning_list, ambiguity_penalty, error_prob):
    """
    Calculates the production probabilities for each of the possible forms_without_noisy_variants given a language and
    topic, as defined by Kirby et al. (2015)

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param error_prob: the probability of making an error in production
    :return: 1D numpy array containing a production probability for each possible form (where the index of the
    probability corresponds to the index of the form in the global variable "forms_without_noisy_variants")
    """
    for m in range(len(meaning_list)):
        if meaning_list[m] == topic:
            topic_index = m
    correct_form = language[topic_index]
    ambiguity = 0
    for f in language:
        if f == correct_form:
            ambiguity += 1
    prop_to_prob_correct_form = ((1./ambiguity) ** ambiguity_penalty) * (1. - error_prob)
    prop_to_prob_error_form = error_prob / (len(forms_without_noise) - 1)
    prop_to_prob_per_form_array = np.zeros(len(forms_without_noise))
    for i in range(len(forms_without_noise)):
        if forms_without_noise[i] == correct_form:
            prop_to_prob_per_form_array[i] = prop_to_prob_correct_form
        else:
            prop_to_prob_per_form_array[i] = prop_to_prob_error_form
    return prop_to_prob_per_form_array


# we also need a function that removes every instance of a given element from a list (to use for
# removing the 'correct' forms_without_noisy_variants from a list of possible forms_without_noisy_variants for a given
# topic:
def remove_all_instances(my_list, element_to_be_removed):
    """
    Takes a list, and removes all instances of a given element from it

    :param my_list: a list
    :param element_to_be_removed: the element to be removed; can be of any type
    :return: the list with all instances of the target element removed
    """
    i = 0  # loop counter
    length = len(my_list)  # list length
    while i < len(my_list):
        if my_list[i] == element_to_be_removed:
            my_list.remove(my_list[i])
            # as an element is removed
            # so decrease the length by 1
            length = length - 1
            # run loop again to check element
            # at same index, when item removed
            # next item will shift to the left
            continue
        i = i + 1
    return my_list


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
def produce(language, topic, ambiguity_penalty, error_prob, prob_of_noise):
    """
    Produces an actual utterance, given a language and a topic

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param topic: the index of the topic (corresponding to an index in the globally defined meaning list) that the
    speaker intends to communicate
    :param ambiguity_penalty: parameter that determines the strength of the penalty on ambiguity (gamma)
    :param error_prob: the probability of making an error in production
    :param prob_of_noise: the probability of noise happening; corresponds to global variable 'noise_prob'
    :return: an utterance. That is, a single form chosen from either the global variable "forms_without_noise" (if
    noise_prob == 0.0) or the global variable "all_forms_including_noisy_variants" (if noise_prob > 0.0).
        """
    if prob_of_noise > 0.0:
        prop_to_prob_per_form_array = production_likelihoods_with_noise(language, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, error_prob, prob_of_noise)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(all_forms_including_noisy_variants, p=prob_per_form_array)
    else:
        prop_to_prob_per_form_array = production_likelihoods_kirby_et_al(language, topic, meanings, ambiguity_penalty, error_prob)
        prob_per_form_array = np.divide(prop_to_prob_per_form_array, np.sum(prop_to_prob_per_form_array))
        utterance = np.random.choice(forms_without_noise, p=prob_per_form_array)
    return utterance


def receive_without_repair(language, utterance):
    """
    Takes a language and an utterance, and returns an interpretation of that utterance, following the language

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param utterance: a form (string)
    :return: an interpretation (string)
    """
    possible_interpretations = []
    for i in range(len(language)):
        if language[i] == utterance:
            possible_interpretations.append(meanings[i])
    if len(possible_interpretations) == 0:
        possible_interpretations = meanings
    interpretation = random.choice(possible_interpretations)
    return interpretation


def noisy_to_complete_forms(noisy_form, complete_forms):
    """
    Takes a noisy form and returns all possible complete forms that it's compatible with.

    :param noisy_form: a noisy form (i.e. a string containing '_' as at least one of the characters)
    :param complete_forms: The full set of possible complete forms (corresponds to global parameter
    'forms_without_noise')
    :return: A list of complete forms that the noisy form is compatible with
    """
    possible_complete_forms = []
    amount_of_noise = noisy_form.count('_')
    for complete_form in complete_forms:
        similarity_score = 0
        for i in range(len(noisy_form)):
            if noisy_form[i] == complete_form[i]:
                similarity_score += 1
        if similarity_score == len(complete_form)-amount_of_noise:
            possible_complete_forms.append(complete_form)
    return possible_complete_forms


def find_possible_interpretations(language, forms):
    """
    Finds all meanings that the forms given as input are mapped to in the language given as input

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param forms: list of forms
    :return: list of meanings (type: string) that the forms given as input are mapped to in the language given as input
    """
    possible_interpretations = []
    for i in range(len(language)):
        if language[i] in forms:
            possible_interpretations.append(meanings[i])
    return possible_interpretations


def find_partial_meaning(language, noisy_form):
    """
    Checks whether the noisy_form given as input maps unambiguously to a partial meaning in the language given as
    input, and if so, returns that partial meaning.

    :param language: list of forms_without_noisy_variants that has same length as list of meanings (global variable),
    where each form is mapped to the meaning at the corresponding index
    :param noisy_form: a noisy form (i.e. a string containing '_' as at least one of the characters)
    :return: a list containing the partial meaning that the noisy_form maps unambiguously to, if there is one
    """
    part_meanings_as_ints = []
    for i in range(len(meanings)):
        for j in range(len(meanings[0])):
            part_meanings_as_ints.append(int(meanings[i][j]))
    max_part_meaning = max(part_meanings_as_ints)
    count_per_partial_meaning = np.zeros(max_part_meaning+1)
    for i in range(len(noisy_form)):
        if noisy_form[i] != '_':
            for j in range(len(language)):
                if language[j][i] == noisy_form[i]:
                    count_per_partial_meaning[int(meanings[j][i])] += 1
    n_features = 0
    for i in range(len(meanings)):
        if meanings[i][0] == meanings[0][0]:
            n_features += 1
    if np.sum(count_per_partial_meaning) == n_features:
        part_meaning_index = np.where(count_per_partial_meaning==n_features)[0]
    else:
        part_meaning_index = []
    if len(part_meaning_index) == 1:
        return part_meaning_index
    else:
        return []


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

def update_posterior(log_posterior, hypotheses, topic, utterance, ambiguity_penalty, prob_of_noise, all_possible_forms):
    """
    Takes a LOG posterior probability distribution and a <topic, utterance> pair, and updates the posterior probability
    distribution accordingly

    :param log_posterior: 1D numpy array containing LOG posterior probability values for each hypothesis
    :param hypotheses: list of all possible languages
    :param topic: a topic (string from the global variable meanings)
    :param utterance: an utterance; string from the global variable forms (can be a noisy form if prob_of_noise > 0.0)
    :param ambiguity_penalty: parameter that determines extent to which speaker tries to avoid ambiguity; corresponds
    to global variable 'gamma'
    :param prob_of_noise: the probability of noise; corresponds to global variable 'noise_prob'
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
        if prob_of_noise > 0.0:
            likelihood_per_form_array = production_likelihoods_with_noise(hypothesis, topic, meanings, forms_without_noise, noisy_forms, ambiguity_penalty, error, prob_of_noise)
        else:
            likelihood_per_form_array = production_likelihoods_kirby_et_al(hypothesis, topic, meanings, ambiguity_penalty, error)
        log_likelihood_per_form_array = np.log(likelihood_per_form_array)
        new_log_posterior.append(log_posterior[j] + log_likelihood_per_form_array[utterance_index])

    new_log_posterior_normalized = np.subtract(new_log_posterior, scipy.special.logsumexp(new_log_posterior))

    return new_log_posterior_normalized


# The function below I borrowed from the code provided in the Simulating Language MSc course that is taught by
# Simon Kirby and Kenny Smith at the University of Edinburgh:
def log_roulette_wheel(normedlogs):
    """
    Samples an index from a list of LOG probabilities, where each index has a probability proportional to their
    probability of being chosen

    :param normedlogs: a list of normalized LOG probabilities
    :return: an index somewhere between 0 and len(normedlogs)
    """
    r = np.log(random.random())  # generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = scipy.special.logsumexp([accumulator, normedlogs[i + 1]])


def sample(hypotheses, log_posterior):
    """
    Samples a language based on the posterior

    :param hypotheses: list of all possible languages; corresponds to global variable 'hypothesis_space'
    :param log_posterior: a list of LOG posterior probabilities
    :return: a language (list of forms_without_noisy_variants that has same length as the global variable meanings,
    where each form is mapped to the meaning at the corresponding index)
    """
    return hypotheses[log_roulette_wheel(log_posterior)]


# NOW THE FUNCTION THAT CREATE A NEW POPULATION, AND MAKES A POPULATION COMMUNICATE (FOR THE INTRA-GENERATIONAL
# INTERACTION ROUNDS):

def new_population(pop_size, log_priors):
    """
    Creates a new population of agents, where each agent simply consists of the prior probability distribution (which
    is assumed to be defined as a global variable called 'priors')

    :param pop_size: the number of agents desired in the new population; corresponds to global variable 'popsize'
    :param log_priors: the LOG prior probability distribution that each agent should be initialised with
    :return: 2D numpy array, with agents on the rows, and hypotheses (or rather their corresponding LOG prior
    probabilities)
    on the columns.
    """
    population = [log_priors for x in range(pop_size)]
    population = np.array(population)
    return population


def population_communication(population, n_rounds, mutual_understanding_pressure, minimal_effort_pressure, ambiguity_penalty, prob_of_noise, communicative_success_pressure, hypotheses):
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
    :param prob_of_noise: the probability of noise happening; corresponds to global variable 'noise_prob'
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
            utterance = produce(speaker_language, topic, ambiguity_penalty, error, prob_of_noise)
            # whenever a speaker is called upon to produce a utterance, they first sample a language from their
            # posterior probability distribution. So each agent keeps updating their language according to the data
            # received from their communication partner.
            listener_response = receive_with_repair(hearer_language, utterance, mutual_understanding_pressure, minimal_effort_pressure)
            counter = 0
            while '?' in listener_response:
                if counter == 3:  # After 3 attempts, the listener stops trying to do repair
                    break
                utterance = produce(speaker_language, topic, ambiguity_penalty, error, prob_of_noise=0.0)
                # For now, we assume that the speaker's response to a repair initiator always comes through without
                # noise.
                listener_response = receive_with_repair(hearer_language, utterance, mutual_understanding_pressure, minimal_effort_pressure)
                counter += 1
            if observed_meaning == 'intended':
                population[hearer_index] = update_posterior(population[hearer_index], hypotheses, topic, utterance, ambiguity_penalty, prob_of_noise, all_forms_including_noisy_variants)
            elif observed_meaning == 'inferred':
                population[hearer_index] = update_posterior(population[hearer_index], hypotheses, listener_response, utterance, ambiguity_penalty, prob_of_noise, all_forms_including_noisy_variants)

        elif mutual_understanding_pressure is False:
            utterance = produce(speaker_language, topic, ambiguity_penalty, error, prob_of_noise)
            # whenever a speaker is called upon to produce a utterance, they first sample a language from their
            # posterior probability distribution. So each agent keeps updating their language according to the data
            # they receive from their communication partner.
            if observed_meaning == 'intended':
                population[hearer_index] = update_posterior(population[hearer_index], hypotheses, topic, utterance, ambiguity_penalty, prob_of_noise, all_forms_including_noisy_variants)
            elif observed_meaning == 'inferred':
                inferred_meaning = receive_without_repair(hearer_language, utterance)
                population[hearer_index] = update_posterior(population[hearer_index], hypotheses, inferred_meaning, utterance, ambiguity_penalty, prob_of_noise, all_forms_including_noisy_variants)

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


# AND NOW FOR THE FUNCTIONS THAT HANDLE CREATING A DATASET FROM A SPECIFIC LANGUAGE TYPE (IN ORDER TO GENERATE THE DATA
# THAT GENERATION 0 WILL LEARN FROM):

def dataset_from_language(language, meaning_list):
    """
    Takes a language and generates a balanced minimal dataset from it, in which each possible meaning occurs exactly
    once, combined with its corresponding form.

    :param language: a language (list of forms_without_noisy_variants that has same length as the global variable
    meanings, where each form is mapped to the meaning at the corresponding index)
    :param meaning_list: list containing all possible meanings; corresponds to global variable 'meanings'
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by
    the form)
    """
    meaning_form_pairs = []
    for i in range(len(language)):
        meaning = meaning_list[i]
        form = language[i]
        meaning_form_pairs.append((meaning, form))
    return meaning_form_pairs


def create_initial_dataset(desired_class, bottleneck, language_list, class_per_language, meaning_list):
    """
    Creates a balanced dataset from a randomly chosen language of the desired class.

    :param desired_class: 'degenerate', 'holistic', 'compositional', or 'other'; category indices as hardcoded
    in classify_language_four_forms function are: 0 = degenerate, 1 = holistic, 2 = compositional, 3 = other
    :param bottleneck: the transmission bottleneck (int); corresponds to global variable 'b'
    :param language_list: list of all languages
    :param class_per_language: list of len(hypothesis_space) which contains an integer indicating the class of the
    language at the corresponding index in the global variable hypothesis_space
    :param meaning_list: list of all possible meanings; corresponds to global variable 'meanings'
    :return: a dataset (list containing tuples, where each tuple is a meaning-form pair, with the meaning followed by
    the form) from a randomly chosen language of the desired class
    """
    if desired_class == 'degenerate':
        class_index = 0
    elif desired_class == 'holistic':
        class_index = 1
    elif desired_class == 'compositional':
        class_index = 2
    elif desired_class == 'other':
        class_index = 3
    language_class_indices = np.where(class_per_language == class_index)[0]
    class_languages = []
    for index in language_class_indices:
        class_languages.append(language_list[index])
    random_language = random.choice(class_languages)
    meaning_form_pairs = dataset_from_language(random_language, meaning_list)
    if bottleneck % len(meaning_form_pairs) != 0:
        raise ValueError("OOPS! b needs to be a multiple of the number of meanings in order for this function to create a balanced dataset.")
    dataset = []
    for i in range(int(bottleneck / len(meaning_form_pairs))):
        dataset = dataset+meaning_form_pairs
    return dataset


# AND NOW A FUNCTION THAT RECORDS THE PROPORTION OF POSTERIOR PROBABILITY THAT IS ASSIGNED TO EACH LANGUAGE CLASS IN A
# GIVEN GENERATION:

def language_stats(population, class_per_language):
    """
    Tracks how well each of the language classes is represented in the populations' posterior probability distributions

    :param population: a population (1D numpy array), where each agent is simply a LOG posterior probability
    distribution
    :param class_per_language: list specifying the class for each corresponding language in the global variable
    \'hypothesis_space'
    :return: a list containing the overall average posterior probability assigned to each class of language in the
    population, where index 0 = degenerate, 1 = holistic, 2 = compositional, 3 = other; these are the category indices
    as hardcoded in the classify_language_four_forms() function.
    """
    stats = np.zeros(n_lang_classes)
    for p in population:
        for i in range(len(p)):
            stats[int(class_per_language[i])] += np.exp(p[i])
    stats = np.divide(stats, len(population))
    return stats


# AND NOW FINALLY FOR THE FUNCTION THAT RUNS THE ACTUAL SIMULATION:

def simulation(population, n_gens, n_rounds, bottleneck, pop_size, hypotheses, class_per_language, log_priors, data, interaction_order, ambiguity_penalty, prob_of_noise, all_possible_forms, mutual_understanding_pressure, minimal_effort_pressure, communicative_success_pressure):
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
    :param ambiguity_penalty: parameter that determines the extent to which the speaker tries to avoid ambiguity;
    corresponds to global variable 'gamma'
    :param prob_of_noise: probability of noise; corresponds to global variable 'noise_prob'
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
                population[j] = update_posterior(population[j], hypotheses, meaning, signal, ambiguity_penalty, prob_of_noise, all_possible_forms)
        data = population_communication(population, n_rounds, mutual_understanding_pressure, minimal_effort_pressure, ambiguity_penalty, prob_of_noise, communicative_success_pressure, hypotheses)
        language_stats_over_gens[i] = language_stats(population, class_per_language)
        data_over_gens.append(data)
        if i == n_gens-1:
            final_pop = population
        if turnover:
            population = new_population(pop_size, log_priors)
        if i % 10 == 0:  # shows the user that the program is making some progress
            print('.')
    return language_stats_over_gens, data_over_gens, final_pop


# AND NOW SOME FUNCTIONS THAT CONVERT VARIABLES TO FORMATS THAT ARE MORE SUITABLE FOR USING IN A FILENAME:

def convert_float_value_to_string(float_value):
    """
    :param float_value: float
    :return: The float converted into a string where dots are removed.
    """
    if float_value % 1. == 0:
        float_value = int(float_value)
    float_string = str(float_value)
    float_string = float_string.replace(".", "")
    return float_string


def convert_array_to_string(array):
    """
    :param array: A 1D numpy array
    :return: The numpy array converted into a string where spaces are replaced by underscores and brackets and dots are
    removed.
    """
    array_string = np.array2string(array, separator=',')
    array_string = array_string.replace(" ", "")
    array_string = array_string.replace(".", "")
    return array_string


###################################################################################################################
if __name__ == '__main__':

    ###################################################################################################################
    # NOW LET'S RUN THE ACTUAL SIMULATION:

    t0 = time.process_time()

    hypothesis_space = create_all_possible_languages(meanings, forms_without_noise)
    print("number of possible languages is:")
    print(len(hypothesis_space))

    class_per_lang = classify_all_languages(hypothesis_space, forms_without_noise, meanings)

    if compressibility_bias:
        priors = prior(hypothesis_space, forms_without_noise, meanings)
    else:
        priors = np.ones(len(hypothesis_space))
        priors = np.divide(priors, np.sum(priors))
        priors = np.log(priors)

    initial_dataset = create_initial_dataset(initial_language_type, b, hypothesis_space, class_per_lang, meanings)  # the data that the first generation learns from

    language_stats_over_gens_per_run = np.zeros((runs, generations, int(max(class_per_lang)+1)))
    data_over_gens_per_run = []
    final_pop_per_run = np.zeros((runs, popsize, len(hypothesis_space)))

    for r in range(runs):
        population = new_population(popsize, priors)

        language_stats_over_gens, data_over_gens, final_pop = simulation(population, generations, rounds, b, popsize, hypothesis_space, class_per_lang, priors, initial_dataset, interaction, gamma, noise_prob, all_forms_including_noisy_variants, mutual_understanding, minimal_effort, communicative_success)

        language_stats_over_gens_per_run[r] = language_stats_over_gens
        data_over_gens_per_run.append(data_over_gens)
        final_pop_per_run[r] = final_pop

    #timestr = time.strftime("%Y%m%d-%H%M%S")

    pickle_file_name = "Pickle_r_" + str(runs) +"_g_" + str(generations) + "_b_" + str(b) + "_rounds_" + str(rounds) + "_size_" + str(popsize) + "_mutual_u_" + str(mutual_understanding) + "_gamma_" + str(gamma) +"_minimal_e_" + str(minimal_effort) + "_c_" + convert_array_to_string(cost_vector) + "_turnover_" + str(turnover) + "_bias_" + str(compressibility_bias) + "_init_" + initial_language_type[:5] + "_noise_prob_" + convert_float_value_to_string(noise_prob) +"_observed_m_" + observed_meaning +"_n_l_classes_" + str(n_lang_classes) +"_CS_" + str(communicative_success) + "_" + convert_float_value_to_string(np.around(communicative_success_pressure_strength, decimals=2)) # + "_" + timestr  # uncomment this last bit and the statement above for defining the timestr in order to add a unique identifier to the filename (to prevent it from being overwritten when running the same simulation again)
    pickle.dump(language_stats_over_gens_per_run, open(pickle_file_path + pickle_file_name + "_lang_stats" + ".p", "wb"))
    pickle.dump(data_over_gens_per_run, open(pickle_file_path+pickle_file_name+"_data"+".p", "wb"))
    pickle.dump(final_pop_per_run, open(pickle_file_path + pickle_file_name + "_final_pop" + ".p", "wb"))

    t1 = time.process_time()

    print('')
    print("number of minutes it took to run simulation:")
    print(round((t1-t0)/60., ndigits=2))

    print('')
    print('results were saved in folder:')
    print(pickle_file_path)

    print('')
    print('using filename:')
    print(pickle_file_name)
