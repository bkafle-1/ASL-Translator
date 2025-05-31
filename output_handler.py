import re
import wordninja
import itertools

custom_model = wordninja.LanguageModel('word_list.txt.gz')

def substitute_placeholders(tokens):
    # Identify positions of placeholders
    choices = []
    for token in tokens:
        if token == '(6W)':
            choices.append(['6', 'w'])
        elif token == '(9F)':
            choices.append(['9', 'f'])
        else:
            choices.append([token])

    # Generate all combinations
    return [''.join(candidate) for candidate in itertools.product(*choices)]

def score(text):
    words = wordninja.split(text)
    long_words = [w for w in words if len(w) > 1]
    return len(long_words), sum(len(w) for w in long_words)

def convert_input_to_output_best(tokens):
    variants = substitute_placeholders(tokens)
    tokens = []
    best = max(variants, key=score)
    tokens.extend(wordninja.split(best))
    return(tokens)
