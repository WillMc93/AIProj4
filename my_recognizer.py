import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # Iterate through Test Set
    for idx in range(0, len(test_set.get_all_Xlengths())):
        sequence, length = test_set.get_item_Xlengths(idx)
        # Dict container for probabilities this loop
        logLs = {}

        for word, model in models.items():
            try:
                score = model.score(sequence, length)
                logLs[word] = score
            except:
                # If model.score throws an exception give it lowest possible score
                logLs[word] = float('-inf')
                continue

        # Append the probability scores
        probabilities.append(logLs)

        # Append the word with the maximum score for each model
        guesses.append(max(logLs, key=logLs.get))

    return probabilities, guesses
