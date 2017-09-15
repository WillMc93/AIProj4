import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3, min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Holder for best score and model
        besties = float('inf'), None

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                num_feats = self.X.shape[1]
                num_params = n_components * (n_components - 1) + 2 * num_feats * n_components
                logN = np.log(self.X.shape[0])
                bic = -2 * logL + num_params * logN
                if bic < besties[0]:
                    besties = bic, model

            # Catch errors
            except Exception as e:
                #print(e)
                continue


        return besties[1] if besties[1] is not None else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    models, values = {}, {}

    # Helper Function
    @classmethod
    def get_dictionary(cls, instance):
        #models, values = {}, {}
        for n_components in range(instance.min_n_components, instance.max_n_components + 1):
            n_components_models, n_components_vals = {}, {}

            for word in instance.words.keys():
                X, lengths = instance.hwords[word]
                try:
                    model = GaussianHMM(n_components=n_components, n_iter=1000,
                                        random_state=instance.random_state).fit(X, lengths)
                    logL = model.score(X, lengths)

                    n_components_models[word] = model
                    n_components_vals[word] = logL

                # Catch errors
                except Exception as e:
                        #print(e)
                        continue

            SelectorDIC.models[n_components] = n_components_models
            SelectorDIC.values[n_components] = n_components_vals


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # get them models 'n values
        #self.models, self.values = self.get_dictionary(self)
        if not len(SelectorDIC.models):
            self.get_dictionary(self)

        # Holder for best score and model
        besties = float('-inf'), None

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            models, vals = SelectorDIC.models[n_components], SelectorDIC.values[n_components]

            if(self.this_word not in vals):
                continue

            mean = np.mean([vals[word] for word in vals.keys() if word != self.this_word])
            dic = vals[self.this_word] - mean

            if dic > besties[0]:
                besties = dic, models[self.this_word]

        return besties[1] if besties[1] is not None else self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    # Class param
    n_splits = 5

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        besties = float('-inf'), None

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []

            if (len(self.sequences) < SelectorCV.n_splits):
                break

            split_method = KFold(random_state=self.random_state, n_splits=SelectorCV.n_splits)
            for train_idx, test_idx in split_method.split(self.sequences):
                X_train, len_train = combine_sequences(train_idx, self.sequences)
                X_test, len_test = combine_sequences(test_idx, self.sequences)

                try:
                    model = GaussianHMM(n_components=n_components, random_state=self.random_state,
                                        n_iter=1000).fit(X_train, len_train)

                    logL = model.score(X_test, len_test)
                    scores.append(logL)

                except Exception as e:
                    break

                mean = np.mean(scores) if len(scores) > 0 else float('-inf')

                if mean > besties[0]:
                    besties = mean, model

        return besties[1] if besties[1] is not None else self.base_model(self.n_constant)
