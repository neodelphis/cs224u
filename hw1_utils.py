from collections import defaultdict
import csv
import itertools
from scipy.stats import spearmanr
from IPython.display import display
import numpy as np
import os
import pandas as pd
import vsm
import utils
from nltk.corpus import wordnet as wn

import platform
if platform.system() == 'Darwin': # un mac
    PATH_TO_DATA = '/Users/pierrejaumier/Data/cs224u'
else:
    PATH_TO_DATA = '/home/neo/Data/cs224u'

VSM_HOME = os.path.join(PATH_TO_DATA, 'vsmdata')

WORDSIM_HOME = os.path.join(PATH_TO_DATA, 'wordsim')



# Dataset readers
def wordsim_dataset_reader(
        src_filename,
        header=False,
        delimiter=',',
        score_col_index=2):
    """
    Basic reader that works for all similarity datasets. They are
    all tabular-style releases where the first two columns give the
    words and a later column (`score_col_index`) gives the score.

    Parameters
    ----------
    src_filename : str
        Full path to the source file.

    header : bool
        Whether `src_filename` has a header.

    delimiter : str
        Field delimiter in `src_filename`.

    score_col_index : int
        Column containing the similarity scores Default: 2

    Yields
    ------
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity
       score in the file so that we are intuitively aligned with our
       distance-based code. To align with our VSMs, all the words are
       downcased.

    """
    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)
        for row in reader:
            w1 = row[0].strip().lower()
            w2 = row[1].strip().lower()
            score = row[score_col_index]
            # Negative of scores to align intuitively with distance functions:
            score = -float(score)
            yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.gabrilovich.com/resources/data/wordsim353/"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'wordsim353', 'combined.csv')
    return wordsim_dataset_reader(
        src_filename, header=True)

def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'MTURK-771.csv')
    return wordsim_dataset_reader(
        src_filename, header=False)

def simverb3500dev_reader():
    """SimVerb-3500: https://www.aclweb.org/anthology/D16-1235/"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'SimVerb-3500', 'SimVerb-500-dev.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=False, score_col_index=3)

def simverb3500test_reader():
    """SimVerb-3500: https://www.aclweb.org/anthology/D16-1235/"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'SimVerb-3500', 'SimVerb-3000-test.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=False, score_col_index=3)


def men_reader():
    """MEN: https://staff.fnwi.uva.nl/e.bruni/MEN"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'MEN', 'MEN_dataset_natural_form_full')
    return wordsim_dataset_reader(
        src_filename, header=False, delimiter=' ')

READERS = (wordsim353_reader, mturk771_reader, simverb3500dev_reader,
           simverb3500test_reader, men_reader)



def get_reader_name(reader):
    """
    Return a cleaned-up name for the dataset iterator `reader`.
    """
    return reader.__name__.replace("_reader", "")

def get_reader_vocab(reader):
    """Return the set of words (str) in `reader`."""
    vocab = set()
    for w1, w2, _ in reader():
        vocab.add(w1)
        vocab.add(w2)
    return vocab

def get_reader_pairs(reader):
    """
    Return the set of alphabetically-sorted word (str) tuples
    in `reader`
    """
    return {tuple(sorted([w1, w2])): score for w1, w2, score in reader()}

def word_similarity_evaluation(reader, df, distfunc=vsm.cosine):
    """
    Word-similarity evaluation framework.

    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).

    df : pd.DataFrame
        The VSM being evaluated.

    distfunc : function mapping vector pairs to floats.
        The measure of distance between vectors. Can also be
        `vsm.euclidean`, `vsm.matching`, `vsm.jaccard`, as well as
        any other float-valued function on pairs of vectors.

    Raises
    ------
    ValueError
        If `df.index` is not a subset of the words in `reader`.

    Returns
    -------
    float, data
        `float` is the Spearman rank correlation coefficient between
        the dataset scores and the similarity values obtained from
        `df` using  `distfunc`. This evaluation is sensitive only to
        rankings, not to absolute values.  `data` is a `pd.DataFrame`
        with columns['word1', 'word2', 'score', 'distance'].

    """
    data = []
    for w1, w2, score in reader():
        d = {'word1': w1, 'word2': w2, 'score': score}
        for w in [w1, w2]:
            if w not in df.index:
                raise ValueError(
                    "Word '{}' is in the similarity dataset {} but not in the "
                    "DataFrame, making this evaluation ill-defined. Please "
                    "switch to a DataFrame with an appropriate vocabulary.".
                    format(w, get_reader_name(reader)))
        d['distance'] = distfunc(df.loc[w1], df.loc[w2])
        data.append(d)
    data = pd.DataFrame(data)
    rho, pvalue = spearmanr(data['score'].values, data['distance'].values)
    return rho, data

def word_similarity_error_analysis(eval_df):
    eval_df['distance_rank'] = _normalized_ranking(eval_df['distance'])
    eval_df['score_rank'] = _normalized_ranking(eval_df['score'])
    eval_df['error'] =  abs(eval_df['distance_rank'] - eval_df['score_rank'])
    return eval_df.sort_values('error')


def _normalized_ranking(series):
    ranks = series.rank(method='dense')
    return ranks / ranks.sum()

def full_word_similarity_evaluation(df, readers=READERS, distfunc=vsm.cosine):
    """
    Evaluate a VSM against all datasets in `readers`.

    Parameters
    ----------
    df : pd.DataFrame

    readers : tuple
        The similarity dataset readers on which to evaluate.

    distfunc : function mapping vector pairs to floats.
        The measure of distance between vectors. Can also be
        `vsm.euclidean`, `vsm.matching`, `vsm.jaccard`, as well as
        any other float-valued function on pairs of vectors.

    Returns
    -------
    pd.Series
        Mapping dataset names to Spearman r values.

    """
    scores = {}
    for reader in readers:
        score, data_df = word_similarity_evaluation(reader, df, distfunc=distfunc)
        scores[get_reader_name(reader)] = score
    series = pd.Series(scores, name='Spearman r')
    series['Macro-average'] = series.mean()
    return series

def ttest(df):
    X = df.to_numpy()
    X_sum = X.sum()
    P_j = X.sum(axis=0) / X_sum
    P_i = X.sum(axis=1) / X_sum
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = (X[i,j] / X_sum - P_i[i] * P_j[j]) / np.sqrt(P_i[i] * P_j[j])
    return pd.DataFrame(X, index=df.index, columns=df.columns)

def subword_enrichment(df, n=4):
    # 1. Use `vsm.ngram_vsm` to create a character-level
    # VSM from `df`, using the above parameter `n` to
    # set the size of the ngrams.

    cf = vsm.ngram_vsm(df, n) # Character level VSM


    # 2. Use `vsm.character_level_rep` to get the representation
    # for every word in `df` according to the character-level
    # VSM you created above.
    
    clr = [] # character level representation
    for w, _ in df.iterrows():
        clr.append(vsm.character_level_rep(w, cf, n))
    clr = np.array(clr)

    # 3. For each representation created at step 2, add in its
    # original representation from `df`. (This should use
    # element-wise addition; the dimensionality of the vectors
    # will be unchanged.)

    # subword enrichment :swe
    swe = df.to_numpy() + clr


    # 4. Return a `pd.DataFrame` with the same index and column
    # values as `df`, but filled with the new representations
    # created at step 3.

    return pd.DataFrame(swe, index=df.index, columns=df.columns)

# retrofitting
def get_wordnet_edges():
    edges = defaultdict(set)
    for ss in wn.all_synsets():
        lem_names = {lem.name() for lem in ss.lemmas()}
        for lem in lem_names:
            edges[lem] |= lem_names
    return edges