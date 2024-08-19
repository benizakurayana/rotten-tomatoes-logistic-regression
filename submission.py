#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = defaultdict(int)  # When the int class is passed as the argument, a defaultdict is created with default value as zero.
    for word in x.strip().split(' '):
        d[word] += 1  # With defaultdict, no need to enter the key first
    return d
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # the weight vector

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    # w_1 = w_0 - alpha * dL_dw_0 = w_0 - alpha*(h_1 - y_1) * FeatureVector_1
    #     = w_0 - alpha * (1 / (1+e^(-k)) - y_1) * FeatureVector_1
    # where k = w_0*FeatureVector_1
    #       scale =  - alpha * (1 / (1+e^(-k)) - y_1)
    trainExamples_updated = [(example[0], 0 if example[1] == -1 else 1) for example in trainExamples]
    validationExamples_updated = [(example[0], 0 if example[1] == -1 else 1) for example in validationExamples]
    for epoch in range(numEpochs):
        for example in trainExamples_updated:
            featureVector = featureExtractor(example[0])

            def sigmoid(k):
                return 1 / (1 + math.exp(-k))
            scale = -1 * alpha * (sigmoid(dotProduct(weights, featureVector)) - example[1])
            increment(weights, scale, featureVector)

        def predictor(x):
            return 1 if dotProduct(weights, featureExtractor(x)) >= 0 else 0
        training_error = evaluatePredictor(trainExamples_updated, predictor)
        validation_error = evaluatePredictor(validationExamples_updated, predictor)
        print(f'Training Error: ({epoch}epoch): {training_error}\nValidation Error: ({epoch}epoch):{validation_error}')
    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        phi = defaultdict(int)
        for i in range(random.randint(1, len(weights))):
            random_word = random.choice(list(weights.keys()))
            phi[random_word] += 1
        y = 1 if dotProduct(weights, phi) >= 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        d = defaultdict(int)
        x_without_space = x.replace(' ', '')
        for i in range(len(x_without_space) - n + 1):
            d[x_without_space[i:i+n]] += 1
        return d
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

