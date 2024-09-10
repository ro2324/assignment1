# models.py
from collections import Counter
import random

import numpy as np

from sentiment_data import *
from utils import *
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def __init__(self):
        self.weights = None
        self.indexer = Indexer()

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        try:
            features = Counter()
            for word in sentence:
                word = word.lower()  # Normalize to lowercase
                if word not in STOPWORDS:
                    if add_to_indexer:
                        # Add the word to the indexer if not present
                        index = self.indexer.add_and_get_index(word)
                    else:
                        # Get the index of the word if it exists
                        index = self.indexer.index_of(word)
                    if index != -1:  # Ensure the word was indexed
                        features[index] += 1
            return features
        except Exception as e:
            raise NotImplementedError("Subclasses should implement this method.")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer=None):
        super().__init__()
        self.indexer = indexer if indexer else Indexer()

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            word = word.lower()
            if word not in STOPWORDS:
                if add_to_indexer:
                    # Add the word to the indexer and get the index
                    index = self.indexer.add_and_get_index(f"Unigram={word}")
                    print(f"Adding word '{word}' with index {index} to indexer")  # Debugging statement
                else:
                    index = self.indexer.add_and_get_index(f"Unigram={word}")
                    print(f"Index of word '{word}' is {index}")  # Debugging statement

            if index != -1:
                features[index] += 1
        print(f"Features extracted: {features}")  # Debugging statement
        print(f"Indexer size: {len(self.indexer)}")  # Debugging statement
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        super().__init__()

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        try:
            features = Counter()
            for i in range(len(sentence) - 1):
                bigram = f"Bigram={sentence[i].lower()}|{sentence[i + 1].lower()}"
                index = self.indexer.add_and_get_index(bigram) if add_to_indexer else self.indexer.index_of(bigram)
                if index != -1:
                    features[index] += 1
            return features
        except Exception as e:
            raise Exception("Exception arisen: {}", e)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        super().__init__()

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            word = word.lower()
            if word not in STOPWORDS:
                index = self.indexer.add_and_get_index(f"Better={word}") if add_to_indexer else self.indexer.index_of(
                    f"Better={word}")
                if index != -1:
                    features[index] += 1
        return features


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, )
        score = sum(self.weights[idx] * value for idx, value in features.items())
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[idx] * value for idx, value in features.items())
        prob = 1 / (1 + np.exp(-score))
        return 1 if prob > 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    num_epochs = 10
    if len(feat_extractor.get_indexer()) == 0:
        raise ValueError("Indexer has no entries, ensure feature extraction is populating indices.")
    weights = np.zeros(len(feat_extractor.get_indexer()))
    for epoch in range(num_epochs):
        random.shuffle(train_exs)  # Make sure train_exs is a list
        for ex in train_exs:
            print(f"Training example words: {ex.words}")  # Debugging statement

            features = feat_extractor.extract_features(ex.words, True)
            prediction = sum(weights[feat] * count for feat, count in features.items()) >= 0
            if prediction != (ex.label == 1):
                for feat, count in features.items():
                    weights[feat] += (ex.label - prediction) * count
    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    num_epochs = 10
    learning_rate = 0.01

    if len(feat_extractor.get_indexer()) == 0:
        raise ValueError("Indexer has no entries, ensure feature extraction is populating indices.")

    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=True)
            if len(features) == 0:
                print(f"No features extracted from: {ex.words}")  # Debugging statement
            score = sum(feat_extractor.weights[idx] * val for idx, val in features.items())
            prob = 1 / (1 + np.exp(-score))
            error = ex.label - prob
            for idx, val in features.items():
                feat_extractor.weights[idx] += learning_rate * error * val
    weights = np.zeros(len(feat_extractor.get_indexer()))
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    indexer = Indexer()
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(indexer)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(indexer)
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(indexer)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
