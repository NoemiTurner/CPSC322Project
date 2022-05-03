from itertools import count
import numpy as np
from mysklearn import myutils
import operator as op
from mysklearn import myutils

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = self.regressor.predict(X_test)
        discretized_predictions = self.discretizer(predictions)
        return discretized_predictions

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        neighbors = []
        distances = []
        all_distances = []
        neighbor_indices = []
        toneighbor_indices = []
        # Calculates all X_train distances from the test instance
        for test_instance in X_test:
            for i, train_coordinates in enumerate(self.X_train):
                neighbors.append([i,myutils.compute_euclidian_distance(train_coordinates,test_instance)])
            # Now we must sort the list based upon the distances
            neighbors.sort(key=op.itemgetter(-1))
            # Now we grab the k closest neighbors to this point
            top_neighbors = neighbors[:self.n_neighbors]
            for neighbor in top_neighbors:
                neighbor_indices.append(neighbor[0])
                distances.append(neighbor[1])
            all_distances.append(distances)
            toneighbor_indices.append(neighbor_indices)
            distances = []
            neighbor_indices = []
            neighbors = []
        return all_distances, toneighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        neighbor_votes = []
        y_predicted = []
        knn_distances, knn_indexes = self.kneighbors(X_test)
        for i, test_instance in enumerate(X_test):
            for index in knn_indexes[i]:
                neighbor_votes.append(self.y_train[index])
            nvalues, nfreqs = myutils.get_frequencies(neighbor_votes)
            # Finding the class label with the maximum frequency, to get the majority vote
            try:
                y_predicted.append(int(nvalues[nfreqs.index(max(nfreqs))]))
            except ValueError:
                y_predicted.append(nvalues[nfreqs.index(max(nfreqs))])
            neighbor_votes = []
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = myutils.get_most_common(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [self.most_common_label for test_instance in X_test]
        return y_predicted


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        priors = []
        posts = []
        indexes = [index for index in range(len(X_train))]
        attribute_values = []
        for index in range(len(X_train[0])):
            values, counts = myutils.get_frequencies(myutils.get_column(X_train, index))
            attribute_values.append(values)

        # frequencies returns a list of y values in alphabetical order and corresponding frequencies
        frequencies = myutils.get_frequencies(y_train.copy())
        for index in range(len(frequencies[0])):
            priors.append(frequencies[1][index] / len(y_train))
            posts.append({})
        
        self.priors = priors
        group_indexes = myutils.group_by_multiple_atts(indexes, y_train)
        index = 0
        index_j = 0
        for item in group_indexes:
            for row in attribute_values:
                for name in row:
                    count = myutils.get_num_instances(name, item, myutils.get_column(X_train, index_j))
                    posts[index].update({name: (count/len(item))})
                    
                
                index_j += 1
            index_j = 0
            index += 1
        self.posteriors = posts

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # multiply all posteriors for the class label then multiply by the prior value at that class label
        y_predicted = []
        posts = []
        indexes = []

        for item in X_test:
            posts = []
            # get classification score
            for index in range(len(self.priors)):
                predicts = []
                # iterate through each attribute
                for val in item: 
                    predicts.append(self.posteriors[index][val])
                prior = self.priors[index]
                for p in predicts:
                    prior = prior * p
                posts.append(prior)
                index += 1
            indexes = myutils.find_max(posts)
            y_predicted.append(indexes)
        return y_predicted
