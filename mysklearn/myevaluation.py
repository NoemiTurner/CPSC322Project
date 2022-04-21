from mysklearn import myutils
import numpy as np
import math

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    np.random.seed(random_state)
    if shuffle == True:
        myutils.randomize_in_place(X, y)
    if type(test_size) == float:
        train_size = len(X) - math.ceil(len(X) * test_size)
        for index in range(train_size):
            X_train.append(X[index])
            y_train.append(y[index])
        for index in range(train_size, len(X)):
            X_test.append(X[index])
            y_test.append(y[index])
    if type(test_size) == int:
        train_size = len(X) - test_size
        for index in range(train_size):
            X_train.append(X[index])
            y_train.append(y[index])
        for index in range(train_size, len(X)):
            X_test.append(X[index])
            y_test.append(y[index])
        
    return X_train, X_test, y_train, y_test 

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    total_folds = []
    list_of_indexes = [*range(0, len(X), 1)]
    current_index = 0
    np.random.seed(random_state)
    if shuffle == True:
        np.random.shuffle(list_of_indexes)
    num_first_folds = len(X) % n_splits
    sizes_of_first_folds = len(X) // n_splits + 1
    equal_size_of_folds = len(X) // n_splits
    num_other_folds = n_splits - num_first_folds
    sizes_of_other_folds = len(X) // n_splits
    # all folds are equal sizes
    if num_first_folds == 0:
        for fold in range(n_splits):
            fold_temp = []
            while len(fold_temp) < equal_size_of_folds:
                fold_temp.append(list_of_indexes[current_index])
                current_index += 1
            total_folds.append(fold_temp)
    else:
        # adding the first folds to the list
        for fold in range(num_first_folds):
            fold_temp = []
            while len(fold_temp) < sizes_of_first_folds:
                fold_temp.append(list_of_indexes[current_index])
                current_index += 1
            total_folds.append(fold_temp)
        # adding the other fold to the list
        for fold in range(num_other_folds):
            fold_temp = []
            while len(fold_temp) < sizes_of_other_folds:
                fold_temp.append(list_of_indexes[current_index])
                current_index += 1
            total_folds.append(fold_temp)
    for fold in total_folds:
        X_test_folds.append(fold)
        train = []
        tmp = []
        for fold_train in total_folds:
            if fold_train != fold:
                train.append(fold_train)
        tmp = [j for sub in train for j in sub]
        X_train_folds.append(tmp)
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_train_folds = []
    X_test_folds = []
    partitioned_x = []
    partitioned_y = []
    class_label = []
    total_folds = []
    list_of_indexes = [*range(0, len(X), 1)]
    np.random.seed(random_state)
    if shuffle == True:
        np.random.shuffle(list_of_indexes)
    # group by class label
    for label in y:
        if label not in class_label:
            class_label.append(label)

    for label in class_label:
        partition_x = []
        partition_y = []
        for item in range(len(y)):
            if label == y[item]:
                partition_x.append(X[item])
                partition_y.append(y[item])
        partitioned_x.append(partition_x)
        partitioned_y.append(partition_y)
    # create a blank 2d list
    for splits in range(n_splits):
        fold = []
        total_folds.append(fold)
    # distribute each partition equally to each fold
    for fold in total_folds:
        index = 0
        for label in partitioned_x:
            fold.append(label[index])
            label.pop(index)
        index += 1
    # join folds into 1 string
    for fold in total_folds:
        X_test_folds.append(fold)
        train = []
        tmp = []
        for fold_train in total_folds:
            if fold_train != fold:
                train.append(fold_train)
        for nums in train:
            for val in nums:
                tmp.append(val)
        X_train_folds.append(tmp)

    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    indexes_used = []
    indexes_not_used = []
    np.random.seed(random_state)
    if n_samples == None:
        n_samples = len(X)
    for sample in range(n_samples):
        index = np.random.randint(0, len(X))
        indexes_used.append(index)
        X_sample.append(X[index])
        if y != None:
            y_sample.append(y[index])

    for index in range(len(X_sample)):
        if index not in indexes_used:
            X_out_of_bag.append(X[index])
            if y != None:
                y_out_of_bag.append(y[index])
    if y == None:
        y_sample = None
        y_out_of_bag = None
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag 

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = np.zeros((len(labels), len(labels)))
    i = 0
    # loop through permutations and add the counts in the matrix
    while i < len(labels):
        for j in range(len(labels)):
            for index in range(len(y_true)):
                if y_true[index] == labels[i] and y_pred[index] == labels[j]:
                    matrix[i][j] += 1
        i += 1   
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    score = 0.0
    true_count = 0
    for index in range(len(y_true)):
        if y_true[index] == y_pred[index]:
            true_count += 1
    if normalize == True:
        score = true_count / len(y_true)
        return score
    score = true_count
    return score

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels == None:
        labels = np.unique(y_true)
    if pos_label == None:
        pos_label = labels[0]
    tp, fp, tn, fn = myutils.get_tp_fp(y_true, y_pred, pos_label)
    if tp + fp == 0:
        return 0.0
    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels == None:
        labels = np.unique(y_true)
    if pos_label == None:
        pos_label = labels[0]
    tp, fp, tn, fn = myutils.get_tp_fp(y_true, y_pred, pos_label)
    if tp + fp == 0:
        return 0.0
    recall = tp / (tp + fn)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
