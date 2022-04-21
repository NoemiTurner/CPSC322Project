# TODO: your reusable general-purpose functions here
import numpy as np
from math import sqrt
from math import pi
from math import exp

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]
                
def get_tp_fp(y_true, y_pred, pos_label):
    """
    Given a list of true labels and a list of predicted labels, return the true positives, true negatives, false positives, and false negatives
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(len(y_true)):
        if y_true[index] == y_pred[index] == pos_label:
            tp += 1
        elif y_true[index] == y_pred[index]:
            tn += 1
        elif y_pred[index] == pos_label:
            fp += 1
        else:
            fn += 1
    return tp, fp, tn, fn

def compute_priors(X_train, y_train):
    """
    Given a list of true labels and a list of predicted labels, return the prior probability of the positive class
    
    Returns:
        priors: a dictionary with the keys being the unique labels in y_train and the values being the prior probability of the label
    """
    priors = {}
    labels = np.unique(y_train)
    for label in labels:
        count = 0
        for instance in y_train:
            if label == instance:
                count += 1
        priors[label].append(count / len(y_train))
    return priors

def adjust_X_test(X_test, table):
    """Adjusts the X_test values so that they can index into the posteriors correctly

        Args:
            X_test : holds the values that need to be adjusted
            table : holds the data that is used to adjust the values

        Returns:
            adjusted_X : holds the adjusted values for X_test

        Notes:
            needed for naive predict
        """
    attribute_values = []
    adjusted_X = []
    attribute_count = 0
    for index in range(len(table[0])):
        values, counts = get_frequencies(get_column(table, index))
        attribute_values.append(values)
    index = 0
    for item in X_test:
        new_X_item = []
        for val in item:
            for item in attribute_values[attribute_count]:
                if val == item:
                    score = index
                    for index in range(attribute_count):
                        score = score + len(attribute_values[index])
                    new_X_item.append(score)
                    attribute_count += 1
                index += 1
            index = 0
        adjusted_X.append(new_X_item)
        attribute_count = 0
    return adjusted_X

def compute_posteriors(X_train, y_train):
    """
    Given a list of true labels and a list of predicted labels, return the posterior probability of the positive class
    """
    posteriors = []
    # compute posteriors on the training set
    for index in range(len(X_train)):
        # compute the posterior probability of the positive class
        posterior = 0
        for label in np.unique(y_train):
            # compute the prior probability of the label
            prior = 0
            for instance in y_train:
                if label == instance:
                    prior += 1
            prior = prior / len(y_train)
            # compute the likelihood of the instance
            likelihood = 0
            for j in range(len(X_train[index])):
                likelihood += X_train[index][j] * 1
            # compute the posterior probability of the positive class
            posterior += prior * likelihood
        posteriors.append(posterior)

    return posteriors

def split_data(X, header, col_to_split_on):

    X_split = X.copy()
    y = []
    for index in range(len(X_split)):

        y.append(X_split[index][col_to_split_on])
        del X_split[index][col_to_split_on]

    return X_split, y

def separate_by_class(X_dataset, y_dataset):
    # Split the dataset by class values, returns a dictionary
    # Assumes the last value in the table is the class label
    X_separated, y_separated = {} , {}
    for index in range(len(y_dataset)):
        class_value = y_dataset[index]
        if (class_value not in y_separated):
            X_separated[class_value] = list()
            y_separated[class_value] = list()
        X_separated[class_value].append(X_dataset[index])
        y_separated[class_value].append(y_dataset[index])
    return X_separated, y_separated

def get_X(table, drop_index):
    """Removes a column from a mypytable object"""
    for row in table:
        row.pop(drop_index)
    return table

def get_frequencies(data):
    """Gets the number of instances of a value from a table

        Args:
            table (2D list): holds the data from a csv file
            header (list): contains the column names in table
            col_name (string): the name of the column to be counted

        Returns:
            two lists containg the item and its frequency
        """
    col = data
    col.sort() # inplace
    # # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts # we can return multiple values in python

def get_num_instances(name, indexes, column):
    """Gets the number of instances for a value

        Args:
            name : holds attribute value
            indexes : holds the indexes to check
            column : holds the values for a given attribtue


        Returns:
            count : number of instances

        Notes:
            needed for naive predict
        """
    count = 0
    for index in indexes:
        if column[index] == name:
            count += 1
    return count

def group_by_multiple_atts(indexes, y):
    groups = []
    group_indexes = []
    for item in y:
        if item not in groups:
            groups.append(item)
            group_indexes.append([])
    groups.sort()
    index = 0
    for item in y:
        for index_j in range(len(groups)):
            if item == groups[index_j]:
                group_indexes[index_j].append(indexes[index])
        index += 1
    return group_indexes

def get_column(table, col_index):
    """Extracts a column from a mypytable object

        Args:
            table (Mypytable()): holds the data from a csv file

        Returns:
            col (list): contains the values extraceted from the table at column

        Notes:
            Changed to NA for vgsales.csv file
        """
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_X_train(table, drop_index):
    for row in table:
        row.pop(drop_index)
    return table

def find_max(values):
    """FInds the max of a list of values

        Args:
            values : list of floats

        Returns:
            max_index : holds the index of the max_value found

        Notes:
            used in naive predict
        """
    max_val = 0
    max_index = 0
    for index in range(len(values)):
        if values[index] > max_val:
            max_val = values[index]
            max_index = index
    return max_index

def compute_euclidean_distance(v1, v2):
    return math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))