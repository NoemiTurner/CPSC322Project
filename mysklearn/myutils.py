# TODO: your reusable general-purpose functions here
import math
import numpy as np
from math import pi

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

def compute_euclidian_distance(v1, v2):
    return math.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def get_most_common(col):
    """ Get the frequencies in a certain column
    Args:
        col (list): the column to be searched
    Returns:
        values[max_count] (item in list): value in list that is the max occurances
    """
    for item in col:
        col[col.index(item)] = str(item)
    col.sort() # inplace 
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)
    max_count = counts.index(max(counts))
    return values[max_count]

def partition_instances(instances, split_attribute, header):
    # lets use a dictionary
    partitions = {}
    att_index = int(split_attribute)
    att_domain = header[att_index]
    for att_value in att_domain:
        partitions[att_value] = []
        # TODO: finish
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def select_attribute(instances, attributes, header):
    #TODO: use entropy to compute and choose the attribute with the smallest Enew
    # for now we will choose randomly
    # get the domain of y
    y = []
    for item in instances:
        if item[-1] not in y:
            y.append(item[-1])
    # get the index of instances left
    index_list = []
    for item in attributes:
        index_list.append(int(item[3:]))
    # find the lowest Enew for each attribute
    Enew = []
    for i in range(len(attributes)):
        Enew_sum = 0
        for dom in header[attributes[i]]:
            nums = []
            for y_dom in y:
                nums.append(0)
            for j in range(len(instances)):
                if instances[j][index_list[i]] == dom:
                    nums[y.index(instances[j][-1])] += 1
            E = 0
            for num in nums:
                if num == 0:
                    E = 0
                    break
                e += (num/sum(nums)) * math.log(num/sum(nums), 2)
            E *= -1
            weight = sum(nums)/len(instances)
            Enew_sum += E*weight
        Enew.append(Enew_sum)
    index = Enew.index(min(Enew))
    return attributes[index]

def all_same_class(instances):
    # returns true if all instances have the same class label
    for instance in instances:
        if instance[-1] != instances[0][-1]:
            return False
    return True

def tdidt(current_instances, available_attributes):
    # basic approach (uses recursion!!):
    print("available attributes", available_attributes)

    # select an attribute to split on
    attribute = select_attribute(current_instances, available_attributes)
    print("splitting on attribute: ", attribute)
    available_attributes.remove(attribute)
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    # group by attribute domain
    partitions = partition_instances(current_instances, attribute)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        print("current attribute value", att_value, len(att_partition))
        value_subtree = ["Value", att_value]
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition) == True:
            #all same class returns true if they have the same class label
            print("Case 1 all same class")
            # TODO: fix the last 2 input numbers
            value_subtree.append(["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)])
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and  len(available_attributes) == 0:
            print("Case 2 no more attributes")
            # TODO: we have a mix of class labels, handle clash with majority vote leaf node
            majority_vote = get_frequencies(att_partition)
            value_subtree.append(["Leaf", majority_vote, len(att_partition), len(current_instances)])
            tree.append(value_subtree)
            # count the number of True and False class labels and choose the one with the most votes

    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            print("Case 3 empty partition")
            # TODO: backtrack to replace the attribute node with a majority vote leaf node
            # replace tree = ["Attribute", attribute] with a majority vote
            majority_vote = get_frequencies(current_instances)
            leaf = ["Leaf", majority_vote, len(current_instances), len(current_instances)]
            return leaf
        else: # the previous conditions were all false, recurse
            subtree = tdidt(att_partition, available_attributes.copy())
            value_subtree.append(subtree)
            # note the copy
        value_subtree.append(subtree)

    return tree