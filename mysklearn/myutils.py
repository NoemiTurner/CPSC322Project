import math
import numpy as np
from math import pi

def randomize_in_place(alist, parallel_list=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        # rand int in [0, len(alist))
        rand_index = np.random.randint(0, len(alist))
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
    X_separated, y_separated = {}, {}
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
    col.sort()  # inplace
    # # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values:  # seen it before
            counts[-1] += 1  # okay because sorted
        else:  # haven't seen it before
            values.append(value)
            counts.append(1)
    return values, counts  # we can return multiple values in python

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
    col.sort()  # inplace
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values:  # seen it before
            counts[-1] += 1  # okay because sorted
        else:  # haven't seen it before
            values.append(value)
            counts.append(1)
    max_count = counts.index(max(counts))
    return values[max_count]

def compute_random_subset(values, num_values, random_state=None):
    # used for F in RandomForest
    # there is a function np.random.choice()
    if random_state is not None:
        np.random.seed(random_state)
    values_copy = values[:]  # shallow copy
    np.random.shuffle(values_copy)  # in place shuffle
    return values_copy[:num_values]

def majority_vote(att_partition):  # working for basic 1 element list
    """Used to determine a clash
        Args:
            att_partition : the attributes partitioned
        Returns:
            the majority vote
        Notes:
            needed for clashes in decision tree
        """
    majority = att_partition[0][-1]
    majority_count = 0
    for vote in att_partition:
        vote_count = 0
        for other_vote in att_partition:
            if vote[-1] == other_vote[-1]:
                vote_count += 1
        if vote_count > majority_count:
            majority = vote[-1]
            majority_count = vote_count
    return majority

def create_table_from_parallel_lists(original_table, list_of_parallel_lists):
    new_cols = []
    new_cols_inner = []
    for index in range(len(original_table.data)):
        for parallel_list in list_of_parallel_lists:
            stats_cols_inner.append(parallel_list[index])
        new_cols.append(stats_cols_inner)
        stats_cols_inner = []
    return new_cols

def print_decision_rules_helper(tree):
    if tree[0] == "Leaf":  # Leaf node case
        print("THEN", tree[1])
        return tree
    else:
        if tree[0] == "Attribute":
            print("IF", tree[1], "=", tree[2][1], "AND", end=" ")
        for node_index in range(len(tree)):
            if node_index > 1:
                print_decision_rules_helper(tree[node_index])
                if tree[0] == "Attribute":
                    print("IF", tree[1], "=", tree[2][1], "AND", end=" ")

def convert_header_to_string(table):
    """ When the header is apart of the table, this is a good function to change just the header to a string
    """
    new_header = []
    new_table = []
    for element in table[0]:
        try:
            new_header.append(str(element))
        except:
            new_header.append(element)
    for i, row in enumerate(table):
        if i == 0:
            new_table.append(new_header)
        else:
            new_table.append(row)
    return new_table

def decision_traverse(tree, X_test_instance):
    if tree[0] == "Leaf":
        return tree[1]
    else:
        if tree[0] == "Attribute":
            attribute_index = tree[1][3]
            attribute_value = X_test_instance[int(attribute_index)]
            for node_index in range(len(tree)):
                if node_index > 1:
                    if tree[node_index][0] == "Value":
                        if tree[node_index][1] == attribute_value:
                            return decision_traverse(tree[node_index], X_test_instance)
        return decision_traverse(tree[2], X_test_instance)

def tdidt_predict(header, tree, instance):
    """Used to traverse the decision tree
        Args:
            header : list of strings holding attribute names
        Returns:
            the leaf value for a given instance
        Notes:
            recursive function
        """
    # recursively traverse tree to make a prediction
    # are we at a leaf node (base case) or attribute node?
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]  # label
    # we are at an attribute
    # find attribute value match for instance
    # for loop
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            # we have a match, recurse
            return tdidt_predict(header, value_list[2], instance)

def traverse_tree(tree, prev, vis, num):
    if tree[0] == "Leaf":  # Leaf node case
        # vis.edge(prev,tree[1],label=tree[1])
        return tree
    else:
        if prev != None:
            if tree[0] == "Value":
                if tree[2][0] == "Attribute":
                    vis.edge(prev, str(tree[2][1]), label=str(tree[1]))
                else:  # Case of a leaf node connection
                    vis.edge(prev, str(tree[2][1] +
                             str(num)), label=str(tree[1]))
        else:
            vis.node(tree[1])
        for node_index in range(len(tree)):
            if node_index > 1:
                # TODO: Configure a way to implement numbering leaf nodes down the recursion
                # process and then back up again, test original tree attempt
                traverse_tree(tree[node_index], tree[1], vis, num + 1)
                num += 1

def partition_instances(instances, split_attribute, attribute_domains, header):
    """Splits the instances into partitions based on an attribute
        Args:
            instances : available instances that could be split
            split_attribute: the attribute that will be split on
            attribute_domains: dictionary holding possible values for an attribute
            header : list of string holding attribute names
        Returns:
            partitions : list of instances according to a specific attribute domain
        Notes:
            needed for recursive descent
        """
    # lets use a dictionary
    partitions = {}  # key (string): value (subtable)
    att_index = header.index(split_attribute)  # e.g. 0 for level
    # e.g. ["Junior", "Mid", "Senior"]
    att_domain = attribute_domains[header[att_index]]
    for att_value in att_domain:
        partitions[att_value] = []
        # TASK: finish
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def all_same_class(attribute_partition):
    """Checks if all remaining attributes are same class for a case 1
        Args:
            attribute_partition : holds the remaining values
        Returns:
            true or false
        Notes:
            needed for determining the case
        """
    label = attribute_partition[0][-1]
    for attribute in attribute_partition:
        if attribute[-1] != label:
            return False
    return True
# def majority_vote(att_partition): # working for basic 1 element list
#     """Used to determine a clash
#         Args:
#             att_partition : the attributes partitioned
#         Returns:
#             the majority vote
#         Notes:
#             needed for clashes in decision tree
#         """
#     majority = att_partition[0][-1]
#     majority_count = 0
#     for vote in att_partition:
#         vote_count = 0
#         for other_vote in att_partition:
#             if vote[-1] == other_vote[-1]:
#                 vote_count += 1
#         if vote_count > majority_count:
#             majority = vote[-1]
#             majority_count = vote_count
#     return majority

def tdidt(current_instances, available_attributes, attribute_domains, header, F=None, random_state=None):
    """Generatest the decision tree
        Args:
            current_instances : instances that haven't been made into a rule
            available_attributes : attributes that can still be split on
        Returns:
            the tree generated
        Notes:
            is a recursive function
        """
    if F == None:
        attribute = select_attribute(
            current_instances, available_attributes, header)
    else:
        K_subsets = compute_random_subset(
            available_attributes, F)  # getting random subset
        # print(K_subsets)
        attribute = select_attribute(current_instances, K_subsets, header)
        # print(attribute)
    # attribute = select_attribute(current_instances, available_attributes, header)
    available_attributes.remove(attribute)
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(
        current_instances, attribute, attribute_domains, header)
    # for each partition, repeat unless one of the following occurs (base case)
    skip = False
    for att_value, att_partition in partitions.items():
        values_subtree = ["Value", att_value]
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            leaf_node = ["Leaf", att_partition[0][-1],
                         len(att_partition), len(current_instances)]
            values_subtree.append(leaf_node)
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif (len(att_partition) > 0 and len(available_attributes) == 0):
            label = majority_vote(att_partition)
            leaf_node = ["Leaf", label, len(
                att_partition), len(current_instances)]
            values_subtree.append(leaf_node)
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            skip = True
            tree = ["Leaf", majority_vote(current_instances), len(
                current_instances), len(current_instances)]  # need to fix len
        else:  # previous conditions are all false... recurse!
            subtree = tdidt(
                att_partition, available_attributes.copy(), attribute_domains, header, F)
            values_subtree.append(subtree)
            # note the copy
        if skip == False:
            tree.append(values_subtree)
    return tree
 
def get_column_ben(table, col_index):
    """ gets the column from a passed in col_name
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        col: (list) column wanted
    """
    col = []
    for row in table:
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])  # was in if statement
    return col


def group_by_ben(table, col_index):
    """ groups the table by the passed in col_index
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        group_names: (list) list of group label names
        group_subtables: (list of lists) 2d list of each group subtable
    """
    col = get_column_ben(table, col_index)
    # get a list of unique values for the column
    group_names = sorted(list(set(col)))  # 75, 76, 77
    group_subtables = [[] for _ in group_names]  # [[], [], []]
    # walk through each row and assign it to the appropriate
    # subtable based on its group by value (model year)
    for row in table:
        group_value = row[col_index]
        # which group_subtable??
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy())  # shallow copy
    return group_names, group_subtables

def select_attribute(instances, available_attributes, header):
    """ Selects attributes using entropy
    Args:
        instances: (list of lists) table of data
        available_attributes: (list) list of available attributes to split on
        header: (list) header of attributes
    Returns:
        attribute: (string) attribute selected to split on
    """
    table_size = len(instances)
    e_new_list = []
    # loops through each available attribute and groups data by each attribute
    for item in available_attributes:
        group_names, group_subtables = group_by_ben(
            instances, header.index(item))
        e_value_list = []
        num_values = []
        # loops through the group subtable and further groups by class name
        for j in range(len(group_subtables)):
            curr_group = group_subtables[j]
            num_attributes = len(curr_group)
            num_values.append(num_attributes)
            class_names, class_subtables = group_by_ben(
                curr_group, len(curr_group[0])-1)
            e_value = 0
            # checks for empty partition for log base 2 of 0 calculations
            if (len(class_subtables) == 1):
                e_value = 0
            else:
                # loops through each group bay attribute class and calculates the entropy
                for k in range(len(class_subtables)):
                    class_num = len(class_subtables[k]) / num_attributes
                    e_value -= (class_num) * (math.log2(class_num))
            e_value_list.append(e_value)
        e_new = 0
        # calculates e_new for each attribute
        for l in range(len(e_value_list)):
            e_new += e_value_list[l] * (num_values[l] / table_size)
        e_new_list.append(e_new)
    # finds attribute with minimum entropy and selects that attribute
    min_entropy = min(e_new_list)
    min_index = e_new_list.index(min_entropy)
    attribute = available_attributes[min_index]
    return attribute

def unique_values(value_list):
    """ Returns a list of unique values from the list of values given. These values are sorted
    in ascending order.
    """
    unique_values, ordered_unique_values = [], []
    for value in value_list:
        if value not in unique_values:
            unique_values.append(value)
    return sorted(unique_values)

def convert_nonpresent_freq_to_zero(possible_values, current_counts):
    """ This function takes a list of possible values and a list of current counts for some of these values,
    converting all of the non-present values to zero.
    """
    INDEX_OF_VALUES = 0
    INDEX_OF_COUNTS = 1
    all_counts = []
    current_counts_index = 0
    for i, value in enumerate(possible_values):
        if value in current_counts[INDEX_OF_VALUES]:
            all_counts.append(
                current_counts[INDEX_OF_COUNTS][current_counts_index])
            current_counts_index += 1
        else:
            all_counts.append(0)
    return all_counts

def convert_freqs_to_table(values_list, counts_list, header):
    """ Converts a list of frequencies into a single table containing posterior values, which is
    the ratio of the counts of a current value to the total counts that have that same class label.
    These two lists should be parallel.
    Some indexing is at plus one because we are setting the first item as the header
    """
    new_table = [[] for tables in range(len(counts_list) + 1)]
    # First column is the classifying attribute always
    new_table[0].append("class")
    for i, attribute_table in enumerate(counts_list):
        new_table[i + 1].append(header[i])
        for j, counts in enumerate(attribute_table):
            for k, count in enumerate(counts):
                if i == 0:  # Only setting the header based on values from the first table
                    try:  # For integer values we should try to conver them back to ints
                        new_table[0].append(int(values_list[i][j][k]))
                    except:
                        new_table[0].append(values_list[i][j][k])
                new_table[i + 1].append(count / sum(counts))
    return new_table

def split_on_header(table):
    """This function takes a table which includes the header and removes the header from the table
    returning a list for the header and the original table without the header
    """
    new_table = []
    for i, row in enumerate(table):
        if i == 0:  # First row
            header = row
        else:
            new_table.append(row)
    if len(new_table) == 1:  # If there is only one table, return it as a list
        return header, new_table[0]
    return header, new_table

def table_by_columns(table, header, column_names):
    """ This function is perfect for accepting a large table and cutting it down to only the certain
    columns you want.
        Args:
            table (list of list of numeric vals): Data contained in the entire table
            header (list of obj): The list of strings that name each column
            column_names (list of obj): The subgroup of the header which you want to make a table of
        Returns:
            table_by_cols (list of list of numeric values) same table but only the necessary columns
    """
    attributes_list = []
    table_by_cols, table_by_cols_row = list(), list()
    for column_name in column_names:
        attributes_list.append(get_column(table, header, column_name))
    for i, attribute in enumerate(attributes_list[0]):
        for attributes in attributes_list:
            table_by_cols_row.append(attributes[i])
        table_by_cols.append(table_by_cols_row)
        table_by_cols_row = []
    return table_by_cols

def randints(noninclusive_max, num_integers, random_state=None):
    """ A simple recreation of the numpy function random.randint() but this version accepts either an integer
    for seeding or a value of None for seeding.
    """
    np.random.seed(0)
    if random_state != None:
        np.random.seed(random_state)
    random_numbers = list(np.random.randint(
        0, noninclusive_max, size=num_integers))
    return random_numbers

def split_folds_to_train_test(folds):
    """ One fold is iterably assigned as the test fold while all others are combined iterably into
    the training folds. These two lists are returned as parallel with the 1 dimensional length
    equivalent to the different tests which can be run.
        Args:
            folds (list of list of values) data placed into a certain number of folds
        Returns:
            X_train_folds (list of list of values) contains lists of all data indexes for training, each list is
            an iteration for testing
            X_test_folds (list of list of values) contains lists of all data indexes for testing, each list is
            an iteration for testing
    """
    X_train_folds = []
    X_test_folds = []
    X_train_fold = []
    for fold_test in folds:
        X_test_folds.append(fold_test)
        for fold_train in folds:
            for instance in fold_train:
                # Last item was the test fold, so do not add this
                if fold_train != X_test_folds[-1]:
                    X_train_fold.append(instance)
        X_train_folds.append(X_train_fold)
        X_train_fold = []
    return X_train_folds, X_test_folds

def normal_round(n):
    """ Simple rounding function that accepts a number and correctly rounds it up
    or down as you would typically expect.
    """
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def display_train_test(label, X_train, X_test, y_train, y_test):
    """
    """
    print("***", label, "***")
    print("train:")
    for i, _ in enumerate(X_train):
        print(X_train[i], "->", y_train[i])
    print("test:")
    for i, _ in enumerate(X_test):
        print(X_test[i], "->", y_test[i])
    print()

def randomize_in_place(alist, parallel_list=None, random_state=0):
    """Shuffles a list of data as well as an optional second list of data
    in parallel using the numpy random integer generator.
        Args:
            alist (list of numeric vals): The list of data to be shuffled
            parallel_list (list of numeric vals): Another list of data to be shuffled in parallel
            random_state (int): number for seeding random number generator
    """
    np.random.seed(random_state)
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        # rand int in [0, len(alist))
        rand_index = np.random.randint(0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
            parallel_list[rand_index], parallel_list[i]

def group_by(table, header, groupby_col_name):
    """Computes subtables specific to various attribute values of the column requested. So, if the
        column to be grouped by contains 3 different attribute options there will be 3 tables.
        Args:
            table (list of numeric vals): The list of x values
            header (list of numeric vals): The list of y values
            group_by_col_name (obj): The string name of the column to be grouped by
        Returns:
            group_names (list of obj) string for each attribute used for grouping
            group_subtables (list of list of obj) list of tables, each should have only a single attribute in
            the given groupby_col_name
    """
    groupby_col_index = header.index(groupby_col_name)  # use this later
    if len(header) > 1:
        groupby_col = get_column(table, header, groupby_col_name)
    else:
        groupby_col = table
    group_names = sorted(list(set(groupby_col)))  # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names]  # e.g. [[], [], []]
    for row in table:
        if len(header) > 1:
            groupby_val = row[groupby_col_index]  # e.g. this row's modelyear
        else:
            groupby_val = row
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        if len(header) > 1:
            group_subtables[groupby_val_subtable_index].append(
                row.copy())  # make a copy
        else:
            group_subtables[groupby_val_subtable_index].append(
                row)  # make a copy
    return group_names, group_subtables