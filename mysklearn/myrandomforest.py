from mysklearn import myutils

class MyRandomForestClassifier:
    """Represents a random forest classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    
    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, N, M, F):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.N = N
        self.M = M
        self.F = F

    """
    Build a forest of trees from the training set (X, y)
    """
    def fit(self, X_train, y_train):
        
        return self
    
    """
    Predict class for X
    """
    def predict(self, X_test):
        y = []
        
        return y # returns the predicted classes 

# 1. Generate a random stratified test set consisting of one third of the original data set, 
# with the remaining two thirds of the instances forming the "remainder set".

# 2. Generate N "random" decision trees using bootstrapping (giving a training and validation set)
#  over the remainder set. At each node, build your decision trees by randomly selecting F of the 
#  remaining attributes as candidates to partition on. This is the standard random forest approach 
#  discussed in class. Note that to build your decision trees you should still use entropy; 
#  however, you are selecting from only a (randomly chosen) subset of the available attributes.

# 3. Select the M most accurate of the N decision trees using the corresponding validation sets.

# 4. Use simple majority voting to predict classes using the M decision trees over the test set.
# BONUS (2 pts): Modify your random forest algorithm to use the "track record" weighted voting scheme 
# (instead of simple majority voting). See the weighted majority voting lab tasks on Github to help 
# with this bonus task. Compare your results to those w/simple majority voting.

# project notes
# implement a random forest (all learners are trees)
# with bagging and with random attribute subsets
# will need to modify tree generation: for each node 
# in our tree, we use random attribute subsets.
# call compute_random_subset() right before a call to
# select_attribute() in tdidt pass the return subset
# (size F) into select_attribute()


# Define a python function that selects F random attributes
# from an attribute list
# (test your function with att_indexes (or header))
def compute_random_subset(values, num_values):
    # there is a function np.random.choice()
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place shuffle
    return values_copy[:num_values]
F = 2
print(compute_random_subset(header, F))
att_indexes = list(range(len(header)))
print(compute_random_subset(att_indexes, F))


# (done on PA5) Ensemble Lab Task 1: 
# Write a bootstrap function to return a random sample of rows with replacement
# (test your function with the interview dataset)
def compute_bootstrapped_sample(table):
    n = len(table)
    sample = []
    for _ in range(n):
        rand_index = np.random.randint(0, n) # Return random integers from low (inclusive) to high (exclusive)
        sample.append(table[rand_index])
    return sample 

sample = compute_bootstrapped_sample(table)
for row in sample:
    print(row)

 