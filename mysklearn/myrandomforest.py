<<<<<<< HEAD
import numpy as np
from mysklearn import myutils
from myclassifiers import MyDecisionTreeClassifier
=======
<<<<<<< HEAD
import numpy as np
from mysklearn import myutils
from myclassifiers import MyDecisionTreeClassifier
=======
from mysklearn import myutils

# TODO: copy your myclassifiers.py solution from PA4-6 here
>>>>>>> 2808c1a793baed00c175478f4e606a1291203af1
>>>>>>> main

class MyRandomForestClassifier:
    """Represents a random forest classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
<<<<<<< HEAD
    
=======
<<<<<<< HEAD
    
    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, N, M, F):
=======
>>>>>>> main
    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
<<<<<<< HEAD
    def __init__(self, N, M, F):
=======
    def __init__(self):
>>>>>>> 2808c1a793baed00c175478f4e606a1291203af1
>>>>>>> main
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> main
        self.N = N # the number of trees in the forest
        self.M = M # the number of classifiers
        self.F = F # size of the random subsets of attributes
        self.forest = None
        self.header = []

<<<<<<< HEAD
=======

    # implement a random forest (all learners are trees)
    # with bagging and with random attribute subsets
    # will need to modify tree generation: for each node 
    # in our tree, we use random attribute subsets.
    # call compute_random_subset() right before a call to
    # select_attribute() in tdidt pass the return subset
    # (size F) into select_attribute()
    def fit(self, X_train, y_train):
        """
        Build a forest of trees from the training set (X, y)
        """
        # 1. Generate a random stratified test set consisting of one third of the original data set, 
        # with the remaining two thirds of the instances forming the "remainder set".

        # 2. Generate N "random" decision trees using bootstrapping (giving a training and validation set)
        #  over the remainder set. At each node, build your decision trees by randomly selecting F of the 
        #  remaining attributes as candidates to partition on. This is the standard random forest approach 
        #  discussed in class. Note that to build your decision trees you should still use entropy; 
        #  however, you are selecting from only a (randomly chosen) subset of the available attributes.

        # 3. Select the M most accurate of the N decision trees using the corresponding validation sets

        header = []

        for i in range(len(X_train[0])):
            att_num = str(i)
            header.append("att" + att_num)
        self.header = header

        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

        self.forest = []

        X_indexes = list(range(len(self.X_train))) # I feel like I need this
        y_indexes = list(range(len(self.y_train))) # I feel like I need this too? :)

        sample = myutils.compute_bootstrapped_sample(train) # no sure where to use this? Maybe in modified tdidt()

        for i in range(self.N):
            
            sample = myutils.compute_bootstrapped_sample(train) 

            tree = MyDecisionTreeClassifier()
            tree.fit(X_train_tree, y_train_tree)
            self.forest.append(tree)
        return self
    
    def predict(self, X_test):
        """
        Predict class for the unseen instances
        """
        # 4. Use simple majority voting to predict classes using the M decision trees over the test set.
        # BONUS (2 pts): Modify your random forest algorithm to use the "track record" weighted voting scheme 
        # (instead of simple majority voting). See the weighted majority voting lab tasks on Github to help 
        # with this bonus task. Compare your results to those w/simple majority voting.

        predictions = [] 
       
        # for each element in X_test
        for element in X_test:
            element_predictions = []
            # for each tree in the forest
            for tree in self.forest:
                # get the prediction
                element_predictions.append(tree.predict(element))

            # then do majority voting for the element in X_test 
            element_predictions.sort()  # inplace
            # parallel lists
            values = []
            counts = []
            for value in element_predictions:
                if value in values:  # seen it before
                    counts[-1] += 1  # okay because sorted
                else:  # haven't seen it before
                    values.append(value)
                    counts.append(1)
            i = counts.index(max(counts)) # get the index of the predicton value that appears most frequently
            
            # append to predictions list
            predictions.append(values[i])

            
        return predictions # returns the predicted classes 
=======
        self.tree = None
>>>>>>> main

    # implement a random forest (all learners are trees)
    # with bagging and with random attribute subsets
    # will need to modify tree generation: for each node 
    # in our tree, we use random attribute subsets.
    # call compute_random_subset() right before a call to
    # select_attribute() in tdidt pass the return subset
    # (size F) into select_attribute()
    def fit(self, X_train, y_train):
        """
        Build a forest of trees from the training set (X, y)
        """
        # 1. Generate a random stratified test set consisting of one third of the original data set, 
        # with the remaining two thirds of the instances forming the "remainder set".

        # 2. Generate N "random" decision trees using bootstrapping (giving a training and validation set)
        #  over the remainder set. At each node, build your decision trees by randomly selecting F of the 
        #  remaining attributes as candidates to partition on. This is the standard random forest approach 
        #  discussed in class. Note that to build your decision trees you should still use entropy; 
        #  however, you are selecting from only a (randomly chosen) subset of the available attributes.

        # 3. Select the M most accurate of the N decision trees using the corresponding validation sets

        header = []

        for i in range(len(X_train[0])):
            att_num = str(i)
            header.append("att" + att_num)
        self.header = header

        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

        self.forest = []

        X_indexes = list(range(len(self.X_train))) # I feel like I need this
        y_indexes = list(range(len(self.y_train))) # I feel like I need this too? :)

        sample = myutils.compute_bootstrapped_sample(train) # no sure where to use this? Maybe in modified tdidt()

        for i in range(self.N):
            
            sample = myutils.compute_bootstrapped_sample(train) 

            tree = MyDecisionTreeClassifier()
            tree.fit(X_train_tree, y_train_tree)
            self.forest.append(tree)
        return self
    
    def predict(self, X_test):
        """
        Predict class for the unseen instances
        """
        # 4. Use simple majority voting to predict classes using the M decision trees over the test set.
        # BONUS (2 pts): Modify your random forest algorithm to use the "track record" weighted voting scheme 
        # (instead of simple majority voting). See the weighted majority voting lab tasks on Github to help 
        # with this bonus task. Compare your results to those w/simple majority voting.

        predictions = [] 
       
        # for each element in X_test
        for element in X_test:
            element_predictions = []
            # for each tree in the forest
            for tree in self.forest:
                # get the prediction
                element_predictions.append(tree.predict(element))

            # then do majority voting for the element in X_test 
            element_predictions.sort()  # inplace
            # parallel lists
            values = []
            counts = []
            for value in element_predictions:
                if value in values:  # seen it before
                    counts[-1] += 1  # okay because sorted
                else:  # haven't seen it before
                    values.append(value)
                    counts.append(1)
            i = counts.index(max(counts)) # get the index of the predicton value that appears most frequently
            
            # append to predictions list
            predictions.append(values[i])

<<<<<<< HEAD
            
        return predictions # returns the predicted classes 
=======
 
>>>>>>> 2808c1a793baed00c175478f4e606a1291203af1
>>>>>>> main
