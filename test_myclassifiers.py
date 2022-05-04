import numpy as np
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> main
import random
from mysklearn import myutils
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier
from mysklearn.myrandomforest import MyRandomForestClassifier
<<<<<<< HEAD
=======

# interview dataset
header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
    "lang": ["R", "Python", "Java"],
    "tweets": ["yes", "no"], 
    "phd": ["yes", "no"]}
X_train = [
    ["Senior", "Java", "no", "no"], # used in predict()
    ["Senior", "Java", "no", "yes"], # used in predict()
    ["Mid", "Python", "no", "no"],  # used in predict()
    ["Junior", "Python", "no", "no"], # used in predict()
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# stitch X and y together to make one table
table = [X_train[i] + [y_train[i]] for i in range(len(X_train))]


# For testing your MyRandomForestClassifier fit() I recommend using a small dataset from PA7's test 
# cases that you can easily calculate entropy for. Then choose small values of N, M, and F. 
# Then seed your random number generator and see what attributes will be selected in the F-sized subsets 
# and what bootstrap samples will be generated for that seed. Then you can determine what the N trees 
# will look like and what the M best ones are based on the validation sets.
def test_random_forest_classifier_fit():
    N = 4
    M = 3
    F = 2
    random_num = random.seed(0)
    print("Random Number " , random_num)
    rfc = MyRandomForestClassifier(N, M, F)
    rfc.fit(X_train, y_train)
    
    # then assert against what the N trees will look like and 
    # what the M best ones are based on the validation sets.

# predict() is much more straightforward, use majority voting amongst the M trees to make a prediction 
# for an unseen instance, asserting it is the correct instance based on the trees.
def test_random_forest_classifier_predict():
    N = 4
    M = 7
    F = 2
    rfc = MyRandomForestClassifier(N, M, F)
    rfc.fit(X_train, y_train)

    X_test = [["Senior", "Java", "no", "no"],
             ["Senior", "Java", "no", "yes"],
             ["Mid", "Python", "no", "no"],
             ["Junior", "Python", "no", "no"]]

    correct_predictions = ["False", "False", "True", "True"]

    predictions = rfc.predict(X_test)

    assert predictions == correct_predictions 

def test_kneighbors_classifier_kneighbors():
    my_knn = MyKNeighborsClassifier()

    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    distances_class_example1 = []
    correct_neighbor_indices = []
    distances, neighbor_indices = my_knn.kneighbors(X_train_class_example1)
    assert distances == distances_class_example1
    assert neighbor_indices == correct_neighbor_indices

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    distances_class_example2 = []
    correct_neighbor_indices = []
    distances, neighbor_indices = my_knn.kneighbors(X_train_class_example2)
    assert distances == distances_class_example2
    assert neighbor_indices == correct_neighbor_indices

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    distances_bramer_example = []
    correct_neighbor_indices = []
    distances, neighbor_indices = my_knn.kneighbors(X_train_bramer_example)
    assert distances == distances_bramer_example
    assert neighbor_indices == correct_neighbor_indices

def test_kneighbors_classifier_predict():
    my_knn = MyKNeighborsClassifier()

    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    example1_correct_prediction = []
    prediction = my_knn.predict(X_train_class_example1)
    assert prediction == example1_correct_prediction

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    example2_correct_prediction = []
    prediction = my_knn.predict(X_train_class_example2)
    assert prediction == example2_correct_prediction

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    bramer_correct_prediction = []
    prediction = my_knn.predict(X_train_bramer_example)
    assert prediction == bramer_correct_prediction

def test_dummy_classifier_fit():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train = []
    test_dummy = MyDummyClassifier()
    test_dummy.fit(X_train, y_train)
    assert test_dummy.most_common_label == "yes"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    test_dummy.fit(X_train, y_train)
    assert test_dummy.most_common_label == "no"

    y_train = list(np.random.choice([1, 2, 3, 4, 5, 6], 100, replace=True, p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1]))
    test_dummy.fit(X_train, y_train)
    assert test_dummy.most_common_label == 4

def test_dummy_classifier_predict():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    y_predicted_answer = ["yes"] * 100
    test_dummy = MyDummyClassifier()
    y_prediction = test_dummy.predict(y_train)
    assert y_prediction == y_predicted_answer

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    y_predicted_answer = ["no"] * 100 
    y_prediction = test_dummy.predict(y_train)
    assert y_prediction == y_predicted_answer

    y_train = list(np.random.choice([1, 2, 3, 4, 5, 6], 100, replace=True, p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1]))
    y_predicted_answer = [4] * 100
    y_prediction = test_dummy.predict(y_train)
    assert y_prediction == y_predicted_answer

def test_naive_bayes_classifier_fit():
    """ Use the 8 instance training set example traced in class on the iPad, asserting against our desk check of the priors and posteriors
        Use the 15 instance training set example from RQ5, asserting against your desk check of the priors and posteriors
        Use Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and posteriors solution in Figure 3.2."""
    nbc = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    # in-class Naive Bayes example (lab task #1)
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    priors = [3/8, 5/8]
    posteriors = [  [2/3, 1/3, 2/3, 1/3],
                    [4/5, 1/5, 2/5, 3/5]
                ]
    # posteriors[0] = no, posts[1] = yes
    nbc.fit(X_train_inclass_example, y_train_inclass_example)
    assert np.allclose(priors, nbc.priors)
    assert np.allclose(posteriors, nbc.posteriors)
    # iphone_table example
    y_train = myutils.get_column(iphone_table, 3)
    X_train = myutils.get_X(iphone_table.copy(), 3)
    # posteriors are organized by class alphabetically
    priors = [5/15, 10/15]
    posteriors = [  [3/5, 2/5, 1/5, 2/5, 2/5, 3/5, 2/5],
                    [2/10, 8/10, 3/10, 4/10, 3/10, 3/10, 7/10]
                ]
    nbc.fit(X_train, y_train)
    assert np.allclose(priors, nbc.priors)
    assert np.allclose(posteriors, nbc.posteriors)
    # Bramer 3.2 train dataset example
    y_train = myutils.get_column(train_table, 4)
    X_train = myutils.get_X(train_table.copy(), 4)
    priors = [0.05, 0.10, 0.70, 0.15]
    posteriors = [  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, .5, 0, .5, 0, 0, 0, 1, .5, 0, .5, .5, .5, 0],
                    [2/14, 2/14, 1/14, 9/14, 2/14, 4/14, 6/14, 2/14, 4/14, 5/14, 5/14, 1/14, 5/14, 8/14],
                    [0, 0, 0, 1, 1/3, 0, 0, 2/3, 1/3, 0, 2/3, 2/3, 1/3, 0]
                ]
    nbc.fit(X_train, y_train)
    assert np.allclose(priors, nbc.priors)
    assert np.allclose(posteriors, nbc.posteriors)

def test_naive_bayes_classifier_predict():
    """Use the 8 instance training set example traced in class on the iPad, asserting against our desk check prediction
        Use the 15 instance training set example from RQ5, asserting against your desk check predictions for the two test instances
        Use Bramer 3.2 unseen instance ["weekday", "winter", "high", "heavy"] and Bramer 3.6 Self-assessment exercise 1 unseen instances,
        asserting against the solution prediction on pg. 28-29 and the exercise solution predictions in Bramer Appendix E"""
    nbc = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    # in-class Naive Bayes example (lab task #1)
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    prediction = ['yes']
    labels, counts = myutils.get_frequencies(y_train_inclass_example)
    nbc.fit(X_train_inclass_example, y_train_inclass_example)
    X_test_unadjusted = [[1,5]]
    X_test = myutils.adjust_X_test(X_test_unadjusted, X_train_inclass_example)
    X_test = [[0,2]]
    indexes = nbc.predict(X_test)
    naive_predictions = []
    for item in indexes:
        naive_predictions.append(labels[item])
    assert np.array_equal(prediction, naive_predictions)
    # RQ5 (fake) iPhone purchases dataset
    y_train = myutils.get_column(iphone_table, 3)
    X_train = myutils.get_X(iphone_table.copy(), 3)
    prediction = ['yes', 'no']
    labels, counts = myutils.get_frequencies(y_train.copy())
    nbc.fit(X_train, y_train)
    X_test_unadjusted = [[2, 2, 'fair'], [1, 1, 'excellent']]
    X_test = myutils.adjust_X_test(X_test_unadjusted, X_train)
    indexes = nbc.predict(X_test)
    naive_predictions = []
    for item in indexes:
        naive_predictions.append(labels[item])
    assert np.array_equal(prediction, naive_predictions)
    # Bramer 3.2 train dataset
    y_train = myutils.get_column(train_table, 4)
    X_train = myutils.get_X(train_table.copy(), 4)
    nbc.fit(X_train, y_train)
    prediction = ['very late', 'on time', 'on time']
    labels, counts = myutils.get_frequencies(y_train.copy())
    nbc.fit(X_train, y_train)
    X_test_unadjusted = [   ['weekday', 'winter', 'high', 'heavy'],
                            ['weekday', 'summer', 'high', 'heavy'], 
                            ['sunday', 'summer', 'normal', 'slight']
                        ]
    X_test = myutils.adjust_X_test(X_test_unadjusted, X_train)
    indexes = nbc.predict(X_test)
    naive_predictions = []
    for item in indexes:
        naive_predictions.append(labels[item])
    assert np.array_equal(prediction, naive_predictions)
=======
>>>>>>> main

# interview dataset
header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
    "lang": ["R", "Python", "Java"],
    "tweets": ["yes", "no"], 
    "phd": ["yes", "no"]}
X_train = [
    ["Senior", "Java", "no", "no"], # used in predict()
    ["Senior", "Java", "no", "yes"], # used in predict()
    ["Mid", "Python", "no", "no"],  # used in predict()
    ["Junior", "Python", "no", "no"], # used in predict()
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]

y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# stitch X and y together to make one table
table = [X_train[i] + [y_train[i]] for i in range(len(X_train))]


# For testing your MyRandomForestClassifier fit() I recommend using a small dataset from PA7's test 
# cases that you can easily calculate entropy for. Then choose small values of N, M, and F. 
# Then seed your random number generator and see what attributes will be selected in the F-sized subsets 
# and what bootstrap samples will be generated for that seed. Then you can determine what the N trees 
# will look like and what the M best ones are based on the validation sets.
def test_random_forest_classifier_fit():
    N = 4
    M = 3
    F = 2
    random_num = random.seed(0)
    print("Random Number " , random_num)
    rfc = MyRandomForestClassifier(N, M, F)
    rfc.fit(X_train, y_train)
    
    # then assert against what the N trees will look like and 
    # what the M best ones are based on the validation sets.

# predict() is much more straightforward, use majority voting amongst the M trees to make a prediction 
# for an unseen instance, asserting it is the correct instance based on the trees.
def test_random_forest_classifier_predict():
<<<<<<< HEAD
    N = 4
    M = 7
    F = 2
    rfc = MyRandomForestClassifier(N, M, F)
    rfc.fit(X_train, y_train)

    X_test = [["Senior", "Java", "no", "no"],
             ["Senior", "Java", "no", "yes"],
             ["Mid", "Python", "no", "no"],
             ["Junior", "Python", "no", "no"]]

    correct_predictions = ["False", "False", "True", "True"]

    predictions = rfc.predict(X_test)

    assert predictions == correct_predictions 

def test_kneighbors_classifier_kneighbors():
    my_knn = MyKNeighborsClassifier()

    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    distances_class_example1 = []
    correct_neighbor_indices = []
    distances, neighbor_indices = my_knn.kneighbors(X_train_class_example1)
    assert distances == distances_class_example1
    assert neighbor_indices == correct_neighbor_indices

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    distances_class_example2 = []
    correct_neighbor_indices = []
    distances, neighbor_indices = my_knn.kneighbors(X_train_class_example2)
    assert distances == distances_class_example2
    assert neighbor_indices == correct_neighbor_indices

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    distances_bramer_example = []
    correct_neighbor_indices = []
    distances, neighbor_indices = my_knn.kneighbors(X_train_bramer_example)
    assert distances == distances_bramer_example
    assert neighbor_indices == correct_neighbor_indices

def test_kneighbors_classifier_predict():
    my_knn = MyKNeighborsClassifier()

    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    example1_correct_prediction = []
    prediction = my_knn.predict(X_train_class_example1)
    assert prediction == example1_correct_prediction

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    example2_correct_prediction = []
    prediction = my_knn.predict(X_train_class_example2)
    assert prediction == example2_correct_prediction

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    bramer_correct_prediction = []
    prediction = my_knn.predict(X_train_bramer_example)
    assert prediction == bramer_correct_prediction

def test_dummy_classifier_fit():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train = []
    test_dummy = MyDummyClassifier()
    test_dummy.fit(X_train, y_train)
    assert test_dummy.most_common_label == "yes"

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    test_dummy.fit(X_train, y_train)
    assert test_dummy.most_common_label == "no"

    y_train = list(np.random.choice([1, 2, 3, 4, 5, 6], 100, replace=True, p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1]))
    test_dummy.fit(X_train, y_train)
    assert test_dummy.most_common_label == 4

def test_dummy_classifier_predict():
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    y_predicted_answer = ["yes"] * 100
    test_dummy = MyDummyClassifier()
    y_prediction = test_dummy.predict(y_train)
    assert y_prediction == y_predicted_answer

    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    y_predicted_answer = ["no"] * 100 
    y_prediction = test_dummy.predict(y_train)
    assert y_prediction == y_predicted_answer

    y_train = list(np.random.choice([1, 2, 3, 4, 5, 6], 100, replace=True, p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1]))
    y_predicted_answer = [4] * 100
    y_prediction = test_dummy.predict(y_train)
    assert y_prediction == y_predicted_answer

def test_naive_bayes_classifier_fit():
    """ Use the 8 instance training set example traced in class on the iPad, asserting against our desk check of the priors and posteriors
        Use the 15 instance training set example from RQ5, asserting against your desk check of the priors and posteriors
        Use Bramer 3.2 Figure 3.1 train dataset example, asserting against the priors and posteriors solution in Figure 3.2."""
    nbc = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    # in-class Naive Bayes example (lab task #1)
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    priors = [3/8, 5/8]
    posteriors = [  [2/3, 1/3, 2/3, 1/3],
                    [4/5, 1/5, 2/5, 3/5]
                ]
    # posteriors[0] = no, posts[1] = yes
    nbc.fit(X_train_inclass_example, y_train_inclass_example)
    assert np.allclose(priors, nbc.priors)
    assert np.allclose(posteriors, nbc.posteriors)
    # iphone_table example
    y_train = myutils.get_column(iphone_table, 3)
    X_train = myutils.get_X(iphone_table.copy(), 3)
    # posteriors are organized by class alphabetically
    priors = [5/15, 10/15]
    posteriors = [  [3/5, 2/5, 1/5, 2/5, 2/5, 3/5, 2/5],
                    [2/10, 8/10, 3/10, 4/10, 3/10, 3/10, 7/10]
                ]
    nbc.fit(X_train, y_train)
    assert np.allclose(priors, nbc.priors)
    assert np.allclose(posteriors, nbc.posteriors)
    # Bramer 3.2 train dataset example
    y_train = myutils.get_column(train_table, 4)
    X_train = myutils.get_X(train_table.copy(), 4)
    priors = [0.05, 0.10, 0.70, 0.15]
    posteriors = [  [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, .5, 0, .5, 0, 0, 0, 1, .5, 0, .5, .5, .5, 0],
                    [2/14, 2/14, 1/14, 9/14, 2/14, 4/14, 6/14, 2/14, 4/14, 5/14, 5/14, 1/14, 5/14, 8/14],
                    [0, 0, 0, 1, 1/3, 0, 0, 2/3, 1/3, 0, 2/3, 2/3, 1/3, 0]
                ]
    nbc.fit(X_train, y_train)
    assert np.allclose(priors, nbc.priors)
    assert np.allclose(posteriors, nbc.posteriors)

def test_naive_bayes_classifier_predict():
    """Use the 8 instance training set example traced in class on the iPad, asserting against our desk check prediction
        Use the 15 instance training set example from RQ5, asserting against your desk check predictions for the two test instances
        Use Bramer 3.2 unseen instance ["weekday", "winter", "high", "heavy"] and Bramer 3.6 Self-assessment exercise 1 unseen instances,
        asserting against the solution prediction on pg. 28-29 and the exercise solution predictions in Bramer Appendix E"""
    nbc = MyNaiveBayesClassifier()
    # in-class Naive Bayes example (lab task #1)
    inclass_example_col_names = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]

    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]

    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    # in-class Naive Bayes example (lab task #1)
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    prediction = ['yes']
    labels, counts = myutils.get_frequencies(y_train_inclass_example)
    nbc.fit(X_train_inclass_example, y_train_inclass_example)
    X_test_unadjusted = [[1,5]]
    X_test = myutils.adjust_X_test(X_test_unadjusted, X_train_inclass_example)
    X_test = [[0,2]]
    indexes = nbc.predict(X_test)
    naive_predictions = []
    for item in indexes:
        naive_predictions.append(labels[item])
    assert np.array_equal(prediction, naive_predictions)
    # RQ5 (fake) iPhone purchases dataset
    y_train = myutils.get_column(iphone_table, 3)
    X_train = myutils.get_X(iphone_table.copy(), 3)
    prediction = ['yes', 'no']
    labels, counts = myutils.get_frequencies(y_train.copy())
    nbc.fit(X_train, y_train)
    X_test_unadjusted = [[2, 2, 'fair'], [1, 1, 'excellent']]
    X_test = myutils.adjust_X_test(X_test_unadjusted, X_train)
    indexes = nbc.predict(X_test)
    naive_predictions = []
    for item in indexes:
        naive_predictions.append(labels[item])
    assert np.array_equal(prediction, naive_predictions)
    # Bramer 3.2 train dataset
    y_train = myutils.get_column(train_table, 4)
    X_train = myutils.get_X(train_table.copy(), 4)
    nbc.fit(X_train, y_train)
    prediction = ['very late', 'on time', 'on time']
    labels, counts = myutils.get_frequencies(y_train.copy())
    nbc.fit(X_train, y_train)
    X_test_unadjusted = [   ['weekday', 'winter', 'high', 'heavy'],
                            ['weekday', 'summer', 'high', 'heavy'], 
                            ['sunday', 'summer', 'normal', 'slight']
                        ]
    X_test = myutils.adjust_X_test(X_test_unadjusted, X_train)
    indexes = nbc.predict(X_test)
    naive_predictions = []
    for item in indexes:
        naive_predictions.append(labels[item])
    assert np.array_equal(prediction, naive_predictions)
=======
    assert False is True # TODO: fix this
>>>>>>> 2808c1a793baed00c175478f4e606a1291203af1
>>>>>>> main
