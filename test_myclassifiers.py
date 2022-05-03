import numpy as np

from mysklearn.myrandomforest import MyRandomForestClassifier

# # interview dataset
# HEADER_INTERVEIW = ["level", "lang", "tweets", "phd", "interviewed_well"]
# X_TRAIN_INTERVIEW = [
#     ["Senior", "Java", "no", "no"],
#     ["Senior", "Java", "no", "yes"],
#     ["Mid", "Python", "no", "no"],
#     ["Junior", "Python", "no", "no"],
#     ["Junior", "R", "yes", "no"],
#     ["Junior", "R", "yes", "yes"],
#     ["Mid", "R", "yes", "yes"],
#     ["Senior", "Python", "no", "no"],
#     ["Senior", "R", "yes", "no"],
#     ["Junior", "Python", "yes", "no"],
#     ["Senior", "Python", "yes", "yes"],
#     ["Mid", "Python", "no", "yes"],
#     ["Mid", "Java", "yes", "no"],
#     ["Junior", "Python", "no", "yes"]
# ]
# Y_TRAIN_INTERVIEW = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
    "lang": ["R", "Python", "Java"],
    "tweets": ["yes", "no"], 
    "phd": ["yes", "no"]}
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
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

y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
# stitch X and y together to make one table
table = [X[i] + [y[i]] for i in range(len(X))]


# For testing your MyRandomForestClassifier fit() I recommend using a small dataset from PA7's test 
# cases that you can easily calculate entropy for. Then choose small values of N, M, and F. 
# Then seed your random number generator and see what attributes will be selected in the F-sized subsets 
# and what bootstrap samples will be generated for that seed. Then you can determine what the N trees 
# will look like and what the M best ones are based on the validation sets.

# predict() is much more straightforward, use majority voting amongst the M trees to make a prediction 
# for an unseen instance, asserting it is the correct instance based on the trees.

def test_random_forest_classifier_fit():
    rfc = MyRandomForestClassifier()
    
    assert False is True # TODO: fix this

def test_random_forest_classifier_predict():
    assert False is True # TODO: fix this