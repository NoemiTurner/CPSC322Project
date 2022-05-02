import numpy as np

from mysklearn.myrandomforest import MyRandomForestClassifier

# interview dataset
HEADER_INTERVEIW = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_TRAIN_INTERVIEW = [
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
Y_TRAIN_INTERVIEW = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

def test_random_forest_classifier_fit():
    rfc = MyRandomForestClassifier()
    
    assert False is True # TODO: fix this

def test_random_forest_classifier_predict():
    assert False is True # TODO: fix this