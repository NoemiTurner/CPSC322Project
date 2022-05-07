from flask import Flask
from flask import jsonify
import os
from flask import request
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier
from mypytable import MyPyTable
import utils

app = Flask(__name__)

@app.route('/', methods=["GET"])
def index():
    return "<h1>Welcome to my forbes app!</h1>", 200

@app.route("/predict", methods=['GET'])
def predict():
    knn = MyKNeighborsClassifier(n_neighbors=10)
    data = MyPyTable()
    data.load_from_file("input_data/forbes_data.csv")
    num_rows, num_cols = data.get_shape()
    data.convert_to_numeric()
    y_data = utils.discritize_data_by_million(data)
    combined_data = []
    sport = utils.get_column(data.data, data.column_names, "Sport")
    country = utils.get_column(data.data, data.column_names, "Nationality")
    sport_values, sport_counts = utils.get_frequencies(data.data, data.column_names, "Sport")
    country_values, country_counts = utils.get_frequencies(data.data, data.column_names, "Nationality")
    # make all lower to stay consistent with other tags
    for i in range(len(sport)):
        country[i] = country[i].lower()
        sport[i] = sport[i].lower()
        combined_data.append([country[i], sport[i]])
    
    # transform the sports and nationalities to numbers
    for index, value in enumerate(sport_values):
        for row in range(len(combined_data)):
            if combined_data[row][1] == value:
                combined_data[row][1] = index + 1
    for index, single_country in enumerate(country_values):
        for row in range(len(combined_data)):
            if combined_data[row][0] == single_country:
                combined_data[row][0] = index + 1
    knn.fit(combined_data, y_data)

    # ***** Prediction Part ***** 

    nationality = request.args.get('nationality', "")
    sport = request.args.get('sport', "")
    print(nationality, sport)
    # input the arguments into the best classifier
    # get the result and return it
    prediction = knn.predict([nationality, sport])
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

if __name__ == "__main__":
    app.run(debug=False, port=3000, host="0.0.0.0")