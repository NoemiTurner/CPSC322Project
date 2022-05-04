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
    dummy = MyDummyClassifier()
    data = MyPyTable()
    data.load_from_file("input_data/forbes_data.csv")
    num_rows, num_cols = data.get_shape()
    data.convert_to_numeric()
    y_data = utils.discritize_data_by_million(data)
    combined_data = []
    sport = utils.get_column(data.data, data.column_names, "Sport")
    country = utils.get_column(data.data, data.column_names, "Nationality")
    for i in range(len(sport)):
        country[i] = country[i].lower()
        sport[i] = sport[i].lower()
        combined_data.append([country[i], sport[i]])
    new_Xtrain = []
    for item in combined_data:
        new_Xtrain.append(item)
    dummy.fit(combined_data, y_data)

    nationality = request.args.get('nationality', "")
    sport = request.args.get('sport', "")
    print(nationality, sport)
    # input the arguments into the best classifier
    # get the result and return it
    prediction = dummy.predict([nationality, sport])
    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    return "Error making prediction", 400

def set_up_data():
    pass

if __name__ == "__main__":
    #port = os.eviron.get("PORT", 5000)
    app.run(debug=False, port=3000, host="0.0.0.0")