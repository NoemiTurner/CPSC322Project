from flask import Flask
from flask import jsonify
from flask import request
app = Flask(__name__)

@app.route("/predict", methods=['GET'])
def index():
    arg1 = request.args.get('arg1')
    arg2 = request.args.get('arg2')
    arg3 = request.args.get('arg3')
    # input the arguments into the best classifier
    # get the result and return it
    y_pred = []
    
    return "<h1>Your preditction is {}</h1>".format(y_pred[0])

if __name__ == "__main__":
    app.run()