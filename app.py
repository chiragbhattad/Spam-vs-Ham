import pandas as pd
from flask import Flask, jsonify, request
import pickle
import joblib

app = Flask(__name__)

filename = "finalized_model.sav"
model = joblib.load(filename)
with open("vectorizer.pickle", "rb") as handle:
	vectorizer = pickle.load(handle)

@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_data()
    new=[data]
    message=vectorizer.transform(new)
    pred = model.predict(message)
    if str(pred[0]) == '1':
        return "SPAM"
    else:
        return "HAM"

if __name__ == '__main__':
    app.run(port = 5000, debug=True)