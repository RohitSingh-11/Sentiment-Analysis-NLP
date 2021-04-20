import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

import pandas as pd
import numpy as np
datanew = pd.read_csv("tweetgot.csv")
arr = np.asarray(datanew["sentiment"])
s = " "
for i in arr:
    s = s + str(i)

app = Flask(__name__)
#model = pickle.load(open('taxi.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

#@app.route('/predict', methods=['POST'])
def predict():
    namegot  = [str(x) for x in request.form.values()]
    prediction = s
    return render_template('index.html',prediction_text="emotion is {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
