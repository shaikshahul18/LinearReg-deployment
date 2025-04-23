import pickle
import numpy as np
import pandas as pd
from flask import Flask,request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


ridge_model = pickle.load(open("models/ridge_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("WS"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        data = scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(data)
        return render_template("home.html", results=result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    
