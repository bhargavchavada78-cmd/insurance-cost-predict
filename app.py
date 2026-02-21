from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    children = float(request.form["children"])

    # ----- Sex Dummy -----
    sex = request.form["sex"]
    if sex == "Male":
        sex_male = 1
    else:
        sex_male = 0

    # ----- Smoker Dummy -----
    smoker = request.form["smoker"]
    if smoker == "Yes":
        smoker_yes = 1
    else:
        smoker_yes = 0

    # ----- Region Dummies -----
    region = request.form["region"]

    region_northwest = 0
    region_southeast = 0
    region_southwest = 0

    if region == "Northwest":
        region_northwest = 1
    elif region == "Southeast":
        region_southeast = 1
    elif region == "Southwest":
        region_southwest = 1
    # If Northeast → all remain 0 (because drop_first=True)

    # Arrange features in SAME order as training
    features = np.array([[age, bmi, children,
                          sex_male, smoker_yes,
                          region_northwest,
                          region_southeast,
                          region_southwest]])

    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template("index.html",
                           prediction_text=f"Predicted Insurance Cost: ₹ {output}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0",port=port)

    
