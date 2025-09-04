from flask import Flask, render_template,request
import numpy as np
import pickle




app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))

@app.route("/")
def home():
    return render_template("HomePage.html")


@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        ph = float(request.form["ph"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                          residual_sugar, chlorides, free_sulfur_dioxide,
                          total_sulfur_dioxide, density, ph, sulphates, alcohol]])
        sha = data.reshape(1, -1)

        prediction = model.predict(sha)






        return render_template("result.html",PREDICTION = prediction)


if __name__ == "__main__":
    app.run(debug=True)




