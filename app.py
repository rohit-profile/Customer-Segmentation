from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
kmeans = pickle.load(open("model.pkl", "rb"))

cluster_labels = {
    0: "Low Income – Low Spending",
    1: "High Income – High Spending",
    2: "Young Low Income – High Spending"
}

@app.route("/", methods=["GET", "POST"])
def home():
    segment = None
    cluster = None

    if request.method == "POST":
        gender = request.form["gender"]
        gender_value = 1 if gender == "Male" else 0

        age = int(request.form["age"])
        income = int(request.form["income"])
        score = int(request.form["score"])

        input_data = np.array([[gender_value, age, income, score]])
        cluster = kmeans.predict(input_data)[0]
        segment = cluster_labels.get(cluster)

    return render_template("index.html", segment=segment, cluster=cluster)

if __name__ == "__main__":
    app.run(debug=True)
