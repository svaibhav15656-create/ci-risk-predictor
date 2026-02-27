from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("risk_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    msg_len = data["msg_len"]
    files_changed = data["files_changed"]

    prediction = model.predict([[msg_len, files_changed]])[0]

    return jsonify({
        "risk": int(prediction)
    })

if __name__ == "__main__":
    app.run(port=5000)