import mlflow.sklearn
import pandas as pd
import os
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient

app = Flask(__name__)

#model_path = r"C:\Users\91994\WaterQualityPrediction\src\mlruns\215628478639028860\a7962bb73f3d4013b26cb4227e428d78\artifacts\WQ_RandomForestModel"
#model = mlflow.sklearn.load_model(model_path)
mlflow_tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient(tracking_uri=mlflow_tracking_uri)

model = mlflow.sklearn.load_model("models:/WaterQualityModel/Production")


@app.route("/")
def home():
    return "Water Quality Prediction is running"

@app.route("/predict",methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if isinstance(data,dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        pred_value = model.predict(df)
        return jsonify({"predictions":pred_value.tolist(),
                        "message":"1 means potable and 0 means non potable"})
        
    except Exception as E:
        return jsonify({"Error":str(E)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)


