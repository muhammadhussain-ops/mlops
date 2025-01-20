from src.mlops.data import CelebADataset
from src.mlops.model import NeuralNetwork
from src.mlops.train import train
from src.mlops.evaluate import evaluate

from flask import Flask, request, jsonify
from google.cloud import storage
from src.mlops.train import train
from src.mlops.evaluate import evaluate
from src.mlops.model import NeuralNetwork
import pickle
import os

app = Flask(__name__)

# Google Cloud Storage Configuration
GCS_BUCKET = "your-gcs-bucket-name"
GCS_DATA_PATH = "path/to/your/data"
GCS_MODEL_PATH = "path/to/your/model.pkl"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"  # Service account key file

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET)

# Helper functions to load and save files from GCS
def download_from_gcs(gcs_path, local_path):
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)

def upload_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

# Load data and model
def load_data():
    local_data_file = "local_data_file"
    download_from_gcs(GCS_DATA_PATH, local_data_file)
    # Replace with your data loading logic
    return "local_data_file"

def load_model():
    local_model_file = "local_model.pkl"
    download_from_gcs(GCS_MODEL_PATH, local_model_file)
    with open(local_model_file, "rb") as f:
        model = pickle.load(f)
    return model

def save_model(model):
    local_model_file = "local_model.pkl"
    with open(local_model_file, "wb") as f:
        pickle.dump(model, f)
    upload_to_gcs(local_model_file, GCS_MODEL_PATH)

# Initialize data and model
data = load_data()
model = load_model()

@app.route("/train", methods=["POST"])
def train_model():
    global model, data
    model = NeuralNetwork()  # Redefiner modellen om nødvendigt
    train(model, data)
    save_model(model)  # Gem den opdaterede model
    return jsonify({"status": "success", "message": "Model training complete."})

@app.route("/evaluate", methods=["GET"])
def evaluate_model():
    global model, data
    results = evaluate(model, data)
    return jsonify({"status": "success", "results": results})

@app.route("/inference", methods=["POST"])
def run_inference():
    global model
    input_data = request.json.get("input", None)
    if input_data is None:
        return jsonify({"status": "error", "message": "Input data is missing."}), 400
    
    try:
        # Konverter input_data til en tensor
        import torch
        input_tensor = torch.tensor(input_data).unsqueeze(0)  # Tilføj batch-dimension

        # Sæt modellen i eval-mode og udfør inferens
        model.eval()
        with torch.no_grad():
            result = model(input_tensor)

        # Konverter resultatet til en JSON-kompatibel struktur
        result_list = result.tolist()
        return jsonify({"status": "success", "result": result_list})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

"""
ekmsempler på api kald:

curl -X POST http://localhost:5000/train

curl -X GET http://localhost:5000/evaluate

curl -X POST http://localhost:5000/inference -H "Content-Type: application/json" -d '{"input": [[[[0.1, 0.2, ...]]]]}'


"""