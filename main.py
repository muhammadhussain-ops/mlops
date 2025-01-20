from flask import Flask, request, jsonify
from src.mlops.data import CelebADataset  # Assuming 'download_data' handles data downloading
from src.mlops.model import NeuralNetwork  # Assuming 'load_model_weights' handles model weights
from src.mlops.train import train
from src.mlops.evaluate import evaluate

import torch

app = Flask(__name__)

# Load data and model
data =  CelebADataset(
    bucket_name="mlops-bucket-224229-1",
    image_folder="raw/img_align_celeba/img_align_celeba",
    labels_path="raw/list_attr_celeba.csv",
    transform=None
)
model = NeuralNetwork()
# load_model_weights(model)  # Loads weights into the model using the function from model.py

@app.route("/train", methods=["POST"])
def train_model():
    global model, data
    train(model, data)  # Train the model with the data
    return jsonify({"status": "success", "message": "Model training complete."})

@app.route("/evaluate", methods=["GET"])
def evaluate_model():
    global model, data
    results = evaluate(model, data)  # Evaluate the model
    return jsonify({"status": "success", "results": results})

@app.route("/inference", methods=["POST"])
def run_inference():
    global model
    input_data = request.json.get("input", None)
    if input_data is None:
        return jsonify({"status": "error", "message": "Input data is missing."}), 400
    
    try:
        # Convert input_data to a tensor
        input_tensor = torch.tensor(input_data).unsqueeze(0)  # Add batch dimension

        # Set the model to evaluation mode and perform inference
        model.eval()
        with torch.no_grad():
            result = model(input_tensor)

        # Convert the result to a JSON-compatible structure
        result_list = result.tolist()
        return jsonify({"status": "success", "result": result_list})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
