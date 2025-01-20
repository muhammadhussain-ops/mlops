from flask import Flask, request, jsonify
from src.mlops.data import CelebADataset  # Assuming 'download_data' handles data downloading
from src.mlops.model import NeuralNetwork  # Assuming 'load_model_weights' handles model weights
from src.mlops.train import train
from src.mlops.evaluate import *
from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

import torch

app = Flask(__name__)

# Load data and model
# DataLoader setup
def create_data_loader():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Example resizing
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(
        bucket_name="mlops-bucket-224229-1",
        image_folder="raw/img_align_celeba/img_align_celeba",
        labels_path="raw/list_attr_celeba.csv",
        transform=transform
    )
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

train_loader = create_data_loader()
model = NeuralNetwork()
# load_model_weights(model)  # Loads weights into the model using the function from model.py

@app.route("/train", methods=["POST"])
def train_model():
    global model, train_loader
    train(model, train_loader)  # Train the model with the data
    local_model_path = "model_weights.pth"
    torch.save(model.state_dict(), local_model_path)

    client = storage.Client()
    bucket = client.bucket("mlops-bucket-224229-1")
    blob = bucket.blob("models/model_weights.pth")
    blob.upload_from_filename(local_model_path)


    return jsonify({"status": "success", "message": "Model training complete."})

@app.route("/evaluate", methods=["GET"])
def evaluate_model():
    global model, train_loader
    
    # Hent vægte fra GCS
    local_model_path = "model_weights.pth"
    bucket_name = "mlops-bucket-224229-1"
    model_path = "models/model_weights.pth"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(local_model_path)
    model.load_state_dict(torch.load(local_model_path))

    criterion = torch.nn.CrossEntropyLoss()
    avg_loss, avg_acc = evaluate(model, train_loader, criterion)
    return jsonify({
        "status": "success",
        "test_loss": avg_loss,
        "test_accuracy": avg_acc
    })


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
