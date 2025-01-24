### main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.mlops.data import create_train_loader  # Assuming 'download_data' handles data downloading
from src.mlops.model import NeuralNetwork  # Assuming 'load_model_weights' handles model weights
from src.mlops.train import train, save_weights
from src.mlops.evaluate import evaluate
from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from omegaconf import OmegaConf

# Initialiser FastAPI
app = FastAPI()

# Load configuration
try:
    config = OmegaConf.load("configs/hyp_config.yaml")
except FileNotFoundError:
    raise RuntimeError("Configuration file not found. Ensure 'configs/main_config.yaml' exists.")



train_loader = create_train_loader()
model = NeuralNetwork()
# load_model_weights(model)  # Uncomment if a pre-trained model is available

# Pydantic model for inference input
class InferenceInput(BaseModel):
    input: List[List[float]]

@app.post("/train")
async def train_model():
    global model, train_loader
    try:
        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparameters.lr)

        # Call train function
        train(model, train_loader, criterion, optimizer, num_epochs=config.hyperparameters.num_epochs)

        # Save model weights to Google Cloud Storage
        save_weights(model, "mlops-bucket-224229-1", "models/model.pth")

        return {"status": "success", "message": "Model training complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.get("/evaluate")
async def evaluate_model():
    global model, train_loader
    try:
        local_model_path = "model_weights.pth"
        bucket_name = "mlops-bucket-224229-1"
        model_path = "models/model.pth"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        blob.download_to_filename(local_model_path)
        model.load_state_dict(torch.load(local_model_path))

        criterion = torch.nn.CrossEntropyLoss()
        avg_loss, avg_acc = evaluate(model, train_loader, criterion)
        return {
            "status": "success",
            "test_loss": avg_loss,
            "test_accuracy": avg_acc
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

@app.post("/inference")
async def run_inference(data: dict):
    global model
    try:
        # Ekstrakter input direkte fra JSON
        input_data = data.get("input")
        if not input_data:
            raise ValueError("Input is required and cannot be empty")

        # Konverter til tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Valider dimensioner
        if input_tensor.dim() != 4 or input_tensor.shape[1:] != (3, 128, 128):  # Tilpas dimensioner
            raise ValueError(f"Invalid input dimensions: {input_tensor.shape}, expected (batch_size, 3, 128, 128)")

        # Inferens
        model.eval()
        with torch.no_grad():
            result = model(input_tensor)

        return {"status": "success", "result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "running", "message": "API is working correctly."}

# Sådan kører du appen:
# uvicorn main:app --host 0.0.0.0 --port 8000
# curl -X POST http://127.0.0.1:8000/train