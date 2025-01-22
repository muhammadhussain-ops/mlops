from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.mlops.data import CelebADataset  # Assuming 'download_data' handles data downloading
from model import NeuralNetwork  # Assuming 'load_model_weights' handles model weights
from train import train
from evaluate import evaluate, accuracy
from google.cloud import storage
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

# Initialiser FastAPI
app = FastAPI()

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

# Pydantic model for inference input
class InferenceInput(BaseModel):
    input: List[List[float]]

@app.post("/train")
async def train_model():
    global model, train_loader
    try:
        train(model, train_loader)  # Train the model with the data
        local_model_path = "model_weights.pth"
        torch.save(model.state_dict(), local_model_path)

        client = storage.Client()
        bucket = client.bucket("mlops-bucket-224229-1")
        blob = bucket.blob("models/model_weights.pth")
        blob.upload_from_filename(local_model_path)

        return {"status": "success", "message": "Model training complete."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.get("/evaluate")
async def evaluate_model():
    global model, train_loader
    try:
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
        return {
            "status": "success",
            "test_loss": avg_loss,
            "test_accuracy": avg_acc
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

@app.post("/inference")
async def run_inference(data: InferenceInput):
    global model
    try:
        input_tensor = torch.tensor(data.input).unsqueeze(0)  # Add batch dimension

        model.eval()
        with torch.no_grad():
            result = model(input_tensor)

        result_list = result.tolist()
        return {"status": "success", "result": result_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "running", "message": "API is working correctly."}

# Sådan kører du appen:
# uvicorn <filnavn>:app --host 0.0.0.0 --port 8000
