import torch
from evaluate import accuracy
from data import create_train_loader
from model import NeuralNetwork
from google.cloud import storage
from torchvision import transforms
from omegaconf import OmegaConf

def save_weights(model, bucket_name, destination_blob_name):
    """
    Save the PyTorch model's weights to a GCS bucket.

    Args:
        model: The PyTorch model to save.
        bucket_name: Name of the GCS bucket.
        destination_blob_name: Path in the bucket to save the model file.
    """
    # Save the model locally
    local_file = "trained_model.pth"
    torch.save(model.state_dict(), local_file)

    # Initialize GCS client and upload the file
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the model file
    blob.upload_from_filename(local_file)
    print(f"Model saved to GCS at: gs://{bucket_name}/{destination_blob_name}")


def train(model, train_loader, criterion, optimizer, num_epochs=5):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
            if (i + 1) % 200 == 0:
                print(f"Epoch {epoch}, Batch {i + 1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}")
                running_loss = 0.0
                running_acc = 0.0
            

# Load the data, model, and hyperparameters
config = OmegaConf.load('configs/train_config.yaml')
train_loader = create_train_loader()
model = NeuralNetwork()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
train(model, train_loader, criterion, optimizer, num_epochs = config.num_epochs)
save_weights(model, "mlops-bucket-224229-1", "models/model.pth")

