### train.py
import logging
from tqdm import tqdm
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_weights(model, bucket_name, destination_blob_name):
    logging.info("Saving model weights...")
    local_file = "trained_model.pth"
    torch.save(model.state_dict(), local_file)

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    try:
        blob.upload_from_filename(local_file)
        logging.info(f"Model saved to GCS at: gs://{bucket_name}/{destination_blob_name}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")


def train(model, train_loader, criterion, optimizer, num_epochs=5):
    logging.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        # Wrap the loader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (inputs, labels) in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += (outputs.argmax(dim=1) == labels).float().mean().item()

            # Update tqdm description with current loss and accuracy
            progress_bar.set_postfix(loss=running_loss / (i + 1), accuracy=running_acc / (i + 1))
