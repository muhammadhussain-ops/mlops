import os
import torch
import pandas as pd
import logging
import google.api_core.exceptions
from torchvision import transforms
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from google.cloud import storage
from torch.utils.data import Subset
from omegaconf import OmegaConf

config = OmegaConf.load("configs/data_config.yaml")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CelebADataset(Dataset):
    def __init__(self, bucket_name, image_folder, labels_path, transform=None):
        logging.info("Initializing CelebADataset...")
        self.bucket_name = bucket_name
        self.image_folder = image_folder
        self.labels_path = labels_path
        self.transform = transform

        try:
            client = storage.Client()  # Create client locally
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(self.labels_path)
            labels_data = blob.download_as_text()
            self.labels_df = pd.read_csv(BytesIO(labels_data.encode("utf-8")))
            self.labels_df = self.labels_df.replace([-1], [0])
            logging.info(f"Labels loaded from {self.labels_path}")
        except Exception as e:
            logging.error(f"Failed to load labels: {e}")
            raise
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        labels = self.labels_df.iloc[idx, 1:].values.astype("float32")
        blob_path = os.path.join(self.image_folder, img_name).replace("\\", "/")

        try:
            client = storage.Client()  # Create client locally
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            image_data = blob.download_as_bytes()
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except google.api_core.exceptions.NotFound:
            logging.error(f"Image not found: {blob_path}")
            raise FileNotFoundError(f"File not found in GCS: {blob_path}")

        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels

    
def create_train_loader(small_subset=config.hyperparameters.small_subset , subset_size=100):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = CelebADataset(
        bucket_name="mlops-bucket-224229-1",
        image_folder="raw/img_align_celeba/img_align_celeba",
        labels_path="raw/list_attr_celeba.csv",
        transform=transform
    )

    if small_subset:
        indices = list(range(subset_size))  # Select the first `subset_size` samples
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size= config.hyperparameters.batch_size , shuffle=True, num_workers=0)