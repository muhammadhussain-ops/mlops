import os
from io import BytesIO
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from google.cloud import storage
import google.api_core.exceptions
from torchvision import transforms



class CelebADataset(Dataset):
    def __init__(self, bucket_name, image_folder, labels_path, transform=None):
        self.bucket_name = bucket_name
        self.image_folder = image_folder
        self.labels_path = labels_path
        self.transform = transform

        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

        # Download and load the labels once during initialization
        blob = self.bucket.blob(self.labels_path)
        labels_data = blob.download_as_text()
        self.labels_df = pd.read_csv(BytesIO(labels_data.encode("utf-8")))  # Convert to file-like object
        self.labels_df = self.labels_df.replace([-1], [0])  # Replace -1 with 0

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        labels = self.labels_df.iloc[idx, 1:].values.astype("float32")
        blob_path = os.path.join(self.image_folder, img_name).replace("\\", "/")

        try:
            # Download image from GCS
            blob = self.bucket.blob(blob_path)
            image_data = blob.download_as_bytes()
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except google.api_core.exceptions.NotFound:
            raise FileNotFoundError(f"File not found in GCS: {blob_path}")

        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels

def create_train_loader():
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