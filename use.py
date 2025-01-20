import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from torchvision import transforms
from src.mlops.data import CelebADataset


dataset = CelebADataset(
    bucket_name="mlops-bucket-224229-1",
    image_folder="raw/img_align_celeba/img_align_celeba",
    labels_path="raw/list_attr_celeba.csv",
    transform=None
)

# Access an item
image, labels = dataset[0]

# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()

# Print the labels
print(labels)



