from torch.utils.data import Dataset
from src.mlops.data import create_train_loader

def test_my_dataset():
    """Test the MyDataset class."""
    data = create_train_loader()
    print(data)
    
test_my_dataset()