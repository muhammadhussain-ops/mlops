from torch.utils.data import Dataset
from mlops import data

def test_my_dataset():
    """Test the MyDataset class."""
    data = create_train_loader()
    print(data)
    
    