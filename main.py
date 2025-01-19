from src.mlops.data import CelebADataset
from src.mlops.model import NeuralNetwork
from src.mlops.train import train
from src.mlops.evaluate import evaluate

if __name__ == "__main__":
    # Load data
    data = CelebADataset()

    # Define model
    model = NeuralNetwork()

    # Train model
    train(model, data)

    # Evaluate model
    evaluate(model, data)
