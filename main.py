from src.mlops.data import load_data
from src.mlops.model import NeuralNetwork
from src.mlops.train import train
from src.mlops.evaluate import evaluate

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = load_data()

    # Define model
    model = NeuralNetwork()

    # Train model
    train(model, train_loader)

    # Evaluate model
    evaluate(model, test_loader)
