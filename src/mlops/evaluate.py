import torch

def accuracy(outputs, labels):
    preds = (outputs > 0.5).float()
    correct = torch.sum(preds == labels).item()
    total = labels.numel()
    return correct / total

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc += accuracy(outputs, labels)
    avg_loss = test_loss / len(test_loader)
    avg_acc = test_acc / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc