import torch
from evaluate import accuracy

def train(model, train_loader, criterion = torch.nn.CrossEntropyLoss(), optimizer = torch.optim.Adam(torch.model.parameters(), lr=0.001), epoch=1):
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