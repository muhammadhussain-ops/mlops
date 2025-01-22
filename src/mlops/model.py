import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size=(3, 128, 128), pretrained_weights=None):
        super(NeuralNetwork, self).__init__()
        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            conv_output = self.conv_stack(dummy_input)
            self.flattened_size = conv_output.numel()
        
        self.flatten = torch.nn.Flatten()
        self.fc_stack = torch.nn.Sequential(
            torch.nn.Linear(self.flattened_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 40),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.fc_stack(x)
        return x
