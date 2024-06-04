import torch
import torch.nn as nn
import time

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Define fully connected layers
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        
        # Combine all layers into a sequential module
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Determine the device dynamically
        device = next(self.parameters()).device

        # Move input to the device
        x = x.to(device)

        # Forward pass through the MLP
        return self.mlp(x)

# Example usage:
input_size = 784  # Size of input features
hidden_sizes = [256, 128]  # Sizes of hidden layers
output_size = 10  # Number of output classes

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device type:", device.type)

# Create an MLP and move it to the device
mlp = MLP(input_size, hidden_sizes, output_size).to(device)

# Generate random input data
input_data = torch.randn(64, input_size).to(device)  # Batch size of 64

# Record the start time
start_time = time.time()

# Forward pass
output = mlp(input_data)

# Record the end time
end_time = time.time()

# Calculate the elapsed time in milliseconds
elapsed_time_ms = (end_time - start_time) * 1000

print("Output shape:", output.shape)  # Example: torch.Size([64, 10])
print("Elapsed time for forward pass: {:.2f} milliseconds".format(elapsed_time_ms))