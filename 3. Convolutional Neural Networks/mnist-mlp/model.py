import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.25):
        super(Network, self).__init__()

        # network parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = drop_p

        # definining model layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_layers[0])
        self.fc2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
        self.fc3 = nn.Linear(self.hidden_layers[1], self.hidden_layers[2])
        # defining output layer
        self.output_layer = nn.Linear(self.hidden_layers[2], self.output_size)

    def forward(self, x):

        x = x.view(x.shape[0], -1)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_rate)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_rate)
        x = F.dropout(F.relu(self.fc3(x)), p=self.dropout_rate)
        x = F.log_softmax(self.output_layer(x), dim=1)

        return x
