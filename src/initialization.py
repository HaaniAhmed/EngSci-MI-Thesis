import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# =================
# === ARGUMENTS ===
# =================

parser = argparse.ArgumentParser(description='NN Initialization')

# ==> Feed Forward Architecture Parameters
parser.add_argument('--input', default=4, type=int)
parser.add_argument('--output', default=2, type=int)
parser.add_argument('--layers', default=2, type=int)
parser.add_argument('--neurons', default=8, type=int)
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'tanh'])

# ==> Data Parameters
parser.add_argument('--samples', default=8, type=int)

# =================================
# === SUPPORT FUNCTIONS/CLASSES ===
# =================================

activation_dict = {
    "relu": F.relu,
    "tanh": torch.tanh
}

class Network(nn.Module):
    '''Source: https://stackoverflow.com/questions/58097924/how-to-create-variable-names-in-loop-for-layers-in-pytorch-neural-network'''
    def __init__(self, input_dim, output_dim, hidden_dim, layers, activation):
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()

        for _ in range(layers):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers.append(nn.Linear(current_dim, output_dim))

        self.act = activation_dict[activation]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        out = F.softmax(self.layers[-1](x), dim=1)
        return out

def init_weights(layer):
    # Function defining initialization
    if isinstance(layer, nn.Linear):
        layer.weight.data.fill_(0.00)
        layer.bias.data.fill_(0.5)

    # Later call:
    # net.apply(init_weights)


# ======================
# === MAIN FUNCTIONS ===
# ======================

def main(args):
    # Create Network
    model = Network(
        input_dim=args.input, 
        output_dim=args.output, 
        hidden_dim=args.neurons, 
        layers=args.layers, 
        activation=args.activation
    )
    # model.apply(init_weights)

    with torch.no_grad():
        input = torch.empty((args.samples, args.input)).normal_(mean=0.5,std=0.2)
        out = model(input)
        
    std = torch.std(out, dim=0)
    mean = torch.mean(out, dim=0)
    print("std: ", std)
    print("mean: ", mean)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
