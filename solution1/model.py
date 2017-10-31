import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from base.model import BaseModel


class Autoencoder(BaseModel):
    def __init__(self, layer_sizes, encoder_features):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):  # last layer has different activation
            print("layer_size", layer_sizes[i], layer_sizes[i + 1])
            fc = nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True)
            nn.init.xavier_normal(fc.weight)
            layers.append(fc)
            layers.append(nn.Tanh())

        print("Last layer_size", layer_sizes[-1], encoder_features)
        last = nn.Linear(layer_sizes[-1], encoder_features)
        nn.init.xavier_normal(last.weight)
        layers.append(last)
        layers.append(nn.Tanh())

        decoder_layers = []
        print("Reversed first layer", encoder_features, layer_sizes[-1])
        first = nn.Linear(encoder_features, layer_sizes[-1])
        nn.init.xavier_normal(first.weight)
        decoder_layers.append(first)
        decoder_layers.append(nn.Tanh())
        for i in range(1, len(layer_sizes)):
            print("reverse layer sizes", layer_sizes[-i], layer_sizes[-(i + 1)])
            layer = nn.Linear(layer_sizes[-i], layer_sizes[-(i+1)])
            nn.init.xavier_normal(layer.weight)
            decoder_layers.append(layer)
            if i != len(layer_sizes) - 1:
                decoder_layers.append(nn.Tanh())

        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    aenc = Autoencoder([52, 30, 15], 10)
    print()
