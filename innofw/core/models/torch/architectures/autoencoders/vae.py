import torch
from rdkit import rdBase
from torch import nn


class Encoder(nn.Module):
    """
    Encoder module.

    Attributes
    ----------
    enc_out_dim : int
        Define the size of the encoded output
    hidden_dim :  int
        Define the size of the hidden state in linear layers
    in_dim : int
        Define the input dimension of the encoder
    encode_nn : nn.Module
        Linear modules with relu activations

    Methods
    -------
    forward():
        Returns the output of the encoder.
    """
    def __init__(self, in_dim, hidden_dim, enc_out_dim):
        super(Encoder, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.enc_out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encode_nn(x)


class GRUDecoder(nn.Module):
    """
    A GRU decoder

    Attributes
    ----------
    latent_dimension : int
        Define the size of the hidden state in the gru
    gru_stack_size : int
        Define the number of layers in the gru
    gru_neurons_num : int
        age of the person
    decode_RNN : nn.Module
        RNN decoder
    Methods
    -------
    forward(z, hidden=None):
        The forward function takes in a tensor z and passes it through the decoder.
        It then returns the output of the fully connected layer, which is a 3D tensor.
        The hidden state is also returned for use in sampling.
    """
    def __init__(
        self, latent_dimension, gru_stack_size, gru_neurons_num, out_dimension
    ):
        super(GRUDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=True,
        )

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def forward(self, z, hidden=None):
        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden


class VAE(dict, torch.nn.Module):
    """
    A simple VAE class to hack innofw logic.
    """
    pass
