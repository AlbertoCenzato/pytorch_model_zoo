from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor

from convlstm import ConvLSTM, HiddenState


class ConvLSTMAutoencoder(nn.Module):
    """
    This model is an implementation of the 'autoencoder' convolutional LSTM
    model proposed in 'Convolutional LSTM Network: A Machine Learning Approach 
    for Precipitation Nowcasting', Shi et al., 2015, http://arxiv.org/abs/1506.04214
    Instead of one decoding network, as proposed in the paper, this model has two
    decoding networks as in 'Unsupervised Learning of Video Representations using LSTMs',
    Srivastava et al., 2016.

    The encoding network receives a sequence of images and outputs its hidden state that
    should represent a compressed representation of the sequence. Its hidden state is then
    used as initial hidden state for the two decoding networks that use the information
    contained in it to respectively reconstruct the input sequence and to predict future 
    frames.
    """
    
    def __init__(self, input_size: Tuple[int, int], input_dim: int, 
                 hidden_dim: List[int], kernel_size: List[Tuple[int, int]], 
                 batch_first: bool=True, bias: bool=True, decoding_steps: int=-1):
        super(ConvLSTMAutoencoder, self).__init__()
        self.decoding_steps = decoding_steps
        self.input_size  = input_size
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.num_layers = len(hidden_dim)

        self.encoder = ConvLSTM(
            input_size=input_size, 
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            kernel_size=kernel_size, 
            num_layers=self.num_layers, 
            batch_first=False, 
            bias=bias, 
            mode=ConvLSTM.SEQUENCE
        )

        # reverse the order of hidden dimensions and kernels
        decoding_hidden_dim  = list(reversed(hidden_dim))
        decoding_kernel_size = list(reversed(kernel_size))
        decoding_hidden_dim .append(input_dim)  # NOTE: we need a num_of_decoding_layers = num_of_encoding_layers+1
        decoding_kernel_size.append((1,1))      #       so we add a 1x1 ConvLSTM as last decoding layer

        self.input_reconstruction = ConvLSTM(
                                        input_size=input_size, 
                                        input_dim=input_dim, 
                                        hidden_dim=decoding_hidden_dim,
                                        kernel_size=decoding_kernel_size, 
                                        num_layers=self.num_layers + 1,
                                        batch_first=False,
                                        bias=bias,
                                        mode=ConvLSTM.STEP_BY_STEP
                                    )
        self.future_prediction = ConvLSTM(
                                    input_size=input_size, 
                                    input_dim=input_dim, 
                                    hidden_dim=decoding_hidden_dim,
                                    kernel_size=decoding_kernel_size, 
                                    num_layers=self.num_layers + 1,
                                    batch_first=False,
                                    bias=bias,
                                    mode=ConvLSTM.STEP_BY_STEP
                                 )
        
    def forward(self, input_sequence: Tensor) -> Tuple[Tensor]:
        sequence = input_sequence.transpose(0,1) if self.batch_first else input_sequence  # always work in sequence-first mode
        sequence_len = sequence.size(0)

        steps = self.decoding_steps if self.decoding_steps != -1 else sequence_len

        # encode        
        _, hidden_state = self.encoder(sequence)

        last_frame = sequence[-1, :]
        h_n, c_n = hidden_state
        representation = (h_n[-1], c_n[-1])

        # decode for input reconstruction
        output_seq_recon = ConvLSTMAutoencoder._decode(self.input_reconstruction, last_frame,
                                                       representation, steps)
        
        # decode for future prediction
        output_seq_pred = ConvLSTMAutoencoder._decode(self.future_prediction, last_frame,
                                                      representation, steps)

        if self.batch_first:  # if input was batch_first restore dimension order
            reconstruction = output_seq_recon.transpose(0,1)
            prediction     = output_seq_pred .transpose(0,1)
        else:
            reconstruction = output_seq_recon
            prediction     = output_seq_pred

        return (reconstruction, prediction)

    @staticmethod
    def _decode(decoder: ConvLSTM, last_frame: Tensor, representation: HiddenState, steps: int) -> Tensor:
        decoded_sequence = []

        h_n, c_n = representation
        h_0, c_0 = decoder.init_hidden(last_frame.size(0))
        h_0[0], c_0[0] = h_n, c_n

        state = (h_0, c_0)
        output = last_frame
        for t in range(steps):
            output, state = decoder(output, state)
            decoded_sequence.append(output)

        return torch.stack(decoded_sequence, dim=0)
