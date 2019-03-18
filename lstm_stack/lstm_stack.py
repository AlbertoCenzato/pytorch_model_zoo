from typing import Tuple, List

import torch
from torch import Tensor
from torch import nn


class LSTMStack(nn.Module):

    def __init__(self, input_size: int, hidden_size: List[int], batch_first: bool=True):
        super(LSTMStack, self).__init__()
        self.num_layers   = len(hidden_size)
        self.batch_first  = batch_first
        self.hidden_sizes = hidden_size
        sizes = [input_size, *hidden_size]
        layers = []
        for l in range(self.num_layers):
            lstm = nn.LSTM(input_size=sizes[l], hidden_size=sizes[l+1])
            layers.append(lstm)
        self.layers = nn.ModuleList(layers)

    def forward(self, input: Tensor, hidden_state: Tuple[List[Tensor], List[Tensor]]=None) \
        -> Tuple[Tensor, Tuple[List[Tensor], List[Tensor]]]:
        # (b, t, c, h, w) -> (t, b, c, h, w)
        input_tensor = input.transpose(0, 1) if self.batch_first else input

        if hidden_state is None:
            hidden_state = self.init_hidden(input.size(1))

        h_0, c_0 = hidden_state
        h_n, c_n = [], []
        for l, lstm in enumerate(self.layers):
            input_tensor, state = lstm(input_tensor, (h_0[l], c_0[l]))
            h_n.append(state[0])
            c_n.append(state[1])
        
        output = input_tensor.transpose(0, 1) if self.batch_first else input_tensor
        return output, (h_n, c_n)

    def init_hidden(self, batch_size: int) -> Tuple[List[Tensor], List[Tensor]]:
        h_0, c_0 = [], []
        for layer, hidden_size in zip(self.layers, self.hidden_sizes):
            dtype  = layer.weight_ih_l0.dtype
            device = layer.weight_ih_l0.device
            shape  = (1, batch_size, hidden_size)
            h = torch.zeros(shape, dtype=dtype, device=device)
            h_0.append(h)
            c_0.append(h)
        return h_0, c_0
