from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor


class LSTMCellStack(nn.Module):
    """
    This module is a stack of LSTMCell: instead of receiving an
    entire sequence in input as torch.nn.LSTM does, it accepts one 
    element at a time.
    """

    def __init__(self, input_size: int, hidden_size: List[int]):
        super(LSTMCellStack, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = len(hidden_size)

        cells = []
        sizes = [input_size, *hidden_size]
        for l in range(self.num_layers):
            layer = nn.LSTMCell(input_size=sizes[l], hidden_size=sizes[l+1])
            cells.append(layer)
        self.cells = nn.ModuleList(cells)

    def forward(self, input: Tensor, hidden_state: Tuple[List[Tensor], List[Tensor]]=None) \
      -> Tuple[Tensor, Tuple[List[Tensor], List[Tensor]]]:
        """
        Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** list of size num_layers that contains tensors of shape 
          `(batch, hidden_size)`: tensor containing the initial hidden 
          state for each element in the batch.
        - **c_0** list of size num_layers that contains tensors of shape 
          `(batch, hidden_size)`: tensor containing the initial cell 
          state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

        Outputs: output, (h_n, c_n)
        - **output** of shape `(batch, hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM. 
        - **h_n** list of size num_layers that contains tensors of shape 
          `(batch, hidden_size)`: tensor containing the hidden state.
        - **c_n** list of size num_layers that contains tensors of shape 
          `(batch, hidden_size)`: tensor containing the cell state
        """
        in_vector = input
        if hidden_state is None:
            hidden_state = self.init_hidden(input.size(0))

        h_0, c_0 = hidden_state
        h_n_list, c_n_list = [], []
        for l, layer in enumerate(self.cells):
            state = (h_0[l], c_0[l])
            h_n, c_n = layer(in_vector, state)
            in_vector = h_n
            h_n_list.append(h_n)
            c_n_list.append(c_n)
        output = in_vector
        return output, (h_n_list, c_n_list)

    def init_hidden(self, batch_size: int) -> Tuple[List[Tensor], List[Tensor]]:
        h_0, c_0 = [], []
        for size, cell in zip(self.hidden_size, self.cells):
            dtype  = cell.weight_ih.dtype
            device = cell.weight_ih.device
            shape = (batch_size, size)
            h = torch.zeros(shape, dtype=dtype, device=device)
            h_0.append(h)
            c_0.append(h)
        return h_0, c_0


class ImgLSTMCellStack(nn.Module):

    def __init__(self, image_size: Tuple[int, int, int], hidden_size: List[int]):
        super(ImgLSTMCellStack, self).__init__()
        self.image_size = image_size
        self.input_size = image_size[0] * image_size[1] * image_size[2]

        self.lstm_cell_stack = LSTMCellStack(self.input_size, hidden_size)

    def forward(self, input: Tensor, hidden_state: Tuple[List[Tensor], List[Tensor]]=None) \
      -> Tuple[Tensor, Tuple[List[Tensor], List[Tensor]]]:

        flattened = input.view((-1, self.input_size))
        output, state = self.lstm_cell_stack(flattened, hidden_state)

        output_img = output.view((-1,) + self.image_size)
        return output_img, state

    def init_hidden(self, batch_size: int) -> Tuple[List[Tensor], List[Tensor]]:
        return self.lstm_cell_stack.init_hidden(batch_size)
