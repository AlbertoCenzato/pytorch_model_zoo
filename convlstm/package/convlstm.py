from typing import Optional, Union, Tuple, List
from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor


# typedefs
HiddenState = Tuple[Tensor, Tensor]
HiddenStateStacked = Tuple[List[Tensor], List[Tensor]]


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size: Tuple[int, int], input_dim: int, hidden_dim: int, 
                 kernel_size: Tuple[int, int], bias: bool, hidden_activation=torch.sigmoid,
                 output_activation=torch.tanh):
        """
        Initialize ConvLSTM cell.
        
        Args:
         @input_size: Height and width of input tensor as (height, width).
         @input_dim: Number of channels of input tensor.
         @hidden_dim: Number of channels of hidden state.
         @kernel_size: Size of the convolutional kernel.
         @bias: Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

    def forward(self, input: Tensor, hx: HiddenState=None) -> HiddenState:
        """
        Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_dim, height, width)`: 
          tensor containing input features
        - **h_0** of shape `(batch, hidden_dim, height, width)`: 
          tensor containing the initial hidden state for each element in the batch.
        - **c_0** of shape `(batch, hidden_dim, height, width)`: 
          tensor containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

        Outputs: h_1, c_1
        - **h_1** of shape `(batch, hidden_dim, height, width)`: 
          tensor containing the next hidden state for each element in the batch
        - **c_1** of shape `(batch, hidden_dim, height, width)`: 
          tensor containing the next cell state for each element in the batch
        """
        if not hx:
            hx = self.init_hidden(input.size(0))
        
        old_h, old_cell = hx
        
        combined = torch.cat([input, old_h], dim=1)  # concatenate along channel axis
        
        gates_activations = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = gates_activations.chunk(4, dim=1)
        i = self.hidden_activation(cc_i)  # torch.sigmoid(cc_i)
        f = self.hidden_activation(cc_f)  # torch.sigmoid(cc_f)
        o = self.hidden_activation(cc_o)  # torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * old_cell + i * g
        h_next = o * self.output_activation(c_next)  # torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size: int) -> HiddenState:
        dtype  = self.conv.weight.dtype
        device = self.conv.weight.device
        shape  = (batch_size, self.hidden_dim, self.height, self.width)
        h = torch.zeros(shape, dtype=dtype).to(device)
        return (h, h)


#class ConvLSTMParams():
#
#    def __init__(self, input_size: Tuple[int, int], input_dim: int, hidden_dim: int, 
#                 kernel_size: Tuple[int, int], num_layers: int, 
#                 batch_first: bool=False, bias: bool=True, mode: str='sequence'):
#        self.input_size  = input_size
#        self.input_dim   = input_dim
#        self.hidden_dim  = hidden_dim
#        self.kernel_size = kernel_size
#        self.num_layers  = num_layers
#        self.batch_first = batch_first
#        self.bias        = bias
#        self.mode        = mode
#        self.model       = ModelType.CONVLSTM


class ConvLSTM(nn.Module):
    """
    2D convolutional LSTM model.
    
    Parameters
    ----------
    input_size: (int, int)
        Height and width of input tensor as (height, width).
    input_dim: int
        Number of channels of each hidden state
    hidden_dim: list of int
        Number of channels of hidden state.
    kernel_size: list of (int, int)
        Size of each convolutional kernel.
    num_layers: int
        number of convolutional LSTM layers
    batch_first: bool (default False)
        input tensor order: (batch_size, sequence_len, channels, height,
        width) if batch_first == True, (sequence_len, batch_size, channels,
        height, width) otherwise
    bias: bool (default True)
        Whether or not to add the bias.
    mode: either 'sequence' or 'item' (default 'sequence')
        if 'sequence' forward() accepts an input tensor of shape
        (batch_size, sequence_len, channels, height, width) and outputs a
        tensor of the same shape;
        if 'item' the model processes one sequence element at a time,
        therefore forward accepts an input tensor of shape (batch_size,
        sequence_len, channels, height, width) and outputs a tensor of the
        same shape.
        When using 'item' mode you should take care of feeing forward() with
        the output of init_hidden() when processing the first element of the
        sequence
    """
    SEQUENCE = 'sequence'
    STEP_BY_STEP = 'step-by-step'

    def __init__(self, input_size: Tuple[int, int], input_dim: int, hidden_dim: List[int], 
                 kernel_size: List[Tuple[int, int]], num_layers: int, batch_first: bool=False, 
                 bias: bool=True, mode: str='sequence'):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.mode = mode

        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers  = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        dims = [input_dim, *hidden_dim]
        for i in range(num_layers):
            layer = ConvLSTMCell(input_size=(self.height, self.width),
                                 input_dim=dims[i],
                                 hidden_dim=dims[i+1],
                                 kernel_size=self.kernel_size[i],
                                 bias=self.bias)
            cell_list.append(layer)
        self.cell_list = nn.ModuleList(cell_list)

        self.set_mode(mode)

    def set_mode(self, mode: str) -> str:
        old_mode = self.mode
        self.mode = mode
        if mode == ConvLSTM.SEQUENCE:
            self.forward = self._forward_sequence
        elif mode == ConvLSTM.STEP_BY_STEP:
            self.forward = self._forward_item
        else:
            raise ValueError("Parameter 'mode' can only be either 'sequence' or 'item'.")

        return old_mode

    def _forward_sequence(self, input: Tensor, hidden_state: HiddenStateStacked=None) \
        -> Tuple[Tensor, HiddenStateStacked]:
        """
        Inputs: input, (h_0, c_0)
        - **input** either of shape `(seq_len, batch, input_dim, height, width)`
          or `(batch, seq_len, channels, height, width)`: tensor containing 
          the features of the input sequence.
        - **h_0** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing the initial 
          hidden state for each element in the batch and for each layer in the model.
        - **c_0** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing the initial 
          cell state for each element in the batch and for each layer in the model.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

        Outputs: output, (h_n, c_n)
        - **output** of shape `(batch, seq_len, channels, height, width)`: 
          tensor containing the output features `(h_t)` from the last layer 
          of the ConvLSTM, for each t. 
        - **h_n** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing the hidden 
          state for `t = seq_len`.
        - **c_n** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing the cell 
          state for `t = seq_len`.
        """
        # (b, t, c, h, w) -> (t, b, c, h, w)
        input_seq = input.transpose(0, 1) if self.batch_first else input

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=input_seq.size(1))

        seq_len = input_seq.size(0)

        h_0, c_0 = hidden_state
        h_n_list, c_n_list = [], []

        prev_layer_output = list(torch.unbind(input_seq))  # [tensor.squeeze(1) for tensor in input_seq.split(1, dim=1)]
        for l, cell in enumerate(self.cell_list):
            state = (h_0[l], c_0[l])
            for t in range(seq_len):
                state = cell(prev_layer_output[t], state)
                prev_layer_output[t] = state[0]
            h_n_list.append(state[0])
            c_n_list.append(state[1])
            
        output = torch.stack(prev_layer_output, dim=1)

        if self.batch_first:
            return output.transpose(0, 1), (h_n_list, c_n_list)

        return output, (h_n_list, c_n_list)

    def _forward_item(self, input: Tensor, hidden_state: HiddenStateStacked) \
        -> Tuple[Tensor, HiddenStateStacked]:
        """
        Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_dim, height, width)`: 
          tensor containing the input image.
        - **h_0** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing 
          the initial hidden state for each element in the batch and for 
          each layer in the model.
        - **c_0** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing the initial 
          cell state for each element in the batch and for each layer in the model.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

        Outputs: output, (h_n, c_n)
        - **output** of shape `(batch, channels, height, width)`: 
          tensor containing the output features `(h_t)` from the last 
          layer of the LSTM. 
        - **h_n** list of size num_layers that contains tensors of shape 
          `(batch, channels, height, width)`: tensor containing the hidden 
          state for each layer.
        - **c_n** list of size num_layers that contains tensors of shape 
          (batch, channels, height, width): tensor containing the cell state
        """
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=input.size(0))

        output = input
        h_0, c_0 = hidden_state
        h_n_list, c_n_list = [], []
        for l, cell in enumerate(self.cell_list):
            h_n, c_n = cell(output, (h_0[l], c_0[l]))
            output = h_n
            h_n_list.append(h_n)
            c_n_list.append(c_n)

        return output, (h_n_list, c_n_list)

    def init_hidden(self, batch_size: int) -> HiddenStateStacked:
        h_0, c_0 = [], []
        for cell in self.cell_list:
            h, c = cell.init_hidden(batch_size)
            h_0.append(h)
            c_0.append(c)
        return (h_0, c_0)  # NOTE: using a list to allow hidden states of different sizes

    @staticmethod
    def _check_kernel_size_consistency(kernel_size: Tuple[int, int]):
        if not (isinstance(kernel_size, tuple) or
           (isinstance(kernel_size, list) and 
           all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers: Union[List[int], int]) -> List[int]:
        if not isinstance(param, list):
            param = [param] * num_layers
        return param