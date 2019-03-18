import math

import torch
import torch.nn as nn

import convlstmcpp

class ConvLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = convlstmcpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = [outputs[1], old_cell] + outputs[2:] + [weights]
        ctx.save_for_backward(*variables)
        
        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = convlstmcpp.backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class ConvLSTMCPPCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCPPCell, self).__init__()
        self.input_features = input_dim
        self.state_size = hidden_dim
        self.weights = nn.Parameter(torch.empty(4 * hidden_dim, input_dim + hidden_dim, kernel_size[0], kernel_size[1]))
        self.bias    = nn.Parameter(torch.empty(4 * hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return ConvLSTMFunction.apply(input, self.weights, self.bias, *state)