import numpy as np

import torch
import torch.nn as nn

from utils import ModelType
from .convlstm import ConvLSTM#, ConvLSTMParams


#class ConvLSTMChannelPoolingParams(ConvLSTMParams):
#    
#    def __init__(self, *args, **kwargs):
#        super(ConvLSTMChannelPoolingParams, self).__init__(*args, **kwargs)
#        self.model = ModelType.CH_POOLING


class ConvLSTMChannelPooling(ConvLSTM):

    def __init__(self, out_dim, *args, **kargs):
        """
            Extends the ConvLSTM model adding a 1x1 convolutional output layer
            that receives as input each channel from each ConvLSTMCell of the model.
            See Shi et al., 'Convolutional LSTM Network: A Machine Learning Approach 
            for Precipitation Nowcasting' for further explanation

            NB: return_all_layers setting is ignored and is always == False
        """
        super(ConvLSTMChannelPooling, self).__init__(*args, **kargs)
        self.output_layer = nn.Conv2d(np.sum(self.hidden_dim), out_dim, (1,1))
        

    #def save_params(self):
    #    return ConvLSTMChannelPoolingParams((self.height, self.width), self.input_dim, self.hidden_dim, 
    #                                        self.kernel_size, self.num_layers, self.batch_first, 
    #                                        self.bias, self.return_all_layers, self.mode)
       
    def _forward_item(self, input_tensor, hidden_state):
        """
            Parameters
            ----------
            input_tensor: todo 
                4-D Tensor either of shape (b, c, h, w) or (c, b, h, w)
            hidden_state: todo
                Tuple of two 4-D Tensor of shape (b, c, h, w)

            Returns
            -------
            output, hidden_state
        """
        layer_output_list = []
        cur_layer_input = input_tensor
        for index, cell in enumerate(self.cell_list):
            hidden_state[index] = cell(input=cur_layer_input,
                                       old_state=hidden_state[index])
            cur_layer_input = hidden_state[index][0]
            layer_output_list.append(cur_layer_input)

        lstms_output = torch.cat([state[0] for state in hidden_state], dim=1)
            
        last_state_list = hidden_state

        #if not self.return_all_layers:
        #    layer_output_list = layer_output_list[-1]

        output = self.output_layer(lstms_output)

        return output, last_state_list


    def _forward_sequence(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self.init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # for each layer compute the entire sequence 
        # and propagate it to the next layer
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            # compute the prediction for each item in the sequence
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        outputs = []
        # concatenate each channel output of each layer at time t in
        # one tensor and feed it to the 1x1 convolutional output layer
        for t in range(seq_len):
            lstms_output_at_t = [tensor[:,t,:,:,:] for tensor in layer_output_list]
            concat = torch.cat(lstms_output_at_t, dim=1)
            output_t = self.output_layer(concat)
            outputs.append(output_t)

        return torch.stack(outputs, dim=1), last_state_list
